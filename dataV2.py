# -*- coding: utf-8 -*-
"""
Data generator for Edge–MEC–Cloud with Poisson arrivals (per-second),
lognormal task features (parameterized by median & sigma_g), and policy-agnostic outputs.

Outputs (both CSV & Parquet):
  - episodes, agents, arrivals, tasks

Includes:
  - time-stamped arrivals
  - HOODIE-style features (b, rho, c, deadline_s/time, ...)
  - non_atomic + split_ratio
  - task class labels (time_sensitive, data_heavy, compute_heavy, normal)
  - dataset_meta.json for reproducibility

No routing/decisions here; purely input data synthesis for later algorithms.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import random, math, os, json

# -------------------------
# reproducibility
# -------------------------
GLOBAL_SEED = 42
rng_global = np.random.default_rng(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# -------------------------
# configuration dataclasses
# -------------------------
@dataclass
class EpisodeConf:
    Delta: float      # seconds per slot
    T_slots: int      # number of slots in the episode
    seed: int

@dataclass
class AgentRanges:
    # arrival rate per SECOND (will be multiplied by Delta per slot)
    lam_sec_min: float
    lam_sec_max: float
    # optional local capacities (kept as meta for later)
    f_local_min: float
    f_local_max: float
    m_local_min: float
    m_local_max: float

@dataclass
class TaskFeatureDist:
    # Lognormal parameterization via median and sigma_g (geometric std).
    b_median: float = 3.0          # MB
    b_sigma_g: float = 0.6

    rho_median: float = 1.2e9      # cycles / MB
    rho_sigma_g: float = 0.5

    L_req_min: float = 1.0
    L_req_max: float = 8.0

    mem_median: float = 64.0       # MB
    mem_sigma_g: float = 0.5

    p_deadline: float = 0.25
    deadline_min: float = 0.3      # seconds (relative)
    deadline_max: float = 1.5

    p_non_atomic: float = 0.35
    split_ratio_min: float = 0.30  # fraction of task size that CAN be split
    split_ratio_max: float = 0.80

@dataclass
class GlobalConfig:
    N_agents: int
    Episode: EpisodeConf
    AgentRanges: AgentRanges
    TaskDist: TaskFeatureDist

# -------------------------
# default sample config (1 hour, Delta=1s)
# -------------------------
HOURS = 1
DELTA = 1.0
T_SLOTS = int(HOURS * 3600 / DELTA)

CFG = GlobalConfig(
    N_agents = 18,
    Episode = EpisodeConf(Delta=DELTA, T_slots=T_SLOTS, seed=GLOBAL_SEED),
    AgentRanges = AgentRanges(
        lam_sec_min=0.02, lam_sec_max=0.8,     # arrivals per second
        f_local_min=0.8e9, f_local_max=2.4e9,  # Hz
        m_local_min=3e9,  m_local_max=8e9      # "capacity metadata" (free to interpret later)
    ),
    TaskDist = TaskFeatureDist()
)

# -------------------------
# helpers
# -------------------------
def lognormal_from_median_sigma_g(rng: np.random.Generator, median: float, sigma_g: float) -> float:
    """
    Draw from LogNormal with given median and geometric std:
      X ~ LogNormal(mu, sigma) where median = exp(mu), sigma_g = exp(sigma).
      => mu = ln(median), sigma = ln(sigma_g)
    """
    mu = math.log(median)
    sigma = math.log(sigma_g)
    return float(rng.lognormal(mean=mu, sigma=sigma))

# -------------------------
# entities
# -------------------------
@dataclass
class Agent:
    agent_id: int
    f_local: float
    m_local: float
    lam_sec: float   # Poisson rate per second (not per-slot)

def build_agents(cfg: GlobalConfig, rng: np.random.Generator) -> List[Agent]:
    agents: List[Agent] = []
    for i in range(cfg.N_agents):
        lam_sec = rng.uniform(cfg.AgentRanges.lam_sec_min, cfg.AgentRanges.lam_sec_max)
        f_loc   = rng.uniform(cfg.AgentRanges.f_local_min, cfg.AgentRanges.f_local_max)
        m_loc   = rng.uniform(cfg.AgentRanges.m_local_min, cfg.AgentRanges.m_local_max)
        agents.append(Agent(agent_id=i, f_local=f_loc, m_local=m_loc, lam_sec=lam_sec))
    return agents

# -------------------------
# task features
# -------------------------
def sample_task_features(cfg: GlobalConfig, rng: np.random.Generator) -> Dict[str, float]:
    d = cfg.TaskDist
    # Size (MB) and compute density (cycles/MB)
    b_mb   = lognormal_from_median_sigma_g(rng, d.b_median,   d.b_sigma_g)
    rho    = lognormal_from_median_sigma_g(rng, d.rho_median, d.rho_sigma_g)
    c_cyc  = b_mb * rho                              # total cycles
    L_req  = float(rng.uniform(d.L_req_min, d.L_req_max))
    mem_mb = lognormal_from_median_sigma_g(rng, d.mem_median, d.mem_sigma_g)

    modality = rng.choice(["image","video","text","sensor"], p=[0.3,0.2,0.3,0.2])

    has_deadline = int(rng.random() < d.p_deadline)
    deadline_s   = np.nan
    if has_deadline:
        deadline_s = float(rng.uniform(d.deadline_min, d.deadline_max))

    non_atomic = int(rng.random() < d.p_non_atomic)
    split_ratio = float(rng.uniform(d.split_ratio_min, d.split_ratio_max)) if non_atomic else 0.0

    return dict(
        b_mb=b_mb, rho=rho, c_cycles=c_cyc, L_req=L_req, mem_mb=mem_mb, modality=modality,
        has_deadline=has_deadline, deadline_s=deadline_s, non_atomic=non_atomic, split_ratio=split_ratio
    )

# -------------------------
# episode generator (arrivals only)
# -------------------------
def run_episode(cfg: GlobalConfig, agents: List[Agent], episode_id: int = 0) -> Dict[str, pd.DataFrame]:
    """
    Generate time-stamped arrivals and task features for one episode.
    """
    rng_local = np.random.default_rng(cfg.Episode.seed + episode_id)

    rows_episodes: List[Dict] = []
    rows_agents:   List[Dict] = [asdict(a) for a in agents]
    rows_arrivals: List[Dict] = []
    rows_tasks:    List[Dict] = []

    Delta   = cfg.Episode.Delta
    T_slots = cfg.Episode.T_slots

    task_id_counter = 0

    for t in range(T_slots):
        t_time = t * Delta
        for a in agents:
            # per-slot rate from per-second rate:
            lam_slot = a.lam_sec * Delta
            n_new = rng_local.poisson(lam=lam_slot)
            if n_new <= 0:
                continue
            for _ in range(n_new):
                feat = sample_task_features(cfg, rng_local)

                # absolute deadline time (NaN if none)
                if np.isnan(feat["deadline_s"]):
                    deadline_time = np.nan
                else:
                    deadline_time = t_time + feat["deadline_s"]

                action_space_hint = "continuous" if feat["non_atomic"] == 1 else "discrete"

                rows_arrivals.append({
                    "episode_id": episode_id,
                    "t_slot": t,
                    "t_time": t_time,
                    "agent_id": a.agent_id,
                    "task_id": task_id_counter
                })

                rows_tasks.append({
                    "episode_id": episode_id,
                    "task_id": task_id_counter,
                    "agent_id": a.agent_id,
                    "t_arrival_slot": t,
                    "t_arrival_time": t_time,
                    "b_mb": feat["b_mb"],
                    "rho_cyc_per_mb": feat["rho"],
                    "c_cycles": feat["c_cycles"],
                    "L_req": feat["L_req"],
                    "mem_mb": feat["mem_mb"],
                    "modality": feat["modality"],
                    "has_deadline": feat["has_deadline"],
                    "deadline_s": feat["deadline_s"],
                    "deadline_time": deadline_time,
                    "non_atomic": feat["non_atomic"],
                    "split_ratio": feat["split_ratio"],
                    "action_space_hint": action_space_hint
                })

                task_id_counter += 1

    rows_episodes.append({
        "episode_id": episode_id,
        "Delta": Delta,
        "T_slots": T_slots,
        "hours": T_slots * Delta / 3600.0,
        "N_agents": len(agents),
        "seed": cfg.Episode.seed + episode_id
    })

    episodes_df = pd.DataFrame(rows_episodes)
    agents_df   = pd.DataFrame(rows_agents)
    arrivals_df = pd.DataFrame(rows_arrivals)
    tasks_df    = pd.DataFrame(rows_tasks)

    # Post-process: assign task classes using quantiles computed on this batch
    tasks_df = assign_task_classes(tasks_df)

    # Optimize dtypes (optional but useful)
    tasks_df["modality"] = tasks_df["modality"].astype("category")
    tasks_df["action_space_hint"] = tasks_df["action_space_hint"].astype("category")

    return {
        "episodes": episodes_df,
        "agents":   agents_df,
        "arrivals": arrivals_df,
        "tasks":    tasks_df
    }

def assign_task_classes(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign class_label based on quantiles in the generated batch:
      - time_sensitive   : has_deadline == 1 and deadline_s <= q30(deadline_s of non-NaN)
      - data_heavy       : b_mb >= q70(b_mb)
      - compute_heavy    : c_cycles >= q70(c_cycles)
      - normal           : otherwise
    Priority: time_sensitive > data_heavy > compute_heavy > normal
    """
    df = tasks_df.copy()

    # quantiles (guard against empty subsets)
    def safe_quantile(series: pd.Series, q: float, default: float) -> float:
        series = series.dropna()
        if len(series) == 0:
            return default
        return float(series.quantile(q))

    q_deadline_30 = safe_quantile(df.loc[df["has_deadline"] == 1, "deadline_s"], 0.30, default=0.5)
    q_b_70        = safe_quantile(df["b_mb"], 0.70, default=float(df["b_mb"].median() if len(df) else 3.0))
    q_c_70        = safe_quantile(df["c_cycles"], 0.70, default=float(df["c_cycles"].median() if len(df) else 1.0))

    # flags
    is_time_sens  = (df["has_deadline"] == 1) & (df["deadline_s"] <= q_deadline_30)
    is_data_heavy = (df["b_mb"] >= q_b_70)
    is_comp_heavy = (df["c_cycles"] >= q_c_70)

    # priority order
    class_label = np.full(len(df), "normal", dtype=object)
    class_label[is_comp_heavy] = "compute_heavy"
    class_label[is_data_heavy] = "data_heavy"
    class_label[is_time_sens]  = "time_sensitive"

    df["class_label"] = pd.Categorical(class_label, categories=[
        "time_sensitive", "data_heavy", "compute_heavy", "normal"
    ], ordered=False)

    return df

# -------------------------
# save utilities
# -------------------------
def save_dataset(dfs: Dict[str, pd.DataFrame], prefix: str = "", out_dir: str = ".") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, df in dfs.items():
        csv_path = os.path.join(out_dir, f"{prefix}{name}.csv")
        pq_path  = os.path.join(out_dir, f"{prefix}{name}.parquet")
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(pq_path, index=False)
        except Exception:
            # Parquet may require pyarrow/fastparquet; ignore if unavailable
            pq_path = ""
        paths[name + "_csv"] = csv_path
        if pq_path:
            paths[name + "_parquet"] = pq_path
    return paths

def save_meta(cfg: GlobalConfig, prefix: str = "", out_dir: str = ".") -> str:
    meta = {
        "seed": cfg.Episode.seed,
        "Delta": cfg.Episode.Delta,
        "T_slots": cfg.Episode.T_slots,
        "N_agents": cfg.N_agents,
        "AgentRanges": asdict(cfg.AgentRanges),
        "TaskDist": asdict(cfg.TaskDist),
        "notes": {
            "rates_unit": "lam_sec is per-second; per-slot rate = lam_sec * Delta",
            "b_unit": "MB",
            "rho_unit": "cycles per MB",
            "c_unit": "cycles",
            "deadline_s": "relative seconds; deadline_time = t_arrival_time + deadline_s",
            "non_atomic": "1 means task can be split; split_ratio = fraction of size that can be split"
        }
    }
    path = os.path.join(out_dir, f"{prefix}dataset_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path

# -------------------------
# main
# -------------------------
def main_generate(cfg: GlobalConfig, episodes: int = 1, prefix: str = "demo_", out_dir: str = ".") -> Dict[str, str]:
    """
    Generate 'episodes' episodes reusing the same agent pool (fixed over episodes).
    """
    # build a dedicated RNG for agents to keep them consistent across episodes
    rng_agents = np.random.default_rng(cfg.Episode.seed + 10_000)
    agents = build_agents(cfg, rng_agents)

    all_paths: Dict[str, str] = {}
    for ep in range(episodes):
        dfs = run_episode(cfg, agents, episode_id=ep)
        ep_prefix = f"{prefix}ep{ep}_"
        paths = save_dataset(dfs, prefix=ep_prefix, out_dir=out_dir)
        all_paths.update({f"ep{ep}_{k}": v for k, v in paths.items()})

    meta_path = save_meta(cfg, prefix=prefix, out_dir=out_dir)
    all_paths["meta"] = meta_path
    return all_paths

if __name__ == "__main__":
    out = main_generate(CFG, episodes=1, prefix="arrivals24h_", out_dir=".")
    print(json.dumps(out, indent=2))

# -*- coding: utf-8 -*-
"""
Data generator for Edge–MEC–Cloud with Poisson arrivals (per-second),
discrete task sizes (uniform integers in [task_size_min_mb, task_size_max_mb]),
lognormal compute & memory features, and policy-agnostic outputs.

HOODIE-style timing:
  - Each episode has T slots:
      * T_decision slots for decision-making (with arrivals)
      * T_drain   slots for draining queues (NO new arrivals)
  - DRL environment can later consume this as event-driven using the time-stamps.

Supports THREE SCENARIOS (light / moderate / heavy).
For each episode we:
  - synthesize time-stamped arrivals and task features for ALL scenarios
  - save CSV + a per-episode dataset_metadata.json
  - plot distribution figures (PNG) for key variables
  - export a summary_stats.csv with quantiles/means per scenario

Directory layout (per episode):
  datasets/
    ep_000/
      light/
        episodes.csv
        agents.csv
        arrivals.csv
        tasks.csv
        summary_stats.csv
        *.png
      moderate/
        ...
      heavy/
        ...
      dataset_metadata.json   # meta info for ALL scenarios in this episode
"""

from __future__ import annotations

import os
import json
import math
import time
import random
import platform
import getpass
from dataclasses import dataclass, asdict, replace
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Load configuration from JSON
# ============================================================
CONFIG_PATH = "taskgen_config.json"  # adjust path if needed

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = json.load(f)

# Global seed and RNGs
GLOBAL_SEED = CFG["global_seed"]
rng_global = np.random.default_rng(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# Dataset-level config (used later in main)
DATASET_CFG = CFG.get("dataset", {})
DATASET_EPISODES_EACH = DATASET_CFG.get("episodes_each", 1)
DATASET_OUT_ROOT = DATASET_CFG.get("out_root", "./datasets")
DATASET_QCAP = DATASET_CFG.get("qcap", 0.99)

# Task-level config (task size range)
TASK_CFG = CFG["task"]
TASK_SIZE_MIN_MB = TASK_CFG["task_size_min_mb"]
TASK_SIZE_MAX_MB = TASK_CFG["task_size_max_mb"]


# ============================================================
# configuration dataclasses
# ============================================================
@dataclass
class EpisodeConf:
    Delta: float        # seconds per slot
    T_slots: int        # total number of slots in the episode
    T_decision: int     # number of slots with arrivals
    T_drain: int        # number of drain slots without arrivals
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
    b_median: float = 3.0          # MB (kept for metadata; actual size is discrete now)
    b_sigma_g: float = 1.5

    rho_median: float = 1.2e9      # cycles / MB
    rho_sigma_g: float = 1.5

    mem_median: float = 64.0       # MB
    mem_sigma_g: float = 1.5

    p_deadline: float = 0.25
    deadline_min: float = 0.3      # seconds (relative)
    deadline_max: float = 1.5

    p_non_atomic: float = 0.35
    split_ratio_min: float = 0.30  # fraction of task size that CAN be split
    split_ratio_max: float = 0.80

    # Optional modality probabilities (image, video, text, sensor)
    modality_probs: Optional[List[float]] = None
    modality_labels: List[str] = None

@dataclass
class GlobalConfig:
    name: str
    N_agents: int
    Episode: EpisodeConf
    AgentRanges: AgentRanges
    TaskDist: TaskFeatureDist


# ============================================================
# asserts / validation
# ============================================================
def _validate_cfg(cfg: GlobalConfig) -> None:
    assert cfg.N_agents > 0
    assert cfg.Episode.Delta > 0
    assert cfg.Episode.T_slots > 0
    assert cfg.Episode.T_decision > 0
    assert cfg.Episode.T_drain >= 0
    assert cfg.Episode.T_slots == cfg.Episode.T_decision + cfg.Episode.T_drain

    ar = cfg.AgentRanges
    assert ar.lam_sec_min >= 0 and ar.lam_sec_max >= ar.lam_sec_min
    td = cfg.TaskDist
    assert td.split_ratio_min > 0 and td.split_ratio_max <= 1.0 and td.split_ratio_max >= td.split_ratio_min
    assert td.deadline_min <= td.deadline_max
    # geometric std must be >= 1.0
    assert td.b_sigma_g >= 1.0 and td.rho_sigma_g >= 1.0 and td.mem_sigma_g >= 1.0


# ============================================================
# helpers (lognormal quantiles)
# ============================================================
_Z_TABLE = {
    0.90: 1.2815515655446004,
    0.95: 1.6448536269514722,
    0.975: 1.959963984540054,
    0.99: 2.3263478740408408,
    0.995: 2.5758293035489004,
    0.999: 3.090232306167813,
}

def _z_from_p(p: float) -> float:
    # Use small table + nearest clamp (no SciPy dependency)
    if p in _Z_TABLE:
        return _Z_TABLE[p]
    # clamp to nearest key
    return _Z_TABLE[min(_Z_TABLE.keys(), key=lambda k: abs(k - p))]

def lognormal_quantile(median: float, sigma_g: float, p: float) -> float:
    # X ~ LogNormal(mu, sigma) with median=exp(mu), sigma_g=exp(sigma)
    # quantile(p) = median * exp( z_p * ln(sigma_g) )
    z = _z_from_p(p)
    return median * math.exp(z * math.log(max(sigma_g, 1.0 + 1e-6)))

def lognormal_from_median_sigma_g(
    rng: np.random.Generator,
    median: float,
    sigma_g: float,
    qcap: float | None = 0.99
) -> float:
    """
    Draw from LogNormal with given median and geometric std:
      X ~ LogNormal(mu, sigma) where median = exp(mu), sigma_g = exp(sigma).
      => mu = ln(median), sigma = ln(sigma_g)
    """
    mu = math.log(max(median, 1e-12))
    sigma = math.log(max(sigma_g, 1.0 + 1e-6))
    x = float(rng.lognormal(mean=mu, sigma=sigma))
    if qcap is not None:
        cap = lognormal_quantile(median, sigma_g, qcap)
        x = min(x, cap)
    return x


# ============================================================
# entities
# ============================================================
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


# ============================================================
# task features
# ============================================================
def _modality_choice(rng: np.random.Generator, d: TaskFeatureDist) -> str:
    # modality labels and probabilities
    if d.modality_labels is None:
        labels = ["image", "video", "text", "sensor"]
    else:
        labels = d.modality_labels

    if d.modality_probs is None:
        probs = [0.3, 0.2, 0.3, 0.2]
    else:
        probs = d.modality_probs
        assert abs(sum(probs) - 1.0) < 1e-6 and len(probs) == len(labels)
    return rng.choice(labels, p=probs)


def sample_task_features(cfg: GlobalConfig, rng: np.random.Generator, qcap: float = 0.99) -> Dict[str, float]:
    """
    Sample a single task's features.

    - Task size (b_mb) is now discrete uniform integer in [TASK_SIZE_MIN_MB, TASK_SIZE_MAX_MB].
    - Compute density (rho) and memory (mem_mb) remain lognormal.
    """
    d = cfg.TaskDist

    # Discrete uniform task size (MB) from JSON config
    task_size_min = TASK_SIZE_MIN_MB
    task_size_max = TASK_SIZE_MAX_MB
    b_mb = int(rng.integers(task_size_min, task_size_max + 1))

    rho    = lognormal_from_median_sigma_g(rng, d.rho_median, d.rho_sigma_g, qcap=qcap)
    c      = b_mb * rho                              # total cycles
    mem_mb = lognormal_from_median_sigma_g(rng, d.mem_median, d.mem_sigma_g, qcap=qcap)
    modality = _modality_choice(rng, d)
    has_deadline = int(rng.random() < d.p_deadline)
    deadline_s   = np.nan
    if has_deadline:
        deadline_s = float(rng.uniform(d.deadline_min, d.deadline_max))
    non_atomic = int(rng.random() < d.p_non_atomic)
    split_ratio = float(rng.uniform(d.split_ratio_min, d.split_ratio_max)) if non_atomic else 0.0
    
    return dict(
        b_mb=b_mb, rho=rho, c_cycles=c, mem_mb=mem_mb, modality=modality,
        has_deadline=has_deadline, deadline_s=deadline_s, non_atomic=non_atomic, split_ratio=split_ratio
    )


# ============================================================
# episode generator (arrivals only)
# ============================================================
def run_episode(
    cfg: GlobalConfig,
    agents: List[Agent],
    episode_id: int = 0,
    qcap: float = 0.99
) -> Dict[str, pd.DataFrame]:
    """
    Generate time-stamped arrivals and task features for ONE episode of ONE scenario.

    HOODIE-style:
      - t = 0 .. T_decision-1 → arrivals via Poisson
      - t = T_decision .. T_slots-1 → NO new arrivals (drain phase)
    """
    _validate_cfg(cfg)
    rng_local = np.random.default_rng(cfg.Episode.seed + episode_id)

    rows_episodes: List[Dict] = []
    rows_agents:   List[Dict] = []
    rows_arrivals: List[Dict] = []
    rows_tasks:    List[Dict] = []

    Delta      = cfg.Episode.Delta
    T_slots    = cfg.Episode.T_slots
    T_decision = cfg.Episode.T_decision

    # agents (include scenario for easier joins)
    for a in agents:
        rows_agents.append({
            "scenario": cfg.name,
            "agent_id": a.agent_id,
            "f_local": a.f_local,
            "m_local": a.m_local,
            "lam_sec": a.lam_sec
        })

    task_id_counter = 0

    for t in range(T_slots):
        t_time = t * Delta

        # Only first T_decision slots have new arrivals (HOODIE style)
        if t >= T_decision:
            continue

        for a in agents:
            # per-slot rate from per-second rate:
            lam_slot = a.lam_sec * Delta
            n_new = rng_local.poisson(lam=lam_slot)
            if n_new <= 0:
                continue
            for _ in range(n_new):
                feat = sample_task_features(cfg, rng_local, qcap=qcap)

                # absolute deadline time (NaN if none)
                if np.isnan(feat["deadline_s"]):
                    deadline_time = np.nan
                else:
                    deadline_time = t_time + feat["deadline_s"]

                action_space_hint = "continuous" if feat["non_atomic"] == 1 else "discrete"

                rows_arrivals.append({
                    "scenario": cfg.name,
                    "episode_id": episode_id,
                    "t_slot": t,
                    "t_time": t_time,
                    "agent_id": a.agent_id,
                    "task_id": task_id_counter
                })

                rows_tasks.append({
                    "scenario": cfg.name,
                    "episode_id": episode_id,
                    "task_id": task_id_counter,
                    "agent_id": a.agent_id,
                    "t_arrival_slot": t,
                    "t_arrival_time": t_time,
                    "b_mb": feat["b_mb"],
                    "rho_cyc_per_mb": feat["rho"],
                    "c_cycles": feat["c_cycles"],
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
        "scenario": cfg.name,
        "episode_id": episode_id,
        "Delta": Delta,
        "T_slots": T_slots,
        "T_decision": T_decision,
        "T_drain": cfg.Episode.T_drain,
        "hours": T_slots * Delta / 3600.0,
        "N_agents": len(agents),
        "seed": cfg.Episode.seed + episode_id
    })

    episodes_df = pd.DataFrame(rows_episodes)
    agents_df   = pd.DataFrame(rows_agents)
    arrivals_df = pd.DataFrame(rows_arrivals)
    tasks_df    = pd.DataFrame(rows_tasks)

    # Optimize dtypes
    if len(tasks_df):
        tasks_df["modality"] = tasks_df["modality"].astype("category")
        tasks_df["action_space_hint"] = tasks_df["action_space_hint"].astype("category")
        # ints
        for col in ["episode_id", "task_id", "agent_id", "t_arrival_slot", "has_deadline", "non_atomic"]:
            if col in tasks_df:
                tasks_df[col] = tasks_df[col].astype("int32")
        # floats
        for col in ["t_arrival_time", "b_mb", "rho_cyc_per_mb", "c_cycles", "mem_mb",
                    "deadline_s", "deadline_time", "split_ratio"]:
            if col in tasks_df:
                tasks_df[col] = tasks_df[col].astype("float32")

    if len(arrivals_df):
        for col in ["episode_id", "t_slot", "agent_id", "task_id"]:
            if col in arrivals_df:
                arrivals_df[col] = arrivals_df[col].astype("int32")
        for col in ["t_time"]:
            if col in arrivals_df:
                arrivals_df[col] = arrivals_df[col].astype("float32")

    if len(agents_df):
        for col in ["agent_id"]:
            agents_df[col] = agents_df[col].astype("int32")
        for col in ["f_local", "m_local", "lam_sec"]:
            agents_df[col] = agents_df[col].astype("float64")

    if len(episodes_df):
        for col in ["episode_id", "N_agents", "T_slots", "T_decision", "T_drain"]:
            if col in episodes_df:
                episodes_df[col] = episodes_df[col].astype("int32")
        for col in ["Delta", "hours"]:
            if col in episodes_df:
                episodes_df[col] = episodes_df[col].astype("float32")

    return {
        "episodes": episodes_df,
        "agents":   agents_df,
        "arrivals": arrivals_df,
        "tasks":    tasks_df
    }


# ============================================================
# save & plotting utilities
# ============================================================
def save_dataset(dfs: Dict[str, pd.DataFrame], out_dir: str = ".") -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for name, df in dfs.items():
        csv_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        paths[name + "_csv"] = csv_path
    return paths

def _config_fingerprint(cfg: GlobalConfig) -> str:
    s = json.dumps({
        "scenario": cfg.name,
        "Episode": asdict(cfg.Episode),
        "AgentRanges": asdict(cfg.AgentRanges),
        "TaskDist": asdict(cfg.TaskDist)
    }, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

def save_episode_meta(
    cfgs: List[GlobalConfig],
    ep_dir: str,
    qcap: float = 0.99
) -> str:
    """
    Save a single dataset_metadata.json inside ep_xxx containing
    config info for ALL scenarios used in this episode.
    """
    meta = {
        "schema_version": "1.0.0",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            "python": platform.python_version(),
            "user": getpass.getuser()
        },
        "episodes_root": os.path.abspath(ep_dir),
        "notes": {
            "policy_agnostic": True,
            "queueing": "not simulated here",
            "timing": {
                "description": "HOODIE-style: T_decision slots with arrivals + T_drain slots without arrivals.",
            },
            "clipping": {
                "enabled": True,
                "method": "lognormal analytic quantile cap",
                "qcap": qcap,
                "z_table_keys": sorted(list(_Z_TABLE.keys()))
            },
            "action_space_hint": "derived from non_atomic; final decision belongs to environment"
        },
        "scenarios": []
    }

    for cfg in cfgs:
        meta["scenarios"].append({
            "name": cfg.name,
            "fingerprint": _config_fingerprint(cfg),
            "Episode": asdict(cfg.Episode),
            "N_agents": cfg.N_agents,
            "AgentRanges": asdict(cfg.AgentRanges),
            "TaskDist": asdict(cfg.TaskDist),
            "units": {
                "Delta": "seconds",
                "lam_sec": "tasks per second (per agent)",
                "per_slot_rate": "lam_sec * Delta",
                "b_mb": "MB",
                "rho_cyc_per_mb": "CPU cycles per MB",
                "c_cycles": "CPU cycles",
                "mem_mb": "MB",
                "deadline_s": "seconds (relative); deadline_time = t_arrival_time + deadline_s",
                "f_local": "Hz",
                "m_local": "MB"
            }
        })

    path = os.path.join(ep_dir, "dataset_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path

def summarize_and_plot(dfs: Dict[str, pd.DataFrame], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tasks = dfs["tasks"].copy()
    arrivals = dfs["arrivals"].copy()
    episodes = dfs["episodes"].copy()

    # ---- summary stats
    def q(series: pd.Series, p: float):
        s = series.dropna()
        return float(np.nanquantile(s, p)) if len(s) else float("nan")

    tasks_per_hour = float(len(tasks)) / float(episodes.iloc[0]["hours"]) if len(episodes) and episodes.iloc[0]["hours"] > 0 else float("nan")

    # modality distribution
    if len(tasks):
        mod_counts = tasks["modality"].value_counts(dropna=False)
        mod_dist = {str(k): int(v) for k, v in mod_counts.to_dict().items()}
    else:
        mod_dist = {}

    summary = {
        "n_tasks": [len(tasks)],
        "n_arrivals": [len(arrivals)],
        "tasks_per_hour": [tasks_per_hour],
        "b_mb_median": [q(tasks["b_mb"], 0.5)] if len(tasks) else [float("nan")],
        "b_mb_p90": [q(tasks["b_mb"], 0.9)] if len(tasks) else [float("nan")],
        "b_mb_p99": [q(tasks["b_mb"], 0.99)] if len(tasks) else [float("nan")],
        "rho_median": [q(tasks["rho_cyc_per_mb"], 0.5)] if len(tasks) else [float("nan")],
        "rho_p90": [q(tasks["rho_cyc_per_mb"], 0.9)] if len(tasks) else [float("nan")],
        "c_cycles_median": [q(tasks["c_cycles"], 0.5)] if len(tasks) else [float("nan")],
        "c_cycles_p90": [q(tasks["c_cycles"], 0.9)] if len(tasks) else [float("nan")],
        "c_cycles_p99": [q(tasks["c_cycles"], 0.99)] if len(tasks) else [float("nan")],
        "deadline_share": [float((tasks["has_deadline"] == 1).mean()) if len(tasks) else float("nan")],
        "non_atomic_share": [float((tasks["non_atomic"] == 1).mean()) if len(tasks) else float("nan")],
        "modality_counts_json": [json.dumps(mod_dist)]
    }
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, "summary_stats.csv"), index=False)

    # ---- plots (each in its own figure)
    def hist_plot(series: pd.Series, title: str, fname: str, logx: bool = False):
        plt.figure()
        s = series.dropna()
        if len(s) == 0:
            plt.title(title + " (no data)")
        else:
            plt.hist(s, bins=50)
            if logx:
                plt.xscale("log")
            plt.title(title)
            plt.xlabel(title)
            plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    if len(tasks):
        hist_plot(tasks["b_mb"],            title="Task size (MB)",                fname="hist_b_mb.png", logx=False)
        hist_plot(tasks["rho_cyc_per_mb"],  title="Compute density (cycles/MB)",   fname="hist_rho.png",  logx=True)
        hist_plot(tasks["c_cycles"],        title="Total cycles",                  fname="hist_c_cycles.png", logx=True)
        hist_plot(tasks["deadline_s"],      title="Deadline (s)",                  fname="hist_deadline_s.png", logx=False)
        hist_plot(tasks.loc[tasks["non_atomic"] == 1, "split_ratio"],
                  title="Split ratio (only non-atomic)", fname="hist_split_ratio.png", logx=False)

    # arrivals per agent
    if len(arrivals):
        per_agent = arrivals.groupby("agent_id").size()
        plt.figure()
        plt.bar(per_agent.index.astype(str), per_agent.values)
        plt.title("Arrivals per agent")
        plt.xlabel("agent_id")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "bar_arrivals_per_agent.png"), dpi=160)
        plt.close()


# ============================================================
# scenario presets (constructed from JSON)
# ============================================================
# Episode base from JSON
EPISODE_CFG = CFG["episode"]
DELTA      = EPISODE_CFG["delta"]
T_DECISION = EPISODE_CFG["t_decision"]
T_DRAIN    = EPISODE_CFG["t_drain"]
T_SLOTS    = T_DECISION + T_DRAIN

BASE_EPISODE = EpisodeConf(
    Delta=DELTA,
    T_slots=T_SLOTS,
    T_decision=T_DECISION,
    T_drain=T_DRAIN,
    seed=GLOBAL_SEED
)

# Agent base ranges (lam_sec_min/max overridden per-scenario)
AGENT_CFG = CFG["agents"]
SC_LIGHT  = CFG["scenarios"]["light"]
SC_MOD    = CFG["scenarios"]["moderate"]
SC_HEAVY  = CFG["scenarios"]["heavy"]

BASE_AGENT_RANGES = AgentRanges(
    lam_sec_min=SC_LIGHT["lam_sec_min"],   # base placeholder; overridden for each scenario
    lam_sec_max=SC_HEAVY["lam_sec_max"],   # base placeholder
    f_local_min=AGENT_CFG["f_local_min"],
    f_local_max=AGENT_CFG["f_local_max"],
    m_local_min=AGENT_CFG["m_local_min"],
    m_local_max=AGENT_CFG["m_local_max"]
)

# Base TaskFeatureDist; scenario-specific overrides applied with replace(...)
BASE_TASK_DIST = TaskFeatureDist()

SCENARIOS: List[GlobalConfig] = [
    # ---- LIGHT ----
    GlobalConfig(
        name="light",
        N_agents=SC_LIGHT["n_agents"],
        Episode=replace(BASE_EPISODE, seed=GLOBAL_SEED + 101),
        AgentRanges=replace(
            BASE_AGENT_RANGES,
            lam_sec_min=SC_LIGHT["lam_sec_min"],
            lam_sec_max=SC_LIGHT["lam_sec_max"]
        ),
        TaskDist=replace(
            BASE_TASK_DIST,
            b_median=SC_LIGHT["b_median"],
            b_sigma_g=SC_LIGHT["b_sigma_g"],
            rho_median=SC_LIGHT["rho_median"],
            rho_sigma_g=SC_LIGHT["rho_sigma_g"],
            mem_median=SC_LIGHT["mem_median"],
            mem_sigma_g=SC_LIGHT["mem_sigma_g"],
            p_deadline=SC_LIGHT["p_deadline"],
            deadline_min=SC_LIGHT["deadline_min"],
            deadline_max=SC_LIGHT["deadline_max"],
            p_non_atomic=SC_LIGHT["p_non_atomic"],
            split_ratio_min=SC_LIGHT["split_ratio_min"],
            split_ratio_max=SC_LIGHT["split_ratio_max"]
        )
    ),

    # ---- MODERATE ----
    GlobalConfig(
        name="moderate",
        N_agents=SC_MOD["n_agents"],
        Episode=replace(BASE_EPISODE, seed=GLOBAL_SEED + 202),
        AgentRanges=replace(
            BASE_AGENT_RANGES,
            lam_sec_min=SC_MOD["lam_sec_min"],
            lam_sec_max=SC_MOD["lam_sec_max"]
        ),
        TaskDist=replace(
            BASE_TASK_DIST,
            b_median=SC_MOD["b_median"],
            b_sigma_g=SC_MOD["b_sigma_g"],
            rho_median=SC_MOD["rho_median"],
            rho_sigma_g=SC_MOD["rho_sigma_g"],
            mem_median=SC_MOD["mem_median"],
            mem_sigma_g=SC_MOD["mem_sigma_g"],
            p_deadline=SC_MOD["p_deadline"],
            deadline_min=SC_MOD["deadline_min"],
            deadline_max=SC_MOD["deadline_max"],
            p_non_atomic=SC_MOD["p_non_atomic"],
            split_ratio_min=SC_MOD["split_ratio_min"],
            split_ratio_max=SC_MOD["split_ratio_max"]
        )
    ),

    # ---- HEAVY ----
    GlobalConfig(
        name="heavy",
        N_agents=SC_HEAVY["n_agents"],
        Episode=replace(BASE_EPISODE, seed=GLOBAL_SEED + 303),
        AgentRanges=replace(
            BASE_AGENT_RANGES,
            lam_sec_min=SC_HEAVY["lam_sec_min"],
            lam_sec_max=SC_HEAVY["lam_sec_max"]
        ),
        TaskDist=replace(
            BASE_TASK_DIST,
            b_median=SC_HEAVY["b_median"],
            b_sigma_g=SC_HEAVY["b_sigma_g"],
            rho_median=SC_HEAVY["rho_median"],
            rho_sigma_g=SC_HEAVY["rho_sigma_g"],
            mem_median=SC_HEAVY["mem_median"],
            mem_sigma_g=SC_HEAVY["mem_sigma_g"],
            p_deadline=SC_HEAVY["p_deadline"],
            deadline_min=SC_HEAVY["deadline_min"],
            deadline_max=SC_HEAVY["deadline_max"],
            p_non_atomic=SC_HEAVY["p_non_atomic"],
            split_ratio_min=SC_HEAVY["split_ratio_min"],
            split_ratio_max=SC_HEAVY["split_ratio_max"]
        )
    ),
]


# ============================================================
# drivers: per-scenario & all-scenarios
# ============================================================
def main_generate_for_scenario(
    cfg: GlobalConfig,
    agents: List[Agent],
    episode_id: int,
    ep_dir: str,
    qcap: float = 0.99
) -> Dict[str, str]:
    """
    Generate ONE episode for ONE scenario under:
        ep_dir/<scenario>/
    For example: datasets/ep_000/heavy/
    """
    _validate_cfg(cfg)

    scenario_dir = os.path.join(ep_dir, cfg.name)
    os.makedirs(scenario_dir, exist_ok=True)

    dfs = run_episode(cfg, agents, episode_id=episode_id, qcap=qcap)
    paths = save_dataset(dfs, out_dir=scenario_dir)
    summarize_and_plot(dfs, out_dir=scenario_dir)

    return paths

def generate_all_scenarios(
    episodes_each: int = 1,
    out_root: str = "./datasets",
    qcap: float = 0.99
) -> Dict[str, Dict[str, str]]:
    """
    Layout:
        out_root/ep_000/<scenario>/
        out_root/ep_001/<scenario>/
        ...
    And inside each ep_xxx we store dataset_metadata.json
    with info for ALL scenarios.

    NOTE:
      - To fully mimic HOODIE training (e.g., many episodes), call:
            generate_all_scenarios(episodes_each=5000, ...)
      - Default episodes_each=1 is for debugging.
    """
    results: Dict[str, Dict[str, str]] = {}

    # For stability, build agents per scenario once (same pool across episodes for that scenario)
    agents_per_scenario: Dict[str, List[Agent]] = {}
    for cfg in SCENARIOS:
        rng_agents = np.random.default_rng(cfg.Episode.seed + 10_000)
        agents_per_scenario[cfg.name] = build_agents(cfg, rng_agents)

    for ep in range(episodes_each):
        ep_name = f"ep_{ep:03d}"
        ep_dir = os.path.join(out_root, ep_name)
        os.makedirs(ep_dir, exist_ok=True)

        episode_paths: Dict[str, str] = {}
        for cfg in SCENARIOS:
            agents = agents_per_scenario[cfg.name]
            paths = main_generate_for_scenario(
                cfg=cfg,
                agents=agents,
                episode_id=ep,
                ep_dir=ep_dir,
                qcap=qcap
            )
            # keys like: heavy_episodes_csv, heavy_tasks_csv, ...
            for k, v in paths.items():
                episode_paths[f"{cfg.name}_{k}"] = v

        # per-episode metadata (ALL scenarios)
        meta_path = save_episode_meta(SCENARIOS, ep_dir=ep_dir, qcap=qcap)
        episode_paths["metadata_json"] = meta_path

        results[ep_name] = episode_paths

    return results


# ============================================================
# sanity checks
# ============================================================
def sanity_check_episode(ep_dir: str, scenario_names: List[str]) -> None:
    """
    Lightweight sanity checks for one ep_xxx:
      - basic file presence
      - tasks vs arrivals count & task_id consistency
      - agent_id consistency
      - basic value ranges (positivity, deadlines, split_ratio)
      - rough consistency between lambda and realized tasks/hour (on decision slots)
    """
    print(f"[sanity] Checking episode directory: {ep_dir}")
    meta_path = os.path.join(ep_dir, "dataset_metadata.json")
    if not os.path.isfile(meta_path):
        print(f"  [WARN] No dataset_metadata.json found in {ep_dir}")
    else:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"  [OK] Loaded metadata with {len(meta.get('scenarios', []))} scenarios")
        except Exception as e:
            print(f"  [WARN] Failed to read metadata: {e}")

    for scen in scenario_names:
        scen_dir = os.path.join(ep_dir, scen)
        if not os.path.isdir(scen_dir):
            print(f"  [WARN] Scenario dir missing: {scen_dir}")
            continue

        try:
            episodes_df = pd.read_csv(os.path.join(scen_dir, "episodes.csv"))
            agents_df   = pd.read_csv(os.path.join(scen_dir, "agents.csv"))
            arrivals_df = pd.read_csv(os.path.join(scen_dir, "arrivals.csv"))
            tasks_df    = pd.read_csv(os.path.join(scen_dir, "tasks.csv"))
        except Exception as e:
            print(f"  [WARN] Failed to load CSVs for scenario '{scen}': {e}")
            continue

        print(f"  [scenario={scen}] n_tasks={len(tasks_df)}, n_arrivals={len(arrivals_df)}, n_agents={len(agents_df)}")

        # 1) non-empty checks
        if len(tasks_df) == 0 or len(arrivals_df) == 0:
            print(f"    [WARN] Empty tasks or arrivals for scenario '{scen}'")
            continue

        # 2) tasks vs arrivals counts & unique task_ids
        if len(tasks_df) != len(arrivals_df):
            print(f"    [WARN] tasks ({len(tasks_df)}) != arrivals ({len(arrivals_df)})")

        if tasks_df["task_id"].nunique() != len(tasks_df):
            print(f"    [WARN] Duplicate task_id in tasks.csv for scenario '{scen}'")

        # 3) agent_id consistency
        agents_set = set(agents_df["agent_id"].tolist())
        arr_agents = set(arrivals_df["agent_id"].tolist())
        task_agents = set(tasks_df["agent_id"].tolist())
        if not arr_agents.issubset(agents_set):
            print(f"    [WARN] Some arrival agent_ids not in agents table for '{scen}'")
        if not task_agents.issubset(agents_set):
            print(f"    [WARN] Some task agent_ids not in agents table for '{scen}'")

        # 4) basic value ranges
        if (tasks_df["b_mb"] <= 0).any():
            print(f"    [WARN] Non-positive b_mb values in tasks for '{scen}'")
        if (tasks_df["mem_mb"] <= 0).any():
            print(f"    [WARN] Non-positive mem_mb values in tasks for '{scen}'")
        if (tasks_df["c_cycles"] <= 0).any():
            print(f"    [WARN] Non-positive c_cycles values in tasks for '{scen}'")

        # deadlines: deadline_time >= t_arrival_time when has_deadline
        with_deadline = tasks_df["has_deadline"] == 1
        if with_deadline.any():
            bad_deadlines = (tasks_df.loc[with_deadline, "deadline_time"] <
                             tasks_df.loc[with_deadline, "t_arrival_time"]).sum()
            if bad_deadlines > 0:
                print(f"    [WARN] {bad_deadlines} rows with deadline_time < t_arrival_time in '{scen}'")

        # split ratio range
        if ((tasks_df["split_ratio"] < 0) | (tasks_df["split_ratio"] > 1)).any():
            print(f"    [WARN] split_ratio out of [0,1] range in '{scen}'")
        # non_atomic=0 => split_ratio==0
        mask = tasks_df["non_atomic"] == 0
        if (~np.isclose(tasks_df.loc[mask, "split_ratio"], 0.0)).any():
            print(f"    [WARN] non_atomic==0 but split_ratio != 0 (float mismatch) in '{scen}'")

        # 5) rough lambda consistency check (using decision horizon length)
        if len(episodes_df):
            hours = float(episodes_df.iloc[0]["T_decision"] * episodes_df.iloc[0]["Delta"]) / 3600.0
            if hours > 0:
                # arrivals only exist in decision slots by construction
                tasks_per_agent = arrivals_df.groupby("agent_id").size() / hours
                merged = pd.merge(
                    agents_df[["agent_id", "lam_sec"]],
                    tasks_per_agent.rename("tasks_per_hour").reset_index(),
                    on="agent_id",
                    how="left"
                )
                merged["lam_per_hour"] = merged["lam_sec"] * 3600.0
                merged["ratio"] = merged["tasks_per_hour"] / merged["lam_per_hour"]
                too_low = (merged["ratio"] < 0.4).sum()
                too_high = (merged["ratio"] > 1.6).sum()
                if too_low + too_high > 0:
                    print(f"    [INFO] Poisson check: some agents have realized rate far from expected "
                          f"(too_low={too_low}, too_high={too_high}) in '{scen}'")

def sanity_check_root(out_root: str, scenario_names: List[str]) -> None:
    """
    Run sanity checks over all ep_* folders under out_root.
    """
    if not os.path.isdir(out_root):
        print(f"[sanity] Root directory '{out_root}' does not exist.")
        return

    episodes = sorted([d for d in os.listdir(out_root) if d.startswith("ep_")])
    if not episodes:
        print(f"[sanity] No ep_* folders found under '{out_root}'.")
        return

    for ep_name in episodes:
        ep_dir = os.path.join(out_root, ep_name)
        if os.path.isdir(ep_dir):
            sanity_check_episode(ep_dir, scenario_names)


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    # Use dataset config from JSON
    results = generate_all_scenarios(
        episodes_each=DATASET_EPISODES_EACH,
        out_root=DATASET_OUT_ROOT,
        qcap=DATASET_QCAP
    )
    print(json.dumps(results, indent=2))

    # basic sanity checks
    scenario_names = [cfg.name for cfg in SCENARIOS]
    sanity_check_root(DATASET_OUT_ROOT, scenario_names=scenario_names)

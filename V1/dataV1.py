# -*- coding: utf-8 -*-
"""
Data generator for Edge–MEC–Cloud with Poisson arrivals and policy-agnostic outputs.
Outputs CSVs: episodes, agents, arrivals, tasks, state_stream, task_snapshots, sequences.
No routing or decisions happen here. Queues are synthetic background loads for timing estimates.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import random
import math
import os

# reproducibility
SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)


# -------------------------
# configuration
# -------------------------
@dataclass
class EpisodeConf:
    Delta: float      # seconds per slot
    T_slots: int      # number of slots in the episode
    seed: int

@dataclass
class AgentRanges:
    lam_min: float
    lam_max: float
    f_local_min: float
    f_local_max: float
    m_local_min: float
    m_local_max: float

@dataclass
class TaskFeatureDist:
    lognorm_b_mean: float = 15.0
    lognorm_b_sigma: float = 0.55
    lognorm_rho_mean: float = 3.6
    lognorm_rho_sigma: float = 0.5
    L_req_min: float = 1.0
    L_req_max: float = 8.0
    lognorm_mem_mean: float = 21.0
    lognorm_mem_sigma: float = 0.3
    p_deadline: float = 0.12
    deadline_min: float = 0.3
    deadline_max: float = 1.5
    p_non_atomic: float = 0.25   # None-atomic tasks rate
    split_ratio_min: float = 0.15
    split_ratio_max: float = 0.85

@dataclass
class GlobalConfig:
    N_agents: int
    Episode: EpisodeConf
    AgentRanges: AgentRanges
    TaskDist: TaskFeatureDist

# samples
HOURS = 1
DELTA = 1
T_SLOTS_24H = int(HOURS * 3600 / DELTA)

CFG = GlobalConfig(
    N_agents = 18,
    Episode = EpisodeConf(Delta=DELTA, T_slots=T_SLOTS_24H, seed=SEED),
    AgentRanges = AgentRanges(
        lam_min=0.05, lam_max=0.8,
        f_local_min=0.8e9, f_local_max=2.4e9,
        m_local_min=3e9, m_local_max=8e9
    ),
    TaskDist = TaskFeatureDist()
)


# -------------------------
# entities
# -------------------------
@dataclass
class Agent:
    agent_id: int
    f_local: float
    m_local: float
    lam_tasks: float

def build_agents(cfg: GlobalConfig) -> List[Agent]:
    agents = []
    for i in range(cfg.N_agents):
        lam = rng.uniform(cfg.AgentRanges.lam_min, cfg.AgentRanges.lam_max)    # poisson rate for each slot
        f_loc = rng.uniform(cfg.AgentRanges.f_local_min, cfg.AgentRanges.f_local_max)
        m_loc = rng.uniform(cfg.AgentRanges.m_local_min, cfg.AgentRanges.m_local_max)
        agents.append(Agent(agent_id=i, f_local=f_loc, m_local=m_loc, lam_tasks=lam))
    return agents


# -------------------------
# task features
# -------------------------
def sample_task_features(cfg: GlobalConfig) -> Dict[str, float]:
    d = cfg.TaskDist
    b = float(rng.lognormal(mean=d.lognorm_b_mean, sigma=d.lognorm_b_sigma))
    rho = float(rng.lognormal(mean=d.lognorm_rho_mean, sigma=d.lognorm_rho_sigma))
    c = b * rho
    L_req = float(rng.uniform(d.L_req_min, d.L_req_max))
    mem_req = float(rng.lognormal(mean=d.lognorm_mem_mean, sigma=d.lognorm_mem_sigma))
    modality = rng.choice(["image", "video", "text", "sensor"], p=[0.3, 0.2, 0.3, 0.2])
    has_deadline = int(rng.random() < d.p_deadline)
    # Atomic and None-Atomic flag
    is_divisible = int(rng.random() < d.p_non_atomic)
    split_ratio = 0.0
    if is_divisible == 1:
        split_ratio = float(rng.uniform(d.split_ratio_min, d.split_ratio_max))
    return dict(
        b=b, rho=rho, c=c, L_req=L_req, mem_req=mem_req, modality=modality,
        has_deadline=has_deadline, is_divisible=is_divisible, split_ratio=split_ratio
    )

# -------------------------
# episode generator, arrivals only
# -------------------------
def run_episode(cfg: GlobalConfig, agents: List[Agent], episode_id: int = 0) -> Dict[str, pd.DataFrame]:
    rng_local = np.random.default_rng(cfg.Episode.seed + episode_id)

    rows_episodes = []
    rows_agents = [asdict(a) for a in agents]
    rows_arrivals = []
    rows_tasks = []

    Delta = cfg.Episode.Delta
    T_slots = cfg.Episode.T_slots

    task_id_counter = 0

    for t in range(T_slots):
        t_time = t * Delta
        for a in agents:
            n_new = rng_local.poisson(lam=a.lam_tasks)
            if n_new <= 0:
                continue
            for _ in range(n_new):
                feat = sample_task_features(cfg)

                deadline_time = 0.0
                if feat["has_deadline"] == 1:
                    dmin, dmax = cfg.TaskDist.deadline_min, cfg.TaskDist.deadline_max
                    deadline_time = t_time + float(rng_local.uniform(dmin, dmax))

                rows_arrivals.append({
                    "episode_id": episode_id,
                    "t_slot": t,
                    "t_time": t_time,
                    "agent_id": a.agent_id,
                    "task_id": task_id_counter
                })

                # algorithm guide
                action_space_hint = "continuous" if feat["is_divisible"] == 1 else "discrete"

                rows_tasks.append({
                    "episode_id": episode_id,
                    "task_id": task_id_counter,
                    "agent_id": a.agent_id,
                    "t_arrival_slot": t,
                    "t_arrival_time": t_time,
                    "b": feat["b"],
                    "rho": feat["rho"],
                    "c": feat["c"],
                    "L_req": feat["L_req"],
                    "mem_req": feat["mem_req"],
                    "modality": feat["modality"],
                    "has_deadline": feat["has_deadline"],
                    "deadline_time": deadline_time,
                    "is_divisible": feat["is_divisible"],
                    "split_ratio": feat["split_ratio"],
                    "action_space_hint": action_space_hint
                })

                task_id_counter += 1

    rows_episodes.append({
        "episode_id": episode_id,
        "Delta": Delta,
        "T_slots": T_slots,
        "hours": HOURS,
        "N_agents": len(agents)
    })

    episodes_df = pd.DataFrame(rows_episodes)
    agents_df   = pd.DataFrame(rows_agents)
    arrivals_df = pd.DataFrame(rows_arrivals)
    tasks_df    = pd.DataFrame(rows_tasks)

    return {
        "episodes": episodes_df,
        "agents": agents_df,
        "arrivals": arrivals_df,
        "tasks": tasks_df
    }

# -------------------------
# save csvs
# -------------------------
def save_dataset(dfs: Dict[str, pd.DataFrame], prefix: str = "") -> Dict[str, str]:
    out = {}
    base = "."
    os.makedirs(base, exist_ok=True)
    for name, df in dfs.items():
        path = os.path.join(base, f"{prefix}{name}.csv")
        df.to_csv(path, index=False)
        out[name] = path
    return out

# -------------------------
# main
# -------------------------
def main_generate(cfg: GlobalConfig, episodes: int = 1, prefix: str = "arrivals_"):
    agents = build_agents(cfg)
    all_paths = {}
    for ep in range(episodes):
        dfs = run_episode(cfg, agents, episode_id=ep)
        paths = save_dataset(dfs, prefix=f"{prefix}ep{ep}_")
        all_paths.update({f"ep{ep}_{k}": v for k, v in paths.items()})
    return all_paths

# if __name__ == "__main__":
paths = main_generate(CFG, episodes=1, prefix="demo24h_")
print(paths)

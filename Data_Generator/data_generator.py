# -*- coding: utf-8 -*-
"""
Data generator for Edge–MEC–Cloud with Poisson arrivals (per-second),
lognormal task features (parameterized by median & sigma_g), and policy-agnostic outputs.

Now supports THREE SCENARIOS (light / moderate / heavy) similar to HOODIE experiments.
For each scenario we:
  - synthesize time-stamped arrivals and task features for one or more episodes
  - save CSV + a dataset_meta.json
  - plot distribution figures (PNG) for key variables
  - export a summary_stats.csv with quantiles/means

Fixes/Improvements:
- Per-episode subfolders (avoid overwrite) → ./datasets/<scenario>/ep_XXX/
- Flexible lognormal quantile with small z-table + clamp
- Parameter asserts & unit consistency
- Richer summary (p90/p99, modality dist, tasks/hour)
- Optional modality probabilities in TaskFeatureDist
- Dtype optimization for large datasets
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import random, math, os, json
import matplotlib.pyplot as plt
import hashlib, time, platform, getpass

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

    mem_median: float = 64.0       # MB
    mem_sigma_g: float = 0.5

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

# -------------------------
# asserts / validation
# -------------------------
def _validate_cfg(cfg: GlobalConfig) -> None:
    assert cfg.N_agents > 0
    assert cfg.Episode.Delta > 0 and cfg.Episode.T_slots > 0
    ar = cfg.AgentRanges
    assert ar.lam_sec_min >= 0 and ar.lam_sec_max >= ar.lam_sec_min
    td = cfg.TaskDist
    assert td.split_ratio_min > 0 and td.split_ratio_max <= 1.0 and td.split_ratio_max >= td.split_ratio_min
    assert td.deadline_min <= td.deadline_max
    # geometric std must be >= 1.0 (clamped later, but assert for awareness)
    assert td.b_sigma_g >= 1.0 and td.rho_sigma_g >= 1.0 and td.mem_sigma_g >= 1.0

# -------------------------
# helpers
# -------------------------
_Z_TABLE = {
    0.90: 1.2815515655446004,
    0.95: 1.6448536269514722,
    0.975: 1.959963984540054,
    0.99: 2.3263478740408408,
    0.995: 2.5758293035489004,
    0.999: 3.090232306167813
}

def _z_from_p(p: float) -> float:
    # Use small table + nearest clamp (no SciPy dependency)
    if p in _Z_TABLE: return _Z_TABLE[p]
    # clamp to nearest key
    return _Z_TABLE[min(_Z_TABLE.keys(), key=lambda k: abs(k - p))]

def lognormal_quantile(median: float, sigma_g: float, p: float) -> float:
    # X ~ LogNormal(mu, sigma) with median=exp(mu), sigma_g=exp(sigma)
    # quantile(p) = median * exp( z_p * ln(sigma_g) )
    z = _z_from_p(p)
    return median * math.exp(z * math.log(max(sigma_g, 1.0 + 1e-6)))

def lognormal_from_median_sigma_g(rng, median: float, sigma_g: float, qcap: float | None = 0.99) -> float:
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
def _modality_choice(rng: np.random.Generator, d: TaskFeatureDist) -> str:
    labels = d.modality_labels or ["image","video","text","sensor"]
    if d.modality_probs is None:
        probs = [0.3, 0.2, 0.3, 0.2]
    else:
        probs = d.modality_probs
        assert abs(sum(probs) - 1.0) < 1e-6 and len(probs) == len(labels)
    return rng.choice(labels, p=probs)

def sample_task_features(cfg: GlobalConfig, rng: np.random.Generator, qcap: float = 0.99) -> Dict[str, float]:
    d = cfg.TaskDist
    b_mb   = lognormal_from_median_sigma_g(rng, d.b_median,   d.b_sigma_g,   qcap=qcap)
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

# -------------------------
# episode generator (arrivals only)
# -------------------------
def run_episode(cfg: GlobalConfig, agents: List[Agent], episode_id: int = 0, qcap: float = 0.99) -> Dict[str, pd.DataFrame]:
    """
    Generate time-stamped arrivals and task features for one episode.
    """
    _validate_cfg(cfg)
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
        "hours": T_slots * Delta / 3600.0,
        "N_agents": len(agents),
        "seed": cfg.Episode.seed + episode_id
    })

    episodes_df = pd.DataFrame(rows_episodes)
    agents_df   = pd.DataFrame(rows_agents)
    arrivals_df = pd.DataFrame(rows_arrivals)
    tasks_df    = pd.DataFrame(rows_tasks)

    # Optimize dtypes (optional but useful)
    if len(tasks_df):
        tasks_df["modality"] = tasks_df["modality"].astype("category")
        tasks_df["action_space_hint"] = tasks_df["action_space_hint"].astype("category")
        # ints
        for col in ["episode_id","task_id","agent_id","t_arrival_slot","has_deadline","non_atomic"]:
            if col in tasks_df:
                tasks_df[col] = tasks_df[col].astype("int32")
        # floats
        for col in ["t_arrival_time","b_mb","rho_cyc_per_mb","c_cycles","mem_mb","deadline_s","deadline_time","split_ratio"]:
            if col in tasks_df:
                tasks_df[col] = tasks_df[col].astype("float32")

    if len(arrivals_df):
        for col in ["episode_id","t_slot","agent_id","task_id"]:
            if col in arrivals_df:
                arrivals_df[col] = arrivals_df[col].astype("int32")
        for col in ["t_time"]:
            if col in arrivals_df:
                arrivals_df[col] = arrivals_df[col].astype("float32")

    if len(agents_df):
        for col in ["agent_id"]:
            agents_df[col] = agents_df[col].astype("int32")
        for col in ["f_local","m_local","lam_sec"]:
            agents_df[col] = agents_df[col].astype("float64")

    return {
        "episodes": episodes_df,
        "agents":   agents_df,
        "arrivals": arrivals_df,
        "tasks":    tasks_df
    }

# -------------------------
# save & plotting utilities
# -------------------------
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

def save_meta(cfg: GlobalConfig, out_dir: str = ".", qcap: float = 0.99) -> str:
    meta = {
        "schema_version": "1.0.0",
        "scenario": cfg.name,
        "seed": cfg.Episode.seed,
        "fingerprint": _config_fingerprint(cfg),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {"python": platform.python_version(), "user": getpass.getuser()},
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
        },
        "notes": {
            "policy_agnostic": True,
            "queueing": "not simulated here",
            "clipping": {
                "enabled": True,
                "method": "lognormal analytic quantile cap",
                "qcap": qcap,
                "z_table_keys": sorted(list(_Z_TABLE.keys()))
            },
            "action_space_hint": "derived from non_atomic; final decision belongs to environment"
        }
    }
    path = os.path.join(out_dir, f"dataset_meta.json")
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
        return float(np.nanquantile(series.dropna(), p)) if len(series.dropna()) else float("nan")

    tasks_per_hour = float(len(tasks)) / float(episodes.iloc[0]["hours"]) if len(episodes) and episodes.iloc[0]["hours"] > 0 else float("nan")

    # modality distribution
    mod_counts = tasks["modality"].value_counts(dropna=False) if len(tasks) else pd.Series(dtype=int)
    mod_dist = {str(k): int(v) for k, v in mod_counts.to_dict().items()}

    summary = {
        "n_tasks": [len(tasks)],
        "n_arrivals": [len(arrivals)],
        "tasks_per_hour": [tasks_per_hour],
        "b_mb_median": [q(tasks["b_mb"], 0.5)],
        "b_mb_p90": [q(tasks["b_mb"], 0.9)],
        "b_mb_p99": [q(tasks["b_mb"], 0.99)],
        "rho_median": [q(tasks["rho_cyc_per_mb"], 0.5)],
        "rho_p90": [q(tasks["rho_cyc_per_mb"], 0.9)],
        "c_cycles_median": [q(tasks["c_cycles"], 0.5)],
        "c_cycles_p90": [q(tasks["c_cycles"], 0.9)],
        "c_cycles_p99": [q(tasks["c_cycles"], 0.99)],
        "deadline_share": [float((tasks["has_deadline"]==1).mean()) if len(tasks) else float("nan")],
        "non_atomic_share": [float((tasks["non_atomic"]==1).mean()) if len(tasks) else float("nan")],
        "modality_counts_json": [json.dumps(mod_dist)]
    }
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, f"summary_stats.csv"), index=False)

    # ---- plots (each in its own figure)
    def hist_plot(series: pd.Series, title: str, fname: str, logx: bool=False):
        plt.figure()
        s = series.dropna()
        if len(s) == 0:
            plt.title(title + " (no data)")
        else:
            plt.hist(s, bins=50)
            if logx:
                plt.xscale('log')
            plt.title(title)
            plt.xlabel(title)
            plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=160)
        plt.close()

    hist_plot(tasks["b_mb"],            title="Task size (MB)",                fname=f"hist_b_mb.png", logx=True)
    hist_plot(tasks["rho_cyc_per_mb"],  title="Compute density (cycles/MB)",   fname=f"hist_rho.png",  logx=True)
    hist_plot(tasks["c_cycles"],        title="Total cycles",                  fname=f"hist_c_cycles.png", logx=True)
    hist_plot(tasks["deadline_s"],      title="Deadline (s)",                  fname=f"hist_deadline_s.png", logx=False)
    hist_plot(tasks.loc[tasks["non_atomic"]==1, "split_ratio"], title="Split ratio (only non-atomic)", fname=f"hist_split_ratio.png", logx=False)

    # arrivals per agent
    if len(arrivals):
        per_agent = arrivals.groupby("agent_id").size()
        plt.figure()
        plt.bar(per_agent.index.astype(str), per_agent.values)
        plt.title("Arrivals per agent")
        plt.xlabel("agent_id")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"bar_arrivals_per_agent.png"), dpi=160)
        plt.close()

# -------------------------
# scenario presets (light / moderate / heavy)
# -------------------------
HOURS = 1
DEFAULT_DELTA = 1.0
DEFAULT_T_SLOTS = int(HOURS * 3600 / DEFAULT_DELTA)

BASE_EPISODE = EpisodeConf(Delta=DEFAULT_DELTA, T_slots=DEFAULT_T_SLOTS, seed=GLOBAL_SEED)
BASE_AGENT_RANGES = AgentRanges(
    lam_sec_min=0.02, lam_sec_max=0.80,
    f_local_min=0.8e9, f_local_max=2.4e9,
    m_local_min=3e3,  m_local_max=8e3 # MB
)
BASE_TASK_DIST = TaskFeatureDist()

SCENARIOS: List[GlobalConfig] = [
    GlobalConfig(
        name="light",
        N_agents=18,
        Episode=replace(BASE_EPISODE, seed=GLOBAL_SEED + 101),
        AgentRanges=replace(BASE_AGENT_RANGES, lam_sec_min=0.01, lam_sec_max=0.05),
        TaskDist=replace(BASE_TASK_DIST,
            b_median=2.0,  b_sigma_g=1.55,
            rho_median=1.0e9, rho_sigma_g=1.45,
            mem_median=64.0, mem_sigma_g=1.40,
            p_deadline=0.15, deadline_min=0.8, deadline_max=2.0,
            p_non_atomic=0.25, split_ratio_min=0.25, split_ratio_max=0.75)
    ),
    GlobalConfig(
        name="moderate",
        N_agents=18,
        Episode=replace(BASE_EPISODE, seed=GLOBAL_SEED + 202),
        AgentRanges=replace(BASE_AGENT_RANGES, lam_sec_min=0.05, lam_sec_max=0.20),
        TaskDist=replace(BASE_TASK_DIST,
            b_median=3.0,  b_sigma_g=1.60,
            rho_median=1.2e9, rho_sigma_g=1.50,
            mem_median=64.0, mem_sigma_g=1.45,
            p_deadline=0.25, deadline_min=0.5, deadline_max=1.5,
            p_non_atomic=0.35, split_ratio_min=0.30, split_ratio_max=0.80)
    ),
    GlobalConfig(
        name="heavy",
        N_agents=18,
        Episode=replace(BASE_EPISODE, seed=GLOBAL_SEED + 303),
        AgentRanges=replace(BASE_AGENT_RANGES, lam_sec_min=0.20, lam_sec_max=0.80),
        TaskDist=replace(BASE_TASK_DIST,
            b_median=5.0,  b_sigma_g=1.70,
            rho_median=1.5e9, rho_sigma_g=1.55,
            mem_median=64.0, mem_sigma_g=1.50,
            p_deadline=0.35, deadline_min=0.3, deadline_max=1.0,
            p_non_atomic=0.45, split_ratio_min=0.40, split_ratio_max=0.85)
    )
]


# -------------------------
# main driver
# -------------------------
def main_generate(cfg: GlobalConfig, episodes: int = 1, out_root: str = "./datasets", qcap: float = 0.99) -> Dict[str, str]:
    """Generate 'episodes' episodes for one scenario (fixed agent pool per scenario)."""
    _validate_cfg(cfg)
    out_dir = os.path.join(out_root, cfg.name)
    os.makedirs(out_dir, exist_ok=True)

    # build agents once per scenario to keep them consistent across its episodes
    rng_agents = np.random.default_rng(cfg.Episode.seed + 10_000)
    agents = build_agents(cfg, rng_agents)

    all_paths: Dict[str, str] = {}
    for ep in range(episodes):
        ep_dir = os.path.join(out_dir, f"ep_{ep:03d}")
        os.makedirs(ep_dir, exist_ok=True)

        dfs = run_episode(cfg, agents, episode_id=ep, qcap=qcap)
        paths = save_dataset(dfs, out_dir=ep_dir)
        summarize_and_plot(dfs, out_dir=ep_dir)
        all_paths.update({f"ep{ep}_{k}": v for k, v in paths.items()})

    meta_path = save_meta(cfg, out_dir=out_dir, qcap=qcap)
    all_paths["meta"] = meta_path
    return all_paths

def generate_all_scenarios(episodes_each: int = 1, out_root: str = "./datasets", qcap: float = 0.99) -> Dict[str, Dict[str, str]]:
    results: Dict[str, Dict[str, str]] = {}
    for cfg in SCENARIOS:
        results[cfg.name] = main_generate(cfg, episodes=episodes_each, out_root=out_root, qcap=qcap)
    return results

if __name__ == "__main__":
    out = generate_all_scenarios(episodes_each=1, out_root="./datasets")
    print(json.dumps(out, indent=2))

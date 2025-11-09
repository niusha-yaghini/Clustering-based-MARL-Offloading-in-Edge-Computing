# Imports
import pandas as pd
import numpy as np
import json
import os
import math
from typing import Dict, Any, Tuple


# Step 1: Prepare data and configure the environment

# 1.1. Data Loading (Data I/O)

# Define the base directories
dataset_dir = '../Data_Generator/datasets'
topology_dir = '../Topology_Generator/topologies'

# Global container
datasets = {}

def load_datasets_from_directory(dataset_dir, verbose=True):
    """
    Build 'episode-first' structure:
    datasets = {
        "ep_000": {
            "light":   { "episodes": df, "agents": df, "arrivals": df, "tasks": df },
            "moderate":{ ... },
            "heavy":   { ... }
        },
        "ep_001": { ... },
        ...
    }
    """
    # Step 1 — detect scenarios (light/moderate/heavy/...)
    scenarios = [
        name for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name))
    ]

    # Step 2 — load per scenario and per episode (ep_XXX)
    scenario_to_episodes = {}
    for scenario in scenarios:
        scn_path = os.path.join(dataset_dir, scenario)
        ep_dirs = sorted([
            ep for ep in os.listdir(scn_path)
            if os.path.isdir(os.path.join(scn_path, ep)) and ep.startswith("ep_")
        ])
        if not ep_dirs and verbose:
            print(f"[warn] no ep_* folders found under scenario '{scenario}'")

        scenario_to_episodes[scenario] = {}
        for ep_name in ep_dirs:
            ep_path = os.path.join(scn_path, ep_name)
            try:
                scenario_to_episodes[scenario][ep_name] = {
                    "episodes": pd.read_csv(os.path.join(ep_path, "episodes.csv")),
                    "agents":   pd.read_csv(os.path.join(ep_path, "agents.csv")),
                    "arrivals": pd.read_csv(os.path.join(ep_path, "arrivals.csv")),
                    "tasks":    pd.read_csv(os.path.join(ep_path, "tasks.csv")),
                }
            except FileNotFoundError as e:
                if verbose:
                    print(f"[error] missing CSV in {ep_path}: {e}")
                continue

    # Step 3 — invert structure: episodes → scenarios
    datasets.clear()
    for scenario, eps in scenario_to_episodes.items():
        for ep_name, dfs in eps.items():
            if ep_name not in datasets:
                datasets[ep_name] = {}
            datasets[ep_name][scenario] = dfs

    # Optional summary printing
    if verbose:
        print("=== Dataset Summary (episode-first) ===")
        print(f"episodes: {len(datasets)}  | scenarios detected: {len(scenarios)} -> {sorted(scenarios)}")
        for ep_name in sorted(datasets.keys()):
            scenarios_here = sorted(datasets[ep_name].keys())
            print(f"  - {ep_name}: scenarios = {scenarios_here}")
            for scn in scenarios_here:
                dfs = datasets[ep_name][scn]
                n_ep   = len(dfs['episodes'])
                n_ag   = len(dfs['agents'])
                n_arr  = len(dfs['arrivals'])
                n_task = len(dfs['tasks'])
                print(f"      {scn:9s} → episodes:{n_ep:3d}  agents:{n_ag:4d}  arrivals:{n_arr:6d}  tasks:{n_task:6d}")
        print("=======================================")

    return datasets

# ---- load all datasets (episode-first) ----
dataset_dir = '../Data_Generator/datasets'
datasets = load_datasets_from_directory(dataset_dir, verbose=True)

# ---- choose an episode and a scenario for printing ----
# pick first available episode if you don't want to hardcode
ep_name = sorted(datasets.keys())[0] if datasets else None
scenario = "heavy"  # you can change to "light"/"moderate" if needed

if ep_name is not None and scenario in datasets[ep_name]:
    print(f"\n[info] printing from episode='{ep_name}', scenario='{scenario}'")

    print("\nagents:")
    print(datasets[ep_name][scenario]['agents'].head())
    datasets[ep_name][scenario]['agents'].info()

    print("\narrivals:")
    print(datasets[ep_name][scenario]['arrivals'].head())
    datasets[ep_name][scenario]['arrivals'].info()

    print("\nepisodes:")
    print(datasets[ep_name][scenario]['episodes'].head())
    datasets[ep_name][scenario]['episodes'].info()

    print("\ntasks:")
    print(datasets[ep_name][scenario]['tasks'].head())
    datasets[ep_name][scenario]['tasks'].info()
else:
    print("[error] no datasets found or requested scenario is missing for the chosen episode.")


# Loading topology
topologies = {}

def load_topologies_from_directory(topology_dir):
    
    for topology_name in os.listdir(topology_dir):
        topology_path = os.path.join(topology_dir, topology_name)
        
        # Only process directories
        if os.path.isdir(topology_path):
            topology_json_path = os.path.join(topology_path, "topology.json")
            meta_json_path = os.path.join(topology_path, "topology_meta.json")
            connection_matrix_csv_path = os.path.join(topology_path, "connection_matrix.csv")
            
             # --- Load JSON & CSV files ---
            topology_data = None
            meta_data = None
            with open(topology_json_path, "r", encoding="utf-8") as f:
                topology_data = json.load(f)
            with open(meta_json_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            
            # The first column is just for displaying row names, not part of the capacity matrix. 
            # So the best way is to index the first column. (index_col=0)
            connection_matrix = pd.read_csv(connection_matrix_csv_path, index_col=0)
            
            # Store the topology details and the loaded CSV
            topologies[topology_name] = {
                "topology_data": topology_data,
                "meta_data": meta_data,
                "connection_matrix": connection_matrix  # Store the loaded CSV data
            }

load_topologies_from_directory(topology_dir)

print('topology clustered -> connection_matrix')
print(topologies['clustered']['connection_matrix'].head())
topologies['clustered']['connection_matrix'].info()

print('\ntopology clustered -> topology_data')
print(topologies['clustered']['topology_data'])

print('\ntopology clustered -> meta_data')
print(topologies['clustered']['meta_data'])





# 1.2. Data Validation (episode-first aware)

# ---------- Generic helpers ----------
def _require(cond: bool, msg: str, errors: list):
    # Collect errors instead of stopping at first failure
    if not cond:
        errors.append(msg)

def _has_cols(df: pd.DataFrame, cols: list) -> bool:
    return all(c in df.columns for c in cols)

# ---------- Dataset-level validation ----------
def validate_one_dataset(dataset_key: str, ds: dict) -> list:
    """
    Validate a single dataset pack (episodes/agents/arrivals/tasks) for one (episode, scenario).
    'dataset_key' is just a label for error messages, e.g. 'ep_000/heavy'.
    """
    errors = []
    episodes = ds.get("episodes")
    agents   = ds.get("agents")
    arrivals = ds.get("arrivals")
    tasks    = ds.get("tasks")

    # 1) Presence checks
    _require(isinstance(episodes, pd.DataFrame), f"[{dataset_key}] episodes missing or not a DataFrame", errors)
    _require(isinstance(agents,   pd.DataFrame), f"[{dataset_key}] agents missing or not a DataFrame", errors)
    _require(isinstance(arrivals, pd.DataFrame), f"[{dataset_key}] arrivals missing or not a DataFrame", errors)
    _require(isinstance(tasks,    pd.DataFrame), f"[{dataset_key}] tasks missing or not a DataFrame", errors)
    if errors:
        return errors

    # 2) Required columns (based on your generators)
    req_ep_cols  = ["scenario", "episode_id", "Delta", "T_slots", "hours", "N_agents", "seed"]
    req_ag_cols  = ["agent_id", "f_local", "m_local", "lam_sec"]
    req_ar_cols  = ["scenario", "episode_id", "t_slot", "t_time", "agent_id", "task_id"]
    req_tk_cols  = [
        "scenario","episode_id","task_id","agent_id","t_arrival_slot","t_arrival_time",
        "b_mb","rho_cyc_per_mb","c_cycles","mem_mb","modality",
        "has_deadline","deadline_s","deadline_time","non_atomic","split_ratio","action_space_hint"
    ]
    _require(_has_cols(episodes, req_ep_cols), f"[{dataset_key}] episodes missing required columns", errors)
    _require(_has_cols(agents,   req_ag_cols), f"[{dataset_key}] agents missing required columns", errors)
    _require(_has_cols(arrivals, req_ar_cols), f"[{dataset_key}] arrivals missing required columns", errors)
    _require(_has_cols(tasks,    req_tk_cols), f"[{dataset_key}] tasks missing required columns", errors)
    if errors:
        return errors

    # 3) Integrity checks
    # unique task_id
    _require(tasks["task_id"].is_unique, f"[{dataset_key}] task_id is not unique", errors)

    # agent id range & count vs episodes.N_agents
    if len(agents):
        min_id = agents["agent_id"].min()
        max_id = agents["agent_id"].max()
        expected_n = int(episodes["N_agents"].iloc[0])
        _require(min_id == 0, f"[{dataset_key}] agent_id should start at 0 (got {min_id})", errors)
        _require(max_id == expected_n - 1,
                 f"[{dataset_key}] agent_id max should be N_agents-1 ({expected_n-1}), got {max_id}", errors)

    # cross refs
    valid_agents = set(agents["agent_id"].tolist())
    bad_arr_agents = set(arrivals["agent_id"]) - valid_agents
    bad_task_agents = set(tasks["agent_id"]) - valid_agents
    _require(len(bad_arr_agents) == 0, f"[{dataset_key}] arrivals contain unknown agent_id(s): {sorted(bad_arr_agents)}", errors)
    _require(len(bad_task_agents) == 0, f"[{dataset_key}] tasks contain unknown agent_id(s): {sorted(bad_task_agents)}", errors)

    # non-negative task numerics
    for col in ["b_mb","rho_cyc_per_mb","c_cycles","mem_mb"]:
        if col in tasks.columns:
            _require((tasks[col] >= 0).all(), f"[{dataset_key}] tasks.{col} has negative values", errors)

    # deadline coherence
    if "has_deadline" in tasks.columns and "deadline_s" in tasks.columns:
        bad_deadline = tasks[(tasks["has_deadline"] == 1) & ((tasks["deadline_s"].isna()) | (tasks["deadline_s"] <= 0))]
        _require(len(bad_deadline) == 0, f"[{dataset_key}] tasks with deadline have invalid deadline_s", errors)

    # single Delta / T_slots inside this (episode, scenario)
    _require(episodes["Delta"].nunique() == 1, f"[{dataset_key}] multiple Delta values in episodes", errors)
    _require(episodes["T_slots"].nunique() == 1, f"[{dataset_key}] multiple T_slots in episodes", errors)

    # arrivals inside range
    T_slots = int(episodes["T_slots"].iloc[0])
    _require(int(tasks["t_arrival_slot"].max()) <= T_slots - 1,
             f"[{dataset_key}] t_arrival_slot exceeds T_slots-1", errors)

    return errors

# ---------- Topology-level validation ----------
def validate_one_topology(topology_name: str, topo_entry: dict) -> list:
    errors = []
    topo = topo_entry.get("topology_data")
    meta = topo_entry.get("meta_data")
    Mdf  = topo_entry.get("connection_matrix")

    _require(isinstance(topo, dict), f"[{topology_name}] topology_data missing or not a dict", errors)
    _require(isinstance(meta, dict), f"[{topology_name}] meta_data missing or not a dict", errors)
    _require(isinstance(Mdf,  pd.DataFrame), f"[{topology_name}] connection_matrix CSV missing or not a DataFrame", errors)
    if errors:
        return errors

    req_keys = [
        "number_of_servers","private_cpu_capacities","public_cpu_capacities",
        "cloud_computational_capacity","connection_matrix","time_step"
    ]
    for k in req_keys:
        _require(k in topo, f"[{topology_name}] topology.json missing key: {k}", errors)
    if errors:
        return errors

    K = int(topo["number_of_servers"])
    _require(len(topo["private_cpu_capacities"]) == K, f"[{topology_name}] private_cpu_capacities length != K", errors)
    _require(len(topo["public_cpu_capacities"])  == K, f"[{topology_name}] public_cpu_capacities length != K", errors)

    Mjson = topo["connection_matrix"]
    _require(isinstance(Mjson, list) and len(Mjson) == K and (K == 0 or len(Mjson[0]) == K+1),
             f"[{topology_name}] connection_matrix in JSON must be K x (K+1)", errors)
    _require(Mdf.shape == (K, K+1), f"[{topology_name}] connection_matrix.csv shape must be K x (K+1)", errors)

    vert_csv = Mdf.iloc[:, K]
    _require((vert_csv > 0).all(), f"[{topology_name}] MEC->Cloud capacities must be > 0", errors)
    horiz_csv = Mdf.iloc[:, :K]
    _require((horiz_csv.values >= 0).all(), f"[{topology_name}] MEC<->MEC capacities contain negatives", errors)

    _require("time_step" in topo, f"[{topology_name}] missing time_step", errors)
    return errors

# ---------- Pairwise validation (dataset <-> topology) ----------
def validate_dataset_topology_pair(ep_name: str, scenario: str, ds: dict,
                                   topology_name: str, topo_entry: dict) -> list:
    """
    Validate alignment between one (episode, scenario) dataset and one topology.
    Ensures Delta == time_step and basic feasibility checks.
    """
    errors = []
    episodes = ds["episodes"]
    topo     = topo_entry["topology_data"]
    K        = int(topo["number_of_servers"])

    # Delta vs time_step
    Delta = float(episodes["Delta"].iloc[0])
    time_step = float(topo["time_step"])
    _require(abs(Delta - time_step) < 1e-9,
             f"[{ep_name}/{scenario} x {topology_name}] Delta ({Delta}) != time_step ({time_step})", errors)

    # Non-negative compute capacities
    priv = topo["private_cpu_capacities"]
    pub  = topo["public_cpu_capacities"]
    cloud = topo["cloud_computational_capacity"]
    _require(all(x >= 0 for x in priv) and all(x >= 0 for x in pub) and cloud >= 0,
             f"[{ep_name}/{scenario} x {topology_name}] negative compute capacities detected", errors)

    # Simple agent→MEC mapping (modulo) is within bounds
    N_agents = int(episodes["N_agents"].iloc[0])
    mapped = [(aid % K) for aid in range(N_agents)] if K > 0 else []
    _require(all(0 <= m < K for m in mapped) if mapped else True,
             f"[{ep_name}/{scenario} x {topology_name}] agent->MEC mapping out of bounds", errors)

    return errors

# ---------- Episode-level Delta consistency across scenarios ----------
def validate_episode_delta_consistency(ep_name: str, ep_dict: dict) -> list:
    """
    Check that all scenarios inside one episode share the same Delta and T_slots.
    """
    errors = []
    deltas = set()
    tslots = set()
    for scenario, ds in ep_dict.items():
        ep_df = ds["episodes"]
        if len(ep_df):
            deltas.add(float(ep_df["Delta"].iloc[0]))
            tslots.add(int(ep_df["T_slots"].iloc[0]))
        else:
            errors.append(f"[{ep_name}/{scenario}] episodes.csv is empty")
    if len(deltas) > 1:
        errors.append(f"[{ep_name}] multiple Delta values across scenarios: {sorted(deltas)}")
    if len(tslots) > 1:
        errors.append(f"[{ep_name}] multiple T_slots values across scenarios: {sorted(tslots)}")
    return errors

# ---------- Orchestrator over ALL datasets (episode-first) and ALL topologies ----------
def validate_everything_episode_first(datasets: dict, topologies: dict) -> dict:
    """
    'datasets' shape:
        {
          "ep_000": {
             "light":   {"episodes": df, "agents": df, "arrivals": df, "tasks": df},
             "moderate":{...},
             "heavy":   {...}
          },
          "ep_001": {...}
        }
    """
    report = {"datasets": {}, "episodes_consistency": {}, "topologies": {}, "pairs": {}}

    # 1) Validate each (episode, scenario)
    for ep_name, ep_pack in datasets.items():
        report["datasets"][ep_name] = {}
        for scenario, dpack in ep_pack.items():
            key = f"{ep_name}/{scenario}"
            errs = validate_one_dataset(key, dpack)
            report["datasets"][ep_name][scenario] = {"ok": len(errs) == 0, "errors": errs}

    # 2) Episode-level Delta/T_slots consistency across scenarios
    for ep_name, ep_pack in datasets.items():
        errs = validate_episode_delta_consistency(ep_name, ep_pack)
        report["episodes_consistency"][ep_name] = {"ok": len(errs) == 0, "errors": errs}

    # 3) Validate each topology
    for tname, tpack in topologies.items():
        errs = validate_one_topology(tname, tpack)
        report["topologies"][tname] = {"ok": len(errs) == 0, "errors": errs}

    # 4) Pairwise validation for every valid (ep, scenario) × valid topology
    for ep_name, ep_pack in datasets.items():
        for scenario, dpack in ep_pack.items():
            d_ok = report["datasets"][ep_name][scenario]["ok"]
            ep_ok = report["episodes_consistency"][ep_name]["ok"]
            for tname, tres in report["topologies"].items():
                key = f"{ep_name}/{scenario}__{tname}"
                if d_ok and ep_ok and tres["ok"]:
                    errs = validate_dataset_topology_pair(ep_name, scenario, dpack, tname, topologies[tname])
                    report["pairs"][key] = {"ok": len(errs) == 0, "errors": errs}
                else:
                    report["pairs"][key] = {"ok": False, "errors": ["Skipped due to upstream invalid dataset/episode/topology."]}

    return report

# ---------- Pretty printer ----------
def print_validation_report_episode_first(report: dict):
    print("=== DATASETS (episode/scenario) ===")
    for ep_name, ep_res in report["datasets"].items():
        for scenario, info in ep_res.items():
            status = "OK" if info["ok"] else "FAIL"
            print(f"[{status}] {ep_name}/{scenario}")
            for e in info["errors"]:
                print(f"  - {e}")

    print("\n=== EPISODE-LEVEL CONSISTENCY (Delta & T_slots) ===")
    for ep_name, info in report["episodes_consistency"].items():
        status = "OK" if info["ok"] else "FAIL"
        print(f"[{status}] {ep_name}")
        for e in info["errors"]:
            print(f"  - {e}")

    print("\n=== TOPOLOGIES ===")
    for name, info in report["topologies"].items():
        status = "OK" if info["ok"] else "FAIL"
        print(f"[{status}] {name}")
        for e in info["errors"]:
            print(f"  - {e}")

    print("\n=== (EPISODE/SCENARIO) × TOPOLOGY PAIRS ===")
    for key, info in report["pairs"].items():
        status = "OK" if info["ok"] else "FAIL"
        print(f"[{status}] {key}")
        for e in info["errors"]:
            print(f"  - {e}")

# ---- run the new validator ----
report = validate_everything_episode_first(datasets, topologies)
print_validation_report_episode_first(report)

all_ok = (
    all(info["ok"] for ep in report["datasets"].values() for info in ep.values())
    and all(info["ok"] for info in report["episodes_consistency"].values())
    and all(info["ok"] for info in report["topologies"].values())
    and all(info["ok"] for info in report["pairs"].values())
)
if not all_ok:
    raise RuntimeError("Validation failed. See printed report for details.")





# 1.3. Units Alignment (episode-first aware)

# In this section, we align units for all dataset episodes and scenarios
# and run consistency checks against all topologies.
# - Datasets: use Delta from episodes.csv; add per-slot helpers:
#     agents.f_local_slot (cycles/slot), tasks.deadline_slots (integer or NaN)
# - Topologies: capacities are already per-slot (generator multiplied by Δ);
#     we only verify time_step == Delta and non-negative capacities.

from typing import Dict, Any, Tuple

# ===== Helpers: safe getters =====
def _get_delta(episodes_df: pd.DataFrame) -> float:
    # Expect a single Delta value in episodes; take the first row
    if "Delta" not in episodes_df.columns:
        raise ValueError("episodes.csv must contain a 'Delta' column.")
    return float(episodes_df["Delta"].iloc[0])

def _ensure_numeric_positive(name: str, arr: np.ndarray):
    # Basic sanity: finite and no negatives for capacities/links
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    if (arr < 0).any():
        raise ValueError(f"{name} contains negative values.")

# ===== Alignment: per-dataset (one (episode, scenario) pack) =====
def align_units_for_dataset(dataset: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Given one dataset dict {"episodes","agents","arrivals","tasks"},
    return a copy with aligned/derived columns (per-slot helpers).
    """
    episodes = dataset["episodes"].copy()
    agents   = dataset["agents"].copy()
    arrivals = dataset["arrivals"].copy()
    tasks    = dataset["tasks"].copy()

    Delta = _get_delta(episodes)

    # Agents: add per-slot compute capacity helper (cycles/slot)
    if "f_local" not in agents.columns:
        raise ValueError("agents.csv must contain 'f_local'.")
    agents["f_local"] = agents["f_local"].astype(float)
    agents["f_local_slot"] = agents["f_local"] * Delta

    # Memory is MB; keep as float
    if "m_local" in agents.columns:
        agents["m_local"] = agents["m_local"].astype(float)

    # Tasks: ensure integer arrival slot
    if "t_arrival_slot" not in tasks.columns:
        raise ValueError("tasks.csv must contain 't_arrival_slot'.")
    tasks["t_arrival_slot"] = tasks["t_arrival_slot"].astype(int)

    # Build deadline_slots = ceil(deadline_s / Delta) when has_deadline == 1, else NaN
    if "has_deadline" in tasks.columns and "deadline_s" in tasks.columns:
        def _to_deadline_slots(row):
            if int(row["has_deadline"]) == 1 and np.isfinite(row["deadline_s"]):
                return int(math.ceil(float(row["deadline_s"]) / Delta))
            return np.nan
        tasks["deadline_slots"] = tasks.apply(_to_deadline_slots, axis=1)
        # Keep as nullable integer when possible
        try:
            tasks["deadline_slots"] = tasks["deadline_slots"].astype("Int64")
        except Exception:
            pass

    # Ensure key numeric task fields are floats
    for col in ["b_mb", "rho_cyc_per_mb", "c_cycles", "mem_mb"]:
        if col in tasks.columns:
            tasks[col] = tasks[col].astype(float)

    return {
        "episodes": episodes,
        "agents":   agents,
        "arrivals": arrivals,
        "tasks":    tasks,
    }

# ===== Verification: per-topology against a target Delta =====
def verify_topology_units(topology: Dict[str, Any], target_Delta: float) -> Tuple[bool, str]:
    """
    Ensure topology capacities are per-slot and consistent with dataset Delta:
    - time_step == target_Delta
    - shapes are valid (K x (K+1))
    - capacities non-negative
    Returns (ok, message).
    """
    # time_step check
    ts = float(topology.get("time_step", -1.0))
    if not np.isclose(ts, target_Delta, atol=1e-9):
        return (False, f"time_step mismatch (topology={ts}, dataset Delta={target_Delta})")

    # K and lists
    K = int(topology.get("number_of_servers", -1))
    priv = np.array(topology.get("private_cpu_capacities", []), dtype=float)
    pub  = np.array(topology.get("public_cpu_capacities", []), dtype=float)
    cloud = float(topology.get("cloud_computational_capacity", -1.0))
    M = np.array(topology.get("connection_matrix", []), dtype=float)

    if K <= 0:
        return (False, "Invalid 'number_of_servers' (K<=0).")
    if priv.shape[0] != K or pub.shape[0] != K:
        return (False, "private/public capacities must have length K.")
    if M.shape != (K, K+1):
        return (False, f"connection_matrix shape must be (K, K+1), got {M.shape}.")

    # Non-negative checks
    _ensure_numeric_positive("private_cpu_capacities", priv)
    _ensure_numeric_positive("public_cpu_capacities",  pub)
    if not np.isfinite(cloud) or cloud < 0:
        return (False, "cloud_computational_capacity must be non-negative and finite.")
    _ensure_numeric_positive("connection_matrix", M)

    return (True, "topology verified (per-slot, consistent).")

# ===== Batch alignment for ALL datasets (episode-first) & ALL topologies =====
def align_all_units_episode_first(
    datasets_ep_first: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    topologies_by_name: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Input 'datasets_ep_first' shape:
        {
          "ep_000": {
             "light":   {"episodes": df, "agents": df, "arrivals": df, "tasks": df},
             "moderate":{...},
             "heavy":   {...}
          },
          "ep_001": {...}
        }

    Returns:
        {
          "datasets_aligned": { ep_name: { scenario: aligned_pack, ... }, ... },
          "topology_checks":  { topo_name: { ep_name: { scenario: {ok, message} } } }
        }
    """
    out = {
        "datasets_aligned": {},
        "topology_checks":  {}
    }

    # Align datasets (episode/scenario)
    for ep_name, ep_pack in datasets_ep_first.items():
        out["datasets_aligned"][ep_name] = {}
        for scenario, ds in ep_pack.items():
            try:
                out["datasets_aligned"][ep_name][scenario] = align_units_for_dataset(ds)
            except Exception as e:
                raise RuntimeError(f"[{ep_name}/{scenario}] dataset alignment failed: {e}") from e

    # Verify each topology against each (episode, scenario) Delta
    for topo_name, topo_bundle in topologies_by_name.items():
        topo_obj = topo_bundle.get("topology_data", None)
        if not isinstance(topo_obj, dict):
            raise RuntimeError(f"[{topo_name}] 'topology_data' missing or not a dict.")
        out["topology_checks"][topo_name] = {}

        for ep_name, ep_pack in out["datasets_aligned"].items():
            out["topology_checks"][topo_name][ep_name] = {}
            for scenario, aligned in ep_pack.items():
                Delta = _get_delta(aligned["episodes"])
                ok, msg = verify_topology_units(topo_obj, Delta)
                out["topology_checks"][topo_name][ep_name][scenario] = {"ok": bool(ok), "message": msg}

    return out

# ===== Pretty printer (episode-first) =====
def print_alignment_summary_episode_first(result: Dict[str, Any]):
    # Datasets
    print("=== DATASETS (aligned, episode/scenario) ===")
    for ep_name in sorted(result["datasets_aligned"].keys()):
        for scenario in sorted(result["datasets_aligned"][ep_name].keys()):
            ds = result["datasets_aligned"][ep_name][scenario]
            Delta = _get_delta(ds["episodes"])
            n_tasks = len(ds["tasks"])
            n_agents = len(ds["agents"])
            print(f"[{ep_name}/{scenario}] Delta={Delta}  tasks={n_tasks}  agents={n_agents}")

    # Topologies
    print("\n=== TOPOLOGIES (checks vs each episode/scenario) ===")
    for topo_name, by_ep in result["topology_checks"].items():
        print(f"Topology: {topo_name}")
        for ep_name in sorted(by_ep.keys()):
            for scenario in sorted(by_ep[ep_name].keys()):
                r = by_ep[ep_name][scenario]
                flag = "OK" if r["ok"] else "FAIL"
                print(f"  - {ep_name}/{scenario}: {flag}  -> {r['message']}")

# ==== Example driver (after your loading step) ====
# datasets: episode-first dict
# topologies: { "full_mesh": {...}, "clustered": {...}, "sparse_ring": {...} }

result_align = align_all_units_episode_first(datasets_ep_first=datasets, topologies_by_name=topologies)
print_alignment_summary_episode_first(result_align)

# Access aligned data for a specific episode/scenario:
# aligned_light_ep0 = result_align["datasets_aligned"]["ep_000"]["light"]
# agents_ep0_light  = aligned_light_ep0["agents"]   # has f_local_slot
# tasks_ep0_light   = aligned_light_ep0["tasks"]    # has deadline_slots





# 1.4. Build Scenario–Topology Pairs (episode-first aware)

# We pair every (episode, scenario) dataset with every topology.
# Output shape:
# pairs_by_topology = {
#   "<topology_name>": {
#       "<ep_XXX>": {
#           "<scenario>": {
#               'scenario': <str>,
#               'episode': <str>,
#               'topology': <str>,
#               'Delta': <float>,
#               'K': <int>,
#               'dataset': {episodes, agents, arrivals, tasks},
#               'topology_data': <dict>,
#               'topology_meta_data': <dict or None>,
#               'connection_matrix_df': <pd.DataFrame>,  # shape (K, K+1)
#               'checks': {'delta_match': bool, 'message': str}
#           }, ...
#       }, ...
#   }, ...
# }

from typing import Dict, Any

def _delta_from_episodes(episodes_df: pd.DataFrame) -> float:
    """Extract a single Delta value from episodes table."""
    if "Delta" not in episodes_df.columns:
        raise ValueError("episodes.csv must contain 'Delta'.")
    return float(episodes_df["Delta"].iloc[0])

def _topology_time_step(topo_json: Dict[str, Any]) -> float:
    """Extract the topology time_step."""
    ts = topo_json.get("time_step", None)
    if ts is None:
        raise ValueError("topology.json must contain 'time_step'.")
    return float(ts)

def build_topology_episode_pairs(
    datasets_ep_first: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    topologies: Dict[str, Dict[str, Any]],
    strict_delta_match: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build pairs between every topology and every (episode, scenario) dataset.
    If strict_delta_match is True, any mismatch between dataset Delta and topology time_step raises an error.
    """
    pairs_by_topology: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Iterate topologies first (topology-centric)
    for topo_name, topo_bundle in topologies.items():
        topo_data = topo_bundle.get("topology_data", None)
        meta_data = topo_bundle.get("meta_data", None)
        cm_df     = topo_bundle.get("connection_matrix", None)

        if not isinstance(topo_data, dict):
            raise ValueError(f"[{topo_name}] topology_data missing or not a dict.")
        if cm_df is None:
            raise ValueError(f"[{topo_name}] connection_matrix DataFrame is missing.")

        # Validate K and connection matrix shape
        K = int(topo_data.get("number_of_servers", -1))
        if K <= 0:
            raise ValueError(f"[{topo_name}] invalid 'number_of_servers' in topology.json")
        if not (cm_df.shape[0] == K and cm_df.shape[1] == K + 1):
            raise ValueError(
                f"[{topo_name}] connection_matrix shape must be (K, K+1); got {cm_df.shape}"
            )

        topo_ts = _topology_time_step(topo_data)

        # Prepare container for this topology
        if topo_name not in pairs_by_topology:
            pairs_by_topology[topo_name] = {}

        # Compare with every (episode, scenario)
        for ep_name, scenarios in datasets_ep_first.items():
            if ep_name not in pairs_by_topology[topo_name]:
                pairs_by_topology[topo_name][ep_name] = {}

            for scen_name, ds in scenarios.items():
                ds_Delta = _delta_from_episodes(ds["episodes"])
                delta_ok = bool(np.isclose(ds_Delta, topo_ts, atol=1e-12))
                msg = "OK" if delta_ok else (
                    f"time_step mismatch (dataset Delta={ds_Delta}, topology time_step={topo_ts})"
                )
                if (not delta_ok) and strict_delta_match:
                    raise ValueError(f"[{topo_name} × {ep_name}/{scen_name}] {msg}")

                # Store bundle
                pairs_by_topology[topo_name][ep_name][scen_name] = {
                    "scenario": scen_name,
                    "episode": ep_name,
                    "topology": topo_name,
                    "Delta": ds_Delta,
                    "K": K,
                    "dataset": ds,
                    "topology_data": topo_data,
                    "topology_meta_data": meta_data,
                    "connection_matrix_df": cm_df,
                    "checks": {"delta_match": delta_ok, "message": msg}
                }

    return pairs_by_topology

def print_pairs_summary_topology_first_ep(
    pairs_by_topology: Dict[str, Dict[str, Dict[str, Any]]]
) -> None:
    """Pretty-print summary as topology → episode → scenario."""
    print("=== TOPOLOGY × EPISODE × SCENARIO ===")
    for topo_name, by_ep in pairs_by_topology.items():
        print(f"[TOPOLOGY] {topo_name}")
        for ep_name in sorted(by_ep.keys()):
            print(f"  ├─ Episode: {ep_name}")
            scen_map = by_ep[ep_name]
            for scen_name in sorted(scen_map.keys()):
                bundle = scen_map[scen_name]
                flag  = "OK" if bundle["checks"]["delta_match"] else "FAIL"
                K     = bundle["K"]
                Delta = bundle["Delta"]
                msg   = bundle["checks"]["message"]
                print(f"  │    - [{flag}] {scen_name:9s} | K={K:2d}  Δ={Delta:g}  -> {msg}")

# --- Example driver (with your current variables) ---
pairs_by_topology = build_topology_episode_pairs(
    datasets_ep_first=datasets,   # episode-first dict you already built
    topologies=topologies,
    strict_delta_match=True
)

print_pairs_summary_topology_first_ep(pairs_by_topology)

# Access examples:
# pairs_by_topology["full_mesh"]["ep_000"]["light"]["dataset"]["tasks"]
# pairs_by_topology["clustered"]["ep_000"]["heavy"]["connection_matrix_df"]





# 1.5. Agent→MEC mapping (for all pairs) — 3-level: topology → episode → scenario

def assign_agents_to_mecs(pairs_by_topology):
    """
    Adds agent→MEC mapping to each (topology / ep / scenario) bundle.
    - Rule: mec_id = agent_id % K
    - Writes:
        bundle["agent_to_mec"]                  (pd.Series, index=agent_id)
        bundle["dataset"]["agents"]["mec_id"]   (added column)
    """
    for topo_name, by_ep in pairs_by_topology.items():
        for ep_name, by_scen in by_ep.items():
            for scen_name, bundle in by_scen.items():

                ds = bundle["dataset"]
                agents = ds["agents"].copy()
                K = int(bundle["K"])

                if "agent_id" not in agents.columns:
                    raise ValueError(f"[{topo_name}/{ep_name}/{scen_name}] agents.csv missing 'agent_id'.")

                # Ensure agent_id is contiguous & sorted (0..N_agents-1)
                agents = agents.sort_values("agent_id").reset_index(drop=True)
                expected_n = int(bundle["dataset"]["episodes"]["N_agents"].iloc[0])
                if agents["agent_id"].min() != 0 or agents["agent_id"].max() != expected_n - 1:
                    raise ValueError(f"[{topo_name}/{ep_name}/{scen_name}] agent_id range not 0..N_agents-1.")

                # Mapping
                mec_ids = (agents["agent_id"].astype(int) % K).astype(int)
                agents["mec_id"] = mec_ids

                # Store: dataset copy + Series with index=agent_id
                ds["agents"] = agents
                bundle["agent_to_mec"] = pd.Series(
                    data=mec_ids.values,
                    index=agents["agent_id"].values,
                    name="mec_id"
                )

    return pairs_by_topology

# Apply mapping
pairs_by_topology = assign_agents_to_mecs(pairs_by_topology)

# Quick sanity peek
pairs_by_topology["clustered"]["ep_000"]["heavy"]["dataset"]["agents"].head()


def _extract_core_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    required = ["dataset", "topology_data", "connection_matrix_df", "Delta", "K"]
    for k in required:
        if k not in bundle:
            raise ValueError(f"Bundle missing required key: '{k}'")

    ds   = bundle["dataset"]
    topo = bundle["topology_data"]
    Mdf  = bundle["connection_matrix_df"]

    private_cpu = np.asarray(topo["private_cpu_capacities"], dtype=float)
    public_cpu  = np.asarray(topo["public_cpu_capacities"],  dtype=float)
    cloud_cpu   = float(topo["cloud_computational_capacity"])
    M           = Mdf.to_numpy(dtype=float)  # (K, K+1), last col MEC→Cloud (MB/slot)

    return dict(
        Delta=float(bundle["Delta"]),
        K=int(bundle["K"]),
        episodes=ds["episodes"],
        agents=ds["agents"],
        arrivals=ds["arrivals"],
        tasks=ds["tasks"],
        private_cpu=private_cpu,
        public_cpu=public_cpu,
        cloud_cpu=cloud_cpu,
        connection_matrix=M,
        topology_type=topo.get("topology_type", "unknown"),
    )

def _build_default_queues(K: int) -> Dict[str, np.ndarray]:
    return {
        "mec_local_cycles":   np.zeros(K, dtype=float),
        "mec_public_cycles":  np.zeros(K, dtype=float),
        "mec_bytes_in_transit": np.zeros(K, dtype=float),
        "cloud_cycles":       np.array([0.0], dtype=float),
    }

def _derive_action_space() -> Dict[str, Any]:
    return {"type": "discrete", "n": 3, "labels": {0: "LOCAL", 1: "MEC", 2: "CLOUD"}}

def _derive_state_spec(K: int) -> Dict[str, Any]:
    return {
        "components": {
            "queues": {
                "mec_local_cycles":  {"shape": (K,),   "dtype": "float"},
                "mec_public_cycles": {"shape": (K,),   "dtype": "float"},
                "cloud_cycles":      {"shape": (1,),   "dtype": "float"},
            },
            "links": {
                "connection_matrix": {"shape": (K, K+1), "dtype": "float"},
            },
            "capacities": {
                "private_cpu": {"shape": (K,), "dtype": "float"},
                "public_cpu":  {"shape": (K,), "dtype": "float"},
                "cloud_cpu":   {"shape": (1,), "dtype": "float"},
            }
        },
        "note": "Declarative spec; tensor assembly happens in the Env at each step."
    }

def build_env_config_for_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    core = _extract_core_from_bundle(bundle)

    if "agent_to_mec" not in bundle:
        raise ValueError("Bundle has no 'agent_to_mec' mapping. Run Stage 5 first.")

    agent_to_mec = bundle["agent_to_mec"]
    if isinstance(agent_to_mec, pd.Series):
        # reorder by agent_id if needed
        if agent_to_mec.index.name != "agent_id":
            agent_to_mec.index.name = "agent_id"
        # enforce index = 0..N_agents-1
        idx = core["agents"].sort_values("agent_id")["agent_id"].to_numpy()
        agent_to_mec = agent_to_mec.reindex(idx)
        agent_to_mec_arr = agent_to_mec.to_numpy(dtype=int)
    else:
        agent_to_mec_arr = np.asarray(agent_to_mec, dtype=int)

    N_agents = int(core["episodes"]["N_agents"].iloc[0])
    if len(agent_to_mec_arr) != N_agents:
        raise ValueError(f"agent_to_mec length ({len(agent_to_mec_arr)}) != N_agents ({N_agents}).")

    queues_init = _build_default_queues(core["K"])
    action_space = _derive_action_space()
    state_spec   = _derive_state_spec(core["K"])

    env_config = {
        "Delta": core["Delta"],
        "K": core["K"],
        "topology_type": core["topology_type"],
        "connection_matrix": core["connection_matrix"],

        "private_cpu": core["private_cpu"],
        "public_cpu":  core["public_cpu"],
        "cloud_cpu":   core["cloud_cpu"],

        "N_agents": N_agents,
        "agent_to_mec": agent_to_mec_arr,

        # aligned dataframes (include f_local_slot, deadline_slots if شما قبلاً افزوده‌اید)
        "episodes": core["episodes"],
        "agents":   core["agents"],
        "arrivals": core["arrivals"],
        "tasks":    core["tasks"],

        "queues_initial": queues_init,
        "action_space": action_space,
        "state_spec": state_spec,

        "checks": bundle.get("checks", {"delta_match": True, "message": "n/a"}),
    }
    return env_config

def build_all_env_configs(pairs_by_topology: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Build env_config for every (topology / episode / scenario) bundle.

    Output shape:
        env_configs[topology][episode][scenario] = env_config
    """
    out: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for topo_name, by_ep in pairs_by_topology.items():
        out[topo_name] = {}
        for ep_name, by_scen in by_ep.items():
            out[topo_name][ep_name] = {}
            for scen_name, bundle in by_scen.items():
                if "agent_to_mec" not in bundle:
                    raise RuntimeError(f"[{topo_name}/{ep_name}/{scen_name}] missing 'agent_to_mec'. Run Stage 5 first.")
                env_cfg = build_env_config_for_bundle(bundle)
                out[topo_name][ep_name][scen_name] = env_cfg
    return out

# Build
env_configs = build_all_env_configs(pairs_by_topology)

# Example access:
# env_configs["clustered"]["ep_000"]["heavy"]["agent_to_mec"]
# env_configs["full_mesh"]["ep_001"]["moderate"]["connection_matrix"]





# 1.6. Environment Configuration

# In this step, we build a unified env_config for each scenario–topology pair.
# It bundles all required information for the MDP/RL environment—such as compute capacities,
    # the Agent→MEC mapping, connection matrix, initial queue states, and action/state specifications—into
    # a single consistent configuration used by the RL training process.
    
def _extract_core_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    required = ["dataset", "topology_data", "connection_matrix_df", "Delta", "K"]
    for k in required:
        if k not in bundle:
            raise ValueError(f"Bundle missing required key: '{k}'")

    ds   = bundle["dataset"]
    topo = bundle["topology_data"]
    Mdf  = bundle["connection_matrix_df"]

    private_cpu = np.asarray(topo["private_cpu_capacities"], dtype=float)
    public_cpu  = np.asarray(topo["public_cpu_capacities"],  dtype=float)
    cloud_cpu   = float(topo["cloud_computational_capacity"])
    M           = Mdf.to_numpy(dtype=float)  # (K, K+1), last col MEC→Cloud (MB/slot)

    return dict(
        Delta=float(bundle["Delta"]),
        K=int(bundle["K"]),
        episodes=ds["episodes"],
        agents=ds["agents"],
        arrivals=ds["arrivals"],
        tasks=ds["tasks"],
        private_cpu=private_cpu,
        public_cpu=public_cpu,
        cloud_cpu=cloud_cpu,
        connection_matrix=M,
        topology_type=topo.get("topology_type", "unknown"),
    )

def _build_default_queues(K: int) -> Dict[str, np.ndarray]:
    return {
        "mec_local_cycles":   np.zeros(K, dtype=float),
        "mec_public_cycles":  np.zeros(K, dtype=float),
        "mec_bytes_in_transit": np.zeros(K, dtype=float),
        "cloud_cycles":       np.array([0.0], dtype=float),
    }

def _derive_action_space() -> Dict[str, Any]:
    return {"type": "discrete", "n": 3, "labels": {0: "LOCAL", 1: "MEC", 2: "CLOUD"}}

def _derive_state_spec(K: int) -> Dict[str, Any]:
    return {
        "components": {
            "queues": {
                "mec_local_cycles":  {"shape": (K,),   "dtype": "float"},
                "mec_public_cycles": {"shape": (K,),   "dtype": "float"},
                "cloud_cycles":      {"shape": (1,),   "dtype": "float"},
            },
            "links": {
                "connection_matrix": {"shape": (K, K+1), "dtype": "float"},
            },
            "capacities": {
                "private_cpu": {"shape": (K,), "dtype": "float"},
                "public_cpu":  {"shape": (K,), "dtype": "float"},
                "cloud_cpu":   {"shape": (1,), "dtype": "float"},
            }
        },
        "note": "Declarative spec; tensor assembly happens in the Env at each step."
    }
    
def build_env_config_for_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    core = _extract_core_from_bundle(bundle)

    if "agent_to_mec" not in bundle:
        raise ValueError("Bundle has no 'agent_to_mec' mapping. Run Stage 5 first.")

    agent_to_mec = bundle["agent_to_mec"]
    if isinstance(agent_to_mec, pd.Series):
        # reorder by agent_id if needed
        if agent_to_mec.index.name != "agent_id":
            agent_to_mec.index.name = "agent_id"
        idx = core["agents"].sort_values("agent_id")["agent_id"].to_numpy()
        agent_to_mec = agent_to_mec.reindex(idx)
        agent_to_mec_arr = agent_to_mec.to_numpy(dtype=int)
    else:
        agent_to_mec_arr = np.asarray(agent_to_mec, dtype=int)

    N_agents = int(core["episodes"]["N_agents"].iloc[0])
    if len(agent_to_mec_arr) != N_agents:
        raise ValueError(f"agent_to_mec length ({len(agent_to_mec_arr)}) != N_agents ({N_agents}).")

    queues_init  = _build_default_queues(core["K"])
    action_space = _derive_action_space()
    state_spec   = _derive_state_spec(core["K"])

    env_config = {
        "Delta": core["Delta"],
        "K": core["K"],
        "topology_type": core["topology_type"],
        "connection_matrix": core["connection_matrix"],

        "private_cpu": core["private_cpu"],
        "public_cpu":  core["public_cpu"],
        "cloud_cpu":   core["cloud_cpu"],

        "N_agents": N_agents,
        "agent_to_mec": agent_to_mec_arr,

        # aligned dataframes
        "episodes": core["episodes"],
        "agents":   core["agents"],
        "arrivals": core["arrivals"],
        "tasks":    core["tasks"],

        "queues_initial": queues_init,
        "action_space": action_space,
        "state_spec": state_spec,

        "checks": bundle.get("checks", {"delta_match": True, "message": "n/a"}),
    }
    return env_config


def build_all_env_configs(
    pairs_by_topology: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Build env_config for every (topology / episode / scenario) bundle.

    Desired output shape (EPISODE-first):
        env_configs[episode][topology][scenario] = env_config

    So you can access:
        env_configs["ep_000"]["clustered"]["heavy"]["agent_to_mec"]
    """
    out: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    # pairs_by_topology: topo -> ep -> scen -> bundle
    for topo_name, by_ep in pairs_by_topology.items():
        for ep_name, by_scen in by_ep.items():
            # ensure episode level exists
            if ep_name not in out:
                out[ep_name] = {}
            # ensure topology level under this episode exists
            if topo_name not in out[ep_name]:
                out[ep_name][topo_name] = {}
            for scen_name, bundle in by_scen.items():
                if "agent_to_mec" not in bundle:
                    raise RuntimeError(
                        f"[{topo_name}/{ep_name}/{scen_name}] missing 'agent_to_mec'. Run Stage 5 first."
                    )
                env_cfg = build_env_config_for_bundle(bundle)
                out[ep_name][topo_name][scen_name] = env_cfg
    return out


# Build
env_configs = build_all_env_configs(pairs_by_topology)

# Example access:
env_configs["clustered"]["ep_000"]["heavy"]["agent_to_mec"]





# 1.7. Sanity Checks

# In this step, we verify that each env_config is internally consistent 
    # (queue shapes, capacities, agent→MEC mapping, and connection matrix are valid and ready for simulation).
    
def sanity_check_env_config(env_config):
    errors = []

    # 1) Agent → MEC alignment
    N_agents = env_config["N_agents"]
    if len(env_config["agent_to_mec"]) != N_agents:
        errors.append("Length of agent_to_mec does not match N_agents.")

    # 2) Queue initial state shapes
    K = env_config["K"]
    q = env_config["queues_initial"]
    if q["mec_local_cycles"].shape != (K,):
        errors.append("mec_local_cycles queue shape mismatch.")
    if q["mec_public_cycles"].shape != (K,):
        errors.append("mec_public_cycles queue shape mismatch.")
    if q["mec_bytes_in_transit"].shape != (K,):
        errors.append("mec_bytes_in_transit queue shape mismatch.")
    if q["cloud_cycles"].shape != (1,):
        errors.append("cloud_cycles shape mismatch (should be (1,)).")

    # 3) Non-negative compute capacities
    if (env_config["private_cpu"] < 0).any():
        errors.append("private_cpu has negative values.")
    if (env_config["public_cpu"] < 0).any():
        errors.append("public_cpu has negative values.")
    if env_config["cloud_cpu"] < 0:
        errors.append("cloud_cpu is negative.")

    # 4) Connection matrix dimension (K x K+1)
    M = env_config["connection_matrix"]
    if M.shape != (K, K+1):
        errors.append("connection_matrix shape mismatch.")

    # 5) Action space correctness
    if env_config["action_space"]["type"] != "discrete":
        errors.append("Action space must be discrete (LOCAL/MEC/CLOUD).")

    return errors

def sanity_check_all(env_configs):
    for topo_name, by_ep in env_configs.items():
        for ep_name, by_scen in by_ep.items():
            for scen_name, env_cfg in by_scen.items():
                errs = sanity_check_env_config(env_cfg)
                if errs:
                    print(f"[FAIL] {topo_name}/{ep_name}/{scen_name}:")
                    for e in errs:
                        print("   -", e)
                else:
                    print(f"[OK]   {topo_name}/{ep_name}/{scen_name}")

# Run all sanity checks
sanity_check_all(env_configs)


# At Step 1, we have loaded the data, aligned the units, assigned agents to MECs, 
# and prepared the environment configuration. Finally, we have performed consistency checks to ensure the data is correct. 
# Next, we can move on to task labeling.





# Step 2: Task Labeling

# 2.1. Basic Task Labeling (buckets, urgency, atomicity, ...)

# ---------- helpers: quantile-based cut points ----------
def _quantile_cutpoints(s: pd.Series, q_low=0.33, q_high=0.66) -> Tuple[float, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return (np.nan, np.nan)
    return (float(s.quantile(q_low)), float(s.quantile(q_high)))

def _bucketize(value: float, q1: float, q2: float) -> str:
    # Returns 'S', 'M', 'L' based on two cut points (q1<=q2)
    if not np.isfinite(value) or not np.isfinite(q1) or not np.isfinite(q2):
        return "U"  # Unknown
    if value <= q1: return "S"
    if value <= q2: return "M"
    return "L"

# ---------- threshold builder (adaptive to each tasks DF) ----------
def build_task_label_thresholds(tasks_df: pd.DataFrame,
                                q_low=0.33, q_high=0.66,
                                urgent_slots_cap: int = 2) -> Dict[str, Any]:
    """
    Build adaptive thresholds from the data itself (per-episode/senario),
    so 'light/moderate/heavy' are handled robustly.
    """
    q_b_mb   = _quantile_cutpoints(tasks_df["b_mb"], q_low, q_high) if "b_mb" in tasks_df else (np.nan, np.nan)
    q_rho    = _quantile_cutpoints(tasks_df["rho_cyc_per_mb"], q_low, q_high) if "rho_cyc_per_mb" in tasks_df else (np.nan, np.nan)
    q_mem    = _quantile_cutpoints(tasks_df["mem_mb"], q_low, q_high) if "mem_mb" in tasks_df else (np.nan, np.nan)
    q_split  = _quantile_cutpoints(tasks_df.loc[tasks_df.get("non_atomic", 0)==1, "split_ratio"], q_low, q_high) \
               if "split_ratio" in tasks_df else (np.nan, np.nan)

    return {
        "b_mb":   {"q1": q_b_mb[0],  "q2": q_b_mb[1]},
        "rho":    {"q1": q_rho[0],   "q2": q_rho[1]},
        "mem":    {"q1": q_mem[0],   "q2": q_mem[1]},
        "split":  {"q1": q_split[0], "q2": q_split[1]},
        # If deadline_slots ≤ urgent_slots_cap → 'hard' (latency sensitive)
        "urgent_slots_cap": int(urgent_slots_cap),
    }

# ---------- main labeling for a single tasks DF ----------
def label_tasks_df(tasks_df: pd.DataFrame, Delta: float, thresholds: Dict[str, Any]) -> pd.DataFrame:
    """
    Add label columns to tasks_df (returns a COPY).
    Columns added:
      - size_bucket, compute_bucket, mem_bucket
      - deadline_slots (if missing), urgency (none/soft/hard)
      - atomicity, split_bucket
      - latency_sensitive, compute_heavy, io_heavy, memory_heavy (bools)
      - routing_hint (LOCAL/MEC/CLOUD)
    """
    df = tasks_df.copy()

    # --- ensure numeric types
    for col in ["b_mb", "rho_cyc_per_mb", "c_cycles", "mem_mb", "deadline_s", "split_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- deadline_slots (if not precomputed in Units Alignment)
    if "deadline_slots" not in df.columns:
        if "has_deadline" in df.columns and "deadline_s" in df.columns:
            df["deadline_slots"] = np.where(
                (df["has_deadline"] == 1) & np.isfinite(df["deadline_s"]),
                np.ceil(df["deadline_s"] / float(Delta)).astype("float"),
                np.nan
            )
        else:
            df["deadline_slots"] = np.nan

    # --- bucketize size/compute/memory
    b_q1, b_q2   = thresholds["b_mb"]["q1"], thresholds["b_mb"]["q2"]
    rho_q1, rho_q2 = thresholds["rho"]["q1"], thresholds["rho"]["q2"]
    mem_q1, mem_q2 = thresholds["mem"]["q1"], thresholds["mem"]["q2"]

    df["size_bucket"]    = df["b_mb"].apply(lambda x: _bucketize(x, b_q1, b_q2)) if "b_mb" in df else "U"
    df["compute_bucket"] = df["rho_cyc_per_mb"].apply(lambda x: _bucketize(x, rho_q1, rho_q2)) if "rho_cyc_per_mb" in df else "U"
    df["mem_bucket"]     = df["mem_mb"].apply(lambda x: _bucketize(x, mem_q1, mem_q2)) if "mem_mb" in df else "U"

    # --- atomicity & split buckets
    if "non_atomic" in df.columns:
        df["atomicity"] = np.where(df["non_atomic"] == 1, "splittable", "atomic")
    else:
        df["atomicity"] = "atomic"

    if "split_ratio" in df.columns:
        sp_q1, sp_q2 = thresholds["split"]["q1"], thresholds["split"]["q2"]
        df["split_bucket"] = np.where(
            df["atomicity"] == "splittable",
            df["split_ratio"].apply(lambda v: _bucketize(v, sp_q1, sp_q2)),
            "NA"
        )
    else:
        df["split_bucket"] = "NA"

    # --- urgency levels
    urgent_cap = int(thresholds.get("urgent_slots_cap", 2))
    def _urg(row):
        if int(row.get("has_deadline", 0)) != 1 or not np.isfinite(row.get("deadline_slots", np.nan)):
            return "none"
        slots = int(row["deadline_slots"])
        if slots <= urgent_cap:  # very tight deadline
            return "hard"
        return "soft"
    df["urgency"] = df.apply(_urg, axis=1)

    # --- boolean convenience labels
    df["latency_sensitive"] = (df["urgency"] == "hard")
    df["compute_heavy"]     = (df["compute_bucket"] == "L")
    df["io_heavy"]          = (df["size_bucket"] == "L")
    df["memory_heavy"]      = (df["mem_bucket"] == "L")

    # --- a very simple routing hint (only for debugging/EDA; not used by the RL policy)
    def _hint(row):
        if row["compute_heavy"] or row["memory_heavy"]:
            return "CLOUD"
        if row["latency_sensitive"]:
            return "MEC"
        return "LOCAL"
    df["routing_hint"] = df.apply(_hint, axis=1)

    return df

# ---------- batch apply to env_configs (topology → episode → scenario) ----------
def label_all_tasks_in_env_configs(env_configs: Dict[str, Dict[str, Dict[str, Any]]],
                                   q_low=0.33, q_high=0.66, urgent_slots_cap=2,
                                   verbose=True) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    For each env_config:
      - build thresholds from its own tasks DF
      - label tasks
      - put labeled DF back into env_config["tasks"]
      - return a concise summary per bundle
    """
    summary = {}

    for topo_name, by_ep in env_configs.items():
        summary[topo_name] = {}
        for ep_name, by_scen in by_ep.items():
            summary[topo_name][ep_name] = {}
            for scen_name, env_cfg in by_scen.items():
                tasks = env_cfg["tasks"]
                Delta = float(env_cfg["Delta"])

                # thresholds adaptive to this bundle
                th = build_task_label_thresholds(tasks, q_low=q_low, q_high=q_high,
                                                 urgent_slots_cap=urgent_slots_cap)
                labeled = label_tasks_df(tasks, Delta=Delta, thresholds=th)
                env_cfg["tasks"] = labeled  # write back

                # tiny summary
                cnt = {
                    "n": len(labeled),
                    "urg_hard": int((labeled["urgency"] == "hard").sum()),
                    "splittable": int((labeled["atomicity"] == "splittable").sum()),
                    "size_L": int((labeled["size_bucket"] == "L").sum()),
                    "compute_L": int((labeled["compute_bucket"] == "L").sum()),
                    "mem_L": int((labeled["mem_bucket"] == "L").sum()),
                }
                summary[topo_name][ep_name][scen_name] = cnt

                if verbose:
                    print(f"[label] {topo_name}/{ep_name}/{scen_name} -> "
                          f"n={cnt['n']}, hard={cnt['urg_hard']}, split={cnt['splittable']}, "
                          f"sizeL={cnt['size_L']}, compL={cnt['compute_L']}, memL={cnt['mem_L']}")

    return env_configs, summary


# env_configs: Produced in Step 6 (structure: episode → topology → scenario)
env_configs, label_summary = label_all_tasks_in_env_configs(
    env_configs,
    q_low=0.33, q_high=0.66, urgent_slots_cap=2,  # tunable thresholds
    verbose=True
)

# Example access:
labeled_tasks = env_configs["ep_000"]["clustered"]["heavy"]["tasks"]
labeled_tasks.head()





# 2.2. Task Type Classification

# === Chapter-4 Task Typing (priority rules) ===
# Pre-req: tasks already labeled by your previous step: 
#   size_bucket, compute_bucket, mem_bucket, urgency, atomicity, split_bucket, routing_hint, etc.

from typing import Dict, Any

def _derive_task_type_row(row: pd.Series) -> tuple[str, str, str, list]:
    """
    Returns (task_type, task_subtype, type_reason, multi_flags)
      task_type   ∈ {"deadline_hard","latency_sensitive","compute_intensive","data_intensive","general"}
      task_subtype: finer note (e.g., "deadline_hard", "deadline_soft", ...)
      type_reason: short human-readable reason
      multi_flags: list of boolean tags that were true (for auditing)
    """

    # Collect boolean flags consistent with your earlier labeling:
    urgency        = str(row.get("urgency", "none"))         # "hard" | "soft" | "none"
    latency_flag   = (urgency == "hard") or (urgency == "soft")
    hard_deadline  = (urgency == "hard")

    compute_heavy  = bool(row.get("compute_heavy", False))   # compute_bucket == "L"
    memory_heavy   = bool(row.get("memory_heavy", False))    # mem_bucket == "L"
    io_heavy       = bool(row.get("io_heavy", False))        # size_bucket == "L"
    non_atomic     = bool(row.get("atomicity", "atomic") == "splittable")

    # Keep all active signals for audit:
    multi_flags = []
    if hard_deadline:  multi_flags.append("deadline_hard")
    elif latency_flag: multi_flags.append("deadline_soft")
    if compute_heavy:  multi_flags.append("compute_heavy")
    if memory_heavy:   multi_flags.append("memory_heavy")
    if io_heavy:       multi_flags.append("io_heavy")
    if non_atomic:     multi_flags.append("splittable")

    # --- Priority resolution (Chapter 4) ---
    # 1) Hard deadline dominates everything
    if hard_deadline:
        return ("deadline_hard", "deadline_hard", "hard deadline (tight slots)", multi_flags)

    # 2) Latency-sensitive (soft deadlines / delay-sensitive)
    if latency_flag:
        return ("latency_sensitive", "deadline_soft", "delay-sensitive (soft deadline)", multi_flags)

    # 3) Compute-intensive (c or rho or mem heavy)
    #    You may decide whether memory_heavy alone pushes to compute_intensive or creates a separate class.
    #    Based on Chapter 4 text we map memory_heavy into compute_intensive family.
    if compute_heavy or memory_heavy:
        return ("compute_intensive", "compute_or_memory_heavy", "high compute/memory demand", multi_flags)

    # 4) Data-intensive (mainly large input size / high IO pressure)
    if io_heavy:
        return ("data_intensive", "large_input_bandwidth", "large data volume / IO heavy", multi_flags)

    # 5) Otherwise general
    return ("general", "general", "no dominant constraint", multi_flags)


def apply_ch4_task_typing(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Chapter-4 level task classes with priority rules into tasks_df (returns a COPY).
    Columns added:
      - task_type            (5-way class)
      - task_subtype         (finer descriptor)
      - type_reason          (short textual rationale)
      - multi_flags          (list of all active boolean traits)
    """
    df = tasks_df.copy()

    # Ensure the expected helper columns exist (created in your previous labeling step).
    required_cols = ["urgency", "compute_heavy", "memory_heavy", "io_heavy", "atomicity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"apply_ch4_task_typing: missing label columns: {missing}")

    out_type, out_sub, out_reason, out_flags = [], [], [], []
    for _, r in df.iterrows():
        t, s, msg, flags = _derive_task_type_row(r)
        out_type.append(t)
        out_sub.append(s)
        out_reason.append(msg)
        out_flags.append(flags)

    df["task_type"]   = out_type
    df["task_subtype"]= out_sub
    df["type_reason"] = out_reason
    df["multi_flags"] = out_flags
    # For convenience: one-hot view (optional)
    df["is_general"]            = (df["task_type"] == "general")
    df["is_deadline_hard"]      = (df["task_type"] == "deadline_hard")
    df["is_latency_sensitive"]  = (df["task_type"] == "latency_sensitive")
    df["is_compute_intensive"]  = (df["task_type"] == "compute_intensive")
    df["is_data_intensive"]     = (df["task_type"] == "data_intensive")

    return df


def apply_task_typing_in_env_configs(env_configs: Dict[str, Dict[str, Dict[str, Any]]],
                                     verbose: bool = True) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    env_configs structure (as we fixed earlier):
      env_configs[ep_name][topology_name][scenario_name]["tasks"] -> DataFrame

    This function:
      - applies Chapter-4 task typing to every tasks DF
      - writes back the enriched DataFrame
      - prints a short summary if verbose=True
    """
    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():
                tasks = env_cfg["tasks"]
                enriched = apply_ch4_task_typing(tasks)
                env_cfg["tasks"] = enriched

                if verbose:
                    n = len(enriched)
                    counts = enriched["task_type"].value_counts().to_dict()
                    print(f"[typing] {ep_name}/{topo_name}/{scen_name}  n={n}  → {counts}")
    return env_configs

# ---- Run typing on your current env_configs (episode → topology → scenario) ----
env_configs = apply_task_typing_in_env_configs(env_configs, verbose=True)

# Example access:
env_configs["ep_000"]["clustered"]["heavy"]["tasks"][["task_id","task_type","task_subtype","type_reason","multi_flags"]].head()





# Step 3 — Agent Profiling (full cell, revised)

# ---- Helper 1: per-agent per-slot arrival counts ----
def _per_agent_slot_counts(arrivals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many tasks each agent generates in each time slot.
    This is used to estimate lambda (arrival rate) statistics.
    """
    if not {"agent_id", "t_slot"}.issubset(arrivals_df.columns):
        raise ValueError("arrivals must contain 'agent_id' and 't_slot'.")
    grp = arrivals_df.groupby(["agent_id", "t_slot"], as_index=False).size()
    grp.rename(columns={"size": "count"}, inplace=True)
    return grp

# ---- Helper 2: estimate λ-mean and λ-variance per agent (tight dtypes) ----
def _lambda_stats_from_counts(counts_df: pd.DataFrame, Delta: float) -> pd.DataFrame:
    """
    Convert per-slot counts to rate statistics:
        lambda_mean = mean(count_per_slot) / Delta
        lambda_var  = var(count_per_slot)  / Delta^2
    """
    if counts_df.empty:
        return pd.DataFrame(columns=["agent_id", "lambda_mean", "lambda_var", "slots_observed"])

    agg = counts_df.groupby("agent_id")["count"].agg(
        lambda_mean_slot="mean",
        lambda_var_slot="var",
        slots_observed="count"
    ).reset_index()

    # If only one observation exists, variance becomes NaN → treat as zero.
    agg["lambda_var_slot"] = agg["lambda_var_slot"].fillna(0.0).astype(float)

    # Convert to per-second rates
    agg["lambda_mean"] = (agg["lambda_mean_slot"] / float(Delta)).astype(float)
    agg["lambda_var"]  = (agg["lambda_var_slot"]  / float(Delta**2)).astype(float)

    return agg[["agent_id", "lambda_mean", "lambda_var", "slots_observed"]]

# ---- Helper 3: task-type distribution per agent (robust + extra stats) ----
_TASK_TYPES = ["general", "latency_sensitive", "deadline_hard", "data_intensive", "compute_intensive"]

def _task_distribution_per_agent(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution of task types per agent (probabilities sum to 1 for agents with tasks).
    Also adds light median features useful for clustering: b_mb_med, rho_med, mem_med, hard_share.
    """
    if not {"agent_id", "task_type"}.issubset(tasks_df.columns):
        raise ValueError("tasks must contain 'agent_id' and 'task_type'.")

    # Raw counts per (agent_id, task_type)
    cnt = tasks_df.groupby(["agent_id", "task_type"], as_index=False).size()
    piv = cnt.pivot(index="agent_id", columns="task_type", values="size").fillna(0.0)

    # Ensure all expected classes exist
    for t in _TASK_TYPES:
        if t not in piv.columns:
            piv[t] = 0.0

    # True count across all seen labels
    piv["n_tasks_agent"] = piv[_TASK_TYPES].sum(axis=1).astype(float)

    # Probabilities
    for t in _TASK_TYPES:
        piv[f"P_{t}"] = np.where(piv["n_tasks_agent"] > 0, piv[t] / piv["n_tasks_agent"], 0.0).astype(float)

    # Optional extra features for clustering (guard on availability)
    feats = {}
    have_feats = {"b_mb", "rho_cyc_per_mb", "mem_mb", "urgency"}
    if have_feats.issubset(tasks_df.columns):
        agg = tasks_df.groupby("agent_id").agg(
            b_mb_med=("b_mb", "median"),
            rho_med=("rho_cyc_per_mb", "median"),
            mem_med=("mem_mb", "median"),
            hard_share=("urgency", lambda s: float((s == "hard").mean()))
        ).reset_index()
        feats = agg.set_index("agent_id")

    # Join extra features (if any)
    piv = piv.join(feats, how="left")
    for c in ["b_mb_med", "rho_med", "mem_med", "hard_share"]:
        if c in piv.columns:
            piv[c] = piv[c].fillna(0.0).astype(float)
        else:
            piv[c] = 0.0

    # Probability mass sum (diagnostic)
    prob_cols = [f"P_{t}" for t in _TASK_TYPES]
    piv["TaskDist_sum"] = piv[prob_cols].sum(axis=1).astype(float)

    keep = ["n_tasks_agent", "TaskDist_sum", "b_mb_med", "rho_med", "mem_med", "hard_share"] + prob_cols
    return piv[keep].reset_index()

# ---- Helper 4: fraction of non-atomic (splittable) tasks ----
def _non_atomic_share_per_agent(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the share of splittable (non-atomic) tasks per agent.
    """
    if not {"agent_id", "non_atomic"}.issubset(tasks_df.columns):
        # If missing, assume zero for all agents that exist in tasks
        agents = tasks_df.get("agent_id")
        if agents is None or len(agents) == 0:
            return pd.DataFrame(columns=["agent_id", "non_atomic_share"])
        return pd.DataFrame({"agent_id": agents.unique(), "non_atomic_share": 0.0})

    grp = tasks_df.groupby("agent_id")["non_atomic"].agg(
        non_atomic_share=lambda s: float((s == 1).mean())
    ).reset_index()
    return grp

# ---- Build agent profiles for ONE env_config ----
def build_agent_profiles_for_env(env_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Construct per-agent profiles combining:
      - Local resource capacity (f_local, m_local, f_local_slot)
      - Arrival rate statistics (lambda_mean, lambda_var)
      - Task type distribution (P_general, P_latency_sensitive, P_deadline_hard, P_data_intensive, P_compute_intensive)
      - Splittability share (non_atomic_share)
      - MEC mapping if available (mec_id)
    """
    agents   = env_cfg["agents"].copy()
    arrivals = env_cfg["arrivals"]
    tasks    = env_cfg["tasks"]
    Delta    = float(env_cfg["Delta"])

    # Ensure cycles/slot exists
    if "f_local_slot" not in agents.columns and "f_local" in agents.columns:
        agents["f_local_slot"] = agents["f_local"].astype(float) * Delta

    # 1) Arrival statistics
    counts_df = _per_agent_slot_counts(arrivals)
    lam_df    = _lambda_stats_from_counts(counts_df, Delta=Delta)

    # 2) Task-type distribution (+ medians & hard_share)
    dist_df   = _task_distribution_per_agent(tasks)

    # 3) Splittable-task share
    na_df     = _non_atomic_share_per_agent(tasks)

    # 4) Agent→MEC mapping (optional)
    mec_map = None
    if "agent_to_mec" in env_cfg:
        a2m = env_cfg["agent_to_mec"]
        if isinstance(a2m, pd.Series):
            mec_map = a2m.rename("mec_id").reset_index()
            # if the index column name is lost, normalize it
            if mec_map.columns.tolist() == ["index", "mec_id"]:
                mec_map.rename(columns={"index": "agent_id"}, inplace=True)
        else:
            mec_map = pd.DataFrame({
                "agent_id": np.arange(len(a2m), dtype=int),
                "mec_id": np.asarray(a2m, dtype=int)
            })

    # Merge all components
    base = agents[["agent_id", "f_local", "f_local_slot", "m_local"]].copy()
    base[["f_local", "f_local_slot", "m_local"]] = base[["f_local", "f_local_slot", "m_local"]].astype(float)

    prof = (base
            .merge(lam_df,  on="agent_id", how="left")
            .merge(dist_df, on="agent_id", how="left")
            .merge(na_df,   on="agent_id", how="left"))

    if mec_map is not None:
        prof = prof.merge(mec_map, on="agent_id", how="left")

    # Fill missing for agents with no arrivals/tasks
    fill_zero = [
        "lambda_mean", "lambda_var", "slots_observed",
        "n_tasks_agent", "non_atomic_share",
        "TaskDist_sum", "b_mb_med", "rho_med", "mem_med", "hard_share"
    ] + [f"P_{t}" for t in _TASK_TYPES]
    for c in fill_zero:
        if c in prof.columns:
            prof[c] = prof[c].fillna(0.0).astype(float)

    # Soft warning if probabilities don't sum to ~1 for agents with tasks
    if "n_tasks_agent" in prof.columns and "TaskDist_sum" in prof.columns:
        mask = (prof["n_tasks_agent"] > 0) & (~np.isclose(prof["TaskDist_sum"], 1.0, atol=1e-6))
        if mask.any():
            n_bad = int(mask.sum())
            print(f"[warn] TaskDist_sum != 1.0 for {n_bad} agent(s). (tolerance 1e-6)")

    return prof

# ---- Batch profiling for ALL env_configs ----
def build_all_agent_profiles(env_configs: Dict[str, Dict[str, Dict[str, Any]]]):
    """
    Compute profiles for every (episode → topology → scenario) environment.
    Stores result both in return dict AND env_configs[...] for convenience.
    Output:
      profiles[ep_name][topology_name][scen_name] = DataFrame
    Also writes back to: env_configs[ep_name][topology_name][scen_name]["agent_profiles"]
    """
    out = {}
    for ep_name, by_topo in env_configs.items():
        out[ep_name] = {}
        for topo_name, by_scen in by_topo.items():
            out[ep_name][topo_name] = {}
            for scen_name, env_cfg in by_scen.items():
                prof = build_agent_profiles_for_env(env_cfg)
                out[ep_name][topo_name][scen_name] = prof
                env_cfg["agent_profiles"] = prof  # attach for direct access
    return out

# ---- Build + quick peek (optional) ----
agent_profiles = build_all_agent_profiles(env_configs)

# Example: Access the profile table for a specific episode / topology / scenario
print(agent_profiles["ep_000"]["clustered"]["heavy"].head())

# Alternatively, read directly from env_configs:
# display(env_configs["ep_000"]["clustered"]["heavy"]["agent_profiles"].head())

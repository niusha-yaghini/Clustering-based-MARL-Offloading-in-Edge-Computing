# Imports
import pandas as pd
import numpy as np
import json
import os
import math
from typing import Dict, Any, Tuple


# Step 1: Prepare data and configure the environment

# 1. Data Loading (Data I/O)

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



# 2. Data Validation (episode-first aware)

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



# 3. Units Alignment (episode-first aware)

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




# 4. Build Scenario–Topology Pairs (episode-first aware)

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



# 5. Agent→MEC mapping (for all pairs) — 3-level: topology → episode → scenario

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




from typing import Dict, Any, Tuple

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

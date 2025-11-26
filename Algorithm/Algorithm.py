import pandas as pd
import numpy as np
import json
import os
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional

import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# Step 1: Prepare data and configure the environment

# 1.1. Data Loading (Data I/O)

# Define the base directories
dataset_dir = '../Data_Generator/datasets'
topology_dir = '../Topology_Generator/topologies'

# Loading dataset

# Global container
datasets = {}

def load_datasets_from_directory(dataset_dir, verbose=True):
    """
    Episode-first loader for structure:

        dataset_dir/
          ep_000/
            light/
              episodes.csv
              agents.csv
              arrivals.csv
              tasks.csv
            moderate/
              ...
            heavy/
              ...
            dataset_metadata.json   (optional, per-episode meta)

    Result:
        datasets = {
            "ep_000": {
                "light":   {"episodes": df, "agents": df, "arrivals": df, "tasks": df},
                "moderate":{"..."},
                "heavy":   {"..."},
                "_meta":   {...}  # if dataset_metadata.json exists
            },
            "ep_001": { ... },
            ...
        }
    """
    global datasets
    datasets = {}

    if not os.path.isdir(dataset_dir):
        raise ValueError(f"dataset_dir does not exist or is not a directory: {dataset_dir}")

    # Step 1 — detect ep_* directories
    ep_dirs = sorted([
        name for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name)) and name.startswith("ep_")
    ])

    if verbose:
        if not ep_dirs:
            print(f"[warn] no ep_* folders found under root '{dataset_dir}'")
        else:
            print(f"[info] detected episodes: {ep_dirs}")

    # Step 2 — per episode, detect scenarios and load CSVs
    for ep_name in ep_dirs:
        ep_path = os.path.join(dataset_dir, ep_name)
        datasets[ep_name] = {}

        # scenarios inside this episode (e.g. light/moderate/heavy)
        scenario_names = sorted([
            name for name in os.listdir(ep_path)
            if os.path.isdir(os.path.join(ep_path, name))
        ])

        if verbose:
            if not scenario_names:
                print(f"[warn] no scenario folders found under episode '{ep_name}'")
            else:
                print(f"[info] {ep_name}: scenarios detected -> {scenario_names}")

        for scenario in scenario_names:
            scn_path = os.path.join(ep_path, scenario)
            try:
                dfs = {
                    "episodes": pd.read_csv(os.path.join(scn_path, "episodes.csv")),
                    "agents":   pd.read_csv(os.path.join(scn_path, "agents.csv")),
                    "arrivals": pd.read_csv(os.path.join(scn_path, "arrivals.csv")),
                    "tasks":    pd.read_csv(os.path.join(scn_path, "tasks.csv")),
                }
                datasets[ep_name][scenario] = dfs
            except FileNotFoundError as e:
                if verbose:
                    print(f"[error] missing CSV in {scn_path}: {e}")
                continue

        # Step 3 — load per-episode metadata if present
        meta_path = os.path.join(ep_path, "dataset_metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                datasets[ep_name]["_meta"] = meta
                if verbose:
                    print(f"[info] loaded metadata for {ep_name} from {meta_path}")
            except Exception as e:
                if verbose:
                    print(f"[warn] could not load metadata for {ep_name}: {e}")

    # Optional summary printing
    if verbose:
        print("\n=== Dataset Summary (episode-first) ===")
        print(f"episodes detected: {len(datasets)}")
        for ep_name in sorted(datasets.keys()):
            keys_here = sorted(datasets[ep_name].keys())
            scenarios_here = [k for k in keys_here if not k.startswith("_")]
            print(f"  - {ep_name}: scenarios = {scenarios_here}")
            for scn in scenarios_here:
                dfs = datasets[ep_name][scn]
                n_ep   = len(dfs["episodes"])
                n_ag   = len(dfs["agents"])
                n_arr  = len(dfs["arrivals"])
                n_task = len(dfs["tasks"])
                print(f"      {scn:9s} → episodes:{n_ep:3d}  agents:{n_ag:4d}  arrivals:{n_arr:6d}  tasks:{n_task:6d}")
            if "_meta" in datasets[ep_name]:
                print(f"      meta: dataset_metadata.json loaded")
        print("=======================================\n")

    return datasets


# ---- load all datasets (episode-first) ----
datasets = load_datasets_from_directory(dataset_dir, verbose=True)

# ---- choose an episode and a scenario for printing ----
ep_name = sorted(datasets.keys())[0] if datasets else None
scenario = "heavy"  # "light"/"moderate"

if ep_name is not None and scenario in datasets[ep_name]:
    print(f"\n[info] printing from episode='{ep_name}', scenario='{scenario}'")

    print("\nagents:")
    print(datasets[ep_name][scenario]["agents"].head())
    datasets[ep_name][scenario]["agents"].info()

    print("\narrivals:")
    print(datasets[ep_name][scenario]["arrivals"].head())
    datasets[ep_name][scenario]["arrivals"].info()

    print("\nepisodes:")
    print(datasets[ep_name][scenario]["episodes"].head())
    datasets[ep_name][scenario]["episodes"].info()

    print("\ntasks:")
    print(datasets[ep_name][scenario]["tasks"].head())
    datasets[ep_name][scenario]["tasks"].info()

    if "_meta" in datasets[ep_name]:
        print("\nmeta (dataset_metadata.json):")
        print(json.dumps(datasets[ep_name]["_meta"], ensure_ascii=False, indent=2))

else:
    print("[error] no datasets found or requested scenario is missing for the chosen episode.")


# Loading topology

topologies = {}

def load_topologies_from_directory(topology_dir, verbose=True):
    """
    Load all topologies under topology_dir, expecting structure:

        topology_dir/
          clustered/
            topology.json
            topology_meta.json
            connection_matrix.csv
          full_mesh/
          sparse_ring/
          ...

    Fills global 'topologies' as:
        {
          "clustered": {
              "topology_data": dict,
              "meta_data": dict,
              "connection_matrix": DataFrame
          },
          ...
        }
    """
    global topologies
    topologies = {}

    if not os.path.isdir(topology_dir):
        raise ValueError(f"topology_dir does not exist or is not a directory: {topology_dir}")

    for topology_name in os.listdir(topology_dir):
        topology_path = os.path.join(topology_dir, topology_name)

        # Only process directories
        if not os.path.isdir(topology_path):
            continue

        topology_json_path = os.path.join(topology_path, "topology.json")
        meta_json_path = os.path.join(topology_path, "topology_meta.json")
        connection_matrix_csv_path = os.path.join(topology_path, "connection_matrix.csv")

        if not (os.path.isfile(topology_json_path) and
                os.path.isfile(meta_json_path) and
                os.path.isfile(connection_matrix_csv_path)):
            if verbose:
                print(f"[warn] skipping '{topology_name}' — missing one of required files.")
            continue

        # --- Load JSON & CSV files ---
        with open(topology_json_path, "r", encoding="utf-8") as f:
            topology_data = json.load(f)
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        # First column is row labels → index_col=0
        connection_matrix = pd.read_csv(connection_matrix_csv_path, index_col=0)

        topologies[topology_name] = {
            "topology_data": topology_data,
            "meta_data": meta_data,
            "connection_matrix": connection_matrix
        }

    if verbose:
        print(f"[info] loaded topologies: {sorted(topologies.keys())}")

    return topologies
            
            
load_topologies_from_directory(topology_dir, verbose=True)

print('topology clustered -> connection_matrix')
print(topologies['clustered']['connection_matrix'].head())
topologies['clustered']['connection_matrix'].info()

print('\ntopology clustered -> topology_data')
print(topologies['clustered']['topology_data'])

print('\ntopology clustered -> meta_data')
print(topologies['clustered']['meta_data'])





# 1.2. Data Validation (episode-first aware)

# Before using the data, we must validate that required columns exist and that IDs match properly.
# **The code below performs three layers of checks:** 
# - Validate each dataset (episodes/agents/arrivals/tasks)
# - Validate each topology (JSON and connection matrix)
# - Validate dataset–topology pairs for unit alignment and overall consistency

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
    Check that all SCENARIOS (light/moderate/heavy/...) inside one episode
    share the same Delta and T_slots.

    ep_dict:
        {
          "light":   {"episodes": df, ...},
          "moderate":{...},
          "heavy":   {...},
          "_meta":   {...}  # we should ignore this
        }
    """
    errors = []
    deltas = set()
    tslots = set()

    for scenario, ds in ep_dict.items():
        # Discard metadata or anything that doesn't have episodes
        if not isinstance(ds, dict) or "episodes" not in ds:
            continue

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
    'datasets' shape (episode-first):

        {
          "ep_000": {
             "light":   {"episodes": df, "agents": df, "arrivals": df, "tasks": df},
             "moderate":{...},
             "heavy":   {...},
             "_meta":   {...}  # optional per-episode metadata
          },
          "ep_001": {...}
        }
    """
    report = {"datasets": {}, "episodes_consistency": {}, "topologies": {}, "pairs": {}}

    # 1) Validate each (episode, scenario)
    for ep_name, ep_pack in datasets.items():
        report["datasets"][ep_name] = {}

        # Only real scenarios (not _meta)
        scenario_names = [
            scn for scn, dpack in ep_pack.items()
            if isinstance(dpack, dict) and "episodes" in dpack
        ]

        for scenario in scenario_names:
            dpack = ep_pack[scenario]
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
        # Real scenarios (same as in report["datasets"][ep_name])
        scenario_names = list(report["datasets"][ep_name].keys())

        for scenario in scenario_names:
            dpack = ep_pack[scenario]
            d_ok  = report["datasets"][ep_name][scenario]["ok"]
            ep_ok = report["episodes_consistency"][ep_name]["ok"]

            for tname, tres in report["topologies"].items():
                key = f"{ep_name}/{scenario}__{tname}"
                if d_ok and ep_ok and tres["ok"]:
                    errs = validate_dataset_topology_pair(ep_name, scenario, dpack, tname, topologies[tname])
                    report["pairs"][key] = {"ok": len(errs) == 0, "errors": errs}
                else:
                    report["pairs"][key] = {
                        "ok": False,
                        "errors": ["Skipped due to upstream invalid dataset/episode/topology."]
                    }

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
             "heavy":   {...},
             "_meta":   {...}   # optional per-episode metadata (NO episodes/agents/tasks)
          },
          "ep_001": {...}
        }

    Returns:
        {
          "datasets_aligned": { ep_name: { scenario: aligned_pack_or_meta, ... }, ... },
          "topology_checks":  { topo_name: { ep_name: { scenario: {ok, message} } } }
        }
    """
    out = {
        "datasets_aligned": {},
        "topology_checks":  {}
    }

    # ---- 1) Align datasets (episode/scenario) ----
    for ep_name, ep_pack in datasets_ep_first.items():
        out["datasets_aligned"][ep_name] = {}

        for scenario, ds in ep_pack.items():
            # If the dataset is real (has episodes) → align
            if isinstance(ds, dict) and "episodes" in ds:
                try:
                    out["datasets_aligned"][ep_name][scenario] = align_units_for_dataset(ds)
                except Exception as e:
                    raise RuntimeError(f"[{ep_name}/{scenario}] dataset alignment failed: {e}") from e
            else:
                # For example "_meta" or anything else → we keep it as is (no changes)
                out["datasets_aligned"][ep_name][scenario] = ds

    # ---- 2) Verify each topology against each (episode, scenario) Delta ----
    for topo_name, topo_bundle in topologies_by_name.items():
        topo_obj = topo_bundle.get("topology_data", None)
        if not isinstance(topo_obj, dict):
            raise RuntimeError(f"[{topo_name}] 'topology_data' missing or not a dict.")
        out["topology_checks"][topo_name] = {}

        for ep_name, ep_pack in out["datasets_aligned"].items():
            out["topology_checks"][topo_name][ep_name] = {}

            for scenario, aligned in ep_pack.items():
                # Only check scenarios that have episodes; ignore metadata
                if not (isinstance(aligned, dict) and "episodes" in aligned):
                    continue

                Delta = _get_delta(aligned["episodes"])
                ok, msg = verify_topology_units(topo_obj, Delta)
                out["topology_checks"][topo_name][ep_name][scenario] = {
                    "ok": bool(ok),
                    "message": msg
                }

    return out

# ===== Pretty printer (episode-first) =====
def print_alignment_summary_episode_first(result: Dict[str, Any]):
    # ===== DATASETS =====
    print("=== DATASETS (aligned, episode/scenario) ===")
    for ep_name in sorted(result["datasets_aligned"].keys()):
        ep_pack = result["datasets_aligned"][ep_name]

        for scenario in sorted(ep_pack.keys()):
            ds = ep_pack[scenario]
            if not (isinstance(ds, dict) and "episodes" in ds):
                continue

            Delta    = _get_delta(ds["episodes"])
            n_tasks  = len(ds["tasks"])
            n_agents = len(ds["agents"])
            print(f"[{ep_name}/{scenario}] Delta={Delta}  tasks={n_tasks}  agents={n_agents}")

    # ===== TOPOLOGIES =====
    print("\n=== TOPOLOGIES (checks vs each episode/scenario) ===")
    for topo_name, by_ep in result["topology_checks"].items():
        print(f"Topology: {topo_name}")
        for ep_name in sorted(by_ep.keys()):
            for scenario, r in sorted(by_ep[ep_name].items()):
                flag = "OK" if r["ok"] else "FAIL"
                print(f"  - {ep_name}/{scenario}: {flag}  -> {r['message']}")
                                
                
# ==== Example driver (after your loading step) ====
# datasets: episode-first dict
# topologies: { "full_mesh": {...}, "clustered": {...}, "sparse_ring": {...} }

result_align = align_all_units_episode_first(
    datasets_ep_first=datasets,
    topologies_by_name=topologies
)
print_alignment_summary_episode_first(result_align)

print("\n ===EXAMPLE===")
aligned_light_ep0 = result_align["datasets_aligned"]["ep_000"]["light"]
agents_ep0_light  = aligned_light_ep0["agents"]   # has f_local_slot
tasks_ep0_light   = aligned_light_ep0["tasks"]    # has deadline_slots





# 1.4. Build Scenario–Topology Pairs

# In this step, all datasets are paired with all topologies (Cartesian product). 
# Each pair is checked for matching time parameters, then a basic bundle is created for further enrichment.

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
    datasets_ep_first: Dict[str, Dict[str, Dict[str, Any]]],
    topologies: Dict[str, Dict[str, Any]],
    strict_delta_match: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build pairs between every topology and every (episode, scenario) dataset.

    datasets_ep_first شکل:
        {
          "ep_000": {
             "light":   {"episodes": df, "agents": df, "arrivals": df, "tasks": df},
             "moderate":{...},
             "heavy":   {...},
             "_meta":   {...}  # only meta data without episodes/agents/tasks
          },
          ...
        }
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
        pairs_by_topology[topo_name] = {}

        # Compare with every (episode, scenario)
        for ep_name, scenarios in datasets_ep_first.items():
            pairs_by_topology[topo_name][ep_name] = {}
            for scen_name, ds in scenarios.items():
                if not (isinstance(ds, dict) and "episodes" in ds):
                    continue

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
            scen_map = by_ep[ep_name]
            if not scen_map:
                print(f"  ├─ Episode: {ep_name}  (no paired scenarios)")
                continue

            print(f"  ├─ Episode: {ep_name}")
            for scen_name in sorted(scen_map.keys()):
                bundle = scen_map[scen_name]
                flag  = "OK" if bundle["checks"]["delta_match"] else "FAIL"
                K     = bundle["K"]
                Delta = bundle["Delta"]
                msg   = bundle["checks"]["message"]
                print(f"  │    - [{flag}] {scen_name:9s} | K={K:2d}  Δ={Delta:g}  -> {msg}")
                

# --- Example driver (with your current variables) ---
result_align = align_all_units_episode_first(datasets_ep_first=datasets,
                                             topologies_by_name=topologies)

datasets_aligned = result_align["datasets_aligned"]

pairs_by_topology = build_topology_episode_pairs(
    datasets_ep_first=datasets_aligned,
    topologies=topologies,
    strict_delta_match=True
)

print_pairs_summary_topology_first_ep(pairs_by_topology)

print("\n ===EXAMPLE===")
tasks_light = pairs_by_topology["full_mesh"]["ep_000"]["light"]["dataset"]["tasks"]
cm_clustered = pairs_by_topology["clustered"]["ep_000"]["heavy"]["connection_matrix_df"]





# 1.5. Agent→MEC mapping (for all pairs)

# Agent → MEC Mapping assigns each agent to a specific MEC server.
# This creates a fixed mec_id for every agent (e.g., agent_id % K), 
# which determines where its tasks are initially queued and processed in the MDP environment.

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
print("\n ===EXAMPLE===")
pairs_by_topology["clustered"]["ep_000"]["heavy"]["dataset"]["agents"].head()





# 1.6. Environment Configuration

# In this step, we build a unified env_config for each scenario–topology pair.
# It bundles all required information for the MDP/RL environment—such as compute capacities, 
    # the Agent→MEC mapping, connection matrix, initial queue states, and action/state specifications—into
    # a single consistent configuration used by the RL training process.

def _extract_core_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract core fields from a (topology × episode × scenario) bundle.
    Ensures required fields exist and converts structures to numpy/DF formats.
    """
    required = ["dataset", "topology_data", "connection_matrix_df", "Delta", "K"]
    for k in required:
        if k not in bundle:
            raise ValueError(f"Bundle missing required key: '{k}'")

    ds = bundle["dataset"]
    topo = bundle["topology_data"]
    Mdf = bundle["connection_matrix_df"]

    private_cpu = np.asarray(topo["private_cpu_capacities"], dtype=float)
    public_cpu = np.asarray(topo["public_cpu_capacities"], dtype=float)
    cloud_cpu = float(topo["cloud_computational_capacity"])
    M = Mdf.to_numpy(dtype=float)  # shape = (K, K+1), last column = MEC→Cloud

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
    """
    Initial queue states for MEC and Cloud tiers, in per-slot units:
      - *_cycles store queued CPU cycles.
      - mec_bytes_in_transit stores bytes currently being transmitted through MEC links.
      - cloud_cycles stores queued cycles at the cloud.
    """
    return {
        "mec_local_cycles": np.zeros(K, dtype=float),
        "mec_public_cycles": np.zeros(K, dtype=float),
        "mec_bytes_in_transit": np.zeros(K, dtype=float),
        "cloud_cycles": np.array([0.0], dtype=float),
    }

def _derive_action_space() -> Dict[str, Any]:
    """
    Basic discrete offloading action space (HOODIE-style):
        0 = Execute locally
        1 = Offload to another MEC server
        2 = Offload to Cloud
    """
    return {
        "type": "discrete",
        "n": 3,
        "labels": {
            0: "LOCAL",
            1: "MEC",
            2: "CLOUD",
        },
    }
    
def _derive_state_spec(K: int) -> Dict[str, Any]:
    """
    Declarative specification of the RL state structure.
    The environment uses this to assemble numerical tensors each step.
    """
    return {
        "components": {
            "queues": {
                "mec_local_cycles": {"shape": (K,), "dtype": "float"},
                "mec_public_cycles": {"shape": (K,), "dtype": "float"},
                "mec_bytes_in_transit": {"shape": (K,), "dtype": "float"},
                "cloud_cycles": {"shape": (1,), "dtype": "float"},
            },
            "links": {
                "connection_matrix": {"shape": (K, K + 1), "dtype": "float"},
            },
            "capacities": {
                "private_cpu": {"shape": (K,), "dtype": "float"},
                "public_cpu": {"shape": (K,), "dtype": "float"},
                "cloud_cpu": {"shape": (1,), "dtype": "float"},
            },
        },
        "note": "Declarative state description; environment assembles numerical tensors at runtime.",
    }
    
def build_env_config_for_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the complete environment configuration structure for a single
    (topology × episode × scenario) bundle.

    Output 'env_config' includes:
        - Time parameters: Delta, T_slots
        - Topology specification: K, connection_matrix, topology_type
        - Resource capacities: private/public/cloud CPU
        - Agent-to-MEC assignment
        - Aligned dataset tables (episodes, agents, arrivals, tasks)
        - Initial queue states
        - Action space and state specification
        - Consistency check results
    """
    core = _extract_core_from_bundle(bundle)

    # Ensure agent-to-MEC mapping is present
    if "agent_to_mec" not in bundle:
        raise ValueError("Bundle missing 'agent_to_mec'. Run Stage 5 first.")

    # Normalize agent_to_mec to numpy array ordered by agent_id
    agent_to_mec = bundle["agent_to_mec"]
    if isinstance(agent_to_mec, pd.Series):
        if agent_to_mec.index.name != "agent_id":
            agent_to_mec.index.name = "agent_id"

        index_order = core["agents"].sort_values("agent_id")["agent_id"].to_numpy()
        agent_to_mec = agent_to_mec.reindex(index_order)
        agent_to_mec_arr = agent_to_mec.to_numpy(dtype=int)
    else:
        agent_to_mec_arr = np.asarray(agent_to_mec, dtype=int)

    # Validate correct length
    N_agents = int(core["episodes"]["N_agents"].iloc[0])
    if len(agent_to_mec_arr) != N_agents:
        raise ValueError(
            f"agent_to_mec length ({len(agent_to_mec_arr)}) != N_agents ({N_agents})"
        )

    # Extract simulation horizon
    if "T_slots" not in core["episodes"].columns:
        raise ValueError("episodes.csv must contain 'T_slots'.")
    T_slots = int(core["episodes"]["T_slots"].iloc[0])

    # Build initial states and specifications
    queues_initial = _build_default_queues(core["K"])
    action_space = _derive_action_space()
    state_spec = _derive_state_spec(core["K"])

    # Final environment configuration object
    env_config = {
        "Delta": core["Delta"],
        "T_slots": T_slots,
        "K": core["K"],
        "topology_type": core["topology_type"],
        "connection_matrix": core["connection_matrix"],

        "private_cpu": core["private_cpu"],
        "public_cpu": core["public_cpu"],
        "cloud_cpu": core["cloud_cpu"],

        "N_agents": N_agents,
        "agent_to_mec": agent_to_mec_arr,

        # Datasets (aligned)
        "episodes": core["episodes"],
        "agents": core["agents"],
        "arrivals": core["arrivals"],
        "tasks": core["tasks"],

        # Initial queue states and specifications
        "queues_initial": queues_initial,
        "action_space": action_space,
        "state_spec": state_spec,

        # Validation results from delta/time-step checks
        "checks": bundle.get("checks", {"delta_match": True, "message": "n/a"}),
    }
    return env_config

def build_all_env_configs(
    pairs_by_topology: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    Build environment configurations for all (topology × episode × scenario) bundles.

    Result shape (episode-first):
        env_configs[episode][topology][scenario] = env_config
    """
    out: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    for topo_name, by_ep in pairs_by_topology.items():
        for ep_name, by_scen in by_ep.items():
            if ep_name not in out:
                out[ep_name] = {}
            if topo_name not in out[ep_name]:
                out[ep_name][topo_name] = {}

            for scen_name, bundle in by_scen.items():
                if "agent_to_mec" not in bundle:
                    raise RuntimeError(
                        f"[{topo_name}/{ep_name}/{scen_name}] missing 'agent_to_mec'. "
                        "Run Stage 5 first."
                    )
                env_cfg = build_env_config_for_bundle(bundle)
                out[ep_name][topo_name][scen_name] = env_cfg

    return out


# Build all environment configs
env_configs = build_all_env_configs(pairs_by_topology)

# Example
print("\n=== EXAMPLE ===")
print(env_configs["ep_000"]["clustered"]["heavy"]["agent_to_mec"])
print("T_slots:", env_configs["ep_000"]["clustered"]["heavy"]["T_slots"])
print("Initial queues:", env_configs["ep_000"]["clustered"]["heavy"]["queues_initial"].keys())





# 1.7. Sanity Checks

# In this step, we verify that each env_config is internally consistent 
    # (queue shapes, capacities, agent→MEC mapping, and connection matrix are valid and ready for simulation).
    
def sanity_check_env_config(env_config: Dict[str, Any]) -> list:
    """
    Run basic sanity checks on a single env_config dictionary.
    Returns a list of error strings; empty list means 'no issues found'.
    """
    errors = []

    # 1) Agent → MEC alignment
    N_agents = env_config["N_agents"]
    agent_to_mec = np.asarray(env_config["agent_to_mec"], dtype=int)
    if len(agent_to_mec) != N_agents:
        errors.append("Length of agent_to_mec does not match N_agents.")
    # All MEC indices must be within [0, K-1]
    K = env_config["K"]
    if (agent_to_mec < 0).any() or (agent_to_mec >= K).any():
        errors.append("agent_to_mec contains indices outside [0, K-1].")

    # 2) Queue initial state shapes
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
    if M.shape != (K, K + 1):
        errors.append("connection_matrix shape mismatch (expected K x (K+1)).")

    # 5) Action space correctness
    action_space = env_config.get("action_space", {})
    if action_space.get("type", None) != "discrete":
        errors.append("Action space must be discrete (LOCAL/MEC/CLOUD).")
    if action_space.get("n", None) != 3:
        errors.append("Action space 'n' must be 3 (LOCAL/MEC/CLOUD).")

    # 6) Basic time parameters
    Delta = float(env_config.get("Delta", -1.0))
    T_slots = int(env_config.get("T_slots", -1))
    if not np.isfinite(Delta) or Delta <= 0:
        errors.append(f"Invalid Delta in env_config (got {Delta}).")
    if T_slots <= 0:
        errors.append(f"Invalid T_slots in env_config (got {T_slots}).")

    return errors

def sanity_check_all(env_configs: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]) -> None:
    """
    Run sanity_check_env_config over all env_config instances.

    env_configs shape (episode-first):
        env_configs[episode][topology][scenario] = env_config
    """
    print("=== SANITY CHECK OVER ALL ENV CONFIGS ===")
    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():
                errs = sanity_check_env_config(env_cfg)
                if errs:
                    print(f"[FAIL] {ep_name}/{topo_name}/{scen_name}:")
                    for e in errs:
                        print("   -", e)
                else:
                    print(f"[OK]   {ep_name}/{topo_name}/{scen_name}")
                    
                    
# Run all sanity checks
sanity_check_all(env_configs)

print("\n=== EXAMPLE TASK TABLE ===")
print(env_configs["ep_000"]["clustered"]["heavy"]["tasks"])


# Saving the Information
def _summarize_array(arr, max_items=6):
    """Return a short, readable summary string for numpy arrays."""
    try:
        arr = np.asarray(arr)
        base = f"ndarray shape={arr.shape}, dtype={arr.dtype}"
        if arr.size == 0:
            return base + " | empty"

        # If small 1D vector, show full values
        if arr.ndim == 1 and arr.size <= max_items:
            return base + f" | values={arr.tolist()}"

        # If numeric, show basic stats
        if np.issubdtype(arr.dtype, np.number):
            return (
                base +
                f" | min={np.nanmin(arr):.4g}, max={np.nanmax(arr):.4g}, mean={np.nanmean(arr):.4g}"
            )

        return base
    except Exception as e:
        return f"(array summary failed: {e})"

def _summarize_df(df: pd.DataFrame, max_cols=10):
    """Return a short summary string for DataFrames."""
    try:
        cols = df.columns.tolist()
        cols_show = cols[:max_cols] + (["..."] if len(cols) > max_cols else [])
        return f"DataFrame shape={df.shape}, columns={cols_show}"
    except Exception as e:
        return f"(dataframe summary failed: {e})"

def _summarize_any(name, obj, indent="    "):
    """
    Produce a few readable summary lines depending on the object type.
    Used recursively for nested dicts (e.g., queues_initial).
    """
    lines = []

    if isinstance(obj, pd.DataFrame):
        lines.append(f"{indent}{name}: {_summarize_df(obj)}")

    elif isinstance(obj, np.ndarray):
        lines.append(f"{indent}{name}: {_summarize_array(obj)}")

    elif isinstance(obj, (list, tuple)):
        preview = obj[:6] if len(obj) > 6 else obj
        lines.append(f"{indent}{name}: list len={len(obj)}, preview={preview}")

    elif isinstance(obj, dict):
        lines.append(f"{indent}{name}: dict keys={list(obj.keys())}")

        # Dive deeper for small dicts or queue dictionaries
        if name == "queues_initial" or len(obj) <= 6:
            for k, v in obj.items():
                sub = _summarize_any(k, v, indent=indent + "  ")
                if isinstance(sub, list):
                    lines.extend(sub)
                else:
                    lines.append(sub)

    elif isinstance(obj, (int, float, str, bool, type(None))):
        lines.append(f"{indent}{name}: {repr(obj)}")

    else:
        # Fallback: try converting to array
        try:
            arr = np.asarray(obj)
            lines.append(f"{indent}{name}: {_summarize_array(arr)}")
        except Exception:
            lines.append(f"{indent}{name}: ({type(obj).__name__})")

    return lines

def save_env_configs_text(env_configs, out_path="./artifacts/env_configs_summary.txt"):
    """
    Save a human-readable summary of all env_configs:
        env_configs[episode][topology][scenario] = env_config

    The summary includes:
    - key scalar parameters (Delta, K, N_agents, topology_type)
    - shapes and stats of numeric arrays
    - summary of DataFrames (episodes, agents, arrivals, tasks)
    - queue initialization
    - RL descriptors (action_space, state_spec)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = []
    lines.append("=== ENV CONFIGS SUMMARY (episode → topology → scenario) ===\n")

    # deterministic ordering for reproducible summaries
    for ep_name in sorted(env_configs.keys()):
        lines.append(f"[EPISODE] {ep_name}")
        by_topo = env_configs[ep_name]

        for topo_name in sorted(by_topo.keys()):
            lines.append(f"  [TOPOLOGY] {topo_name}")
            by_scen = by_topo[topo_name]

            for scen_name in sorted(by_scen.keys()):
                env_cfg = by_scen[scen_name]
                lines.append(f"    [SCENARIO] {scen_name}")

                # -- important scalars --
                for key in ["Delta", "K", "N_agents", "topology_type"]:
                    if key in env_cfg:
                        lines.extend(_summarize_any(key, env_cfg[key], indent="      "))

                # -- main tensors/arrays --
                for key in [
                    "connection_matrix", "private_cpu", "public_cpu",
                    "cloud_cpu", "agent_to_mec"
                ]:
                    if key in env_cfg:
                        lines.extend(_summarize_any(key, env_cfg[key], indent="      "))

                # -- dataframes --
                for key in ["episodes", "agents", "arrivals", "tasks"]:
                    if key in env_cfg:
                        lines.extend(_summarize_any(key, env_cfg[key], indent="      "))

                # -- RL descriptors and queues --
                for key in ["queues_initial", "action_space", "state_spec", "checks"]:
                    if key in env_cfg:
                        lines.extend(_summarize_any(key, env_cfg[key], indent="      "))

                lines.append("")  # blank line after scenario

            lines.append("")  # blank line after topology

        lines.append("")  # blank line after episode

    # Write file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[saved] env_configs summary → {out_path}")

# --------- Usage ---------
save_env_configs_text(env_configs, out_path="./artifacts/env_configs_summary.txt")

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
print("\n ===EXAMPLE===")
labeled_tasks = env_configs["ep_000"]["clustered"]["heavy"]["tasks"]
print(labeled_tasks.head())
print(labeled_tasks.info())





# 2.2. Task Type Classification

# Pre-req: tasks already labeled by your previous step: 
#   size_bucket, compute_bucket, mem_bucket, urgency, atomicity, split_bucket, routing_hint, etc.

def _derive_task_type_row(row: pd.Series) -> tuple[str, str, str, list, str]:
    """
    Returns (task_type, task_subtype, type_reason, multi_flags, final_flag)
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
        final_flag = "deadline_hard"
        return ("deadline_hard", "deadline_hard", "hard deadline (tight slots)", multi_flags, final_flag)

    # 2) Latency-sensitive (soft deadlines / delay-sensitive)
    if latency_flag:
        final_flag = "latency_sensitive"
        return ("latency_sensitive", "deadline_soft", "delay-sensitive (soft deadline)", multi_flags, final_flag)

    # 3) Compute-intensive (c or rho or mem heavy)
    if compute_heavy or memory_heavy:
        final_flag = "compute_intensive"
        return ("compute_intensive", "compute_or_memory_heavy", "high compute/memory demand", multi_flags, final_flag)

    # 4) Data-intensive (mainly large input size / high IO pressure)
    if io_heavy:
        final_flag = "data_intensive"
        return ("data_intensive", "large_input_bandwidth", "large data volume / IO heavy", multi_flags, final_flag)

    # 5) Otherwise general
    final_flag = "general"
    return ("general", "general", "no dominant constraint", multi_flags, final_flag)

def apply_ch4_task_typing(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Chapter-4 level task classes with priority rules into tasks_df (returns a COPY).
    Columns added:
      - task_type            (5-way class)
      - task_subtype         (finer descriptor)
      - type_reason          (short textual rationale)
      - multi_flags          (list of all active boolean traits)
      - final_flag           (single flag representing the task's priority class)
    """
    df = tasks_df.copy()

    # Ensure the expected helper columns exist (created in your previous labeling step).
    required_cols = ["urgency", "compute_heavy", "memory_heavy", "io_heavy", "atomicity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"apply_ch4_task_typing: missing label columns: {missing}")

    out_type, out_sub, out_reason, out_flags, out_final_flag = [], [], [], [], []
    for _, r in df.iterrows():
        t, s, msg, flags, final_flag = _derive_task_type_row(r)
        out_type.append(t)
        out_sub.append(s)
        out_reason.append(msg)
        out_flags.append(flags)
        out_final_flag.append(final_flag)

    df["task_type"]   = out_type
    df["task_subtype"]= out_sub
    df["type_reason"] = out_reason
    df["multi_flags"] = out_flags
    df["final_flag"]  = out_final_flag  # Add the final flag to represent the primary category

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
print("\n ===EXAMPLE===")
print(env_configs["ep_000"]["clustered"]["heavy"]["tasks"][["task_id","task_type","task_subtype","type_reason","multi_flags", "final_flag"]].head(25))


# none → Tasks that do not have a specific deadline or time sensitivity </br>
# hard → Tasks that have a very limited deadline and delay is very important to them

labeled_tasks_completed = env_configs["ep_000"]["clustered"]["heavy"]["tasks"]
print(labeled_tasks_completed["urgency"].value_counts())
print("\n", labeled_tasks_completed["task_type"].value_counts())
print("\n", labeled_tasks_completed.groupby("task_type")[["b_mb","rho_cyc_per_mb","mem_mb"]].median())

print(labeled_tasks_completed.head())
print(labeled_tasks_completed.info())





# Step 3: Agent Profiling

# In this step, we construct a behavioral profile for each agent, capturing its local compute resources, 
    # task arrival rate, and the distribution of task types it generates. These profiles are later used for 
    # clustering agents and assigning suitable reinforcement learning strategies to each group.

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
print("\n ===EXAMPLE===")
print(agent_profiles["ep_000"]["clustered"]["heavy"].head())

# Alternatively, read directly from env_configs:
print(env_configs["ep_000"]["clustered"]["heavy"]["agent_profiles"].head(25))


ep   = "ep_000"
topo = "clustered"
scen = "heavy"

env_cfg = env_configs[ep][topo][scen]

print(env_cfg.keys())

prof = env_cfg["agent_profiles"]
print(prof.head())
print(prof.columns)


# Saving Information
# def _ensure_dir(path: str):
#     """Create a folder if it does not already exist."""
#     os.makedirs(path, exist_ok=True)

# def _serialize_non_df_components(env_cfg: dict) -> dict:
#     """
#     Prepare a JSON-serializable dictionary for all non-DataFrame parts
#     of env_config. Arrays are converted to lists.
#     """
#     out = {}
#     for key, value in env_cfg.items():
#         if isinstance(value, pd.DataFrame):
#             continue  # handled separately

#         # numpy arrays → lists
#         if isinstance(value, np.ndarray):
#             out[key] = value.tolist()
#             continue

#         # dicts (queues, action_space, state_spec, checks)
#         if isinstance(value, dict):
#             try:
#                 # recursively convert numpy arrays inside dicts
#                 def _convert(obj):
#                     if isinstance(obj, np.ndarray):
#                         return obj.tolist()
#                     if isinstance(obj, dict):
#                         return {k: _convert(v) for k, v in obj.items()}
#                     return obj
#                 out[key] = _convert(value)
#             except Exception as e:
#                 out[key] = f"(serialization error: {e})"
#             continue

#         # scalars (int, float, str, None)
#         if isinstance(value, (int, float, str, bool, type(None))):
#             out[key] = value
#             continue

#         # Handle StandardScaler separately by serializing only its mean and scale
#         if isinstance(value, StandardScaler):
#             out[key] = {
#                 "mean": value.mean_.tolist(),
#                 "scale": value.scale_.tolist()
#             }
#             continue

#         # fallback
#         try:
#             out[key] = json.loads(json.dumps(value))
#         except Exception:
#             out[key] = f"(unserializable type: {type(value).__name__})"

#     return out

# def save_all_env_configs(env_configs, out_root: str = "./artifacts/env_configs"):
#     """
#     Save all env_configs to disk in a structured layout:
#         artifacts/env_configs/ep_xxx/topology/scenario/
#             tasks_env_config.csv
#             agents_env_config.csv
#             arrivals_env_config.csv
#             episodes_env_config.csv
#             env_meta.json   <-- (non-DF components)
#     """
#     n_saved = 0

#     for ep_name, by_topo in env_configs.items():
#         for topo_name, by_scen in by_topo.items():
#             for scen_name, env_cfg in by_scen.items():

#                 out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
#                 _ensure_dir(out_dir)

#                 # ---- Save DataFrame components ----
#                 for df_name, df in env_cfg.items():
#                     if isinstance(df, pd.DataFrame):
#                         file_path_csv = os.path.join(out_dir, f"{df_name}_env_config.csv")
#                         df.to_csv(file_path_csv, index=False)

#                         print(f"[saved] {file_path_csv}  (rows={len(df)})")
#                         n_saved += 1

#                 # ---- Save non-DataFrame metadata ----
#                 meta = _serialize_non_df_components(env_cfg)

#                 meta_path = os.path.join(out_dir, "env_meta.json")
#                 with open(meta_path, "w", encoding="utf-8") as f:
#                     json.dump(meta, f, indent=2)

#                 print(f"[saved] {meta_path}")

#     print(f"\nDone. Saved {n_saved} DataFrames + meta files for all env_configs.")
    

# save_all_env_configs(env_configs, out_root="./artifacts/env_configs")






# Step 4: Clustering Agents

# STEP 4.1 – Agent Feature Matrix Construction

# The feature matrix combines two key components:
#   (1) Local resource capacities (hardware characteristics)
#   (2) Task generation patterns (behavioral characteristics)
# 
# The resulting matrix (X) is used for clustering agents in Step 4.2.

AGENT_FEATURES_V1 = [
    # ---- (1) Local resources ----
    "f_local_slot",   # Local CPU cycles per slot
    "m_local",        # Local memory capacity
    "lambda_mean",    # Mean task arrival rate
    "lambda_var",     # Variance of task arrival rate

    # ---- (2) Task generation pattern ----
    # Derived from labeled task distribution across final_flag categories
    "P_deadline_hard",
    "P_latency_sensitive",
    "P_compute_intensive",
    "P_data_intensive",
    "P_general",

    # ---- (optional statistical descriptors of generated tasks) ----
    "b_mb_med",       # Median input size
    "rho_med",        # Median compute demand (cycles/MB)
    "mem_med",        # Median memory demand (MB)
    "non_atomic_share", # Share of splittable tasks
    "hard_share"        # Share of hard-deadline tasks
]

# Utility: Keep only existing columns; others will be filled with zeros
def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

# Build feature matrix for one environment configuration
# def make_agent_feature_matrix_for_env(
#     env_cfg: Dict[str, Any],
#     feature_list: Optional[List[str]] = None,
#     standardize: bool = True,
# ) -> Tuple[np.ndarray, List[str], np.ndarray, Optional[StandardScaler]]:
#     """
#     Build the feature matrix (X) for all agents in one environment configuration.
#     Each row represents an agent; each column a numerical feature.
    
#     Returns:
#         X_scaled       : np.ndarray (n_agents × n_features)
#         used_cols      : list of feature names in order
#         agent_ids      : np.ndarray of agent identifiers
#         scaler         : fitted StandardScaler object (or None if not standardized)
#     """
#     if "agent_profiles" not in env_cfg or not isinstance(env_cfg["agent_profiles"], pd.DataFrame):
#         raise ValueError("env_cfg['agent_profiles'] must contain a valid DataFrame.")

#     prof = env_cfg["agent_profiles"].copy()
#     if "agent_id" not in prof.columns:
#         raise ValueError("agent_profiles must include column 'agent_id'.")

#     if feature_list is None:
#         feature_list = AGENT_FEATURES_V1

#     # Keep valid features and fill missing ones with zeros
#     cols = _safe_cols(prof, feature_list)
#     X = prof.reindex(columns=cols).fillna(0.0).astype(float).to_numpy()
#     agent_ids = prof["agent_id"].to_numpy(dtype=int)

#     # Standardize features (mean=0, std=1) for clustering stability
#     scaler = None
#     if standardize and X.shape[0] > 0:
#         scaler = StandardScaler()
#         X = scaler.fit_transform(X)

#     return X, cols, agent_ids, scaler

# Build feature matrix for one environment configuration
def make_agent_feature_matrix_for_env(
    env_cfg: Dict[str, Any],
    feature_list: Optional[List[str]] = None,
    standardize: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Build the feature matrix (X) for all agents in one environment configuration.
    Each row represents an agent; each column a numerical feature.
    
    Returns:
        X_scaled       : np.ndarray (n_agents × n_features)
        used_cols      : list of feature names in order
        agent_ids      : np.ndarray of agent identifiers
    """
    if "agent_profiles" not in env_cfg or not isinstance(env_cfg["agent_profiles"], pd.DataFrame):
        raise ValueError("env_cfg['agent_profiles'] must contain a valid DataFrame.")

    prof = env_cfg["agent_profiles"].copy()
    if "agent_id" not in prof.columns:
        raise ValueError("agent_profiles must include column 'agent_id'.")

    if feature_list is None:
        feature_list = AGENT_FEATURES_V1

    # Keep valid features and fill missing ones with zeros
    cols = _safe_cols(prof, feature_list)
    X = prof.reindex(columns=cols).fillna(0.0).astype(float).to_numpy()
    agent_ids = prof["agent_id"].to_numpy(dtype=int)

    # Standardize features (mean=0, std=1) for clustering stability
    if standardize and X.shape[0] > 0:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Nomalize the data
        # Don't store scaler, we will only store the scaled data
    else:
        scaler = None

    return X, cols, agent_ids

# Attach computed features to the environment configuration
# def attach_features_to_env(env_cfg: Dict[str, Any],
#                            feature_list: Optional[List[str]] = None,
#                            standardize: bool = True) -> Dict[str, Any]:
#     """
#     Attach the constructed feature matrix and related metadata
#     to env_cfg["clustering"]["features"].
#     """
#     X, cols, agent_ids, scaler = make_agent_feature_matrix_for_env(env_cfg, feature_list, standardize)

#     env_cfg.setdefault("clustering", {})
#     env_cfg["clustering"]["features"] = {
#         "X": X,                          # Feature matrix (scaled)
#         "feature_cols": cols,            # List of column names
#         "agent_ids": agent_ids,          # Agent identifiers
#         "scaler": scaler,                # StandardScaler (for later inverse transform)
#         "n_agents": int(X.shape[0]),
#         "n_features": int(X.shape[1]),
#     }
#     return env_cfg

def attach_features_to_env(env_cfg: Dict[str, Any],
                           feature_list: Optional[List[str]] = None,
                           standardize: bool = True) -> Dict[str, Any]:
    """
    Attach the constructed feature matrix and related metadata
    to env_cfg["clustering"]["features"].
    """
    X, cols, agent_ids, scaler = make_agent_feature_matrix_for_env(env_cfg, feature_list, standardize)

    # We do not store 'scaler' as it is not JSON serializable
    env_cfg.setdefault("clustering", {})
    env_cfg["clustering"]["features"] = {
        "X": X,                          # Feature matrix (scaled)
        "feature_cols": cols,            # List of column names
        "agent_ids": agent_ids,          # Agent identifiers
        "n_agents": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return env_cfg

# Apply feature construction to all topology-scenario combinations
def attach_features_to_all_envs(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    feature_list: Optional[List[str]] = None,
    standardize: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Iterate through all (episode → topology → scenario) combinations
    and build the feature matrix for each one.
    """
    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():
                env_configs[ep_name][topo_name][scen_name] = attach_features_to_env(
                    env_cfg, feature_list, standardize
                )
                fz = env_configs[ep_name][topo_name][scen_name]["clustering"]["features"]
                print(f"[features] {ep_name}/{topo_name}/{scen_name} "
                      f"-> X.shape={fz['X'].shape}  (agents={fz['n_agents']}, feats={fz['n_features']})")
    return env_configs


def _assert_no_nan_inf(X: np.ndarray, where: str):
    if not np.isfinite(X).all():
        bad = np.isnan(X).sum(), np.isinf(X).sum()
        raise AssertionError(f"{where}: Feature matrix contains NaN or Inf. counts={bad}")

def _assert_agent_count_match(env_cfg: Dict[str, Any], where: str):
    n_agents_ep = int(env_cfg["episodes"]["N_agents"].iloc[0])
    n_agents_prof = len(env_cfg["agent_profiles"])
    fz = env_cfg["clustering"]["features"]
    if not (fz["n_agents"] == n_agents_prof == n_agents_ep):
        raise AssertionError(
            f"{where}: Agent count mismatch. episodes={n_agents_ep}, "
            f"profiles={n_agents_prof}, X={fz['n_agents']}"
        )

def _assert_feature_prob_sum_hint(env_cfg: Dict[str, Any], tol=1e-3):
    prof = env_cfg["agent_profiles"]
    if "TaskDist_sum" in prof.columns and "n_tasks_agent" in prof.columns:
        mask = prof["n_tasks_agent"] > 0
        if mask.any():
            mean_sum = float(prof.loc[mask, "TaskDist_sum"].mean())
            if abs(mean_sum - 1.0) > tol:
                print(f"[warn] Mean(TaskDist_sum)={mean_sum:.4f} ≠ 1 (tol={tol})")

def run_feature_matrix_sanity_checks(env_configs: Dict[str, Dict[str, Dict[str, Any]]]):
    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():
                where = f"{ep_name}/{topo_name}/{scen_name}"
                if "clustering" not in env_cfg or "features" not in env_cfg["clustering"]:
                    raise AssertionError(f"{where}: Missing features.")
                X = env_cfg["clustering"]["features"]["X"]
                _assert_no_nan_inf(X, where)
                _assert_agent_count_match(env_cfg, where)
                _assert_feature_prob_sum_hint(env_cfg)
                if X.shape[1] == 0:
                    raise AssertionError(f"{where}: Empty feature matrix.")
    print("[checks] All sanity checks passed successfully.")
        

# Build feature matrices for all envs
env_configs = attach_features_to_all_envs(env_configs, feature_list=AGENT_FEATURES_V1, standardize=True)

# Run sanity checks
run_feature_matrix_sanity_checks(env_configs)

# Example inspection
print("\n=== EXAMPLE: features of ep_000 / clustered / heavy ===")
fz = env_configs["ep_000"]["clustered"]["heavy"]["clustering"]["features"]
print("X.shape:", fz["X"].shape)
print("feature_cols:", fz["feature_cols"])
print("agent_ids (first 10):", fz["agent_ids"][:10])


print("missing features:",
      [c for c in AGENT_FEATURES_V1 if c not in prof.columns])

print(prof[ [c for c in AGENT_FEATURES_V1 if c in prof.columns] ].head())


fz = env_configs["ep_000"]["clustered"]["heavy"]["clustering"]["features"]
print("X.shape:", fz["X"].shape)
print("feature_cols actually used:", fz["feature_cols"])
print("agent_ids[:10]:", fz["agent_ids"][:10])





# 4.2. Optimal Number of Clusters

# Uses feature matrix from Step 4.1:
#   env_cfg["clustering"]["features"] = {
#       "X", "feature_cols", "agent_ids",
#       "scaler", "n_agents", "n_features"
#   }
#
# For each (episode → topology → scenario) we:
#   1) Build candidate K set based on n_agents.
#   2) Run KMeans for each K.
#   3) Compute metrics:
#        - inertia (WCSS)
#        - silhouette
#        - Calinski–Harabasz
#        - Davies–Bouldin
#   4) Normalize them and compute composite score:
#        Score(K) = α·Sil_norm(K) + β·CH_norm(K) − γ·DB_norm(K)
#   5) Compute elbow_score from inertia curvature.
#   6) Select:
#        best_K  = argmax(Score(K))
#        K_elbow = argmax(elbow_score)
#   7) Save plot: elbow + composite score per triple:
#        ./artifacts/clustering/<ep>/<topology>/<scenario>/elbow_and_score.png
#
# Results are attached to:
#   env_cfg["clustering"]["k_selection"] = {
#       "metrics_df", "best_K", "K_elbow", "elbow_plot_path"
#   }
# And returned as:
#   K_selection[ep][topology][scenario] = {...}
# ===============================================

# ---------- 4.2.1 Candidate K values ----------
def _candidate_K_values(
    n_agents: int,
    k_min: int = 2,
    max_K_fraction: float = 0.25,
    max_K_abs: int = 10
) -> List[int]:
    """
    Build a reasonable candidate set for K given n_agents.

    - Lower bound is k_min (default 2).
    - Upper bound is min(max_K_abs, floor(max_K_fraction * n_agents), n_agents - 1).
    - If n_agents is too small, returns an empty list.
    """
    if n_agents <= k_min:
        return []

    k_max_by_fraction = int(np.floor(max_K_fraction * n_agents))
    k_max = min(max_K_abs, n_agents - 1, max(k_min, k_max_by_fraction))

    if k_max < k_min:
        return []

    return list(range(k_min, k_max + 1))

# ---------- 4.2.2 Evaluate KMeans for a single K ----------
def _evaluate_kmeans_for_K(
    X: np.ndarray,
    K: int,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Run KMeans for a given K and compute clustering metrics.

    Returns:
        {
          "K": int,
          "inertia": float,
          "silhouette": float or np.nan,
          "davies_bouldin": float or np.nan,
          "calinski_harabasz": float or np.nan,
        }
    """
    n_samples = X.shape[0]
    result = {
        "K": int(K),
        "inertia": np.nan,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }

    if K <= 1 or K > n_samples:
        return result

    try:
        km = KMeans(
            n_clusters=K,
            random_state=random_state,
            n_init="auto"
        )
        labels = km.fit_predict(X)
        result["inertia"] = float(km.inertia_)

        unique_labels = np.unique(labels)
        if unique_labels.shape[0] > 1:
            # Silhouette
            try:
                result["silhouette"] = float(silhouette_score(X, labels))
            except Exception:
                result["silhouette"] = np.nan

            # Davies–Bouldin
            try:
                result["davies_bouldin"] = float(davies_bouldin_score(X, labels))
            except Exception:
                result["davies_bouldin"] = np.nan

            # Calinski–Harabasz
            try:
                result["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
            except Exception:
                result["calinski_harabasz"] = np.nan

    except Exception as e:
        print(f"[warn] KMeans failed for K={K}: {e}")

    return result

# ---------- 4.2.3 Min-max normalization ----------
def _min_max_normalize(
    arr: np.ndarray,
    invert: bool = False
) -> np.ndarray:
    """
    Min-max normalize a 1D array to [0, 1].

    - If all values are NaN or the range is zero, returns NaN array.
    - If invert=True, larger original values map to lower normalized ones.
      (Useful if 'smaller is better' in the original metric.)
    """
    arr = np.asarray(arr, dtype=float)
    if np.all(np.isnan(arr)):
        return np.full_like(arr, np.nan)

    valid = ~np.isnan(arr)
    if valid.sum() <= 1:
        return np.full_like(arr, np.nan)

    vmin = np.nanmin(arr[valid])
    vmax = np.nanmax(arr[valid])
    if vmax - vmin == 0:
        return np.full_like(arr, np.nan)

    norm = (arr - vmin) / (vmax - vmin)
    if invert:
        norm = 1.0 - norm
    return norm

# ---------- 4.2.4 Add composite score ----------
def _add_composite_score(
    metrics_df: pd.DataFrame,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2
) -> pd.DataFrame:
    """
    Given metrics_df with:
        K, inertia, silhouette, calinski_harabasz, davies_bouldin

    Add:
        sil_norm, ch_norm, db_norm, score
    where:
        score = alpha * sil_norm + beta * ch_norm - gamma * db_norm
    """
    df = metrics_df.copy().reset_index(drop=True)

    sil = df["silhouette"].to_numpy(dtype=float)
    ch  = df["calinski_harabasz"].to_numpy(dtype=float)
    db  = df["davies_bouldin"].to_numpy(dtype=float)

    # Silhouette: higher is better
    df["sil_norm"] = _min_max_normalize(sil, invert=False)

    # Calinski–Harabasz: higher is better
    df["ch_norm"] = _min_max_normalize(ch, invert=False)

    # Davies–Bouldin: lower is better → we subtract db_norm in the score
    df["db_norm"] = _min_max_normalize(db, invert=False)

    df["score"] = (
        alpha * df["sil_norm"].fillna(0.0)
        + beta * df["ch_norm"].fillna(0.0)
        - gamma * df["db_norm"].fillna(0.0)
    )

    return df

# ---------- 4.2.5 Elbow score from curvature ----------
def _compute_elbow_rank(metrics_df: pd.DataFrame) -> np.ndarray:
    """
    Compute a simple 'elbow_score' based on curvature of inertia vs K.

    - Normalize inertia to [0,1].
    - Use discrete second derivative:
        curvature_i ≈ |y_{i-1} - 2*y_i + y_{i+1}|
    - Endpoints get curvature 0.
    """
    df = metrics_df.sort_values("K").reset_index(drop=True)
    inertia = df["inertia"].to_numpy(dtype=float)

    if inertia.shape[0] < 3:
        return np.zeros_like(inertia, dtype=float)

    inertia_norm = _min_max_normalize(inertia, invert=False)
    if np.all(np.isnan(inertia_norm)):
        return np.zeros_like(inertia, dtype=float)

    curv = np.zeros_like(inertia_norm, dtype=float)
    for i in range(1, len(inertia_norm) - 1):
        y_prev = inertia_norm[i - 1]
        y_curr = inertia_norm[i]
        y_next = inertia_norm[i + 1]
        if np.isnan(y_prev) or np.isnan(y_curr) or np.isnan(y_next):
            curv[i] = 0.0
        else:
            curv[i] = abs(y_prev - 2.0 * y_curr + y_next)

    curv_norm = _min_max_normalize(curv, invert=False)
    curv_norm = np.nan_to_num(curv_norm, nan=0.0)
    return curv_norm

# ---------- 4.2.6 Select best_K and K_elbow ----------
def _select_best_K_from_df(metrics_with_scores: pd.DataFrame) -> Tuple[int, int]:
    """
    Given metrics_with_scores with columns:
        K, score, elbow_score

    Returns:
        best_K  : K with max composite score  (tie → smallest K)
        K_elbow : K with max elbow_score     (tie → smallest K)
    """
    df = metrics_with_scores.sort_values("K").reset_index(drop=True)

    # best_K from composite score
    if df["score"].notna().any():
        idx_best = df["score"].idxmax()
    else:
        idx_best = df["K"].idxmin()
    best_K = int(df.loc[idx_best, "K"])

    # K_elbow from elbow_score
    if "elbow_score" in df.columns and df["elbow_score"].notna().any():
        idx_elb = df["elbow_score"].idxmax()
    else:
        idx_elb = idx_best
    K_elbow = int(df.loc[idx_elb, "K"])

    return best_K, K_elbow

# ---------- 4.2.7 Plot elbow + composite score ----------
def _plot_elbow_and_scores(
    metrics_df: pd.DataFrame,
    ep_name: str,
    topo_name: str,
    scen_name: str,
    out_root: str = "./artifacts/clustering"
) -> str:
    """
    Plot:
      - inertia (WCSS) vs K  → classic elbow
      - composite score vs K

    Save under:
      ./artifacts/clustering/<ep>/<topology>/<scenario>/elbow_and_score.png
    """
    df = metrics_df.sort_values("K").reset_index(drop=True)

    Ks      = df["K"].to_numpy(dtype=int)
    inertia = df["inertia"].to_numpy(dtype=float)
    scores  = df["score"].to_numpy(dtype=float)

    out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "elbow_and_score.png")

    plt.figure(figsize=(6, 4))

    # Left axis: inertia (WCSS)
    ax1 = plt.gca()
    ax1.plot(Ks, inertia, marker="o", linestyle="-", label="WCSS (inertia)")
    ax1.set_xlabel("Number of clusters K")
    ax1.set_ylabel("WCSS (inertia)")

    # Right axis: composite score
    ax2 = ax1.twinx()
    ax2.plot(Ks, scores, marker="s", linestyle="--", label="Composite score")
    ax2.set_ylabel("Composite score")

    title = f"Elbow & Score: {ep_name} / {topo_name} / {scen_name}"
    ax1.set_title(title)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path

# ---------- 4.2.8 Main driver over all env_configs ----------
def step4_2_select_K_for_all_envs(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    random_state: int = 42,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Step 4.2: K-selection for all environments.

    For each:
        env_configs[ep_name][topology_name][scenario_name]["clustering"]["features"]["X"]
    we:
      - build candidate K list
      - run KMeans for each K
      - compute metrics + composite score
      - compute elbow_score
      - select best_K and K_elbow
      - plot & save elbow figure
      - store results in:
            env_cfg["clustering"]["k_selection"]

    Returns:
      K_selection[ep_name][topology_name][scenario_name] = {
        "best_K", "K_elbow", "metrics_df", "elbow_plot_path"
      }
    """
    K_selection: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ep_name, by_topo in env_configs.items():
        K_selection[ep_name] = {}
        for topo_name, by_scen in by_topo.items():
            K_selection[ep_name][topo_name] = {}
            for scen_name, env_cfg in by_scen.items():

                # Check clustering features exist
                clust = env_cfg.get("clustering", {})
                feats = clust.get("features", None)
                if feats is None or "X" not in feats:
                    if verbose:
                        print(f"[4.2/skip] {ep_name}/{topo_name}/{scen_name}: "
                              f"no clustering features found.")
                    continue

                X = np.asarray(feats["X"], dtype=float)
                if X.ndim != 2 or X.shape[0] == 0:
                    if verbose:
                        print(f"[4.2/skip] {ep_name}/{topo_name}/{scen_name}: "
                              f"empty or invalid feature matrix.")
                    continue

                n_agents = X.shape[0]
                K_candidates = _candidate_K_values(n_agents)
                if not K_candidates:
                    if verbose:
                        print(f"[4.2/skip] {ep_name}/{topo_name}/{scen_name}: "
                              f"not enough agents for clustering (n_agents={n_agents}).")
                    continue

                # 1) Evaluate KMeans for all candidate K
                metrics_list = []
                for K in K_candidates:
                    m = _evaluate_kmeans_for_K(X, K, random_state=random_state)
                    metrics_list.append(m)
                metrics_df_raw = pd.DataFrame(metrics_list).sort_values("K").reset_index(drop=True)

                # 2) Add composite score
                metrics_df_full = _add_composite_score(
                    metrics_df_raw,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
                )

                # 3) Add elbow_score
                metrics_df_full["elbow_score"] = _compute_elbow_rank(metrics_df_full)

                # 4) Select best_K and K_elbow
                best_K, K_elbow = _select_best_K_from_df(metrics_df_full)

                # 5) Plot elbow + composite score
                elbow_plot_path = _plot_elbow_and_scores(
                    metrics_df_full,
                    ep_name=ep_name,
                    topo_name=topo_name,
                    scen_name=scen_name,
                    out_root="./artifacts/clustering"
                )

                # 6) Attach to env_config
                env_cfg.setdefault("clustering", {})
                env_cfg["clustering"]["k_selection"] = {
                    "metrics_df": metrics_df_full,
                    "best_K": best_K,
                    "K_elbow": K_elbow,
                    "elbow_plot_path": elbow_plot_path,
                }

                # 7) Record summary
                K_selection[ep_name][topo_name][scen_name] = {
                    "best_K": best_K,
                    "K_elbow": K_elbow,
                    "metrics_df": metrics_df_full,
                    "elbow_plot_path": elbow_plot_path,
                }

                if verbose:
                    print(f"[4.2] {ep_name}/{topo_name}/{scen_name}: "
                          f"n_agents={n_agents}, candidates={K_candidates}")
                    print(f"      → best_K  (composite score) = {best_K}")
                    print(f"      → K_elbow (inertia elbow)    = {K_elbow}")
                    print(f"      → elbow plot saved at: {elbow_plot_path}")
                    cols_show = [
                        "K", "inertia", "silhouette",
                        "calinski_harabasz", "davies_bouldin", "score"
                    ]
                    print(metrics_df_full[cols_show].round(4))
                    print("-" * 60)

    return K_selection


# ---------- 4.2.9 Example driver ----------

# After Step 4.1 (attach_features_to_all_envs + sanity checks), run:
K_selection = step4_2_select_K_for_all_envs(
    env_configs,
    random_state=42,
    alpha=0.4,
    beta=0.4,
    gamma=0.2,
    verbose=True
)

print("\n=== STEP 4.2 EXAMPLE: ep_000 / clustered / heavy ===")
ex_ep   = "ep_000"
ex_topo = "clustered"
ex_scen = "heavy"

if (ex_ep in K_selection and
    ex_topo in K_selection[ex_ep] and
    ex_scen in K_selection[ex_ep][ex_topo]):

    ex_sel = K_selection[ex_ep][ex_topo][ex_scen]
    print("Chosen best_K  (composite score):", ex_sel["best_K"])
    print("Elbow-based K_elbow             :", ex_sel["K_elbow"])
    print("Elbow plot path                 :", ex_sel["elbow_plot_path"])
    print("\nMetrics per K:")
    print(
        ex_sel["metrics_df"][
            ["K", "inertia", "silhouette", "calinski_harabasz", "davies_bouldin", "score"]
        ].round(4)
    )
else:
    print("[warn] Example triple (ep_000/clustered/heavy) not found in K_selection.")
        
    
    
# The checkups !!! (the charts are alike)

# 1. profile differences between light/moderate/heavy
ep   = "ep_000"
topo = "clustered"

X_light = env_configs[ep][topo]["light"]["clustering"]["features"]["X"]
X_mod   = env_configs[ep][topo]["moderate"]["clustering"]["features"]["X"]
X_heavy = env_configs[ep][topo]["heavy"]["clustering"]["features"]["X"]

print("X_light vs X_heavy allclose:", np.allclose(X_light, X_heavy))
print("X_light vs X_mod   allclose:", np.allclose(X_light, X_mod))
print("shapes:", X_light.shape, X_mod.shape, X_heavy.shape)


prof_light = env_configs[ep][topo]["light"]["agent_profiles"]
prof_mod   = env_configs[ep][topo]["moderate"]["agent_profiles"]
prof_heavy = env_configs[ep][topo]["heavy"]["agent_profiles"]

print("profiles equal (light vs heavy):", prof_light.equals(prof_heavy))
print("profiles equal (light vs mod)  :", prof_light.equals(prof_mod))



# 2. Distributions differences
targets = [
    ("clustered", "light"),
    ("clustered", "moderate"),
    ("clustered", "heavy"),
]

for topo, scen in targets:
    print(f"\n=== {ep} / {topo} / {scen} ===")
    prof = env_configs[ep][topo][scen]["agent_profiles"]

    print("\nlambda stats:")
    print(prof[["lambda_mean", "lambda_var"]].describe())

    print("\nP(task_type) stats:")
    cols_p = [f"P_{t}" for t in ["deadline_hard","latency_sensitive",
                                  "compute_intensive","data_intensive","general"]]
    print(prof[cols_p].describe())

    print("\nmedian task resource stats:")
    print(prof[["b_mb_med","rho_med","mem_med"]].describe())
    
    

# 3. metrics similarity
for topo in ["clustered", "full_mesh", "sparse_ring"]:
    for scen in ["light", "moderate", "heavy"]:
        sel = K_selection["ep_000"][topo][scen]
        dfm = sel["metrics_df"]
        print(f"\n=== {topo} / {scen} ===")
        print(dfm[["K","inertia","silhouette",
                   "calinski_harabasz","davies_bouldin","score"]].round(4))        
        




# 4.3. Implementing K-Means Clustering

# For each (episode → topology → scenario), we:
#   1) Fetch best_K from step 4.2
#   2) Run K-Means (once) using best_K
#   3) Store:
#        - cluster_labels (per agent)
#        - cluster_centers (scaled feature space)
#        - agent_ids for mapping
#
# Results written to:
#   env_cfg["clustering"]["final"] = {
#       "K": best_K,
#       "labels": np.ndarray shape (n_agents,),
#       "centers": np.ndarray shape (K, n_features),
#       "agent_ids": np.ndarray
#   }
# ======================================================

def step4_3_run_final_kmeans_for_all_envs(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:

    clustering_results = {}

    for ep_name, by_topo in env_configs.items():
        clustering_results[ep_name] = {}

        for topo_name, by_scen in by_topo.items():
            clustering_results[ep_name][topo_name] = {}

            for scen_name, env_cfg in by_scen.items():

                clust = env_cfg.get("clustering", {})
                feats = clust.get("features", None)
                k_sel = clust.get("k_selection", None)

                # sanity check
                if feats is None or "X" not in feats:
                    if verbose:
                        print(f"[4.3/skip] {ep_name}/{topo_name}/{scen_name}: no feature matrix.")
                    continue

                if k_sel is None or "best_K" not in k_sel:
                    if verbose:
                        print(f"[4.3/skip] {ep_name}/{topo_name}/{scen_name}: no K chosen.")
                    continue

                X = feats["X"]
                agent_ids = feats["agent_ids"]
                best_K = int(k_sel["best_K"])

                if best_K <= 1 or best_K > X.shape[0]:
                    if verbose:
                        print(f"[4.3/skip] invalid best_K={best_K} for {ep_name}/{topo_name}/{scen_name}.")
                    continue

                # Final K-Means fit
                km = KMeans(
                    n_clusters=best_K,
                    random_state=random_state,
                    n_init="auto"
                )
                labels = km.fit_predict(X)
                centers = km.cluster_centers_

                # Store results
                env_cfg["clustering"]["final"] = {
                    "K": best_K,
                    "labels": labels,
                    "centers": centers,
                    "agent_ids": agent_ids,
                }

                clustering_results[ep_name][topo_name][scen_name] = {
                    "K": best_K,
                    "labels": labels,
                    "centers": centers,
                    "agent_ids": agent_ids,
                }

                if verbose:
                    print(f"[4.3] {ep_name}/{topo_name}/{scen_name}:")
                    print(f"      best_K = {best_K}")
                    print(f"      labels distribution:", np.bincount(labels))
                    print(f"      centers shape:", centers.shape)
                    print("-" * 50)

    return clustering_results


clustering_final = step4_3_run_final_kmeans_for_all_envs(
    env_configs,
    random_state=42,
    verbose=True
)

# Example inspection
print("\n=== STEP 4.3 EXAMPLE: ep_000 / clustered / heavy ===")
ex = clustering_final["ep_000"]["clustered"]["heavy"]
print("K:", ex["K"])
print("Label counts:", np.bincount(ex["labels"]))
print("Centers shape:", ex["centers"].shape)



# Visualization — PCA: Display clusters in 2D space
def step4_3_plot_clusters_pca(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    out_root: str = "./artifacts/clustering",
    verbose: bool = True
):
    """
    For each environment (ep/topology/scenario), take:
        - X (scaled feature matrix)
        - labels (final K-Means labels)
    Project X to 2D via PCA and save scatter plot.

    Output saved as:
        <out_root>/<ep>/<topology>/<scenario>/cluster_plot_pca.png
    """
    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():

                # Must have clustering results
                clust = env_cfg.get("clustering", {})
                feats = clust.get("features", None)
                final = clust.get("final", None)

                if feats is None or "X" not in feats:
                    continue
                if final is None or "labels" not in final:
                    if verbose:
                        print(f"[PCA/skip] {ep_name}/{topo_name}/{scen_name}: no final KMeans labels.")
                    continue

                X = feats["X"]
                labels = final["labels"]
                K = final["K"]

                n_agents = X.shape[0]
                # PCA requires n_samples >= n_components (here 2)
                if n_agents < 2:
                    if verbose:
                        print(f"[PCA/skip] {ep_name}/{topo_name}/{scen_name}: "
                              f"n_agents={n_agents} < 2, cannot run PCA.")
                    continue

                # PCA projection
                pca = PCA(n_components=2, random_state=42)
                X_2d = pca.fit_transform(X)

                # Plot
                out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "cluster_plot_pca.png")

                plt.figure(figsize=(6, 5))

                for cl in np.unique(labels):
                    mask = (labels == cl)
                    plt.scatter(
                        X_2d[mask, 0],
                        X_2d[mask, 1],
                        label=f"Cluster {cl}",
                        alpha=0.75,
                        s=50
                    )

                plt.title(f"PCA Clusters: {ep_name} / {topo_name} / {scen_name}  (K={K})")
                plt.xlabel("PCA Component 1")
                plt.ylabel("PCA Component 2")
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()

                if verbose:
                    print(f"[PCA] Saved PCA cluster plot → {out_path}")
                                        
                    
step4_3_plot_clusters_pca(env_configs, verbose=True)



# Visualization — spacet-SNE: Display clusters in 2D
def step4_3_plot_clusters_tsne(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    out_root: str = "./artifacts/clustering",
    perplexity: int = 5,
    early_exaggeration: int = 12,
    n_iter: int = 1500,
    verbose: bool = True,
):
    """
    Draw 2D t-SNE visualization for final KMeans clusters.
    
    Saves figure as:
        <out_root>/<ep>/<topology>/<scenario>/cluster_plot_tsne.png
    """

    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():

                clust = env_cfg.get("clustering", {})
                feats = clust.get("features", None)
                final = clust.get("final", None)

                if feats is None or "X" not in feats:
                    continue
                if final is None or "labels" not in final:
                    if verbose:
                        print(f"[t-SNE/skip] {ep_name}/{topo_name}/{scen_name}: no cluster labels.")
                    continue

                X = feats["X"]
                labels = final["labels"]
                K = final["K"]

                n_agents = X.shape[0]
                if n_agents <= perplexity:
                    if verbose:
                        print(f"[t-SNE/skip] {ep_name}/{topo_name}/{scen_name}: "
                              f"n_agents={n_agents} <= perplexity={perplexity}")
                    continue

                # Run t-SNE
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
                    n_iter=n_iter,
                    init='pca',
                    learning_rate='auto',
                    random_state=42,
                    metric='euclidean'
                )

                X_2d = tsne.fit_transform(X)

                # Plot
                out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, "cluster_plot_tsne.png")

                plt.figure(figsize=(6, 5))

                for cl in np.unique(labels):
                    mask = (labels == cl)
                    plt.scatter(
                        X_2d[mask, 0],
                        X_2d[mask, 1],
                        label=f"Cluster {cl}",
                        s=50,
                        alpha=0.8
                    )

                plt.title(f"t-SNE Clusters: {ep_name} / {topo_name} / {scen_name}  (K={K})")
                plt.xlabel("t-SNE Dim 1")
                plt.ylabel("t-SNE Dim 2")
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()

                if verbose:
                    print(f"[t-SNE] Saved t-SNE cluster plot → {out_path}")
                                        

step4_3_plot_clusters_tsne(env_configs, verbose=True)





# 4.4. Cluster Interpretation & Profiling

# Interpretation, Summaries, and Cluster Profiles

def build_cluster_profiles_for_env(
    ep_name: str,
    topo_name: str,
    scen_name: str,
    env_cfg: Dict[str, Any],
    out_root: str = "./artifacts/clustering"
):
    """
    Build cluster representative profiles using:
      - labels from Step 4.3 (env_cfg['clustering']['final'])
      - scaled centers from Step 4.3
      - inverse-transformed centers using scaler from Step 4.1
      - agent_profiles from Step 3
    """

    # 1) Extract dependencies
    clust = env_cfg.get("clustering", {})
    feats = clust.get("features", None)
    final = clust.get("final", None)   # This must exist (Step 4.3)

    if feats is None or "X" not in feats:
        raise ValueError(f"[4.4] Missing features for {ep_name}/{topo_name}/{scen_name}")

    if final is None or "labels" not in final or "centers" not in final:
        raise ValueError(
            f"[4.4] Missing final KMeans results for {ep_name}/{topo_name}/{scen_name}. "
            f"Did you forget to run Step 4.3?"
        )

    best_K         = int(final["K"])
    labels         = np.asarray(final["labels"], dtype=int)
    centers_scaled = np.asarray(final["centers"], dtype=float)

    agent_ids    = feats["agent_ids"]
    scaler       = feats["scaler"]
    feature_cols = feats["feature_cols"]

    prof = env_cfg["agent_profiles"].copy()

    # 2) Build assignment table (agent_id → cluster_id)
    assign_df = pd.DataFrame({
        "agent_id": agent_ids,
        "cluster_id": labels
    })

    prof = prof.merge(assign_df, on="agent_id", how="left")

    # === NEW: Inject cluster_id into tasks DataFrame ===
    tasks_df = env_cfg.get("tasks", None)
    if tasks_df is not None and "agent_id" in tasks_df.columns:
        tasks_df = tasks_df.merge(assign_df, on="agent_id", how="left")
        # Convert to int (and fill agents with no tasks with -1)
        tasks_df["cluster_id"] = tasks_df["cluster_id"].fillna(-1).astype(int)
        # write back
        env_cfg["tasks"] = tasks_df
        print(f"[4.4] Added 'cluster_id' to tasks for {ep_name}/{topo_name}/{scen_name} "
              f"(rows={len(tasks_df)})")

    # 3) Cluster-level summary (numeric columns only, excluding cluster_id from aggregation)
    numeric_cols = prof.select_dtypes(include=[np.number]).columns.tolist()
    # Separate group key from aggregated columns
    agg_cols = [c for c in numeric_cols if c != "cluster_id"]

    cluster_summary = (
        prof[["cluster_id"] + agg_cols]
        .groupby("cluster_id", as_index=False)
        .mean()
        .sort_values("cluster_id")
    )

    cluster_sizes = (
        prof.groupby("cluster_id")["agent_id"]
        .count()
        .rename("n_agents_cluster")
        .reset_index()
    )
    cluster_summary = cluster_summary.merge(cluster_sizes, on="cluster_id", how="left")

    # 4) Decode centroids back to original scale
    if scaler is not None:
        centers_original = scaler.inverse_transform(centers_scaled)
    else:
        centers_original = centers_scaled.copy()

    centroids_scaled_df = pd.DataFrame(centers_scaled, columns=feature_cols)
    centroids_scaled_df.insert(0, "cluster_id", np.arange(best_K))

    centroids_original_df = pd.DataFrame(centers_original, columns=feature_cols)
    centroids_original_df.insert(0, "cluster_id", np.arange(best_K))

    # 5) Save to disk
    out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
    os.makedirs(out_dir, exist_ok=True)

    assign_path    = os.path.join(out_dir, "cluster_assignments.csv")
    summary_path   = os.path.join(out_dir, "cluster_summary.csv")
    cent_sc_path   = os.path.join(out_dir, "centroids_scaled.csv")
    cent_orig_path = os.path.join(out_dir, "centroids_original.csv")

    assign_df.to_csv(assign_path, index=False)
    cluster_summary.to_csv(summary_path, index=False)
    centroids_scaled_df.to_csv(cent_sc_path, index=False)
    centroids_original_df.to_csv(cent_orig_path, index=False)

    print(f"[4.4] {ep_name}/{topo_name}/{scen_name} → cluster profiles built.")
    print(cluster_sizes.set_index("cluster_id")["n_agents_cluster"])

    # 6) Attach final results to env_cfg
    env_cfg["clustering"]["profiles"] = {
        "K": best_K,
        "cluster_assignments": assign_df,
        "cluster_summary": cluster_summary,
        "centroids_scaled_df": centroids_scaled_df,
        "centroids_original_df": centroids_original_df,
        "centroids_scaled": centers_scaled,
        "centroids_original": centers_original,
    }

    return env_cfg["clustering"]["profiles"]

def build_all_cluster_profiles(env_configs):
    out = {}
    for ep_name, by_topo in env_configs.items():
        out[ep_name] = {}
        for topo_name, by_scen in by_topo.items():
            out[ep_name][topo_name] = {}
            for scen_name, env_cfg in by_scen.items():
                try:
                    prof = build_cluster_profiles_for_env(
                        ep_name, topo_name, scen_name, env_cfg
                    )
                    out[ep_name][topo_name][scen_name] = prof
                except Exception as e:
                    print(f"[4.4/warn] skipping {ep_name}/{topo_name}/{scen_name}: {e}")
    return out


# ---- Run Step 4.4 on all environments ----
cluster_profiles = build_all_cluster_profiles(env_configs)

print("\n=== EXAMPLE: cluster summary for ep_000 / clustered / heavy ===")
ex_ep   = "ep_000"
ex_topo = "clustered"
ex_scen = "heavy"

if (ex_ep in cluster_profiles and
    ex_topo in cluster_profiles[ex_ep] and
    ex_scen in cluster_profiles[ex_ep][ex_topo]):

    ex_prof = cluster_profiles[ex_ep][ex_topo][ex_scen]
    print("K =", ex_prof["K"])
    print("\nCluster summary (first few cols):")
    print(ex_prof["cluster_summary"].iloc[:, :10])
else:
    print("[warn] Example triple not found in cluster_profiles.")
    

test = env_configs["ep_000"]["clustered"]["heavy"]["clustering"]
print(test.keys())

print(env_configs["ep_000"]["clustered"]["heavy"]["clustering"]["profiles"]["cluster_summary"])


# Heatmap Visualization
def plot_cluster_profile_heatmap(env_cfg,
                                 ep_name: str,
                                 topo_name: str,
                                 scen_name: str,
                                 out_root="./artifacts/clustering"):
    """
    Draw heatmap of cluster profile means for a given env_cfg.

    Uses:
        env_cfg["clustering"]["profiles"]["cluster_summary"]
    """

    # 1) Extract cluster profiles
    clust = env_cfg.get("clustering", {})
    profs = clust.get("profiles", None)

    if profs is None or "cluster_summary" not in profs:
        print(f"[heatmap/skip] No cluster profiles for {ep_name}/{topo_name}/{scen_name}")
        return None

    cluster_summary = profs["cluster_summary"].copy()
    K = profs["K"]

    # remove non-feature columns if present
    drop_cols = ["cluster_id", "n_agents_cluster"]
    feature_cols = [c for c in cluster_summary.columns if c not in drop_cols]

    df = cluster_summary[feature_cols].copy()

    # 2) Normalize per-column
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-9)

    # 3) Output directory
    out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cluster_profile_heatmap.png")

    # 4) Plot heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        df_norm,
        annot=False,
        cmap="viridis",
        xticklabels=df_norm.columns,
        yticklabels=[f"Cluster {i}" for i in range(K)]
    )

    plt.title(f"Cluster Profile Heatmap: {ep_name}/{topo_name}/{scen_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[heatmap] Saved → {out_path}")
    return out_path


for ep in env_configs:
    for topo in env_configs[ep]:
        for scen in env_configs[ep][topo]:
            plot_cluster_profile_heatmap(
                env_configs[ep][topo][scen],
                ep, topo, scen
            )
 

# # Saving Information
# def _ensure_dir(path: str):
#     """Create a folder if it does not already exist."""
#     os.makedirs(path, exist_ok=True)

# def _serialize_non_df_components(env_cfg: dict) -> dict:
#     """
#     Prepare a JSON-serializable dictionary for all non-DataFrame parts
#     of env_config. Arrays are converted to lists.
#     """
#     out = {}
#     for key, value in env_cfg.items():
#         if isinstance(value, pd.DataFrame):
#             continue  # handled separately

#         # numpy arrays → lists
#         if isinstance(value, np.ndarray):
#             out[key] = value.tolist()
#             continue

#         # dicts (queues, action_space, state_spec, checks)
#         if isinstance(value, dict):
#             try:
#                 # recursively convert numpy arrays inside dicts
#                 def _convert(obj):
#                     if isinstance(obj, np.ndarray):
#                         return obj.tolist()
#                     if isinstance(obj, dict):
#                         return {k: _convert(v) for k, v in obj.items()}
#                     return obj
#                 out[key] = _convert(value)
#             except Exception as e:
#                 out[key] = f"(serialization error: {e})"
#             continue

#         # scalars (int, float, str, None)
#         if isinstance(value, (int, float, str, bool, type(None))):
#             out[key] = value
#             continue

#         # Handle StandardScaler separately by serializing only its mean and scale
#         if isinstance(value, StandardScaler):
#             out[key] = {
#                 "mean": value.mean_.tolist(),
#                 "scale": value.scale_.tolist()
#             }
#             continue

#         # fallback
#         try:
#             out[key] = json.loads(json.dumps(value))
#         except Exception:
#             out[key] = f"(unserializable type: {type(value).__name__})"

#     return out

# def save_all_env_configs(env_configs, out_root: str = "./artifacts/env_configs"):
#     """
#     Save all env_configs to disk in a structured layout:
#         artifacts/env_configs/ep_xxx/topology/scenario/
#             tasks_env_config.csv
#             agents_env_config.csv
#             arrivals_env_config.csv
#             episodes_env_config.csv
#             env_meta.json   <-- (non-DF components)
#     """
#     n_saved = 0

#     for ep_name, by_topo in env_configs.items():
#         for topo_name, by_scen in by_topo.items():
#             for scen_name, env_cfg in by_scen.items():

#                 out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
#                 _ensure_dir(out_dir)

#                 # ---- Save DataFrame components ----
#                 for df_name, df in env_cfg.items():
#                     if isinstance(df, pd.DataFrame):
#                         file_path_csv = os.path.join(out_dir, f"{df_name}_env_config.csv")
#                         df.to_csv(file_path_csv, index=False)

#                         print(f"[saved] {file_path_csv}  (rows={len(df)})")
#                         n_saved += 1

#                 # ---- Save non-DataFrame metadata ----
#                 meta = _serialize_non_df_components(env_cfg)

#                 meta_path = os.path.join(out_dir, "env_meta.json")
#                 with open(meta_path, "w", encoding="utf-8") as f:
#                     json.dump(meta, f, indent=2)

#                 print(f"[saved] {meta_path}")

#     print(f"\nDone. Saved {n_saved} DataFrames + meta files for all env_configs.")
    

# save_all_env_configs(env_configs, out_root="./artifacts/env_configs")





# Step 5: MDP Environment


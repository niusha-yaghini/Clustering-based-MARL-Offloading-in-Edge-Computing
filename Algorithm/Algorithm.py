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
# Define the base directories
dataset_dir = '../Data_Generator/datasets'
topology_dir = '../Topology_Generator/topologies'
environment_dir = '../Environment_Generator/simulation_output'


# Loading dataset
def load_datasets_from_directory(dataset_dir: str, verbose: bool = True):
    """
    Episode-first loader for the structure:

        dataset_dir/
          ep_000/
            light/
              episodes.csv
              arrivals.csv
              tasks.csv
              summary_stats.csv      (optional for this loader)
            moderate/
              ...
            heavy/
              ...
            dataset_metadata.json   (optional, per-episode metadata)

    Returns:
        datasets = {
            "ep_000": {
                "light":   {"episodes": df, "arrivals": df, "tasks": df},
                "moderate":{"..."},
                "heavy":   {"..."},
                "_meta":   {...}  # if dataset_metadata.json exists
            },
            "ep_001": { ... },
            ...
        }
    """
    datasets = {}

    if not os.path.isdir(dataset_dir):
        raise ValueError(f"dataset_dir does not exist or is not a directory: {dataset_dir}")

    # Step 1 — find ep_* directories
    ep_dirs = sorted([
        name for name in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, name)) and name.startswith("ep_")
    ])

    if verbose:
        if not ep_dirs:
            print(f"[warn] no ep_* folders found under root '{dataset_dir}'")
        else:
            print(f"[info] detected episodes: {ep_dirs}")

    # Step 2 — for each episode, detect scenarios and load CSVs
    for ep_name in ep_dirs:
        ep_path = os.path.join(dataset_dir, ep_name)
        datasets[ep_name] = {}

        # Scenario names (e.g., light / moderate / heavy)
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
                episodes_csv = os.path.join(scn_path, "episodes.csv")
                arrivals_csv = os.path.join(scn_path, "arrivals.csv")
                tasks_csv    = os.path.join(scn_path, "tasks.csv")

                dfs = {
                    "episodes": pd.read_csv(episodes_csv),
                    "arrivals": pd.read_csv(arrivals_csv),
                    "tasks":    pd.read_csv(tasks_csv),
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
        print("\n=== Dataset Summary ===")
        print(f"episodes detected: {len(datasets)}")
        for ep_name in sorted(datasets.keys()):
            keys_here = sorted(datasets[ep_name].keys())
            scenarios_here = [k for k in keys_here if not k.startswith("_")]
            print(f"  - {ep_name}: scenarios = {scenarios_here}")
            for scn in scenarios_here:
                dfs = datasets[ep_name][scn]
                n_ep   = len(dfs["episodes"])
                n_arr  = len(dfs["arrivals"])
                n_task = len(dfs["tasks"])
                print(
                    f"      {scn:9s} → "
                    f"episodes:{n_ep:3d}  "
                    f"arrivals:{n_arr:6d}  "
                    f"tasks:{n_task:6d}"
                )
            if "_meta" in datasets[ep_name]:
                print(f"      meta: dataset_metadata.json loaded")
        print("=======================================\n")

    return datasets


datasets = load_datasets_from_directory(dataset_dir, verbose=True)

# Pick an episode and a scenario for inspection
ep_name = sorted(datasets.keys())[0] if datasets else None
scenario = "heavy"  # or "light" / "moderate"

if ep_name is not None and scenario in datasets[ep_name]:
    print(f"\n[info] printing from episode='{ep_name}', scenario='{scenario}'")

    dfs = datasets[ep_name][scenario]

    print("\narrivals:")
    display(dfs["arrivals"].head())
    dfs["arrivals"].info()

    print("\nepisodes:")
    display(dfs["episodes"].head())
    dfs["episodes"].info()

    print("\ntasks:")
    display(dfs["tasks"].head())
    dfs["tasks"].info()

    # Example: check how many arrivals per mec_id
    if "mec_id" in dfs["arrivals"].columns:
        print("\narrivals per mec_id:")
        print(dfs["arrivals"]["mec_id"].value_counts().sort_index())

    # Show metadata if available
    if "_meta" in datasets[ep_name]:
        print("\nmeta (dataset_metadata.json):")
        print(json.dumps(datasets[ep_name]["_meta"], ensure_ascii=False, indent=2))
else:
    print("[error] no datasets found or requested scenario is missing for the chosen episode.")
    
    

# Loading environment
def load_environment_from_directory(
    environment_dir: str,
    mec_filename: str = "environment.csv",
    cloud_filename: str = "cloud_info.csv",
    verbose: bool = True
):
    """
    Load MEC & Cloud environment from CSV files generated by Environment_Generator.

    Expected structure:
        environment_dir/
          environment.csv   # servers: Server ID, Private CPU Capacity, Public CPU Capacity
          cloud_info.csv    # cloud:   id, computational_capacity

    Returns:
        environment = {
            "servers_df":    DataFrame,
            "cloud_df":      DataFrame,
            "num_servers":   int,
            "num_clouds":    int,
            "private_cpu":   np.ndarray,
            "public_cpu":    np.ndarray,
            "cloud_capacity": np.ndarray,
        }
    """
    if not os.path.isdir(environment_dir):
        raise ValueError(f"environment_dir does not exist or is not a directory: {environment_dir}")

    mec_path = os.path.join(environment_dir, mec_filename)
    cloud_path = os.path.join(environment_dir, cloud_filename)

    if not os.path.isfile(mec_path):
        raise FileNotFoundError(f"MEC environment CSV not found: {mec_path}")
    if not os.path.isfile(cloud_path):
        raise FileNotFoundError(f"Cloud info CSV not found: {cloud_path}")

    # --- Load CSVs ---
    servers_df = pd.read_csv(mec_path)
    cloud_df   = pd.read_csv(cloud_path)

    # --- Basic sanity on required columns ---
    required_server_cols = {"Server ID", "Private CPU Capacity", "Public CPU Capacity"}
    if not required_server_cols.issubset(servers_df.columns):
        missing = required_server_cols - set(servers_df.columns)
        raise ValueError(f"servers_df is missing required columns: {missing}")

    required_cloud_cols = {"id", "computational_capacity"}
    if not required_cloud_cols.issubset(cloud_df.columns):
        missing = required_cloud_cols - set(cloud_df.columns)
        raise ValueError(f"cloud_df is missing required columns: {missing}")

    # --- Basic shapes and arrays ---
    num_servers = len(servers_df)
    num_clouds  = len(cloud_df)

    private_cpu = servers_df["Private CPU Capacity"].to_numpy()
    public_cpu  = servers_df["Public CPU Capacity"].to_numpy()
    cloud_cap   = cloud_df["computational_capacity"].to_numpy()

    # --- Optional: enforce that Server ID are 0..num_servers-1 (consistent with mec_id) ---
    server_ids = servers_df["Server ID"].to_numpy()
    expected_ids = np.arange(num_servers, dtype=server_ids.dtype)
    if not np.array_equal(server_ids, expected_ids):
        raise ValueError(
            "Server ID column is not a simple 0..num_servers-1 sequence. "
            "This may break consistency with mec_id in datasets/topology. "
            f"Found IDs: {server_ids}"
        )

    environment = {
        "servers_df": servers_df,
        "cloud_df": cloud_df,
        "num_servers": num_servers,
        "num_clouds": num_clouds,
        "private_cpu": private_cpu,
        "public_cpu": public_cpu,
        "cloud_capacity": cloud_cap,
    }

    if verbose:
        print(f"[info] loaded environment from '{environment_dir}'")
        print(f"  - num_servers: {num_servers}")
        print(f"  - num_clouds : {num_clouds}")
        print("  - private_cpu (first 5):", private_cpu[:5])
        print("  - public_cpu  (first 5):", public_cpu[:5])
        print("  - cloud_capacity:", cloud_cap)

    return environment


# ---- Load environment (MEC + Cloud) and quick inspection ----
environment = load_environment_from_directory(environment_dir, verbose=True)

print("\n[environment] servers_df.head():")
print(environment["servers_df"].head())
environment["servers_df"].info()

print("\n[environment] cloud_df:")
print(environment["cloud_df"])
environment["cloud_df"].info()



# Loading topology
def load_topologies_from_directory(topology_dir: str, verbose: bool = True):
    """
    Load all topologies from a root directory.

    Expected structure:
        topology_dir/
          <topology_name>/
            topology.json
            topology_meta.json
            connection_matrix.csv
    """
    topologies = {}

    if not os.path.isdir(topology_dir):
        raise ValueError(f"topology_dir does not exist or is not a directory: {topology_dir}")

    # Iterate over subdirectories (each representing a topology variant)
    for topology_name in os.listdir(topology_dir):
        topology_path = os.path.join(topology_dir, topology_name)

        # Only process directories
        if not os.path.isdir(topology_path):
            continue

        topology_json_path = os.path.join(topology_path, "topology.json")
        meta_json_path = os.path.join(topology_path, "topology_meta.json")
        connection_matrix_csv_path = os.path.join(topology_path, "connection_matrix.csv")

        # Check for required files
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

        # First column is row labels (mec_i), so we use index_col=0
        connection_matrix = pd.read_csv(connection_matrix_csv_path, index_col=0)

        # Optional sanity check: match matrix shape with number_of_servers
        if "number_of_servers" in topology_data:
            K = int(topology_data["number_of_servers"])
            if connection_matrix.shape[0] != K:
                raise ValueError(
                    f"Topology '{topology_name}': number_of_servers={K} "
                    f"but connection_matrix has {connection_matrix.shape[0]} rows."
                )
            if connection_matrix.shape[1] != K + 1:
                raise ValueError(
                    f"Topology '{topology_name}': expected {K+1} columns in "
                    f"connection_matrix (K MEC + 1 cloud), got {connection_matrix.shape[1]}."
                )

        topologies[topology_name] = {
            "topology_data": topology_data,
            "meta_data": meta_data,
            "connection_matrix": connection_matrix
        }

    if verbose:
        print(f"[info] loaded topologies: {sorted(topologies.keys())}")

    return topologies
            
            
topologies = load_topologies_from_directory(topology_dir, verbose=True)

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
    """Collect errors instead of stopping at first failure."""
    if not cond:
        errors.append(msg)

def _has_cols(df: pd.DataFrame, cols: list) -> bool:
    """Check that all required columns exist in a DataFrame."""
    return all(c in df.columns for c in cols)

# ---------- Dataset-level validation ----------
def validate_one_dataset(dataset_key: str, ds: dict) -> list:
    """
    Validate a single dataset pack (episodes/arrivals/tasks) for one (episode, scenario).
    'dataset_key' is just a label for error messages, e.g. 'ep_000/heavy'.

    Assumes the new dataset structure (no agents.csv, MEC-based):
      - episodes.csv: has N_mecs instead of N_agents
      - arrivals.csv: uses mec_id
      - tasks.csv:    uses mec_id
    """
    errors = []
    episodes = ds.get("episodes")
    arrivals = ds.get("arrivals")
    tasks    = ds.get("tasks")

    # 1) Presence checks
    _require(isinstance(episodes, pd.DataFrame), f"[{dataset_key}] episodes missing or not a DataFrame", errors)
    _require(isinstance(arrivals, pd.DataFrame), f"[{dataset_key}] arrivals missing or not a DataFrame", errors)
    _require(isinstance(tasks,    pd.DataFrame), f"[{dataset_key}] tasks missing or not a DataFrame", errors)
    if errors:
        return errors

    # 2) Required columns (aligned with new Dataset_Generator)
    req_ep_cols  = ["scenario", "episode_id", "Delta",
                    "T_slots", "T_decision", "T_drain",
                    "hours", "N_mecs", "seed"]
    req_ar_cols  = ["scenario", "episode_id", "t_slot", "t_time", "mec_id", "task_id"]
    req_tk_cols  = [
        "scenario", "episode_id", "task_id", "mec_id",
        "t_arrival_slot", "t_arrival_time",
        "b_mb", "rho_cyc_per_mb", "c_cycles", "mem_mb", "modality",
        "has_deadline", "deadline_s", "deadline_time",
        "non_atomic", "split_ratio", "action_space_hint"
    ]

    _require(_has_cols(episodes, req_ep_cols),
             f"[{dataset_key}] episodes missing required columns", errors)
    _require(_has_cols(arrivals, req_ar_cols),
             f"[{dataset_key}] arrivals missing required columns", errors)
    _require(_has_cols(tasks,    req_tk_cols),
             f"[{dataset_key}] tasks missing required columns", errors)
    if errors:
        return errors

    # 3) Basic integrity checks

    # 3.1) unique task_id
    _require(tasks["task_id"].is_unique,
             f"[{dataset_key}] task_id is not unique", errors)

    # 3.2) arrivals and tasks should have the same number of rows
    _require(len(arrivals) == len(tasks),
             f"[{dataset_key}] arrivals ({len(arrivals)}) != tasks ({len(tasks)})", errors)

    # 3.3) mec_id range against N_mecs
    N_mecs = int(episodes["N_mecs"].iloc[0])
    _require(N_mecs > 0, f"[{dataset_key}] N_mecs must be > 0 (got {N_mecs})", errors)

    if len(tasks):
        mec_min = int(tasks["mec_id"].min())
        mec_max = int(tasks["mec_id"].max())
        _require(mec_min >= 0,
                 f"[{dataset_key}] mec_id minimum must be >= 0 (got {mec_min})", errors)
        _require(mec_max <= N_mecs - 1,
                 f"[{dataset_key}] mec_id maximum must be <= N_mecs-1 ({N_mecs-1}), got {mec_max}", errors)

    if len(arrivals):
        mec_min_a = int(arrivals["mec_id"].min())
        mec_max_a = int(arrivals["mec_id"].max())
        _require(mec_min_a >= 0,
                 f"[{dataset_key}] arrivals.mec_id minimum must be >= 0 (got {mec_min_a})", errors)
        _require(mec_max_a <= N_mecs - 1,
                 f"[{dataset_key}] arrivals.mec_id maximum must be <= N_mecs-1 ({N_mecs-1}), got {mec_max_a}", errors)

    # 3.4) non-negative task numerics
    for col in ["b_mb", "rho_cyc_per_mb", "c_cycles", "mem_mb"]:
        if col in tasks.columns:
            _require((tasks[col] >= 0).all(),
                     f"[{dataset_key}] tasks.{col} has negative values", errors)

    # 3.5) deadline coherence
    if "has_deadline" in tasks.columns and "deadline_s" in tasks.columns:
        bad_deadline = tasks[
            (tasks["has_deadline"] == 1) &
            ((tasks["deadline_s"].isna()) | (tasks["deadline_s"] <= 0))
        ]
        _require(len(bad_deadline) == 0,
                 f"[{dataset_key}] tasks with deadline have invalid deadline_s", errors)

    # 3.6) single Delta / T_slots / T_decision / T_drain inside this (episode, scenario)
    _require(episodes["Delta"].nunique() == 1,
             f"[{dataset_key}] multiple Delta values in episodes", errors)
    _require(episodes["T_slots"].nunique() == 1,
             f"[{dataset_key}] multiple T_slots in episodes", errors)
    _require(episodes["T_decision"].nunique() == 1,
             f"[{dataset_key}] multiple T_decision values in episodes", errors)
    _require(episodes["T_drain"].nunique() == 1,
             f"[{dataset_key}] multiple T_drain values in episodes", errors)

    # 3.7) arrivals inside slot range [0, T_slots-1] and only in decision horizon
    T_slots    = int(episodes["T_slots"].iloc[0])
    T_decision = int(episodes["T_decision"].iloc[0])

    if len(tasks):
        _require(int(tasks["t_arrival_slot"].max()) <= T_slots - 1,
                 f"[{dataset_key}] t_arrival_slot exceeds T_slots-1", errors)

    if len(arrivals):
        _require(int(arrivals["t_slot"].max()) <= T_slots - 1,
                 f"[{dataset_key}] arrivals.t_slot exceeds T_slots-1", errors)
        _require(int(arrivals["t_slot"].max()) <= T_decision - 1,
                 f"[{dataset_key}] arrivals.t_slot exceeds T_decision-1 (should only arrive during decision horizon)", errors)

    return errors

# ---------- Environment-level validation ----------
def validate_environment(environment: dict) -> list:
    """
    Validate the MEC + Cloud environment loaded from CSVs.

    Expected structure (from load_environment_from_directory):

        environment = {
            "servers_df":    DataFrame,
            "cloud_df":      DataFrame,
            "num_servers":   int,
            "num_clouds":    int,
            "private_cpu":   np.ndarray,
            "public_cpu":    np.ndarray,
            "cloud_capacity": np.ndarray
        }
    """
    errors = []

    servers_df  = environment.get("servers_df")
    cloud_df    = environment.get("cloud_df")
    num_servers = environment.get("num_servers")
    num_clouds  = environment.get("num_clouds")
    private_cpu = environment.get("private_cpu")
    public_cpu  = environment.get("public_cpu")
    cloud_cap   = environment.get("cloud_capacity")

    # --- presence / type checks ---
    _require(isinstance(servers_df, pd.DataFrame),
             "[env] servers_df missing or not a DataFrame", errors)
    _require(isinstance(cloud_df, pd.DataFrame),
             "[env] cloud_df missing or not a DataFrame", errors)
    if errors:
        return errors

    # --- required columns (According to Environment_Generator) ---
    required_server_cols = {"Server ID", "Private CPU Capacity", "Public CPU Capacity"}
    _require(required_server_cols.issubset(servers_df.columns),
             f"[env] servers_df missing required columns: "
             f"{required_server_cols - set(servers_df.columns)}", errors)

    required_cloud_cols = {"id", "computational_capacity"}
    _require(required_cloud_cols.issubset(cloud_df.columns),
             f"[env] cloud_df missing required columns: "
             f"{required_cloud_cols - set(cloud_df.columns)}", errors)

    # --- num_servers / num_clouds consistency ---
    if num_servers is not None:
        _require(num_servers == len(servers_df),
                 f"[env] num_servers ({num_servers}) != len(servers_df) ({len(servers_df)})", errors)
    else:
        num_servers = len(servers_df)

    if num_clouds is not None:
        _require(num_clouds == len(cloud_df),
                 f"[env] num_clouds ({num_clouds}) != len(cloud_df) ({len(cloud_df)})", errors)
    else:
        num_clouds = len(cloud_df)

    # --- Server ID sanity (unique and from 0 to num_servers-1) ---
    server_ids = servers_df["Server ID"].to_numpy()
    _require(len(np.unique(server_ids)) == len(server_ids),
             "[env] duplicate Server ID values detected", errors)

    if np.issubdtype(server_ids.dtype, np.number):
        _require(server_ids.min() == 0,
                 f"[env] Server ID should start from 0 (got {server_ids.min()})", errors)
        _require(server_ids.max() == num_servers - 1,
                 f"[env] Server ID max should be num_servers-1 ({num_servers-1}), "
                 f"got {server_ids.max()}", errors)

    # --- arrays shapes ---
    if private_cpu is not None:
        _require(len(private_cpu) == num_servers,
                 "[env] private_cpu length != num_servers", errors)
    if public_cpu is not None:
        _require(len(public_cpu) == num_servers,
                 "[env] public_cpu length != num_servers", errors)
    if cloud_cap is not None:
        _require(len(cloud_cap) == num_clouds,
                 "[env] cloud_capacity length != num_clouds", errors)

    # --- non-negativity ---
    if private_cpu is not None:
        _require((private_cpu >= 0).all(),
                 "[env] private_cpu contains negative values", errors)
    if public_cpu is not None:
        _require((public_cpu >= 0).all(),
                 "[env] public_cpu contains negative values", errors)
    if cloud_cap is not None:
        _require((cloud_cap >= 0).all(),
                 "[env] cloud_capacity contains negative values", errors)

    return errors

# ---------- Topology-level validation ----------
def validate_one_topology(topology_name: str, topo_entry: dict) -> list:
    """
    Validate a single topology pack: topology.json + topology_meta.json + connection_matrix.csv.
    """
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
        "number_of_servers", "private_cpu_capacities", "public_cpu_capacities",
        "cloud_computational_capacity", "connection_matrix", "time_step"
    ]
    for k in req_keys:
        _require(k in topo, f"[{topology_name}] topology.json missing key: {k}", errors)
    if errors:
        return errors

    K = int(topo["number_of_servers"])
    _require(len(topo["private_cpu_capacities"]) == K,
             f"[{topology_name}] private_cpu_capacities length != K", errors)
    _require(len(topo["public_cpu_capacities"])  == K,
             f"[{topology_name}] public_cpu_capacities length != K", errors)

    Mjson = topo["connection_matrix"]
    _require(
        isinstance(Mjson, list)
        and len(Mjson) == K
        and (K == 0 or len(Mjson[0]) == K + 1),
        f"[{topology_name}] connection_matrix in JSON must be K x (K+1)",
        errors
    )
    _require(Mdf.shape == (K, K + 1),
             f"[{topology_name}] connection_matrix.csv shape must be K x (K+1)", errors)

    # MEC->Cloud capacities (last column) must be > 0
    vert_csv = Mdf.iloc[:, K]
    _require((vert_csv > 0).all(),
             f"[{topology_name}] MEC->Cloud capacities must be > 0", errors)

    # MEC<->MEC capacities (first K columns) must be >= 0
    horiz_csv = Mdf.iloc[:, :K]
    _require((horiz_csv.values >= 0).all(),
             f"[{topology_name}] MEC<->MEC capacities contain negatives", errors)

    _require("time_step" in topo, f"[{topology_name}] missing time_step", errors)

    return errors

# ---------- Pairwise validation (environment <-> topology) ----------
def validate_environment_topology_pair(environment: dict,
                                       topology_name: str,
                                       topo_entry: dict,
                                       atol: float = 1e-6) -> list:
    """
    Validate alignment between environment (MEC/Cloud CSVs)
    and one topology (topology.json + connection_matrix).

    Checks:
      - number_of_servers == env.num_servers
      - private/public CPU capacities match (up to tolerance)
      - cloud_computational_capacity matches env.cloud_capacity
    """
    errors = []

    num_servers  = environment.get("num_servers")
    private_cpu  = environment.get("private_cpu")
    public_cpu   = environment.get("public_cpu")
    cloud_cap    = environment.get("cloud_capacity")

    topo = topo_entry.get("topology_data")

    _require(isinstance(topo, dict),
             f"[env x {topology_name}] topology_data missing or not a dict", errors)
    if errors:
        return errors

    K = int(topo["number_of_servers"])
    _require(K == num_servers,
             f"[env x {topology_name}] number_of_servers ({K}) != env.num_servers ({num_servers})", errors)

    topo_priv  = np.array(topo["private_cpu_capacities"], dtype=float)
    topo_pub   = np.array(topo["public_cpu_capacities"], dtype=float)
    topo_cloud = float(topo["cloud_computational_capacity"])

    if private_cpu is not None:
        _require(topo_priv.shape == private_cpu.shape,
                 f"[env x {topology_name}] shape mismatch in private CPU capacities "
                 f"topo:{topo_priv.shape}, env:{private_cpu.shape}", errors)
        if topo_priv.shape == private_cpu.shape:
            _require(np.allclose(topo_priv, private_cpu, atol=atol),
                     f"[env x {topology_name}] private CPU capacities differ (topology vs environment)", errors)

    if public_cpu is not None:
        _require(topo_pub.shape == public_cpu.shape,
                 f"[env x {topology_name}] shape mismatch in public CPU capacities "
                 f"topo:{topo_pub.shape}, env:{public_cpu.shape}", errors)
        if topo_pub.shape == public_cpu.shape:
            _require(np.allclose(topo_pub, public_cpu, atol=atol),
                     f"[env x {topology_name}] public CPU capacities differ (topology vs environment)", errors)

    if cloud_cap is not None and len(cloud_cap) > 0:
        env_cloud_val = float(cloud_cap[0])   # currently we have 1 cloud
        _require(abs(topo_cloud - env_cloud_val) <= atol,
                 f"[env x {topology_name}] cloud capacity differs: "
                 f"topology={topo_cloud}, env={env_cloud_val}", errors)

    return errors

# ---------- Pairwise validation (dataset <-> topology) ----------
def validate_dataset_topology_pair(ep_name: str, scenario: str, ds: dict,
                                   topology_name: str, topo_entry: dict) -> list:
    """
    Validate alignment between one (episode, scenario) dataset and one topology.

    Ensures:
      - Delta == time_step
      - N_mecs == number_of_servers
      - mec_id values are within [0, K-1]
      - compute capacities are non-negative
    """
    errors = []
    episodes = ds["episodes"]
    arrivals = ds["arrivals"]
    tasks    = ds["tasks"]
    topo     = topo_entry["topology_data"]
    K        = int(topo["number_of_servers"])

    # Delta vs time_step
    Delta     = float(episodes["Delta"].iloc[0])
    time_step = float(topo["time_step"])
    _require(abs(Delta - time_step) < 1e-9,
             f"[{ep_name}/{scenario} x {topology_name}] Delta ({Delta}) != time_step ({time_step})", errors)

    # N_mecs vs number_of_servers
    N_mecs = int(episodes["N_mecs"].iloc[0])
    _require(N_mecs == K,
             f"[{ep_name}/{scenario} x {topology_name}] N_mecs ({N_mecs}) != number_of_servers ({K})", errors)

    # mec_id range inside dataset vs topology K
    if len(tasks):
        min_mec_t = int(tasks["mec_id"].min())
        max_mec_t = int(tasks["mec_id"].max())
        _require(min_mec_t >= 0 and max_mec_t <= K - 1,
                 f"[{ep_name}/{scenario} x {topology_name}] tasks.mec_id out of range [0, {K-1}]", errors)

    if len(arrivals):
        min_mec_a = int(arrivals["mec_id"].min())
        max_mec_a = int(arrivals["mec_id"].max())
        _require(min_mec_a >= 0 and max_mec_a <= K - 1,
                 f"[{ep_name}/{scenario} x {topology_name}] arrivals.mec_id out of range [0, {K-1}]", errors)

    # Non-negative compute capacities in topology
    priv  = topo["private_cpu_capacities"]
    pub   = topo["public_cpu_capacities"]
    cloud = topo["cloud_computational_capacity"]
    _require(all(x >= 0 for x in priv) and all(x >= 0 for x in pub) and cloud >= 0,
             f"[{ep_name}/{scenario} x {topology_name}] negative compute capacities detected", errors)

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
        # Ignore metadata or entries without 'episodes'
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

# ---------- Global validation entrypoint ----------
def validate_everything(datasets: dict,
                        topologies: dict,
                        environment: dict) -> dict:
    """
    'datasets' shape (episode-first, new structure):
        {
          "ep_000": {
             "light":   {"episodes": df, "arrivals": df, "tasks": df},
             "moderate":{...},
             "heavy":   {...},
             "_meta":   {...}  # optional per-episode metadata
          },
          "ep_001": {...}
        }
    """
    report = {
        "datasets": {},
        "episodes_consistency": {},
        "topologies": {},
        "pairs": {},
        "environment": {},
        "env_topology_pairs": {}
    }

    # 0) Validate environment (MEC + Cloud)
    env_errs = validate_environment(environment)
    report["environment"] = {"ok": len(env_errs) == 0, "errors": env_errs}

    # 1) Validate each (episode, scenario) dataset
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
            report["datasets"][ep_name][scenario] = {
                "ok": len(errs) == 0,
                "errors": errs
            }

    # 2) Episode-level Delta/T_slots consistency across scenarios
    for ep_name, ep_pack in datasets.items():
        errs = validate_episode_delta_consistency(ep_name, ep_pack)
        report["episodes_consistency"][ep_name] = {
            "ok": len(errs) == 0,
            "errors": errs
        }

    # 3) Validate each topology
    for tname, tpack in topologies.items():
        errs = validate_one_topology(tname, tpack)
        report["topologies"][tname] = {
            "ok": len(errs) == 0,
            "errors": errs
        }

    # 4) Pairwise validation: ENVIRONMENT × each topology
    for tname, tpack in topologies.items():
        if report["environment"]["ok"] and report["topologies"][tname]["ok"]:
            errs = validate_environment_topology_pair(environment, tname, tpack)
            report["env_topology_pairs"][tname] = {
                "ok": len(errs) == 0,
                "errors": errs
            }
        else:
            report["env_topology_pairs"][tname] = {
                "ok": False,
                "errors": ["Skipped due to invalid environment or topology."]
            }

    # 5) Pairwise validation for every valid (ep, scenario) × valid topology
    for ep_name, ep_pack in datasets.items():
        scenario_names = list(report["datasets"][ep_name].keys())

        for scenario in scenario_names:
            dpack = ep_pack[scenario]
            d_ok  = report["datasets"][ep_name][scenario]["ok"]
            ep_ok = report["episodes_consistency"][ep_name]["ok"]

            for tname, tres in report["topologies"].items():
                key = f"{ep_name}/{scenario}__{tname}"
                if d_ok and ep_ok and tres["ok"]:
                    errs = validate_dataset_topology_pair(
                        ep_name, scenario, dpack, tname, topologies[tname]
                    )
                    report["pairs"][key] = {
                        "ok": len(errs) == 0,
                        "errors": errs
                    }
                else:
                    report["pairs"][key] = {
                        "ok": False,
                        "errors": ["Skipped due to upstream invalid dataset/episode/topology."]
                    }

    return report

# ---------- Pretty printer ----------
def print_validation_report(report: dict):
    print("=== ENVIRONMENT (MEC + Cloud) ===")
    env_info = report.get("environment")
    if env_info:
        status = "OK" if env_info["ok"] else "FAIL"
        print(f"[{status}] environment")
        for e in env_info["errors"]:
            print(f"  - {e}")
    print()
    
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
            
    print("\n=== ENVIRONMENT × TOPOLOGY PAIRS ===")
    for tname, info in report.get("env_topology_pairs", {}).items():
        status = "OK" if info["ok"] else "FAIL"
        print(f"[{status}] env x {tname}")
        for e in info["errors"]:
            print(f"  - {e}")
            

# ---------- Run validation ----------
report = validate_everything(datasets, topologies, environment)
print_validation_report(report)

all_ok = (
    # datasets
    all(info["ok"] for ep in report["datasets"].values() for info in ep.values())
    # per-episode Delta/T_slots consistency
    and all(info["ok"] for info in report["episodes_consistency"].values())
    # topologies
    and all(info["ok"] for info in report["topologies"].values())
    # dataset × topology pairs
    and all(info["ok"] for info in report["pairs"].values())
    # environment itself
    and report["environment"]["ok"]
    # environment × topology pairs
    and all(info["ok"] for info in report["env_topology_pairs"].values())
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
    Given one dataset dict {"episodes","arrivals","tasks"},
    return a copy with aligned/derived columns (per-slot helpers).

    Notes:
      - This version assumes MEC-based datasets (no agents.csv).
      - Deadline-related NaNs are replaced with -1 so that the
        final table is RL-friendly (no NaNs in deadline fields).
    """
    episodes = dataset["episodes"].copy()
    arrivals = dataset["arrivals"].copy()
    tasks    = dataset["tasks"].copy()

    Delta = _get_delta(episodes)

    # Tasks: ensure integer arrival slot
    if "t_arrival_slot" not in tasks.columns:
        raise ValueError("tasks.csv must contain 't_arrival_slot'.")
    tasks["t_arrival_slot"] = tasks["t_arrival_slot"].astype(int)

    # Compute deadline_slots (in slots) and replace missing deadlines with -1
    if "has_deadline" in tasks.columns and "deadline_s" in tasks.columns:
        # Normalize types first
        tasks["has_deadline"] = tasks["has_deadline"].astype(int)
        tasks["deadline_s"] = tasks["deadline_s"].astype("float32")

        def _to_deadline_slots(row):
            # only tasks with has_deadline == 1 and valid deadline_s
            if int(row["has_deadline"]) == 1 and np.isfinite(row["deadline_s"]):
                return int(math.ceil(float(row["deadline_s"]) / Delta))
            # no deadline → use -1 sentinel
            return -1

        tasks["deadline_slots"] = tasks.apply(_to_deadline_slots, axis=1).astype("int32")

        # For tasks that effectively have no valid deadline, set -1 in deadline_s and deadline_time
        no_valid_deadline_mask = (tasks["has_deadline"] == 0) | (~np.isfinite(tasks["deadline_s"]))

        tasks.loc[no_valid_deadline_mask, "deadline_s"] = -1.0

        if "deadline_time" in tasks.columns:
            # deadline_time is absolute time; for 'no deadline' we also put -1
            tasks["deadline_time"] = tasks["deadline_time"].astype("float32")
            tasks.loc[no_valid_deadline_mask, "deadline_time"] = -1.0

    # Ensure key numeric task fields are floats
    for col in ["b_mb", "rho_cyc_per_mb", "c_cycles", "mem_mb"]:
        if col in tasks.columns:
            tasks[col] = tasks[col].astype(float)

    return {
        "episodes": episodes,
        "arrivals": arrivals,
        "tasks":    tasks,
    }

# ===== Alignment: per-environment against a target Delta =====
def align_environment_units(environment: dict, target_Delta: float) -> dict:
    """
    Take raw environment dict from load_environment_from_directory and
    return an aligned copy with per-slot capacities.

    Adds:
      - private_cpu_slot, public_cpu_slot, cloud_capacity_slot (numpy arrays)
      - columns in servers_df / cloud_df reflecting per-slot capacities
    """
    if environment is None:
        raise ValueError("environment is None in align_environment_units.")

    env_aligned = dict(environment)  # shallow copy of dict; we'll copy DFs below

    servers_df = env_aligned["servers_df"].copy()
    cloud_df   = env_aligned["cloud_df"].copy()

    private_cpu = np.asarray(env_aligned["private_cpu"], dtype=float)
    public_cpu  = np.asarray(env_aligned["public_cpu"], dtype=float)
    cloud_cap   = np.asarray(env_aligned["cloud_capacity"], dtype=float)

    # basic sanity (non-finite / negative)
    _ensure_numeric_positive("env.private_cpu", private_cpu)
    _ensure_numeric_positive("env.public_cpu", public_cpu)
    _ensure_numeric_positive("env.cloud_capacity", cloud_cap)

    # per-slot capacities
    private_cpu_slot = private_cpu * float(target_Delta)
    public_cpu_slot  = public_cpu  * float(target_Delta)
    cloud_slot       = cloud_cap   * float(target_Delta)

    # update numpy arrays in dict
    env_aligned["private_cpu"] = private_cpu
    env_aligned["public_cpu"]  = public_cpu
    env_aligned["cloud_capacity"] = cloud_cap
    env_aligned["private_cpu_slot"] = private_cpu_slot
    env_aligned["public_cpu_slot"]  = public_cpu_slot
    env_aligned["cloud_capacity_slot"] = cloud_slot

    # also add columns to DataFrames for convenience
    servers_df["Private CPU Capacity"] = private_cpu
    servers_df["Public CPU Capacity"]  = public_cpu
    servers_df["Private CPU per_slot"] = private_cpu_slot
    servers_df["Public CPU per_slot"]  = public_cpu_slot

    cloud_df["computational_capacity"] = cloud_cap
    cloud_df["capacity_per_slot"]      = cloud_slot

    env_aligned["servers_df"] = servers_df
    env_aligned["cloud_df"]   = cloud_df

    return env_aligned

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
def align_all_units(
    datasets_ep_first: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
    topologies_by_name: Dict[str, Dict[str, Any]],
    environment: dict
) -> Dict[str, Any]:
    """
    Input 'datasets_ep_first' shape:

        {
          "ep_000": {
             "light":   {"episodes": df, "arrivals": df, "tasks": df},
             "moderate":{...},
             "heavy":   {...},
             "_meta":   {...}   # optional per-episode metadata (NO episodes/tasks)
          },
          "ep_001": {...}
        }

    Returns:
        {
          "datasets_aligned": { ep_name: { scenario: aligned_pack_or_meta, ... }, ... },
          "topology_checks":  { topo_name: { ep_name: { scenario: {ok, message} } } },
          "environment_aligned": dict,
          "environment_Delta": float
        }
    """
    out = {
        "datasets_aligned": {},
        "topology_checks":  {},
        "environment_aligned": None,
        "environment_Delta": None,
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
                # For example "_meta" or anything else → keep as is
                out["datasets_aligned"][ep_name][scenario] = ds

    # ---- 2) Infer a reference Delta for environment alignment ----
    Delta_ref = None
    for ep_name, ep_pack in out["datasets_aligned"].items():
        for scenario, ds in ep_pack.items():
            if isinstance(ds, dict) and "episodes" in ds and len(ds["episodes"]):
                Delta_ref = _get_delta(ds["episodes"])
                break
        if Delta_ref is not None:
            break

    if Delta_ref is None:
        raise RuntimeError("Could not infer a reference Delta for environment alignment (no episodes found).")

    # ---- 3) Align environment w.r.t this Delta ----
    env_aligned = align_environment_units(environment, Delta_ref)
    out["environment_aligned"] = env_aligned
    out["environment_Delta"]   = Delta_ref
    
    # ---- 4) Verify each topology against each (episode, scenario) Delta ----
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

# ===== Pretty printer =====
def print_alignment_summary(result: Dict[str, Any]):
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
            print(f"[{ep_name}/{scenario}] Delta={Delta}  tasks={n_tasks}")

    # ===== TOPOLOGIES =====
    print("\n=== TOPOLOGIES (checks vs each episode/scenario) ===")
    for topo_name, by_ep in result["topology_checks"].items():
        print(f"Topology: {topo_name}")
        for ep_name in sorted(by_ep.keys()):
            for scenario, r in sorted(by_ep[ep_name].items()):
                flag = "OK" if r["ok"] else "FAIL"
                print(f"  - {ep_name}/{scenario}: {flag}  -> {r['message']}")
    
    # ===== ENVIRONMENT =====
    print("\n=== ENVIRONMENT (aligned) ===")
    env_aligned = result.get("environment_aligned")
    Delta_env   = result.get("environment_Delta", None)

    if env_aligned is None:
        print("No environment_aligned found in result.")
        return

    print(f"Reference Delta used for environment per-slot capacities: {Delta_env}")
    print(f"num_servers = {env_aligned['num_servers']}, num_clouds = {env_aligned['num_clouds']}")

    priv = env_aligned["private_cpu"]
    pub  = env_aligned["public_cpu"]
    priv_slot = env_aligned["private_cpu_slot"]
    pub_slot  = env_aligned["public_cpu_slot"]
    cloud_cap = env_aligned["cloud_capacity"]
    cloud_slot = env_aligned["cloud_capacity_slot"]

    print("  private_cpu (first 5):      ", priv[:5])
    print("  private_cpu_per_slot (5):   ", priv_slot[:5])
    print("  public_cpu (first 5):       ", pub[:5])
    print("  public_cpu_per_slot (5):    ", pub_slot[:5])
    print("  cloud_capacity:             ", cloud_cap)
    print("  cloud_capacity_per_slot:    ", cloud_slot)
    

# ===== Example usage =====
result_align = align_all_units(
    datasets_ep_first=datasets,
    topologies_by_name=topologies,
    environment=environment
)
print_alignment_summary(result_align)

print("\n ===EXAMPLE===")
aligned_light_ep0 = result_align["datasets_aligned"]["ep_000"]["light"]
tasks_ep0_light   = aligned_light_ep0["tasks"]    # has deadline_slots
print(tasks_ep0_light[["task_id", "mec_id", "t_arrival_slot", "deadline_s", "deadline_slots"]].head())





# 1.4. Build Scenario–Topology Pairs

# In this step, we create a Cartesian product between:
#   - all (episode, scenario) datasets
#   - all topologies
#
# For each (topology, episode, scenario) triple we build a "bundle" that contains:
#   - the aligned dataset (episodes, arrivals, tasks)
#   - the topology (JSON + connection matrix)
#   - the aligned environment (MEC + Cloud capacities)
#
# Output structure (topology-first):
# pairs_by_topology = {
#   "<topology_name>": {
#       "<ep_XXX>": {
#           "<scenario>": {
#               'scenario': <str>,
#               'episode': <str>,
#               'topology': <str>,
#               'Delta': <float>,
#               'K': <int>,  # number_of_servers (MECs)
#               'dataset': { 'episodes': df, 'arrivals': df, 'tasks': df },
#               'topology_data': <dict>,
#               'topology_meta_data': <dict or None>,
#               'connection_matrix_df': <pd.DataFrame>,  # shape (K, K+1)
#               'environment': <dict>,  # aligned environment (same for all pairs)
#               'checks': {
#                   'delta_match': bool,
#                   'env_servers_match': bool,
#                   'message': str
#               }
#           }, ...
#       }, ...
#   }, ...
# }

def _delta_from_episodes(episodes_df: pd.DataFrame) -> float:
    """Extract a single Delta value from episodes table."""
    if "Delta" not in episodes_df.columns:
        raise ValueError("episodes.csv must contain a 'Delta' column.")
    return float(episodes_df["Delta"].iloc[0])

def _topology_time_step(topo_json: Dict[str, Any]) -> float:
    """Extract the topology time_step from topology.json."""
    ts = topo_json.get("time_step", None)
    if ts is None:
        raise ValueError("topology.json must contain 'time_step'.")
    return float(ts)

def build_topology_episode_pairs(
    datasets_ep_first: Dict[str, Dict[str, Dict[str, Any]]],
    topologies: Dict[str, Dict[str, Any]],
    environment: dict,
    strict_delta_match: bool = True,
    strict_env_match: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build pairs between every topology and every (episode, scenario) dataset.

    Parameters
    ----------
    datasets_ep_first :
        Episode-first datasets, shape:
        {
          "ep_000": {
             "light":   {"episodes": df, "arrivals": df, "tasks": df},
             "moderate":{...},
             "heavy":   {...},
             "_meta":   {...}  # metadata only (no episodes/arrivals/tasks)
          },
          ...
        }

    topologies :
        Dict of topologies as returned by load_topologies_from_directory, e.g.:
        {
          "clustered": {
              "topology_data": dict,
              "meta_data": dict,
              "connection_matrix": DataFrame
          },
          ...
        }

    environment :
        Aligned environment dictionary as returned by align_all_units(...)
        under key "environment_aligned".

    strict_delta_match :
        If True, raise an error when dataset Delta != topology time_step.

    strict_env_match :
        If True, raise an error when topology.number_of_servers != environment.num_servers.
    """
    if environment is None:
        raise ValueError("environment must not be None in build_topology_episode_pairs.")

    env_num_servers = int(environment["num_servers"])

    pairs_by_topology: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Iterate over topologies first (topology-centric view)
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

        # Check consistency between topology and environment (number of MEC servers)
        env_match_ok = (K == env_num_servers)
        if (not env_match_ok) and strict_env_match:
            raise ValueError(
                f"[{topo_name}] number_of_servers ({K}) != environment.num_servers ({env_num_servers})"
            )

        topo_ts = _topology_time_step(topo_data)

        # Initialize container for this topology
        pairs_by_topology[topo_name] = {}

        # Compare against every (episode, scenario)
        for ep_name, scenarios in datasets_ep_first.items():
            pairs_by_topology[topo_name][ep_name] = {}

            for scen_name, ds in scenarios.items():
                # Skip metadata entries such as "_meta"
                if not (isinstance(ds, dict) and "episodes" in ds):
                    continue

                ds_Delta = _delta_from_episodes(ds["episodes"])
                delta_ok = bool(np.isclose(ds_Delta, topo_ts, atol=1e-12))
                msg_delta = "OK" if delta_ok else (
                    f"time_step mismatch (dataset Delta={ds_Delta}, topology time_step={topo_ts})"
                )

                msg_env = "OK" if env_match_ok else (
                    f"env.num_servers ({env_num_servers}) != topology.K ({K})"
                )

                # If Delta mismatch is not tolerated, raise immediately
                if (not delta_ok) and strict_delta_match:
                    raise ValueError(f"[{topo_name} × {ep_name}/{scen_name}] {msg_delta}")

                # Build final message from delta + environment checks
                if delta_ok and env_match_ok:
                    final_msg = "OK"
                else:
                    final_msg = f"{msg_delta}; {msg_env}"

                # Store bundle for this (topology, episode, scenario)
                pairs_by_topology[topo_name][ep_name][scen_name] = {
                    "scenario": scen_name,
                    "episode": ep_name,
                    "topology": topo_name,
                    "Delta": ds_Delta,
                    "K": K,
                    "dataset": ds,                         # aligned dataset (episodes/arrivals/tasks)
                    "topology_data": topo_data,
                    "topology_meta_data": meta_data,
                    "connection_matrix_df": cm_df,
                    "environment": environment,            # same aligned environment for all pairs
                    "checks": {
                        "delta_match": delta_ok,
                        "env_servers_match": env_match_ok,
                        "message": final_msg
                    }
                }

    return pairs_by_topology

def print_pairs_summary_topology_first_ep(
    pairs_by_topology: Dict[str, Dict[str, Dict[str, Any]]]
) -> None:
    """
    Pretty-print summary of pairs in the form:

        TOPOLOGY -> EPISODE -> SCENARIO
    """
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
                checks = bundle["checks"]
                flag   = "OK" if (checks["delta_match"] and checks["env_servers_match"]) else "FAIL"
                K      = bundle["K"]
                Delta  = bundle["Delta"]
                msg    = checks["message"]
                print(f"  │    - [{flag}] {scen_name:9s} | K={K:2d}  Δ={Delta:g}  -> {msg}")
                
                
# --- Example driver (using current variables) ---
result_align = align_all_units(
    datasets_ep_first=datasets,
    topologies_by_name=topologies,
    environment=environment
)

datasets_aligned    = result_align["datasets_aligned"]
environment_aligned = result_align["environment_aligned"]

pairs_by_topology = build_topology_episode_pairs(
    datasets_ep_first=datasets_aligned,
    topologies=topologies,
    environment=environment_aligned,
    strict_delta_match=True,
    strict_env_match=True
)

print_pairs_summary_topology_first_ep(pairs_by_topology)

print("\n ===EXAMPLE===")
# Example access:
#   - tasks for light scenario under fully_connected topology and ep_000
tasks_light = pairs_by_topology["fully_connected"]["ep_000"]["light"]["dataset"]["tasks"]
print(tasks_light)

#   - connection matrix for clustered topology and heavy scenario, ep_000
cm_clustered = pairs_by_topology["clustered"]["ep_000"]["heavy"]["connection_matrix_df"]
print(cm_clustered)

#   - aligned environment for the same pair
env_for_pair = pairs_by_topology["clustered"]["ep_000"]["heavy"]["environment"]
print(env_for_pair)





# 1.5. Environment Configuration

# In this step, we build a unified env_config for each scenario–topology pair.
# It bundles all required information for the MDP/RL environment—such as compute capacities, 
    # the Agent→MEC mapping, connection matrix, initial queue states, and action/state specifications—into
    # a single consistent configuration used by the RL training process.

def _extract_core_from_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract core fields from a (topology × episode × scenario) bundle.
    Ensures required fields exist and converts structures to numpy/DF formats.
    This version is MEC-based (no agents / agent_to_mec).
    """
    required = ["dataset", "topology_data", "connection_matrix_df", "Delta", "K"]
    for k in required:
        if k not in bundle:
            raise ValueError(f"Bundle missing required key: '{k}'")

    ds   = bundle["dataset"]
    topo = bundle["topology_data"]
    Mdf  = bundle["connection_matrix_df"]

    # Dataset tables (already aligned in previous stage)
    if not {"episodes", "arrivals", "tasks"}.issubset(ds.keys()):
        raise ValueError("dataset in bundle must contain 'episodes', 'arrivals', 'tasks'.")

    episodes = ds["episodes"]
    arrivals = ds["arrivals"]
    tasks    = ds["tasks"]

    # Capacities from topology (validated earlier against environment)
    private_cpu = np.asarray(topo["private_cpu_capacities"], dtype=float)  # shape (K,)
    public_cpu  = np.asarray(topo["public_cpu_capacities"],  dtype=float)  # shape (K,)
    cloud_cpu   = float(topo["cloud_computational_capacity"])             # scalar

    # Connection matrix: shape (K, K+1), last column = MEC→Cloud
    M = Mdf.to_numpy(dtype=float)

    return dict(
        Delta=float(bundle["Delta"]),
        K=int(bundle["K"]),
        episodes=episodes,
        arrivals=arrivals,
        tasks=tasks,
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
    All queues start empty (=0).
    """
    return {
        "mec_local_cycles":      np.zeros(K, dtype=float),
        "mec_public_cycles":     np.zeros(K, dtype=float),
        "mec_bytes_in_transit":  np.zeros(K, dtype=float),
        "cloud_cycles":          np.array([0.0], dtype=float),
    }

def _derive_action_space() -> Dict[str, Any]:
    """
    Basic discrete offloading action space (HOODIE-style):
        0 = Execute locally on the MEC where the task arrived
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
                "mec_local_cycles":     {"shape": (K,),      "dtype": "float"},
                "mec_public_cycles":    {"shape": (K,),      "dtype": "float"},
                "mec_bytes_in_transit": {"shape": (K,),      "dtype": "float"},
                "cloud_cycles":         {"shape": (1,),      "dtype": "float"},
            },
            "links": {
                "connection_matrix":    {"shape": (K, K + 1), "dtype": "float"},
            },
            "capacities": {
                "private_cpu": {"shape": (K,), "dtype": "float"},
                "public_cpu":  {"shape": (K,), "dtype": "float"},
                "cloud_cpu":   {"shape": (1,), "dtype": "float"},
            },
        },
        "note": "Declarative state description; environment assembles numerical tensors at runtime.",
    }

def build_env_config_for_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the complete environment configuration structure for a single
    (topology × episode × scenario) bundle.

    MEC-based version (no agents / no agent_to_mec):

    env_config includes:
        - Time parameters: Delta, T_slots
        - Topology specification: K, connection_matrix, topology_type
        - Resource capacities: private/public/cloud CPU (per MEC / cloud)
        - Aligned dataset tables (episodes, arrivals, tasks)
        - Initial queue states
        - Action space and state specification
        - Basic consistency info from previous checks (bundle["checks"])
    """
    core = _extract_core_from_bundle(bundle)

    episodes = core["episodes"]

    # Simulation horizon
    if "T_slots" not in episodes.columns:
        raise ValueError("episodes.csv must contain 'T_slots'.")
    T_slots = int(episodes["T_slots"].iloc[0])

    # Number of MECs (from episodes.csv) should match topology K
    N_mecs = None
    if "N_mecs" in episodes.columns:
        N_mecs = int(episodes["N_mecs"].iloc[0])
        if N_mecs != core["K"]:
            raise ValueError(
                f"Mismatch between episodes.N_mecs ({N_mecs}) and topology K ({core['K']})."
            )
    else:
        # If N_mecs column is missing, fall back to K
        N_mecs = core["K"]

    # Build initial queues and specs
    queues_initial = _build_default_queues(core["K"])
    action_space   = _derive_action_space()
    state_spec     = _derive_state_spec(core["K"])

    # Final environment configuration object
    env_config = {
        # Time / horizon
        "Delta":   core["Delta"],
        "T_slots": T_slots,

        # Topology
        "K":              core["K"],
        "N_mecs":         N_mecs,
        "topology_type":  core["topology_type"],
        "connection_matrix": core["connection_matrix"],

        # Capacities (per-slot units, as defined in topology/environment)
        "private_cpu": core["private_cpu"],   # shape (K,)
        "public_cpu":  core["public_cpu"],    # shape (K,)
        "cloud_cpu":   core["cloud_cpu"],     # scalar

        # Datasets (aligned, MEC-based)
        "episodes": core["episodes"],
        "arrivals": core["arrivals"],
        "tasks":    core["tasks"],

        # Initial queue states and specifications
        "queues_initial": queues_initial,
        "action_space":   action_space,
        "state_spec":     state_spec,

        # Validation / consistency info propagated from previous stage
        "checks": bundle.get("checks", {"delta_match": True, "env_servers_match": True, "message": "n/a"}),
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
                # No more 'agent_to_mec' required here (MEC-based).
                env_cfg = build_env_config_for_bundle(bundle)
                out[ep_name][topo_name][scen_name] = env_cfg

    return out


env_configs = build_all_env_configs(pairs_by_topology)

print("\n=== EXAMPLE ENV CONFIG ===")
example_cfg = env_configs["ep_000"]["clustered"]["heavy"]
print("Delta:", example_cfg["Delta"])
print("T_slots:", example_cfg["T_slots"])
print("K (number of MECs):", example_cfg["K"])
print("queues_initial keys:", list(example_cfg["queues_initial"].keys()))
print("tasks shape:", example_cfg["tasks"].shape)





# 1.6. Sanity Checks

# In this step, we verify that each env_config is internally consistent 
# (queue shapes, capacities, agent→MEC mapping, and connection matrix are valid and ready for simulation).
    
def sanity_check_env_config(env_config: Dict[str, Any]) -> list:
    """
    Run basic sanity checks on a single env_config dictionary (MEC-based).
    Returns a list of error strings; empty list means 'no issues found'.
    """
    errors = []

    # ---- 0) Basic required keys ----
    required_keys = [
        "K",
        "Delta",
        "T_slots",
        "connection_matrix",
        "private_cpu",
        "public_cpu",
        "cloud_cpu",
        "queues_initial",
        "action_space",
        "episodes",
        "arrivals",
        "tasks",
    ]
    for k in required_keys:
        if k not in env_config:
            errors.append(f"Missing required key in env_config: '{k}'")
    if errors:
        return errors

    K       = int(env_config["K"])
    Delta   = float(env_config["Delta"])
    T_slots = int(env_config["T_slots"])

    episodes = env_config["episodes"]
    arrivals = env_config["arrivals"]
    tasks    = env_config["tasks"]

    # ---- 1) MEC count alignment (K vs N_mecs / episodes) ----
    N_mecs_cfg = env_config.get("N_mecs", None)
    if N_mecs_cfg is not None:
        if int(N_mecs_cfg) != K:
            errors.append(f"N_mecs ({N_mecs_cfg}) != K ({K}) in env_config.")

    if isinstance(episodes, pd.DataFrame) and "N_mecs" in episodes.columns and len(episodes):
        N_mecs_ep = int(episodes["N_mecs"].iloc[0])
        if N_mecs_ep != K:
            errors.append(f"episodes.N_mecs ({N_mecs_ep}) != K ({K}).")

    # ---- 2) MEC IDs in tasks / arrivals ----
    if isinstance(tasks, pd.DataFrame):
        if "mec_id" not in tasks.columns:
            errors.append("tasks table is missing 'mec_id' column.")
        else:
            mec_ids_tasks = tasks["mec_id"].to_numpy()
            if len(mec_ids_tasks):
                if (mec_ids_tasks < 0).any() or (mec_ids_tasks >= K).any():
                    errors.append("tasks.mec_id contains values outside [0, K-1].")
    else:
        errors.append("tasks is not a DataFrame.")

    if isinstance(arrivals, pd.DataFrame):
        if "mec_id" not in arrivals.columns:
            errors.append("arrivals table is missing 'mec_id' column.")
        else:
            mec_ids_arr = arrivals["mec_id"].to_numpy()
            if len(mec_ids_arr):
                if (mec_ids_arr < 0).any() or (mec_ids_arr >= K).any():
                    errors.append("arrivals.mec_id contains values outside [0, K-1].")
    else:
        errors.append("arrivals is not a DataFrame.")

    # ---- 3) Queue initial state shapes ----
    q = env_config["queues_initial"]
    if q["mec_local_cycles"].shape != (K,):
        errors.append("mec_local_cycles queue shape mismatch.")
    if q["mec_public_cycles"].shape != (K,):
        errors.append("mec_public_cycles queue shape mismatch.")
    if q["mec_bytes_in_transit"].shape != (K,):
        errors.append("mec_bytes_in_transit queue shape mismatch.")
    if q["cloud_cycles"].shape != (1,):
        errors.append("cloud_cycles shape mismatch (should be (1,)).")

    # ---- 4) Non-negative compute capacities ----
    private_cpu = np.asarray(env_config["private_cpu"], dtype=float)
    public_cpu  = np.asarray(env_config["public_cpu"],  dtype=float)
    cloud_cpu   = float(env_config["cloud_cpu"])

    if (private_cpu < 0).any():
        errors.append("private_cpu has negative values.")
    if (public_cpu < 0).any():
        errors.append("public_cpu has negative values.")
    if cloud_cpu < 0:
        errors.append("cloud_cpu is negative.")

    if private_cpu.shape != (K,):
        errors.append(f"private_cpu shape mismatch, expected ({K},).")
    if public_cpu.shape != (K,):
        errors.append(f"public_cpu shape mismatch, expected ({K},).")

    # ---- 5) Connection matrix dimension (K x K+1) ----
    M = np.asarray(env_config["connection_matrix"], dtype=float)
    if M.shape != (K, K + 1):
        errors.append("connection_matrix shape mismatch (expected K x (K+1)).")
    if (M < 0).any():
        errors.append("connection_matrix contains negative values.")

    # ---- 6) Action space correctness ----
    action_space = env_config.get("action_space", {})
    if action_space.get("type", None) != "discrete":
        errors.append("Action space must be discrete (LOCAL/MEC/CLOUD).")
    if action_space.get("n", None) != 3:
        errors.append("Action space 'n' must be 3 (LOCAL/MEC/CLOUD).")

    # ---- 7) Basic time parameters + consistency with episodes ----
    if not np.isfinite(Delta) or Delta <= 0:
        errors.append(f"Invalid Delta in env_config (got {Delta}).")
    if T_slots <= 0:
        errors.append(f"Invalid T_slots in env_config (got {T_slots}).")

    if isinstance(episodes, pd.DataFrame) and len(episodes):
        if "Delta" in episodes.columns:
            Delta_ep = float(episodes["Delta"].iloc[0])
            if not np.isclose(Delta_ep, Delta, atol=1e-9):
                errors.append(f"episodes.Delta ({Delta_ep}) != env_config.Delta ({Delta}).")
        if "T_slots" in episodes.columns:
            T_slots_ep = int(episodes["T_slots"].iloc[0])
            if T_slots_ep != T_slots:
                errors.append(f"episodes.T_slots ({T_slots_ep}) != env_config.T_slots ({T_slots}).")

    # ---- 8) Arrival slot range sanity ----
    if isinstance(tasks, pd.DataFrame) and "t_arrival_slot" in tasks.columns and len(tasks):
        max_slot = int(tasks["t_arrival_slot"].max())
        if max_slot >= T_slots:
            errors.append(
                f"tasks.t_arrival_slot has values >= T_slots (max={max_slot}, T_slots={T_slots})."
            )

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

def save_env_configs_text(env_configs: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
                          out_path: str = "./artifacts/env_configs_summary.txt"):
    """
    Save a human-readable summary of all env_configs:
        env_configs[episode][topology][scenario] = env_config

    The summary includes:
    - key scalar parameters (Delta, K, N_mecs, topology_type)
    - shapes and stats of numeric arrays
    - summary of DataFrames (episodes, arrivals, tasks)
    - queue initialization
    - RL descriptors (action_space, state_spec, checks)
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
                for key in ["Delta", "K", "N_mecs", "topology_type"]:
                    if key in env_cfg:
                        lines.extend(_summarize_any(key, env_cfg[key], indent="      "))

                # -- main tensors/arrays --
                for key in [
                    "connection_matrix",
                    "private_cpu",
                    "public_cpu",
                    "cloud_cpu",
                ]:
                    if key in env_cfg:
                        lines.extend(_summarize_any(key, env_cfg[key], indent="      "))

                # -- dataframes (no agents in MEC-based design) --
                for key in ["episodes", "arrivals", "tasks"]:
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
    """
    Map a scalar value into a bucket:
      - 'S' = small
      - 'M' = medium
      - 'L' = large
      - 'U' = unknown (if any input is non-finite)
    """
    if not np.isfinite(value) or not np.isfinite(q1) or not np.isfinite(q2):
        return "U"  # Unknown
    if value <= q1:
        return "S"
    if value <= q2:
        return "M"
    return "L"

# ---------- threshold builder (adaptive to each tasks DF) ----------
def build_task_label_thresholds(
    tasks_df: pd.DataFrame,
    q_low: float = 0.33,
    q_high: float = 0.66,
    urgent_slots_cap: int = 2,
) -> Dict[str, Any]:
    """
    Build adaptive thresholds from the data itself (per-episode/scenario),
    so 'light/moderate/heavy' are handled robustly.
    """
    q_b_mb  = _quantile_cutpoints(tasks_df["b_mb"], q_low, q_high) if "b_mb" in tasks_df else (np.nan, np.nan)
    q_rho   = _quantile_cutpoints(tasks_df["rho_cyc_per_mb"], q_low, q_high) if "rho_cyc_per_mb" in tasks_df else (np.nan, np.nan)
    q_mem   = _quantile_cutpoints(tasks_df["mem_mb"], q_low, q_high) if "mem_mb" in tasks_df else (np.nan, np.nan)

    if "split_ratio" in tasks_df.columns:
        mask_split = tasks_df.get("non_atomic", 0) == 1
        q_split = _quantile_cutpoints(tasks_df.loc[mask_split, "split_ratio"], q_low, q_high)
    else:
        q_split = (np.nan, np.nan)

    return {
        "b_mb":   {"q1": q_b_mb[0],  "q2": q_b_mb[1]},
        "rho":    {"q1": q_rho[0],   "q2": q_rho[1]},
        "mem":    {"q1": q_mem[0],   "q2": q_mem[1]},
        "split":  {"q1": q_split[0], "q2": q_split[1]},
        # If deadline_slots ≤ urgent_slots_cap → 'hard' (latency sensitive)
        "urgent_slots_cap": int(urgent_slots_cap),
    }

# ---------- main labeling for a single tasks DF ----------
def label_tasks_df(
    tasks_df: pd.DataFrame,
    Delta: float,
    thresholds: Dict[str, Any]
) -> pd.DataFrame:
    """
    Add label columns to tasks_df (returns a COPY).

    Columns added:
      - size_bucket, compute_bucket, mem_bucket
      - (if missing) deadline_slots, and then urgency: none/soft/hard
      - atomicity, split_bucket
      - latency_sensitive, compute_heavy, io_heavy, memory_heavy (bools)
      - routing_hint (LOCAL/MEC/CLOUD) – only for EDA / debugging
    """
    df = tasks_df.copy()

    # --- ensure numeric types for main features
    for col in ["b_mb", "rho_cyc_per_mb", "c_cycles", "mem_mb", "deadline_s", "split_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- deadline_slots (only if not precomputed earlier)
    # In our current pipeline, Units Alignment already created `deadline_slots`
    # and used -1 as sentinel for "no deadline".
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
    b_q1, b_q2       = thresholds["b_mb"]["q1"],  thresholds["b_mb"]["q2"]
    rho_q1, rho_q2   = thresholds["rho"]["q1"],   thresholds["rho"]["q2"]
    mem_q1, mem_q2   = thresholds["mem"]["q1"],   thresholds["mem"]["q2"]

    df["size_bucket"] = (
        df["b_mb"].apply(lambda x: _bucketize(x, b_q1, b_q2))
        if "b_mb" in df else "U"
    )
    df["compute_bucket"] = (
        df["rho_cyc_per_mb"].apply(lambda x: _bucketize(x, rho_q1, rho_q2))
        if "rho_cyc_per_mb" in df else "U"
    )
    df["mem_bucket"] = (
        df["mem_mb"].apply(lambda x: _bucketize(x, mem_q1, mem_q2))
        if "mem_mb" in df else "U"
    )

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
        # `deadline_slots` in our pipeline:
        #   - positive integer => has a valid deadline
        #   - 0 or negative    => sentinel (no deadline)
        has_deadline_flag = int(row.get("has_deadline", 0)) == 1
        slots_val = row.get("deadline_slots", -1)

        # Try to cast to int; if it fails, treat as "no deadline"
        try:
            slots = int(slots_val)
        except Exception:
            return "none"

        if (not has_deadline_flag) or (slots <= 0):
            return "none"

        if slots <= urgent_cap:  # very tight deadline → hard
            return "hard"
        return "soft"

    df["urgency"] = df.apply(_urg, axis=1)

    # --- boolean convenience labels
    df["latency_sensitive"] = (df["urgency"] == "hard")
    df["compute_heavy"]     = (df["compute_bucket"] == "L")
    df["io_heavy"]          = (df["size_bucket"] == "L")
    df["memory_heavy"]      = (df["mem_bucket"] == "L")

    # --- a very simple routing hint (for EDA only; not used by RL)
    def _hint(row):
        if row["compute_heavy"] or row["memory_heavy"]:
            return "CLOUD"
        if row["latency_sensitive"]:
            return "MEC"
        return "LOCAL"

    df["routing_hint"] = df.apply(_hint, axis=1)

    return df

# ---------- batch apply to env_configs (episode → topology → scenario) ----------
def label_all_tasks_in_env_configs(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    q_low: float = 0.33,
    q_high: float = 0.66,
    urgent_slots_cap: int = 2,
    verbose: bool = True
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]],
           Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    For each env_config (episode → topology → scenario):
      - build thresholds from its own tasks DF
      - label tasks
      - put labeled DF back into env_config["tasks"]
      - return a concise summary per bundle

    env_configs structure (as built earlier):
        env_configs[ep_name][topology_name][scenario_name] = env_config
    """
    summary: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ep_name, by_topo in env_configs.items():
        summary[ep_name] = {}
        for topo_name, by_scen in by_topo.items():
            summary[ep_name][topo_name] = {}
            for scen_name, env_cfg in by_scen.items():
                tasks = env_cfg["tasks"]
                Delta = float(env_cfg["Delta"])

                # Build thresholds for this tasks DF
                th = build_task_label_thresholds(
                    tasks, q_low=q_low, q_high=q_high, urgent_slots_cap=urgent_slots_cap
                )

                labeled = label_tasks_df(tasks, Delta=Delta, thresholds=th)
                env_cfg["tasks"] = labeled  # write back into env_config

                # Small summary
                cnt = {
                    "n": len(labeled),
                    "urg_hard": int((labeled["urgency"] == "hard").sum()),
                    "splittable": int((labeled["atomicity"] == "splittable").sum()),
                    "size_L": int((labeled["size_bucket"] == "L").sum()),
                    "compute_L": int((labeled["compute_bucket"] == "L").sum()),
                    "mem_L": int((labeled["mem_bucket"] == "L").sum()),
                }
                summary[ep_name][topo_name][scen_name] = cnt

                if verbose:
                    print(
                        f"[label] {ep_name}/{topo_name}/{scen_name} -> "
                        f"n={cnt['n']}, hard={cnt['urg_hard']}, split={cnt['splittable']}, "
                        f"sizeL={cnt['size_L']}, compL={cnt['compute_L']}, memL={cnt['mem_L']}"
                    )

    return env_configs, summary


# ---- Run labeling on current env_configs (episode → topology → scenario) ----
env_configs, label_summary = label_all_tasks_in_env_configs(
    env_configs,
    q_low=0.33,
    q_high=0.66,
    urgent_slots_cap=2,  # tunable
    verbose=True
)

print("\n ===EXAMPLE (labeled tasks) ===")
labeled_tasks = env_configs["ep_000"]["clustered"]["heavy"]["tasks"]
print(labeled_tasks.head())
print(labeled_tasks.info())





# 2.2. Task Type Classification

# Pre-req: tasks already labeled by your previous step: 
# size_bucket, compute_bucket, mem_bucket, urgency, atomicity, split_bucket, routing_hint, etc.

def _derive_task_type_row(row: pd.Series) -> tuple[str, str, str, list, str]:
    """
    Returns (task_type, task_subtype, type_reason, multi_flags, final_flag),
    where task_type is one of:
        - 'deadline_hard'
        - 'latency_sensitive'
        - 'compute_intensive'
        - 'data_intensive'
        - 'general'
    """
    # Flags based on previous labeling
    urgency        = str(row.get("urgency", "none"))     # "hard" | "soft" | "none"
    latency_flag   = (urgency == "hard") or (urgency == "soft")
    hard_deadline  = (urgency == "hard")

    compute_heavy  = bool(row.get("compute_heavy", False))
    memory_heavy   = bool(row.get("memory_heavy", False))
    io_heavy       = bool(row.get("io_heavy", False))
    non_atomic     = bool(row.get("atomicity", "atomic") == "splittable")

    # Collect all active traits for audit
    multi_flags = []
    if hard_deadline:
        multi_flags.append("deadline_hard")
    elif latency_flag:
        multi_flags.append("deadline_soft")
    if compute_heavy:
        multi_flags.append("compute_heavy")
    if memory_heavy:
        multi_flags.append("memory_heavy")
    if io_heavy:
        multi_flags.append("io_heavy")
    if non_atomic:
        multi_flags.append("splittable")

    # --- Priority resolution (simple, Chapter-4 style) ---

    # 1) Hard deadline dominates everything
    if hard_deadline:
        final_flag = "deadline_hard"
        return ("deadline_hard", "deadline_hard", "hard deadline (tight slots)", multi_flags, final_flag)

    # 2) Latency-sensitive (soft deadlines)
    if latency_flag:
        final_flag = "latency_sensitive"
        return ("latency_sensitive", "deadline_soft", "delay-sensitive (soft deadline)", multi_flags, final_flag)

    # 3) Compute-intensive (compute or memory heavy)
    if compute_heavy or memory_heavy:
        final_flag = "compute_intensive"
        return ("compute_intensive", "compute_or_memory_heavy", "high compute/memory demand", multi_flags, final_flag)

    # 4) Data-intensive (large input size / IO heavy)
    if io_heavy:
        final_flag = "data_intensive"
        return ("data_intensive", "large_input_bandwidth", "large data volume / IO heavy", multi_flags, final_flag)

    # 5) Otherwise general
    final_flag = "general"
    return ("general", "general", "no dominant constraint", multi_flags, final_flag)

def apply_ch4_task_typing(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Chapter-4 level task classes with priority rules into tasks_df (returns a COPY).

    Requires that tasks_df already has:
        - urgency
        - compute_heavy
        - memory_heavy
        - io_heavy
        - atomicity

    Columns added:
      - task_type       (5-way class)
      - task_subtype    (finer descriptor)
      - type_reason     (short textual rationale)
      - multi_flags     (list of all active boolean traits)
      - final_flag      (single primary flag)
      - is_* one-hot convenience columns
    """
    df = tasks_df.copy()

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

    df["task_type"]    = out_type
    df["task_subtype"] = out_sub
    df["type_reason"]  = out_reason
    df["multi_flags"]  = out_flags
    df["final_flag"]   = out_final_flag

    # One-hot convenience view
    df["is_general"]           = (df["task_type"] == "general")
    df["is_deadline_hard"]     = (df["task_type"] == "deadline_hard")
    df["is_latency_sensitive"] = (df["task_type"] == "latency_sensitive")
    df["is_compute_intensive"] = (df["task_type"] == "compute_intensive")
    df["is_data_intensive"]    = (df["task_type"] == "data_intensive")

    return df

def apply_task_typing_in_env_configs(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    verbose: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    env_configs structure (episode-first):
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


# ---- Run typing on current env_configs (episode → topology → scenario) ----
env_configs = apply_task_typing_in_env_configs(env_configs, verbose=True)

print("\n ===EXAMPLE (task typing) ===")
print(
    env_configs["ep_000"]["clustered"]["heavy"]["tasks"][
        ["task_id", "task_type", "task_subtype", "type_reason", "multi_flags", "final_flag"]
    ].head(25)
)


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

# ---- Helper 1: per-MEC per-slot arrival counts ----
def _per_mec_slot_counts(arrivals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many tasks each MEC receives in each time slot.
    This is used to estimate lambda (arrival rate) statistics per MEC.
    """
    if not {"mec_id", "t_slot"}.issubset(arrivals_df.columns):
        raise ValueError("arrivals must contain 'mec_id' and 't_slot'.")
    grp = arrivals_df.groupby(["mec_id", "t_slot"], as_index=False).size()
    grp.rename(columns={"size": "count"}, inplace=True)
    return grp

# ---- Helper 2: estimate λ-mean and λ-variance per MEC ----
def _lambda_stats_from_counts_mec(counts_df: pd.DataFrame, Delta: float) -> pd.DataFrame:
    """
    Convert per-slot counts to rate statistics per MEC:
        lambda_mean = mean(count_per_slot) / Delta
        lambda_var  = var(count_per_slot)  / Delta^2
    """
    if counts_df.empty:
        return pd.DataFrame(columns=["mec_id", "lambda_mean", "lambda_var", "slots_observed"])

    agg = counts_df.groupby("mec_id")["count"].agg(
        lambda_mean_slot="mean",
        lambda_var_slot="var",
        slots_observed="count"
    ).reset_index()

    # If only one observation exists, variance becomes NaN → treat as zero.
    agg["lambda_var_slot"] = agg["lambda_var_slot"].fillna(0.0).astype(float)

    # Convert to per-second rates
    agg["lambda_mean"] = (agg["lambda_mean_slot"] / float(Delta)).astype(float)
    agg["lambda_var"]  = (agg["lambda_var_slot"]  / float(Delta**2)).astype(float)

    return agg[["mec_id", "lambda_mean", "lambda_var", "slots_observed"]]

# ---- Helper 3: task-type distribution per MEC ----
_TASK_TYPES = ["general", "latency_sensitive", "deadline_hard", "data_intensive", "compute_intensive"]

def _task_distribution_per_mec(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distribution of task types per MEC (probabilities sum to 1
    for MECs that actually have tasks).

    Also adds light median features useful for clustering:
      - b_mb_med, rho_med, mem_med, hard_share
    """
    if not {"mec_id", "task_type"}.issubset(tasks_df.columns):
        raise ValueError("tasks must contain 'mec_id' and 'task_type'.")

    if tasks_df.empty:
        # Empty DF: return an empty frame with all expected columns
        piv = pd.DataFrame(index=pd.Index([], name="mec_id"))
        for t in _TASK_TYPES:
            piv[t] = 0.0
        piv["n_tasks_mec"] = 0.0
    else:
        # Raw counts per (mec_id, task_type)
        cnt = tasks_df.groupby(["mec_id", "task_type"], as_index=False).size()
        piv = cnt.pivot(index="mec_id", columns="task_type", values="size").fillna(0.0)

        # Ensure all expected classes exist
        for t in _TASK_TYPES:
            if t not in piv.columns:
                piv[t] = 0.0

        # Total tasks per MEC
        piv["n_tasks_mec"] = piv[_TASK_TYPES].sum(axis=1).astype(float)

    # Probabilities
    for t in _TASK_TYPES:
        piv[f"P_{t}"] = np.where(
            piv["n_tasks_mec"] > 0,
            piv[t] / piv["n_tasks_mec"],
            0.0
        ).astype(float)

    # Optional extra features (medians, hard deadline share)
    feats = {}
    needed = {"b_mb", "rho_cyc_per_mb", "mem_mb", "urgency"}
    if needed.issubset(tasks_df.columns) and not tasks_df.empty:
        agg = tasks_df.groupby("mec_id").agg(
            b_mb_med=("b_mb", "median"),
            rho_med=("rho_cyc_per_mb", "median"),
            mem_med=("mem_mb", "median"),
            hard_share=("urgency", lambda s: float((s == "hard").mean()))
        ).reset_index()
        feats = agg.set_index("mec_id")

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

    keep = ["n_tasks_mec", "TaskDist_sum", "b_mb_med", "rho_med", "mem_med", "hard_share"] + prob_cols
    return piv[keep].reset_index()  # reset_index → get mec_id as a column

# ---- Helper 4: fraction of non-atomic (splittable) tasks per MEC ----
def _non_atomic_share_per_mec(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the share of splittable (non-atomic) tasks per MEC.
    """
    if "mec_id" not in tasks_df.columns:
        raise ValueError("tasks must contain 'mec_id'.")

    if "non_atomic" not in tasks_df.columns or tasks_df.empty:
        # If missing or no tasks, assume zero share for each MEC that appears
        mec_ids = tasks_df.get("mec_id")
        if mec_ids is None or len(mec_ids) == 0:
            return pd.DataFrame(columns=["mec_id", "non_atomic_share"])
        return pd.DataFrame({"mec_id": mec_ids.unique(), "non_atomic_share": 0.0})

    grp = tasks_df.groupby("mec_id")["non_atomic"].agg(
        non_atomic_share=lambda s: float((s == 1).mean())
    ).reset_index()
    return grp

# ---- Build MEC profiles for ONE env_config ----
def build_mec_profiles_for_env(env_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Construct per-MEC profiles combining:
      - Resource capacities:
            private_cpu, public_cpu
            private_cpu_slot, public_cpu_slot (approx: capacity * Delta)
      - Arrival rate statistics:
            lambda_mean, lambda_var, slots_observed
      - Task type distribution:
            P_general, P_latency_sensitive, P_deadline_hard,
            P_data_intensive, P_compute_intensive
            + n_tasks_mec, TaskDist_sum, b_mb_med, rho_med, mem_med, hard_share
      - Splittability share:
            non_atomic_share
    """
    tasks    = env_cfg["tasks"]
    arrivals = env_cfg["arrivals"]
    Delta    = float(env_cfg["Delta"])
    K        = int(env_cfg["K"])

    # Capacities from topology/env_config
    private_cpu = np.asarray(env_cfg["private_cpu"], dtype=float)
    public_cpu  = np.asarray(env_cfg["public_cpu"], dtype=float)

    if private_cpu.shape[0] != K or public_cpu.shape[0] != K:
        raise ValueError("Length of private_cpu/public_cpu must equal K in env_config.")

    # Per-slot capacities (assuming private_cpu/public_cpu are per-second-like units)
    private_cpu_slot = private_cpu * Delta
    public_cpu_slot  = public_cpu  * Delta

    # 1) Arrival statistics per MEC
    counts_df = _per_mec_slot_counts(arrivals)
    lam_df    = _lambda_stats_from_counts_mec(counts_df, Delta=Delta)

    # 2) Task-type distribution (+ medians & hard_share)
    dist_df   = _task_distribution_per_mec(tasks)

    # 3) Splittable-task share per MEC
    na_df     = _non_atomic_share_per_mec(tasks)

    # Base table: one row per MEC
    base = pd.DataFrame({
        "mec_id": np.arange(K, dtype=int),
        "private_cpu": private_cpu.astype(float),
        "public_cpu": public_cpu.astype(float),
        "private_cpu_slot": private_cpu_slot.astype(float),
        "public_cpu_slot": public_cpu_slot.astype(float),
    })

    # Merge all components
    prof = (base
            .merge(lam_df,  on="mec_id", how="left")
            .merge(dist_df, on="mec_id", how="left")
            .merge(na_df,   on="mec_id", how="left"))

    # Fill missing values for MECs with no arrivals/tasks
    fill_zero = [
        "lambda_mean", "lambda_var", "slots_observed",
        "n_tasks_mec", "non_atomic_share",
        "TaskDist_sum", "b_mb_med", "rho_med", "mem_med", "hard_share"
    ] + [f"P_{t}" for t in _TASK_TYPES]

    for c in fill_zero:
        if c in prof.columns:
            prof[c] = prof[c].fillna(0.0).astype(float)

    # Soft warning if probabilities don't sum to ~1 for MECs that do have tasks
    if "n_tasks_mec" in prof.columns and "TaskDist_sum" in prof.columns:
        mask = (prof["n_tasks_mec"] > 0) & (~np.isclose(prof["TaskDist_sum"], 1.0, atol=1e-6))
        if mask.any():
            n_bad = int(mask.sum())
            print(f"[warn] TaskDist_sum != 1.0 for {n_bad} MEC(s). (tolerance 1e-6)")

    return prof

# ---- Batch profiling for ALL env_configs ----
def build_all_mec_profiles(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Compute MEC profiles for every (episode → topology → scenario) environment.

    Input:
        env_configs[ep_name][topology_name][scenario_name]["tasks"] / ["arrivals"] / ...

    Output:
        mec_profiles[ep_name][topology_name][scen_name] = DataFrame

    Also writes back to:
        env_configs[ep_name][topology_name][scen_name]["mec_profiles"]
    """
    out: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

    for ep_name, by_topo in env_configs.items():
        out[ep_name] = {}
        for topo_name, by_scen in by_topo.items():
            out[ep_name][topo_name] = {}
            for scen_name, env_cfg in by_scen.items():
                prof = build_mec_profiles_for_env(env_cfg)
                out[ep_name][topo_name][scen_name] = prof
                env_cfg["mec_profiles"] = prof  # attach for direct access

    return out


# ---- Build + quick peek (optional) ----
mec_profiles = build_all_mec_profiles(env_configs)

print("\n ===EXAMPLE MEC PROFILES===")
print(mec_profiles["ep_000"]["clustered"]["heavy"].head())

# Or directly from env_configs:
print(env_configs["ep_000"]["clustered"]["heavy"]["mec_profiles"].head(25))


ep   = "ep_000"
topo = "clustered"
scen = "heavy"

env_cfg = env_configs[ep][topo][scen]

print(env_cfg.keys())

prof = env_cfg["mec_profiles"]
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

MEC_FEATURES_V1 = [
    # ---- (1) Local resources / capacities ----
    "private_cpu_slot",     # Private CPU cycles per slot
    "public_cpu_slot",      # Public CPU cycles per slot

    # ---- (2) Arrival statistics ----
    "lambda_mean",          # Mean task arrival rate (tasks / second)
    "lambda_var",           # Variance of task arrival rate

    # ---- (3) Task generation pattern (probabilities over types) ----
    "P_deadline_hard",
    "P_latency_sensitive",
    "P_compute_intensive",
    "P_data_intensive",
    "P_general",

    # ---- (4) Statistical descriptors of tasks handled by this MEC ----
    "b_mb_med",             # Median input size
    "rho_med",              # Median compute demand (cycles / MB)
    "mem_med",              # Median memory demand (MB)
    "non_atomic_share",     # Share of splittable tasks
    "hard_share",           # Share of hard-deadline tasks
]

# Features to standardize (others like probabilities are already 0–1)
FEATURES_TO_STANDARDIZE = [
    "private_cpu_slot",
    "public_cpu_slot",
    "lambda_mean",
    "lambda_var",
    "b_mb_med",
    "rho_med",
    "mem_med",
]

# --------------------------------------------
# Utility: keep only existing columns
# --------------------------------------------
def _safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """Return only those columns from 'cols' that actually exist in df."""
    return [c for c in cols if c in df.columns]

# --------------------------------------------
# Normalize selected feature columns
# --------------------------------------------
def normalize_features(
    X: np.ndarray,
    cols: List[str],
    feature_list: List[str]
) -> np.ndarray:
    """
    Apply standardization (zero mean, unit variance) only to selected columns.
    """
    standardize_cols = [col for col in feature_list if col in FEATURES_TO_STANDARDIZE]
    if len(standardize_cols) > 0:
        scaler = StandardScaler()
        col_indices = [cols.index(col) for col in standardize_cols if col in cols]
        if col_indices:
            X[:, col_indices] = scaler.fit_transform(X[:, col_indices])
    return X

# --------------------------------------------
# Build MEC feature matrix for one env_config
# --------------------------------------------
def make_mec_feature_matrix_for_env(
    env_cfg: Dict[str, Any],
    feature_list: Optional[List[str]] = None,
    standardize: bool = True,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Build the feature matrix (X) for all MECs in one environment configuration.
    Each row represents a MEC; each column a numerical feature.

    Returns:
        X_scaled   : np.ndarray (n_mecs × n_features)
        used_cols  : list of feature names in order
        mec_ids    : np.ndarray of MEC identifiers
    """
    if "mec_profiles" not in env_cfg or not isinstance(env_cfg["mec_profiles"], pd.DataFrame):
        raise ValueError("env_cfg['mec_profiles'] must contain a valid DataFrame.")

    prof = env_cfg["mec_profiles"].copy()
    if "mec_id" not in prof.columns:
        raise ValueError("mec_profiles must include column 'mec_id'.")

    if feature_list is None:
        feature_list = MEC_FEATURES_V1

    # Keep valid features and fill missing values with zeros
    cols = _safe_cols(prof, feature_list)
    X = prof.reindex(columns=cols).fillna(0.0).astype(float).to_numpy()
    mec_ids = prof["mec_id"].to_numpy(dtype=int)

    # Standardize selected features
    if standardize:
        X = normalize_features(X, cols, feature_list)

    return X, cols, mec_ids

# --------------------------------------------
# Attach MEC features to a single env_config
# --------------------------------------------
def attach_mec_features_to_env(
    env_cfg: Dict[str, Any],
    feature_list: Optional[List[str]] = None,
    standardize: bool = True
) -> Dict[str, Any]:
    """
    Attach the constructed MEC feature matrix and metadata to:
        env_cfg["clustering"]["features"]
    """
    X, cols, mec_ids = make_mec_feature_matrix_for_env(env_cfg, feature_list, standardize)

    env_cfg.setdefault("clustering", {})
    env_cfg["clustering"]["features"] = {
        "X": X,                      # Feature matrix (possibly standardized)
        "feature_cols": cols,        # List of feature names
        "mec_ids": mec_ids,          # MEC identifiers
        "n_mecs": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return env_cfg

# --------------------------------------------
# Apply feature construction to all envs
# --------------------------------------------
def attach_mec_features_to_all_envs(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    feature_list: Optional[List[str]] = None,
    standardize: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Iterate through all (episode → topology → scenario) combinations
    and build the MEC feature matrix for each one.
    """
    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():
                env_configs[ep_name][topo_name][scen_name] = attach_mec_features_to_env(
                    env_cfg, feature_list, standardize
                )
                fz = env_configs[ep_name][topo_name][scen_name]["clustering"]["features"]
                print(
                    f"[features] {ep_name}/{topo_name}/{scen_name} "
                    f"-> X.shape={fz['X'].shape}  (mecs={fz['n_mecs']}, feats={fz['n_features']})"
                )
    return env_configs



# --------------------------------------------
# Sanity checks
# --------------------------------------------
def _assert_no_nan_inf(X: np.ndarray, where: str):
    if not np.isfinite(X).all():
        n_nan = int(np.isnan(X).sum())
        n_inf = int(np.isinf(X).sum())
        raise AssertionError(f"{where}: Feature matrix contains NaN or Inf. counts=(NaN={n_nan}, Inf={n_inf})")

def _assert_mec_count_match(env_cfg: Dict[str, Any], where: str):
    """
    Check that:
      - episodes.N_mecs
      - mec_profiles rows
      - feature matrix rows
    all match.
    """
    if "episodes" not in env_cfg or "mec_profiles" not in env_cfg:
        raise AssertionError(f"{where}: Missing episodes or mec_profiles in env_cfg.")

    n_mecs_ep = int(env_cfg["episodes"]["N_mecs"].iloc[0])
    n_mecs_prof = len(env_cfg["mec_profiles"])
    fz = env_cfg["clustering"]["features"]
    n_mecs_X = fz["n_mecs"]

    if not (n_mecs_X == n_mecs_prof == n_mecs_ep):
        raise AssertionError(
            f"{where}: MEC count mismatch. episodes={n_mecs_ep}, "
            f"profiles={n_mecs_prof}, X={n_mecs_X}"
        )

def _assert_feature_prob_sum_hint_mec(env_cfg: Dict[str, Any], tol: float = 1e-3):
    """
    Hint check: for MECs that have tasks, TaskDist_sum ≈ 1 on average.
    """
    prof = env_cfg["mec_profiles"]
    if "TaskDist_sum" in prof.columns and "n_tasks_mec" in prof.columns:
        mask = prof["n_tasks_mec"] > 0
        if mask.any():
            mean_sum = float(prof.loc[mask, "TaskDist_sum"].mean())
            if abs(mean_sum - 1.0) > tol:
                print(f"[warn] Mean(TaskDist_sum)={mean_sum:.4f} ≠ 1 (tol={tol})")

def run_mec_feature_matrix_sanity_checks(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]]
):
    """
    Run sanity checks over all MEC feature matrices in env_configs.
    """
    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():
                where = f"{ep_name}/{topo_name}/{scen_name}"
                if "clustering" not in env_cfg or "features" not in env_cfg["clustering"]:
                    raise AssertionError(f"{where}: Missing clustering.features.")
                X = env_cfg["clustering"]["features"]["X"]
                _assert_no_nan_inf(X, where)
                _assert_mec_count_match(env_cfg, where)
                _assert_feature_prob_sum_hint_mec(env_cfg)
                if X.shape[1] == 0:
                    raise AssertionError(f"{where}: Empty feature matrix.")
    print("[checks] All MEC feature matrices passed sanity checks.")
                

# Build feature matrices for all envs
env_configs = attach_mec_features_to_all_envs(
    env_configs,
    feature_list=MEC_FEATURES_V1,
    standardize=True
)

# Run sanity checks
run_mec_feature_matrix_sanity_checks(env_configs)

# Example inspection
print("\n=== EXAMPLE: MEC features of ep_000 / clustered / heavy ===")
fz = env_configs["ep_000"]["clustered"]["heavy"]["clustering"]["features"]
print("X.shape:", fz["X"].shape)
print("feature_cols:", fz["feature_cols"])
print("mec_ids (first 10):", fz["mec_ids"][:10])


print("missing features:",
      [c for c in MEC_FEATURES_V1 if c not in prof.columns])

print(prof[ [c for c in MEC_FEATURES_V1 if c in prof.columns] ].head())


fz = env_configs["ep_000"]["clustered"]["heavy"]["clustering"]["features"]
print("X.shape:", fz["X"].shape)
print("feature_cols actually used:", fz["feature_cols"])
print("mec_ids[:10]:", fz["mec_ids"][:10])





# 4.2. Optimal Number of Clusters

# ---------- 4.2.1 Candidate K values ----------
def _candidate_K_values(
    n_mecs: int,
    k_min: int = 2,
    max_K_fraction: float = 0.25,
    max_K_abs: int = 10
) -> List[int]:
    """
    Build a reasonable candidate set for K given n_mecs.

    - Lower bound is k_min (default 2).
    - Upper bound is min(max_K_abs, floor(max_K_fraction * n_mecs), n_mecs - 1).
    - If n_mecs is too small, returns an empty list.
    """
    if n_mecs <= k_min:
        return []

    k_max_by_fraction = int(np.floor(max_K_fraction * n_mecs))
    k_max = min(max_K_abs, n_mecs - 1, max(k_min, k_max_by_fraction))

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
    Step 4.2: K-selection for all environments (MEC clustering).

    For each:
        env_configs[ep_name][topology_name][scenario_name]["clustering"]["features"]["X"]
    we:
      - build candidate K list (based on n_mecs = X.shape[0])
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

                n_mecs = X.shape[0]
                K_candidates = _candidate_K_values(n_mecs)
                if not K_candidates:
                    if verbose:
                        print(f"[4.2/skip] {ep_name}/{topo_name}/{scen_name}: "
                              f"not enough MECs for clustering (n_mecs={n_mecs}).")
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
                          f"n_mecs={n_mecs}, candidates={K_candidates}")
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
# After Step 4.1 (attach_mec_features_to_all_envs + sanity checks), run:
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


prof_light = env_configs[ep][topo]["light"]["mec_profiles"]
prof_mod   = env_configs[ep][topo]["moderate"]["mec_profiles"]
prof_heavy = env_configs[ep][topo]["heavy"]["mec_profiles"]

print("MEC profiles equal (light vs heavy):", prof_light.equals(prof_heavy))
print("MEC profiles equal (light vs mod)  :", prof_light.equals(prof_mod))



# 2. Distributions differences
targets = [
    ("clustered", "light"),
    ("clustered", "moderate"),
    ("clustered", "heavy"),
]

for topo, scen in targets:
    print(f"\n=== {ep} / {topo} / {scen} ===")
    prof = env_configs[ep][topo][scen]["mec_profiles"]

    print("\nλ (arrival rate) stats per MEC:")
    print(prof[["lambda_mean", "lambda_var"]].describe())

    print("\nP(task_type) stats per MEC:")
    cols_p = [f"P_{t}" for t in [
        "deadline_hard",
        "latency_sensitive",
        "compute_intensive",
        "data_intensive",
        "general"
    ]]
    existing_p = [c for c in cols_p if c in prof.columns]
    print(prof[existing_p].describe())

    print("\nmedian task resource stats per MEC:")
    cols_med = [c for c in ["b_mb_med","rho_med","mem_med"] if c in prof.columns]
    print(prof[cols_med].describe())    
    


# 3. metrics similarity
if ep not in K_selection:
    print(f"[warn] episode {ep} not in K_selection.")
else:
    for topo_name, by_scen in K_selection[ep].items():
        for scen_name, sel in by_scen.items():
            print(f"\n=== {topo_name} / {scen_name} ===")
            dfm = sel["metrics_df"]
            print(
                dfm[[
                    "K",
                    "inertia",
                    "silhouette",
                    "calinski_harabasz",
                    "davies_bouldin",
                    "score"
                ]].round(4)
            )        





# 4.3. Implementing K-Means Clustering

def step4_3_run_final_kmeans_for_all_envs(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Step 4.3: Run final K-Means clustering for all environments.

    Assumes:
      - Step 4.1 has already built feature matrices under:
            env_cfg["clustering"]["features"] = {
                "X": np.ndarray (n_mecs, n_features),
                "feature_cols": list[str],
                "mec_ids": np.ndarray (n_mecs,),
                "n_agents": int,
                "n_features": int,
            }
      - Step 4.2 has already selected best_K under:
            env_cfg["clustering"]["k_selection"]["best_K"]

    For each (episode / topology / scenario), this function:
      - runs KMeans with K = best_K
      - stores the result under env_cfg["clustering"]["final"]

    Returns:
      clustering_results[ep_name][topology_name][scen_name] = {
        "K": int,
        "labels": np.ndarray (n_mecs,),
        "centers": np.ndarray (K, n_features),
        "mec_ids": np.ndarray (n_mecs,)
      }
    """
    clustering_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for ep_name, by_topo in env_configs.items():
        clustering_results[ep_name] = {}

        for topo_name, by_scen in by_topo.items():
            clustering_results[ep_name][topo_name] = {}

            for scen_name, env_cfg in by_scen.items():

                clust = env_cfg.get("clustering", {})
                feats = clust.get("features", None)
                k_sel = clust.get("k_selection", None)

                # --- Sanity: feature matrix must be present ---
                if feats is None or "X" not in feats:
                    if verbose:
                        print(f"[4.3/skip] {ep_name}/{topo_name}/{scen_name}: no feature matrix.")
                    continue

                # mec_ids must exist (MEC-based clustering only)
                if "mec_ids" not in feats:
                    raise ValueError(
                        f"[4.3] {ep_name}/{topo_name}/{scen_name}: "
                        "'mec_ids' missing in clustering.features (expected MEC-based features)."
                    )

                X = np.asarray(feats["X"], dtype=float)
                mec_ids = np.asarray(feats["mec_ids"], dtype=int)

                # --- Sanity: K selection must be available ---
                if k_sel is None or "best_K" not in k_sel:
                    if verbose:
                        print(f"[4.3/skip] {ep_name}/{topo_name}/{scen_name}: no K chosen.")
                    continue

                best_K = int(k_sel["best_K"])

                # --- Sanity: K must be valid ---
                if best_K <= 1 or best_K > X.shape[0]:
                    if verbose:
                        print(
                            f"[4.3/skip] invalid best_K={best_K} for "
                            f"{ep_name}/{topo_name}/{scen_name} (n_mecs={X.shape[0]})."
                        )
                    continue

                # --- Final K-Means fit over MECs ---
                km = KMeans(
                    n_clusters=best_K,
                    random_state=random_state,
                    n_init="auto"
                )
                labels = km.fit_predict(X)
                centers = km.cluster_centers_

                # --- Store results into env_config ---
                env_cfg.setdefault("clustering", {})
                env_cfg["clustering"]["final"] = {
                    "K": best_K,
                    "labels": labels,
                    "centers": centers,
                    "mec_ids": mec_ids,
                }

                # --- Also store in return dictionary ---
                clustering_results[ep_name][topo_name][scen_name] = {
                    "K": best_K,
                    "labels": labels,
                    "centers": centers,
                    "mec_ids": mec_ids,
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

print("\n=== STEP 4.3 EXAMPLE: ep_000 / clustered / heavy ===")
ex = clustering_final["ep_000"]["clustered"]["heavy"]
print("K:", ex["K"])
print("Label counts:", np.bincount(ex["labels"]))
print("Centers shape:", ex["centers"].shape)
print("MEC IDs (first 10):", ex["mec_ids"][:10])



# Visualization — PCA: Display clusters in 2D space
def step4_3_plot_clusters_pca(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    out_root: str = "./artifacts/clustering",
    verbose: bool = True
):
    """
    For each environment (ep/topology/scenario), take:
        - X (scaled MEC feature matrix)
        - labels (final K-Means cluster labels)
    Project X to 2D via PCA and save a scatter plot.

    Output saved as:
        <out_root>/<ep>/<topology>/<scenario>/cluster_plot_pca.png
    """
    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():

                clust = env_cfg.get("clustering", {})
                feats = clust.get("features", None)
                final = clust.get("final", None)

                # Need feature matrix and final clustering
                if feats is None or "X" not in feats:
                    continue
                if final is None or "labels" not in final:
                    if verbose:
                        print(f"[PCA/skip] {ep_name}/{topo_name}/{scen_name}: no final KMeans labels.")
                    continue

                X = np.asarray(feats["X"], dtype=float)
                labels = np.asarray(final["labels"], dtype=int)
                K = int(final["K"])

                n_mecs = X.shape[0]

                # PCA requires n_samples >= n_components (2 here)
                if n_mecs < 2:
                    if verbose:
                        print(f"[PCA/skip] {ep_name}/{topo_name}/{scen_name}: "
                              f"n_mecs={n_mecs} < 2, cannot run PCA.")
                    continue

                # PCA projection to 2D
                pca = PCA(n_components=2, random_state=42)
                X_2d = pca.fit_transform(X)

                # Output path
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
                        s=50,
                    )

                plt.title(f"PCA Clusters (MEC): {ep_name} / {topo_name} / {scen_name}  (K={K})")
                plt.xlabel("PCA Component 1")
                plt.ylabel("PCA Component 2")
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()

                if verbose:
                    print(f"[PCA] Saved PCA MEC cluster plot → {out_path}")


step4_3_plot_clusters_pca(env_configs, verbose=True)



# Visualization — spacet-SNE: Display clusters in 2D
def step4_3_plot_clusters_tsne(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    out_root: str = "./artifacts/clustering",
    perplexity: int = 5,
    early_exaggeration: int = 12,
    n_iter: int = 1500,   # kept in signature for compatibility, NOT used (depends on sklearn version)
    verbose: bool = True,
):
    """
    Draw 2D t-SNE visualization for final KMeans clusters over MEC feature space.

    NOTE: Some sklearn versions do not support `n_iter` in TSNE.__init__.
          To keep compatibility, we rely on the library's default n_iter.

    Saves figure as:
        <out_root>/<ep>/<topology>/<scenario>/cluster_plot_tsne.png
    """

    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():

                clust = env_cfg.get("clustering", {})
                feats = clust.get("features", None)
                final = clust.get("final", None)

                # Need feature matrix and final clustering
                if feats is None or "X" not in feats:
                    continue
                if final is None or "labels" not in final:
                    if verbose:
                        print(f"[t-SNE/skip] {ep_name}/{topo_name}/{scen_name}: no cluster labels.")
                    continue

                X = np.asarray(feats["X"], dtype=float)
                labels = np.asarray(final["labels"], dtype=int)
                K = int(final["K"])

                n_mecs = X.shape[0]
                if n_mecs <= perplexity:
                    if verbose:
                        print(f"[t-SNE/skip] {ep_name}/{topo_name}/{scen_name}: "
                              f"n_mecs={n_mecs} <= perplexity={perplexity}")
                    continue

                # Run t-SNE (n_iter omitted for sklearn compatibility)
                tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    early_exaggeration=early_exaggeration,
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
                        alpha=0.8,
                    )

                plt.title(f"t-SNE Clusters (MEC): {ep_name} / {topo_name} / {scen_name}  (K={K})")
                plt.xlabel("t-SNE Dim 1")
                plt.ylabel("t-SNE Dim 2")
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close()

                if verbose:
                    print(f"[t-SNE] Saved t-SNE MEC cluster plot → {out_path}")


step4_3_plot_clusters_tsne(env_configs, verbose=True)





# 4.4. Cluster Interpretation & Profiling

def build_cluster_profiles_for_env(
    ep_name: str,
    topo_name: str,
    scen_name: str,
    env_cfg: Dict[str, Any],
    out_root: str = "./artifacts/clustering"
):
    """
    Build MEC-cluster representative profiles using:
      - final KMeans labels from Step 4.3 (env_cfg['clustering']['final'])
      - cluster centers (scaled feature space) from Step 4.3
      - mec_profiles from Step 3
      - feature metadata from Step 4.1

    Effects:
      - Adds `cluster_id` to mec_profiles (per MEC).
      - Adds `cluster_id` to tasks (via mec_id).
      - Computes per-cluster summary statistics.
      - Saves CSVs for assignments, summaries, and centroids.
      - Attaches results under env_cfg["clustering"]["profiles"].
    """

    # 1) Extract dependencies
    clust = env_cfg.get("clustering", {})
    feats = clust.get("features", None)
    final = clust.get("final", None)   # must exist after Step 4.3

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

    mec_ids     = np.asarray(feats["mec_ids"], dtype=int)
    feature_cols = list(feats["feature_cols"])

    if "mec_profiles" not in env_cfg or not isinstance(env_cfg["mec_profiles"], pd.DataFrame):
        raise ValueError(f"[4.4] env_cfg['mec_profiles'] must be a DataFrame for {ep_name}/{topo_name}/{scen_name}")

    mec_prof = env_cfg["mec_profiles"].copy()
    if "mec_id" not in mec_prof.columns:
        raise ValueError(f"[4.4] mec_profiles is missing 'mec_id' for {ep_name}/{topo_name}/{scen_name}")

    # 2) Build assignment table (mec_id → cluster_id)
    assign_df = pd.DataFrame({
        "mec_id": mec_ids,
        "cluster_id": labels
    })

    mec_prof = mec_prof.merge(assign_df, on="mec_id", how="left")

    # 2bis) Inject cluster_id into tasks DataFrame based on mec_id
    tasks_df = env_cfg.get("tasks", None)
    if tasks_df is not None and "mec_id" in tasks_df.columns:
        tasks_df = tasks_df.merge(assign_df, on="mec_id", how="left")
        # MECs with no cluster (should not happen) get -1
        tasks_df["cluster_id"] = tasks_df["cluster_id"].fillna(-1).astype(int)
        env_cfg["tasks"] = tasks_df
        print(f"[4.4] Added 'cluster_id' to tasks for {ep_name}/{topo_name}/{scen_name} "
              f"(rows={len(tasks_df)})")

        # Debug: show first few rows of tasks with cluster_id
        print(f"Debug: Tasks DataFrame head for {ep_name}/{topo_name}/{scen_name}:")
        print(tasks_df.head())

    # 3) Cluster-level summary over MEC profiles (numeric columns only)
    numeric_cols = mec_prof.select_dtypes(include=[np.number]).columns.tolist()
    # Group key is cluster_id; do not aggregate it
    agg_cols = [c for c in numeric_cols if c != "cluster_id"]

    cluster_summary = (
        mec_prof[["cluster_id"] + agg_cols]
        .groupby("cluster_id", as_index=False)
        .mean()
        .sort_values("cluster_id")
    )

    cluster_sizes = (
        mec_prof.groupby("cluster_id")["mec_id"]
        .count()
        .rename("n_mecs_cluster")
        .reset_index()
    )
    cluster_summary = cluster_summary.merge(cluster_sizes, on="cluster_id", how="left")

    # Debug: show cluster summary head
    print(f"Debug: MEC cluster summary for {ep_name}/{topo_name}/{scen_name}:")
    print(cluster_summary.head())

    # 4) Decode centroids back to "original" scale
    #    (since we no longer keep a scaler, we keep the scaled space as-is;
    #     centers_original is equal to centers_scaled here.)
    centers_original = centers_scaled.copy()

    centroids_scaled_df = pd.DataFrame(centers_scaled, columns=feature_cols)
    centroids_scaled_df.insert(0, "cluster_id", np.arange(best_K))

    centroids_original_df = pd.DataFrame(centers_original, columns=feature_cols)
    centroids_original_df.insert(0, "cluster_id", np.arange(best_K))

    # 5) Save to disk
    out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
    os.makedirs(out_dir, exist_ok=True)

    assign_path    = os.path.join(out_dir, "mec_cluster_assignments.csv")
    summary_path   = os.path.join(out_dir, "mec_cluster_summary.csv")
    cent_sc_path   = os.path.join(out_dir, "mec_centroids_scaled.csv")
    cent_orig_path = os.path.join(out_dir, "mec_centroids_original.csv")

    assign_df.to_csv(assign_path, index=False)
    cluster_summary.to_csv(summary_path, index=False)
    centroids_scaled_df.to_csv(cent_sc_path, index=False)
    centroids_original_df.to_csv(cent_orig_path, index=False)

    print(f"[4.4] {ep_name}/{topo_name}/{scen_name} → MEC cluster profiles built.")
    print(cluster_sizes.set_index("cluster_id")["n_mecs_cluster"])

    # 6) Attach final results back to env_cfg
    env_cfg["mec_profiles"] = mec_prof
    env_cfg.setdefault("clustering", {})
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

def build_all_cluster_profiles(env_configs: Dict[str, Dict[str, Dict[str, Any]]]):
    """
    Run cluster profile construction (Step 4.4) for all ep/topology/scenario combos.

    Returns:
      out[ep_name][topo_name][scen_name] = clustering profiles dict
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}

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

print("\n=== EXAMPLE: MEC cluster summary for ep_000 / clustered / heavy ===")
ex_ep   = "ep_000"
ex_topo = "clustered"
ex_scen = "heavy"

if (ex_ep in cluster_profiles and
    ex_topo in cluster_profiles[ex_ep] and
    ex_scen in cluster_profiles[ex_ep][ex_topo]):

    ex_prof = cluster_profiles[ex_ep][ex_topo][ex_scen]
    print("K =", ex_prof["K"])
    print("\nCluster summary (first few columns):")
    print(ex_prof["cluster_summary"].iloc[:, :10])
else:
    print("[warn] Example triple not found in cluster_profiles.")
    

# Quick debug peek
test = env_configs["ep_000"]["clustered"]["heavy"]["clustering"]
print("\nclustering keys:", test.keys())
print("\nMEC cluster summary:")
print(env_configs["ep_000"]["clustered"]["heavy"]["clustering"]["profiles"]["cluster_summary"])



# Heatmap Visualization
def plot_cluster_profile_heatmap(
    env_cfg: Dict[str, Any],
    ep_name: str,
    topo_name: str,
    scen_name: str,
    out_root: str = "./artifacts/clustering"
):
    """
    Draw a heatmap of MEC cluster profile means for a given env_cfg.

    Uses:
        env_cfg["clustering"]["profiles"]["cluster_summary"]
    Each row = one cluster_id
    Each column = one numeric cluster-level feature (mean over MECs in that cluster)
    """

    # 1) Extract cluster profiles
    clust = env_cfg.get("clustering", {})
    profs = clust.get("profiles", None)

    if profs is None or "cluster_summary" not in profs:
        print(f"[heatmap/skip] No cluster profiles for {ep_name}/{topo_name}/{scen_name}")
        return None

    cluster_summary = profs["cluster_summary"].copy()
    K = int(profs["K"])

    # 2) Drop non-feature columns
    #   - cluster_id: used only for row labels
    #   - n_mecs_cluster: cluster size, not a feature
    drop_cols = ["cluster_id", "n_mecs_cluster"]
    feature_cols = [c for c in cluster_summary.columns if c not in drop_cols]

    # Keep only numeric feature columns (avoid issues if any column is non-numeric)
    numeric_feature_cols = [
        c for c in feature_cols
        if np.issubdtype(cluster_summary[c].dtype, np.number)
    ]

    if not numeric_feature_cols:
        print(f"[heatmap/skip] No numeric cluster-level features for {ep_name}/{topo_name}/{scen_name}")
        return None

    df = cluster_summary[numeric_feature_cols].copy()

    # 3) Normalize each feature column to [0, 1] for better visualization
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-9)

    # 4) Prepare output path
    out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cluster_profile_heatmap.png")

    # 5) Row labels based on actual cluster_id values
    cluster_ids = cluster_summary["cluster_id"].tolist()
    yticklabels = [f"Cluster {cid}" for cid in cluster_ids]

    # 6) Plot heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        df_norm,
        annot=False,
        cmap="viridis",
        xticklabels=df_norm.columns,
        yticklabels=yticklabels
    )

    plt.title(f"Cluster Profile Heatmap (MEC): {ep_name}/{topo_name}/{scen_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[heatmap] Saved → {out_path}")
    return out_path


# ---- Run heatmap for all environments ----
for ep in env_configs:
    for topo in env_configs[ep]:
        for scen in env_configs[ep][topo]:
            plot_cluster_profile_heatmap(
                env_configs[ep][topo][scen],
                ep, topo, scen
            )



# Saving Information
def save_env_configs_to_csv(
    env_configs: Dict[str, Dict[str, Dict[str, Any]]],
    out_root: str = "./artifacts/env_configs"
) -> None:
    """
    Save env_configs content to CSV files in a structured folder hierarchy:

        <out_root>/<episode>/<topology>/<scenario>/

    For each env_cfg, this function saves (when available):
      - episodes.csv
      - agents.csv
      - arrivals.csv
      - tasks.csv              (labeled + cluster_id, etc.)
      - agent_profiles.csv
      - clustering_k_selection_metrics.csv
      - cluster_assignments.csv
      - cluster_summary.csv
      - centroids_scaled.csv
      - centroids_original.csv

    Note:
      Only pandas DataFrames are saved as CSV. Nested non-DataFrame
      objects inside env_cfg["clustering"] (e.g., numpy arrays) are
      not saved here.
    """
    for ep_name, by_topo in env_configs.items():
        for topo_name, by_scen in by_topo.items():
            for scen_name, env_cfg in by_scen.items():

                # Create directory for this (episode, topology, scenario)
                out_dir = os.path.join(out_root, ep_name, topo_name, scen_name)
                os.makedirs(out_dir, exist_ok=True)

                # --- Core tables ---
                if "episodes" in env_cfg and isinstance(env_cfg["episodes"], pd.DataFrame):
                    path = os.path.join(out_dir, "episodes.csv")
                    env_cfg["episodes"].to_csv(path, index=False)
                    print(f"[save] episodes → {path}")

                if "agents" in env_cfg and isinstance(env_cfg["agents"], pd.DataFrame):
                    path = os.path.join(out_dir, "agents.csv")
                    env_cfg["agents"].to_csv(path, index=False)
                    print(f"[save] agents → {path}")

                if "arrivals" in env_cfg and isinstance(env_cfg["arrivals"], pd.DataFrame):
                    path = os.path.join(out_dir, "arrivals.csv")
                    env_cfg["arrivals"].to_csv(path, index=False)
                    print(f"[save] arrivals → {path}")

                if "tasks" in env_cfg and isinstance(env_cfg["tasks"], pd.DataFrame):
                    path = os.path.join(out_dir, "tasks.csv")
                    env_cfg["tasks"].to_csv(path, index=False)
                    print(f"[save] tasks → {path}")

                # --- Agent profiles (Step 3) ---
                if "agent_profiles" in env_cfg and isinstance(env_cfg["agent_profiles"], pd.DataFrame):
                    path = os.path.join(out_dir, "agent_profiles.csv")
                    env_cfg["agent_profiles"].to_csv(path, index=False)
                    print(f"[save] agent_profiles → {path}")

                # --- Clustering-related artifacts ---
                clust = env_cfg.get("clustering", {})

                # 1) K-selection metrics (Step 4.2)
                k_sel = clust.get("k_selection", None)
                if isinstance(k_sel, dict):
                    metrics_df = k_sel.get("metrics_df", None)
                    if isinstance(metrics_df, pd.DataFrame):
                        path = os.path.join(out_dir, "clustering_k_selection_metrics.csv")
                        metrics_df.to_csv(path, index=False)
                        print(f"[save] k_selection metrics → {path}")

                # 2) Cluster profiles (Step 4.4)
                profs = clust.get("profiles", None)
                if isinstance(profs, dict):
                    # cluster_assignments: DataFrame(agent_id, mec_id, cluster_id) if present
                    ca = profs.get("cluster_assignments", None)
                    if isinstance(ca, pd.DataFrame):
                        path = os.path.join(out_dir, "cluster_assignments.csv")
                        ca.to_csv(path, index=False)
                        print(f"[save] cluster_assignments → {path}")

                    # cluster_summary: per-cluster aggregated stats
                    cs = profs.get("cluster_summary", None)
                    if isinstance(cs, pd.DataFrame):
                        path = os.path.join(out_dir, "cluster_summary.csv")
                        cs.to_csv(path, index=False)
                        print(f"[save] cluster_summary → {path}")

                    # centroids (scaled)
                    cent_scaled_df = profs.get("centroids_scaled_df", None)
                    if isinstance(cent_scaled_df, pd.DataFrame):
                        path = os.path.join(out_dir, "centroids_scaled.csv")
                        cent_scaled_df.to_csv(path, index=False)
                        print(f"[save] centroids_scaled → {path}")

                    # centroids (original scale)
                    cent_orig_df = profs.get("centroids_original_df", None)
                    if isinstance(cent_orig_df, pd.DataFrame):
                        path = os.path.join(out_dir, "centroids_original.csv")
                        cent_orig_df.to_csv(path, index=False)
                        print(f"[save] centroids_original → {path}")

                # If you ever want to also save the raw feature matrix X or labels/centers
                # (which are numpy arrays), it's better to use np.save in a separate helper
                # rather than forcing them into CSV here.

    print(f"[save] All data has been saved under '{out_root}'.")


# Usage
save_env_configs_to_csv(env_configs)




# اینجاااااااممممممممم
# Step 5: MDP Environment

# Step 5.2 - Action Semantics (Contextual Bandit)
def select_algo_for_cluster(cluster_profile):
    """
    Select an RL algorithm for each cluster based on its profile.
    The profile can include information like lambda_mean, task types, etc.
    """
    if cluster_profile["lambda_mean"] > 0.5 and cluster_profile["P_compute_intensive"] > 0.5:
        return "PPO"
    elif cluster_profile["lambda_mean"] < 0.5 and cluster_profile["P_general"] > 0.5:
        return "DQN"
    else:
        return "A2C"


# Step 5.1 - State Construction
# Example of scaling a feature array
def normalize_feature_array(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features.reshape(-1, 1))  # Assuming features are in a 1D array

class OffloadingEnv:
    def __init__(self, env_cfg):
        self.env_cfg = env_cfg
        self.delta = env_cfg["Delta"]
        self.t_slots = env_cfg["T_slots"]
        self.agent_profiles = env_cfg["agent_profiles"]
        self.alpha = 1.0  # Weight for delay penalty
        self.beta = 1.0   # Weight for drop penalty
        
        # Initialize the queues for each agent
        self.queues = {agent_id: {'local': [], 'mec': [], 'cloud': []} for agent_id in self.agent_profiles["agent_id"]}
        
        # Initialize other environment variables like CPU/memory
        self.cpu_capacity = env_cfg["private_cpu"]
        self.cloud_capacity = env_cfg["cloud_cpu"]
        
        # Initialize the algorithm mapping for each cluster
        self.algo_map = self.assign_algorithms_to_clusters()

    def assign_algorithms_to_clusters(self):
        """
        Assign RL algorithms to clusters based on their profiles.
        """
        algo_map = {}
        for cluster_id, cluster_profile in self.env_cfg["clustering"]["profiles"].items():
            algo_type = select_algo_for_cluster(cluster_profile)
            algo_map[cluster_id] = {"type": algo_type, "agent": None}
        return algo_map

    def reset(self):
        """Reset the environment for a new episode."""
        for agent_id in self.queues:
            self.queues[agent_id] = {'local': [], 'mec': [], 'cloud': []}
        return self._get_state()

    def step(self, actions):
        """
        Perform one step in the environment for each agent.
        actions: dict of {agent_id: action}
        """
        # Update the queues based on actions
        for agent_id, action in actions.items():
            if action == 0:  # LOCAL
                # Add task to local queue
                self.queues[agent_id]['local'].append(self._get_task(agent_id))
            elif action == 1:  # MEC
                # Add task to MEC queue
                self.queues[agent_id]['mec'].append(self._get_task(agent_id))
            elif action == 2:  # CLOUD
                # Add task to Cloud queue
                self.queues[agent_id]['cloud'].append(self._get_task(agent_id))

        # Process tasks in queues (FIFO)
        self._process_queues()

        # Calculate reward for each agent based on task completions, delays, etc.
        rewards = self._calculate_rewards()

        # Update state
        next_state = self._get_state()

        # Check if episode is done (based on T_slots)
        done = self._check_done()

        return next_state, rewards, done, {}
  
    def _process_queues(self):
        """Process tasks in the queues based on GPS and FIFO."""
        total_active_queues = sum([1 for queues in self.queues.values() if any(queue) for queue in queues.values()])
        cpu_share = self.cpu_capacity / total_active_queues if total_active_queues > 0 else 0

        for agent_id, queues in self.queues.items():
            for queue_name, queue in queues.items():
                # If there are tasks in the queue, process them
                if queue:
                    task = queue.pop(0)  # FIFO
                    task_processing_time = task["c_cycles"] / cpu_share  # Assuming equal CPU share
                    # Process the task (you may want to add logic for processing based on GPS share here)
                    self._task_complete(agent_id, task, queue_name)

                    # Update the queue state after processing the task
                    print(f"Processed task for agent {agent_id} in {queue_name} queue, estimated time: {task_processing_time:.2f} cycles.")

    def _task_complete(self, agent_id, task, queue_name):
        """Handle task completion (can be expanded with delay/drop logic)."""
        print(f"Task completed for agent {agent_id} in {queue_name} queue.")
        # Here you can implement delay or drop logic if required.
    
    def _calculate_rewards(self):
        """Calculate reward based on task delays, drop rates, etc."""
        rewards = {}
        for agent_id, queues in self.queues.items():
            for queue_name, queue in queues.items():
                if queue:  # If there are tasks in the queue
                    task = queue[0]  # Take the first task for reward calculation
                    task_completion_time = self._calculate_task_time(task)
                    delay = self._calculate_delay(task, task_completion_time)
                    drop = self._check_task_drop(task)

                    # Reward calculation based on delay and drop
                    rewards[agent_id] = -self.alpha * delay - self.beta * drop
        return rewards
    
    def _calculate_task_time(self, task):
        """Calculate the total processing time for a task."""
        # Here we calculate total time based on task's computation cycles
        # For now, assuming constant computation power, you can update based on more detailed logic
        processing_time = task["c_cycles"] / self.cpu_capacity  # Example logic
        return processing_time

    def _calculate_delay(self, task, completion_time):
        """Calculate delay based on the task's total execution time and its deadline."""
        # Assuming task has a deadline attribute
        if "deadline_slots" in task and task["deadline_slots"] > 0:
            delay = max(0, completion_time - task["deadline_slots"])  # Time past the deadline
            return delay
        return 0  # No delay if no deadline
    
    def _check_task_drop(self, task):
        """Check if the task is dropped due to exceeding its deadline."""
        if "deadline_slots" in task and task["deadline_slots"] > 0:
            if task["deadline_slots"] <= 0:  # If deadline is violated
                return 1  # Drop penalty
        return 0  # No drop if deadline is not violated

    # Update _get_state to normalize relevant features
    def _get_state(self):
        """Return the state for each agent, which includes queue lengths, resources, etc."""
        state = {}
        for agent_id, queues in self.queues.items():
            # Retrieve features from tasks and agent profile for state
            task_features = self._get_task_features(agent_id)
            agent_profile = self.agent_profiles.loc[self.agent_profiles['agent_id'] == agent_id]
            cluster_id = agent_profile["cluster_id"].values[0]  # Assuming cluster_id exists in agent_profiles
            
            # Normalize the relevant task features and agent profile features
            task_features["b_mb"] = normalize_feature_array(task_features["b_mb"])
            task_features["c_cycles"] = normalize_feature_array(task_features["c_cycles"])
            task_features["mem_mb"] = normalize_feature_array(task_features["mem_mb"])
            task_features["rho_cyc_per_mb"] = normalize_feature_array(task_features["rho_cyc_per_mb"])

            state[agent_id] = {
                "local_queue_length": len(queues['local']),
                "mec_queue_length": len(queues['mec']),
                "cloud_queue_length": len(queues['cloud']),
                "task_features": task_features,  # Features related to the current task
                "agent_profile": agent_profile,  # Agent's resource profile (lambda, etc.)
                "cluster_id": self._one_hot_cluster(cluster_id)  # One-hot encode cluster_id
            }
        return state

    # One-hot encoding for task type and cluster id
    def _one_hot_task_type(task_type):
        task_types = ['general', 'latency_sensitive', 'compute_intensive', 'data_intensive', 'deadline_hard']
        task_type_onehot = [0] * len(task_types)
        if task_type in task_types:
            task_type_onehot[task_types.index(task_type)] = 1
        return task_type_onehot

    def _one_hot_cluster(self, cluster_id):
        """One-hot encode the cluster_id."""
        num_clusters = self.env_cfg["clustering"]["profiles"]["K"]
        cluster_onehot = [0] * num_clusters
        cluster_onehot[cluster_id] = 1
        return cluster_onehot

    # Update task features to include one-hot encoding for task_type
    def _get_task_features(self, agent_id):
        """Get the current task features for the agent from the pre-generated tasks."""
        tasks_df = self.env_cfg['tasks']
        agent_tasks = tasks_df[tasks_df['agent_id'] == agent_id]
        
        if not agent_tasks.empty:
            task = agent_tasks.iloc[0]
            task_features = {
                "b_mb": task["b_mb"],
                "c_cycles": task["c_cycles"],
                "mem_mb": task["mem_mb"],
                "rho_cyc_per_mb": task["rho_cyc_per_mb"],
                "deadline_slots": task["deadline_slots"],
                "task_type": self._one_hot_task_type(task["task_type"]),  # One-hot encode task_type
            }
            return task_features
        else:
            return {}  # If no tasks for this agent, return an empty dictionary
    
    def _check_done(self):
        """Check if the episode is done (based on T_slots)."""
        # If the number of slots (T_slots) has been completed, end the episode
        if self.t_slots <= 0:
            # Save the statistics of the episode like delays, drop rate, etc.
            self._log_episode_statistics()
            return True
        return False

    def _log_episode_statistics(self):
        """Log the statistics of the current episode."""
        # Calculate and log relevant statistics such as average delay, drop rate, and reward
        print("Logging statistics for the episode...")
        # Example (you can modify this based on your requirement):
        avg_delay = np.mean([self._calculate_delay(task, self._calculate_task_time(task)) for task in self._get_all_tasks()])
        avg_drop_rate = np.mean([self._check_task_drop(task) for task in self._get_all_tasks()])
        
        print(f"Average Delay: {avg_delay}, Average Drop Rate: {avg_drop_rate}")
        # You can also save these statistics to a file or a list for further analysis
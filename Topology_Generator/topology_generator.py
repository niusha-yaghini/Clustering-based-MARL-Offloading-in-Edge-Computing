# -*- coding: utf-8 -*-
"""
Topology generator for Edge–MEC–Cloud, aligned with HOODIE's topology generators.
Outputs: nodes.csv, links.csv, routing.csv, topology_meta.json

- Layering: Edge (agents) — MEC (K_MEC) — Cloud (1)
- MEC↔MEC + MEC→Cloud capacities are generated like HOODIE's FullyConnected or SkipConnections.
- Edge→MEC links (bandwidth/latency) and queue policies are added per assumptions in Chapter 4.
- Reads agents from your dataset (agents.csv) to import f_local per agent.

This is a policy-agnostic, static topology description for section 5.2.2.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import os, json, time, platform, hashlib, getpass

# -------------------------
# reproducibility
# -------------------------
GLOBAL_SEED = 12345
rng_global = np.random.default_rng(GLOBAL_SEED)

# -------------------------
# units helpers
# -------------------------
def mbps_to_bps(x_mbps: float) -> float:
    return float(x_mbps) * 1e6

def ms_to_sec(x_ms: float) -> float:
    return float(x_ms) / 1000.0

# -------------------------
# config dataclasses
# -------------------------
@dataclass
class EpisodeTiming:
    Delta: float        # seconds per slot (keep same as dataset)
    T_slots: int        # used for consistency/meta; no dynamics here

@dataclass
class MECCompute:
    f_mec_min: float    # Hz
    f_mec_max: float    # Hz

@dataclass
class CloudCompute:
    f_cloud: float      # Hz (single cloud)

@dataclass
class EdgeToMECLink:
    bandwidth_mbps: float
    latency_ms: float

@dataclass
class MECToCloudLink:
    latency_ms: float   # capacity sampled by topology model per-MEC

@dataclass
class MECToMECLink:
    latency_ms: float   # capacity sampled by topology model per-pair

@dataclass
class CapacitySampler:
    """Capacity sampler settings for HOODIE-style generators."""
    # common min/max for all horizontal MEC↔MEC capacities (in Mbps)
    horiz_min_mbps: float
    horiz_max_mbps: float
    # common min/max for vertical MEC→Cloud capacities (in Mbps)
    vert_min_mbps: float
    vert_max_mbps: float
    # distribution: "uniform" | "loguniform"
    distribution: str = "uniform"
    symmetric: bool = True  # whether MEC↔MEC capacities should be symmetric

@dataclass
class TopologyConfig:
    name: str
    seed: int
    # layers
    K_MEC: int
    # timing (match dataset meta for consistency)
    timing: EpisodeTiming
    # compute
    mec_compute: MECCompute
    cloud_compute: CloudCompute
    # links
    edge2mec: EdgeToMECLink
    mec2cloud: MECToCloudLink
    mec2mec: MECToMECLink
    # capacity generator
    caps: CapacitySampler
    # style: "fully_connected" | "skip_connections"
    style: str = "fully_connected"
    skip_connections: int = 2  # used if style == "skip_connections"

# -------------------------
# capacity samplers (HOODIE-like)
# -------------------------
def _sample_capacity(rng, lo_mbps: float, hi_mbps: float, dist: str) -> float:
    lo, hi = float(lo_mbps), float(hi_mbps)
    if dist == "loguniform":
        lo_ = np.log(max(lo, 1e-6))
        hi_ = np.log(max(hi, lo + 1e-6))
        return float(np.exp(rng.uniform(lo_, hi_)))
    # default: uniform
    return float(rng.uniform(lo, hi))

def build_connection_matrix(cfg: TopologyConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Returns a (#MEC) x (#MEC + 1) matrix:
      cols 0..K_MEC-1  : horizontal MEC↔MEC capacities (Mbps)
      col  K_MEC       : vertical MEC→Cloud capacities (Mbps)
    style: fully_connected or skip_connections
    """
    K = cfg.K_MEC
    M = np.zeros((K, K + 1), dtype=float)

    # vertical capacities MEC->Cloud
    for i in range(K):
        M[i, K] = _sample_capacity(
            rng, cfg.caps.vert_min_mbps, cfg.caps.vert_max_mbps, cfg.caps.distribution
        )

    # horizontal capacities MEC<->MEC
    if cfg.style == "fully_connected":
        for i in range(K):
            for j in range(K):
                if i == j: 
                    continue
                cap = _sample_capacity(
                    rng, cfg.caps.horiz_min_mbps, cfg.caps.horiz_max_mbps, cfg.caps.distribution
                )
                M[i, j] = cap
                if cfg.caps.symmetric:
                    M[j, i] = cap
    else:  # skip_connections
        step = max(1, int(cfg.skip_connections))
        for i in range(K):
            j = (i + step) % K
            if i != j:
                cap = _sample_capacity(
                    rng, cfg.caps.horiz_min_mbps, cfg.caps.horiz_max_mbps, cfg.caps.distribution
                )
                M[i, j] = cap
                if cfg.caps.symmetric:
                    M[j, i] = cap

    return M

# -------------------------
# IO helpers
# -------------------------
def _fingerprint(obj: dict) -> str:
    s = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

def save_csv(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

def save_json(obj: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

# -------------------------
# main builder
# -------------------------
def build_topology(cfg: TopologyConfig,
                   agents_csv: str,
                   out_dir: str = "./topology") -> Dict[str, str]:
    """
    Build topology files for section 5.2.2
    - agents_csv: the CSV from your dataset generator (per scenario), to import f_local per agent.
    """
    rng = np.random.default_rng(cfg.seed)
    os.makedirs(out_dir, exist_ok=True)

    # --- load agents to get Edge nodes & f_local
    agents_df = pd.read_csv(agents_csv)
    if "agent_id" not in agents_df.columns:
        raise ValueError("agents.csv must contain 'agent_id' column")
    # optional fallbacks
    if "f_local" not in agents_df.columns:
        # if missing, set a default or raise error
        agents_df["f_local"] = 1.5e9  # Hz default
    if "m_local" not in agents_df.columns:
        agents_df["m_local"] = 4e9     # bytes or MB; just for meta

    # --- MEC compute
    mec_ids = [f"MEC_{i}" for i in range(cfg.K_MEC)]
    f_mec = rng.uniform(cfg.mec_compute.f_mec_min, cfg.mec_compute.f_mec_max, size=cfg.K_MEC)

    # --- Cloud compute
    cloud_id = "CLOUD"
    f_cloud = cfg.cloud_compute.f_cloud

    # --- connection matrix (HOODIE-like)
    C = build_connection_matrix(cfg, rng)   # Mbps

    # ========== nodes.csv ==========
    nodes_rows = []
    # Edge nodes
    for _, r in agents_df.iterrows():
        nodes_rows.append({
            "node_id": f"EDGE_{int(r['agent_id'])}",
            "layer": "edge",
            "cpu_hz": float(r["f_local"]),
            "queue_policy": "FCFS",
            "queue_capacity": "inf"
        })
    # MEC nodes
    for i, mec in enumerate(mec_ids):
        nodes_rows.append({
            "node_id": mec,
            "layer": "mec",
            "cpu_hz": float(f_mec[i]),
            "queue_policy": "FCFS",
            "queue_capacity": "inf"
        })
    # Cloud node
    nodes_rows.append({
        "node_id": cloud_id,
        "layer": "cloud",
        "cpu_hz": float(f_cloud),
        "queue_policy": "FCFS",
        "queue_capacity": "inf"
    })
    nodes_df = pd.DataFrame(nodes_rows)

    # ========== routing.csv (Edge→MEC assignment) ==========
    # simple stable mapping: agent_id % K_MEC
    routing_rows = []
    for _, r in agents_df.iterrows():
        aid = int(r["agent_id"])
        mec_idx = aid % cfg.K_MEC
        routing_rows.append({
            "agent_id": aid,
            "mec_id": mec_ids[mec_idx],
            "rule": "agent_id_mod_KMEC"
        })
    routing_df = pd.DataFrame(routing_rows)

    # ========== links.csv ==========
    links_rows = []

    # Edge→MEC links (fixed capacity & latency)
    bw_e2m_bps = mbps_to_bps(cfg.edge2mec.bandwidth_mbps)
    lat_e2m_s  = ms_to_sec(cfg.edge2mec.latency_ms)
    for _, r in routing_df.iterrows():
        src = f"EDGE_{int(r['agent_id'])}"
        dst = r["mec_id"]
        links_rows.append({
            "src": src, "dst": dst,
            "bandwidth_bps": bw_e2m_bps,
            "latency_sec": lat_e2m_s,
            "kind": "edge_to_mec"
        })

    # MEC↔MEC horizontal links (from matrix C, columns 0..K-1)
    lat_m2m_s  = ms_to_sec(cfg.mec2mec.latency_ms)
    for i, src_mec in enumerate(mec_ids):
        for j, dst_mec in enumerate(mec_ids):
            if i == j:
                continue
            cap_mbps = C[i, j]
            if cap_mbps <= 0:
                continue
            links_rows.append({
                "src": src_mec, "dst": dst_mec,
                "bandwidth_bps": mbps_to_bps(cap_mbps),
                "latency_sec": lat_m2m_s,
                "kind": "mec_to_mec"
            })

    # MEC→Cloud vertical links (from matrix C, last column)
    lat_m2c_s  = ms_to_sec(cfg.mec2cloud.latency_ms)
    for i, src_mec in enumerate(mec_ids):
        cap_mbps = C[i, cfg.K_MEC]     # vertical col
        links_rows.append({
            "src": src_mec, "dst": cloud_id,
            "bandwidth_bps": mbps_to_bps(cap_mbps),
            "latency_sec": lat_m2c_s,
            "kind": "mec_to_cloud"
        })

    links_df = pd.DataFrame(links_rows)

    # ========== meta ==========
    meta = {
        "schema_version": "1.0.0",
        "name": cfg.name,
        "seed": cfg.seed,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {"python": platform.python_version(), "user": getpass.getuser()},
        "timing": asdict(cfg.timing),
        "K_MEC": cfg.K_MEC,
        "compute": {
            "edge_f_local_source": "agents.csv (dataset)",
            "mec_compute": asdict(cfg.mec_compute),
            "cloud_compute": asdict(cfg.cloud_compute)
        },
        "links": {
            "edge2mec": asdict(cfg.edge2mec),
            "mec2mec": asdict(cfg.mec2mec),
            "mec2cloud": asdict(cfg.mec2cloud),
            "capacity_sampler": asdict(cfg.caps),
            "style": cfg.style,
            "skip_connections": cfg.skip_connections
        },
        "units": {
            "cpu_hz": "Hz",
            "bandwidth_bps": "bits per second",
            "bandwidth_input_mbps": "Mbps (converted to bps)",
            "latency_sec": "seconds",
            "latency_input_ms": "milliseconds"
        }
    }
    meta["fingerprint"] = _fingerprint(meta)

    # ========== save ==========
    paths = {}
    paths["nodes_csv"]    = save_csv(nodes_df, os.path.join(out_dir, "nodes.csv"))
    paths["links_csv"]    = save_csv(links_df, os.path.join(out_dir, "links.csv"))
    paths["routing_csv"]  = save_csv(routing_df, os.path.join(out_dir, "routing.csv"))
    paths["topo_meta"]    = save_json(meta, os.path.join(out_dir, "topology_meta.json"))

    return paths

# -------------------------
# example usage
# -------------------------
if __name__ == "__main__":
    # Example config (put real numbers after you paste HOODIE's hyperparameters)
    cfg = TopologyConfig(
        name="moderate_topology",
        seed=20251026,
        K_MEC=3,
        timing=EpisodeTiming(Delta=1.0, T_slots=3600),
        mec_compute=MECCompute(f_mec_min=4.0e9, f_mec_max=6.0e9),
        cloud_compute=CloudCompute(f_cloud=20.0e9),
        edge2mec=EdgeToMECLink(bandwidth_mbps=50.0, latency_ms=10.0),
        mec2cloud=MECToCloudLink(latency_ms=25.0),
        mec2mec=MECToMECLink(latency_ms=5.0),
        caps=CapacitySampler(
            horiz_min_mbps=200.0, horiz_max_mbps=800.0,
            vert_min_mbps=500.0,  vert_max_mbps=2000.0,
            distribution="uniform",
            symmetric=True
        ),
        style="fully_connected",     # or "skip_connections"
        skip_connections=2
    )

    # point to your dataset's agents.csv (e.g., datasets/moderate/moderate_ep0_agents.csv)
    AGENTS_CSV = "./datasets/moderate/moderate_ep0_agents.csv"
    out = build_topology(cfg, agents_csv=AGENTS_CSV, out_dir="./topology/moderate")
    print(json.dumps(out, indent=2))

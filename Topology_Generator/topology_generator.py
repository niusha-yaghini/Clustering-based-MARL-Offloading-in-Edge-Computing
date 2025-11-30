from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np
import json
import os
import time
import hashlib
import platform
import getpass
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
print("OK")


# Optional deps (for graph)
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    _GRAPH_OK = True
except Exception:
    _GRAPH_OK = False


# ============================================================
# Utility functions
# ============================================================
def _fp(obj: dict) -> str:
    """Compute a short fingerprint for a dictionary."""
    s = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

def _save_json(obj: dict, path: str) -> str:
    """Save a dictionary as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

def _save_text(text: str, path: str) -> str:
    """Save plain text to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def _save_matrix_csv(M: np.ndarray, path: str) -> str:
    """Save connection matrix as a CSV with MEC labels and a cloud column."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    K = M.shape[0]
    header = [f"mec_{i}" for i in range(K)] + ["cloud"]
    lines = [",".join([""] + header)]
    for i in range(K):
        row = ",".join([f"mec_{i}"] + [str(float(x)) for x in M[i, :]])
        lines.append(row)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ============================================================
# Data classes (TopologyHyper)
# ============================================================
@dataclass
class TopologyHyper:
    """
    Hyperparameters for topology generation.
    MEC / Cloud compute capacities are loaded from environment files,
    so this class only stores structural and link-level parameters.
    """

    time_step: float

    # Link bandwidths (to be provided as inputs by the user)
    bw_mec_mec: float = 3.0   # MEC↔MEC bandwidth
    bw_mec_cloud: float = 5.0 # MEC→Cloud bandwidth

    # Topology structure
    topology_type: str = "skip_connections"  # fully_connected | clustered | skip_connections
    skip_k: int = 3                          # used in skip_connections (k-nearest ring)
    symmetric: bool = True
    num_clusters: int = 3
    inter_cluster_frac: float = 0.0          # for weak inter-cluster links in clustered topology

    # Optional: store environment CSV paths (not strictly required for logic)
    environment_mec_file: Optional[str] = None
    environment_cloud_file: Optional[str] = None

    seed: int = 2025


# ============================================================
# Reading MEC / Cloud data
# ============================================================
def read_environment_files(mec_file: str, cloud_file: str):
    """
    Read MEC and Cloud capacities from CSV files.
    Expected columns:
      - MEC CSV: 'Private CPU Capacity', 'Public CPU Capacity'
      - Cloud CSV: 'computational_capacity'
    """
    
    print("inside read_environment_files")
    mec_df = pd.read_csv(mec_file)
    cloud_df = pd.read_csv(cloud_file)

    num_mec = len(mec_df)
    private_caps = mec_df["Private CPU Capacity"].tolist()
    public_caps = mec_df["Public CPU Capacity"].tolist()

    cloud_cap = float(cloud_df["computational_capacity"].iloc[0])

    return num_mec, private_caps, public_caps, cloud_cap


# ============================================================
# Build Connection Matrix
# ============================================================
def _build_connection_matrix(h: TopologyHyper, K: int) -> np.ndarray:
    """
    Build connection matrix of shape (K, K+1):

    - Columns 0..K-1  : MEC↔MEC horizontal links
    - Column K        : MEC→Cloud vertical links

    Link capacities are NOT random here:
    - MEC↔MEC links use h.bw_mec_mec
    - MEC→Cloud links use h.bw_mec_cloud
    """
    
    print("inside _build_connection_matrix")

    # Initialize matrix with zeros
    M = np.zeros((K, K + 1), dtype=float)

    # Read bandwidth parameters from hyper
    bw_mm = float(h.bw_mec_mec)     # MEC↔MEC bandwidth
    bw_mc = float(h.bw_mec_cloud)   # MEC→Cloud bandwidth

    # ---------------------------
    # Horizontal MEC↔MEC links
    # ---------------------------
    if h.topology_type == "fully_connected":
        # Fully connected: all MECs interconnected
        for i in range(K):
            for j in range(i + 1, K):
                M[i, j] = bw_mm
                if h.symmetric:
                    M[j, i] = bw_mm

    elif h.topology_type == "skip_connections":
        # k-nearest ring
        step = max(1, int(h.skip_k))
        for i in range(K):
            for s in range(1, step + 1):
                j = (i + s) % K
                if i == j:
                    continue
                M[i, j] = bw_mm
                if h.symmetric:
                    M[j, i] = bw_mm

    elif h.topology_type == "clustered":
        # Clustered topology: fully connected inside clusters
        C = max(1, int(h.num_clusters))

        # Compute cluster sizes
        sizes = [K // C] * C
        for idx in range(K % C):
            sizes[idx] += 1
        starts = np.cumsum([0] + sizes[:-1])
        clusters = [(int(s), int(s + sz)) for s, sz in zip(starts, sizes)]  # [(start, end), ...]

        # Intra-cluster links (fully connected)
        for (a, b) in clusters:
            for i in range(a, b):
                for j in range(i + 1, b):
                    M[i, j] = bw_mm
                    if h.symmetric:
                        M[j, i] = bw_mm

        # Optional weak inter-cluster links
        if h.inter_cluster_frac > 0.0:
            weak = bw_mm * float(h.inter_cluster_frac)
            for c1 in range(len(clusters)):
                for c2 in range(c1 + 1, len(clusters)):
                    a1, b1 = clusters[c1]
                    a2, b2 = clusters[c2]
                    i = a1   # representative node of cluster 1
                    j = a2   # representative node of cluster 2
                    if i == j:
                        continue
                    M[i, j] = max(M[i, j], weak)
                    if h.symmetric:
                        M[j, i] = max(M[j, i], weak)
    else:
        # Unknown topology_type → no horizontal links
        pass

    # Ensure zero diagonal for MEC↔MEC
    for i in range(K):
        M[i, i] = 0.0

    # ---------------------------
    # Vertical MEC→Cloud links
    # ---------------------------
    for i in range(K):
        M[i, K] = bw_mc

    return M


# ============================================================
# Graph drawing
# ============================================================
def _draw_graph_png(M: np.ndarray,
                    out_png: str,
                    title: str = "MEC Graph (MB/slot)",
                    with_cloud: bool = True):
    """Draw a simple network graph of MECs and Cloud and save as PNG."""
    if not _GRAPH_OK:
        return None

    K = M.shape[0]

    G = nx.Graph()

    # MEC nodes
    for i in range(K):
        G.add_node(f"MEC_{i}", layer="mec")

    # MEC↔MEC edges
    for i in range(K):
        for j in range(i + 1, K):
            cap = max(M[i, j], M[j, i])
            if cap > 0:
                G.add_edge(f"MEC_{i}", f"MEC_{j}", weight=cap)

    # Layout for MEC nodes
    pos = nx.circular_layout([f"MEC_{i}" for i in range(K)])

    # Optional cloud node
    if with_cloud:
        G.add_node("CLOUD", layer="cloud")
        pos["CLOUD"] = np.array([0.0, 1.25])
        for i in range(K):
            cap_cloud = M[i, K]
            if cap_cloud > 0:
                G.add_edge(f"MEC_{i}", "CLOUD", weight=cap_cloud)

    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[n for n, d in G.nodes(data=True) if d.get("layer") == "mec"]
    )
    if with_cloud:
        nx.draw_networkx_nodes(G, pos, nodelist=["CLOUD"], node_shape="s")

    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)

    edge_labels = {(u, v): f"{G[u][v]['weight']:.1f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png


# ============================================================
# Markdown Report Writer
# ============================================================
def _write_markdown_report(topo: dict, meta: dict, graph_png: Optional[str], out_md: str):
    """Write a simple Markdown report summarizing the topology."""
    K = topo["number_of_servers"]
    priv = topo["private_cpu_capacities"]
    pub = topo["public_cpu_capacities"]
    cloud = topo["cloud_computational_capacity"]

    M = np.array(topo["connection_matrix"], dtype=float)
    horiz = M[:, :K]
    vert = M[:, K]

    nonzero = int((horiz > 0).sum())
    density = nonzero / float(K * (K - 1)) if K > 1 else 0.0

    hyper = meta.get("hyperparameters", {})
    bw_mm = hyper.get("bw_mec_mec", "n/a")
    bw_mc = hyper.get("bw_mec_cloud", "n/a")

    md = []
    md.append(f"# Topology Report\n")
    md.append(f"- **Servers (MEC)**: {K}")
    md.append(f"- **Topology type**: {topo['topology_type']}")
    md.append(f"- **Link density (MEC↔MEC)**: {density:.3f}")
    md.append("")
    md.append(f"## Compute Capacities")
    md.append(f"- Private: {priv}")
    md.append(f"- Public:  {pub}")
    md.append(f"- Cloud:   {cloud}")
    md.append("")
    md.append(f"## Link Capacities")
    md.append(f"- MEC↔MEC bandwidth: {bw_mm}")
    md.append(f"- MEC→Cloud bandwidth: {bw_mc}")
    md.append(f"- MEC→Cloud stats: min={vert.min():.3g}, mean={vert.mean():.3g}, max={vert.max():.3g}")
    md.append("")
    if graph_png:
        md.append(f"## Graph")
        md.append(f"![Topology Graph]({os.path.basename(graph_png)})")
        md.append("")

    _save_text("\n".join(md), out_md)
    return out_md


# ============================================================
# Main builder
# ============================================================
def build_topology(
    h: TopologyHyper,
    mec_csv_path: str,
    cloud_csv_path: str,
    out_topology: str = "./topology/topology.json",
    out_meta: str = "./topology/topology_meta.json"
) -> Dict[str, str]:
    """
    Build a single topology using:
      - MEC / Cloud capacities from CSV
      - structural parameters from TopologyHyper
    """

    
    print("inside build_topology")
    # Optional RNG in case you later want stochastic behaviors
    _ = np.random.default_rng(h.seed)

    # STEP 1 — Read MEC + Cloud data
    K, private_caps, public_caps, cloud_cap = read_environment_files(
        mec_csv_path, cloud_csv_path
    )

    # STEP 2 — Build connection matrix
    M = _build_connection_matrix(h, K)

    # STEP 3 — Construct topology dict
    topo = {
        "number_of_servers": K,
        "private_cpu_capacities": private_caps,
        "public_cpu_capacities": public_caps,
        "cloud_computational_capacity": cloud_cap,
        "connection_matrix": M.tolist(),
        "time_step": h.time_step,
        "topology_type": h.topology_type,
        "skip_k": h.skip_k,
        "symmetric": h.symmetric,
        "num_clusters": h.num_clusters
    }

    # Meta info
    meta = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fingerprint": _fp(topo),
        "env": {"python": platform.python_version(), "user": getpass.getuser()},
        "hyperparameters": asdict(h),
    }

    # Save data
    _save_json(topo, out_topology)
    _save_json(meta, out_meta)

    out_dir = os.path.dirname(out_topology) or "."
    cm_csv = os.path.join(out_dir, "connection_matrix.csv")
    _save_matrix_csv(M, cm_csv)

    # Optional Graph
    graph_png = None
    if _GRAPH_OK:
        graph_png = os.path.join(out_dir, "topology_graph.png")
        _draw_graph_png(M, graph_png, with_cloud=True)

    # Markdown report
    report_md = os.path.join(out_dir, "topology_report.md")
    _write_markdown_report(topo, meta, graph_png, report_md)

    print("inside build_topology after savings")
    return {
        "topology_json": out_topology,
        "meta_json": out_meta,
        "connection_matrix_csv": cm_csv,
        "graph_png": graph_png if graph_png else "",
        "report_md": report_md
    }


# ============================================================
# Build multiple variants
# ============================================================
def build_three_topologies_variants(
    mec_csv_path: str,
    cloud_csv_path: str,
    delta: float,
    seed_base: int,
    bw_mec_mec: float,
    bw_mec_cloud: float,
    out_root: str = "./topologies",
) -> Dict[str, Dict[str, str]]:
    """
    Build three topologies (fully_connected, clustered, skip_connections)
    using the same MEC/Cloud environment and link bandwidths.
    """

    os.makedirs(out_root, exist_ok=True)
    print("inside build_three_topologies_variants")

    variants = ["fully_connected", "clustered", "skip_connections"]
    results: Dict[str, Dict[str, str]] = {}

    for idx, topo_type in enumerate(variants):
        h = TopologyHyper(
            time_step=delta,
            topology_type=topo_type,
            skip_k=3,
            symmetric=True,
            num_clusters=3,
            bw_mec_mec=bw_mec_mec,
            bw_mec_cloud=bw_mec_cloud,
            environment_mec_file=mec_csv_path,
            environment_cloud_file=cloud_csv_path,
            seed=seed_base + idx * 100
        )

        out_dir = os.path.join(out_root, topo_type)
        os.makedirs(out_dir, exist_ok=True)

        paths = build_topology(
            h,
            mec_csv_path=mec_csv_path,
            cloud_csv_path=cloud_csv_path,
            out_topology=os.path.join(out_dir, "topology.json"),
            out_meta=os.path.join(out_dir, "topology_meta.json")
        )

        results[topo_type] = paths

    return results


# ============================================================
# Example entrypoint (optional)
# ============================================================
if __name__ == "__main__":
    print("hi")
    # Example usage (you can adjust paths and parameters as needed)
    mec_csv = '../Environment_Generator/simulation_output/environment.csv'
    cloud_csv = "../Environment_Generator/simulation_output/cloud_info.csv"

    # Example bandwidths (can be changed by the user)
    bw_mm = 3.0   # MEC↔MEC
    bw_mc = 5.0   # MEC→Cloud

    delta = 1.0
    seed_base = 20251129

    out_dirs = build_three_topologies_variants(
        mec_csv_path=mec_csv,
        cloud_csv_path=cloud_csv,
        delta=delta,
        seed_base=seed_base,
        bw_mec_mec=bw_mm,
        bw_mec_cloud=bw_mc,
        out_root="./topologies"
    )
    print(json.dumps(out_dirs, indent=2))
    
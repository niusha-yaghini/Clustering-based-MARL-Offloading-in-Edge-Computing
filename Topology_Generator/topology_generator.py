# -*- coding: utf-8 -*-
"""
HOODIE–style Topology Builder (enhanced)
- Separate private/public capacities (not merged)
- Connection matrix shape: (K, K+1), last column = MEC→Cloud
- Styles: 'fully_connected' | 'skip_connections'
- Inputs per-second -> scaled by Delta to per-slot (HOODIE-compatible)

Extras (without changing the core structure):
  * Save connection_matrix.csv
  * Draw network graph as PNG (NetworkX + Matplotlib)
  * Write a lightweight Markdown report

Outputs (core): topology.json, topology_meta.json
Extras: connection_matrix.csv, topology_graph.png, topology_report.md
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Optional
import numpy as np
import json, os, time, hashlib, platform, getpass

# Optional deps for graph/report
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    _GRAPH_OK = True
except Exception:
    _GRAPH_OK = False

# ----------------------------
# Data classes
# ----------------------------
@dataclass
class TopologyHyper:
    number_of_servers: int              # K (MEC count)
    time_step: float                    # Δ (sec per slot)

    # ----- Compute capacities (per second); we scale by Δ -> per slot
    private_cpu_min: Optional[float] = None
    private_cpu_max: Optional[float] = None
    public_cpu_min: Optional[float] = None
    public_cpu_max: Optional[float] = None

    # If you don't have separate ranges, provide totals + public_share in [0,1]
    cpu_total_min: Optional[float] = None
    cpu_total_max: Optional[float] = None
    public_share: Optional[float] = None

    # Cloud capacity (per second) — fixed or range
    cloud_capacity: Optional[float] = None
    cloud_capacity_min: Optional[float] = None
    cloud_capacity_max: Optional[float] = None

    # ----- Links (per second); we scale by Δ -> per slot
    horiz_cap_min: float = 8.0         # MB/s (MEC↔MEC)
    horiz_cap_max: float = 12.0
    cloud_cap_min: float = 50.0        # MB/s (MEC→Cloud)
    cloud_cap_max: float = 200.0

    # ----- Generator
    topology_type: str = "skip_connections"  # 'fully_connected' | 'skip_connections'
    skip_k: int = 5
    symmetric: bool = True

    # ----- RNG
    seed: int = 2025

# ----------------------------
# Utils
# ----------------------------
def _fp(obj: dict) -> str:
    s = json.dumps(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]

def _save_json(obj: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

def _save_text(text: str, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def _save_matrix_csv(M: np.ndarray, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # header: mec_0,...,mec_{K-1}, cloud
    K = M.shape[0]
    header = [f"mec_{i}" for i in range(K)] + ["cloud"]
    lines = [",".join([""] + header)]
    for i in range(K):
        row = ",".join([f"mec_{i}"] + [str(float(x)) for x in M[i, :]])
        lines.append(row)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

# ----------------------------
# Builders
# ----------------------------
def _sample_cloud_capacity(h: TopologyHyper, rng: np.random.Generator) -> float:
    if h.cloud_capacity is not None:
        return float(h.cloud_capacity)
    if h.cloud_capacity_min is not None and h.cloud_capacity_max is not None:
        return float(rng.uniform(h.cloud_capacity_min, h.cloud_capacity_max))
    return 3.0e10  # fallback per-second

def _build_compute_caps(h: TopologyHyper, rng: np.random.Generator):
    K = h.number_of_servers
    if (h.private_cpu_min is not None and h.private_cpu_max is not None and
        h.public_cpu_min  is not None and h.public_cpu_max  is not None):
        priv_sec = rng.uniform(h.private_cpu_min, h.private_cpu_max, size=K)
        pub_sec  = rng.uniform(h.public_cpu_min,  h.public_cpu_max,  size=K)
    else:
        tot_sec  = rng.uniform(float(h.cpu_total_min or 2.0e9),
                               float(h.cpu_total_max or 3.0e9),
                               size=K)
        share = float(h.public_share if h.public_share is not None else 0.3)
        pub_sec  = tot_sec * share
        priv_sec = tot_sec - pub_sec

    priv_slot = (priv_sec * h.time_step).astype(float).tolist()
    pub_slot  = (pub_sec  * h.time_step).astype(float).tolist()
    return priv_slot, pub_slot

def _build_connection_matrix(h: TopologyHyper, rng: np.random.Generator):
    """
    Returns M of shape (K, K+1), MB/slot.
    cols 0..K-1 : MEC↔MEC horizontal capacities
    col  K      : MEC→Cloud vertical capacity
    """
    K = h.number_of_servers
    M = np.zeros((K, K + 1), dtype=float)

    # vertical MEC→Cloud (per slot)
    for i in range(K):
        cap_sec = rng.uniform(h.cloud_cap_min, h.cloud_cap_max)
        M[i, K] = float(cap_sec * h.time_step)

    # horizontal MEC↔MEC (per slot)
    if h.topology_type == "fully_connected":
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                cap_sec = rng.uniform(h.horiz_cap_min, h.horiz_cap_max)
                M[i, j] = float(cap_sec * h.time_step)
        if h.symmetric:
            M[:, :K] = np.maximum(M[:, :K], M[:, :K].T)
    else:  # skip_connections
        step = max(1, int(h.skip_k))
        for i in range(K):
            for s in range(1, step + 1):
                j = (i + s) % K
                if i == j:
                    continue
                cap_sec = rng.uniform(h.horiz_cap_min, h.horiz_cap_max)
                M[i, j] = float(cap_sec * h.time_step)
                if h.symmetric:
                    M[j, i] = M[i, j]

    return M

# ----------------------------
# Graph drawing (optional)
# ----------------------------
def _draw_graph_png(M: np.ndarray,
                    out_png: str,
                    title: str = "MEC Graph (MB/slot)",
                    with_cloud: bool = True):
    """
    Draw an undirected MEC↔MEC graph + (optionally) MEC→Cloud spokes.
    Edge labels show capacity (MB/slot). Cloud drawn on top.
    """
    if not _GRAPH_OK:
        return None

    K = M.shape[0]
    G = nx.Graph()

    # MEC nodes
    for i in range(K):
        G.add_node(f"MEC_{i}", layer="mec")

    # MEC↔MEC edges (only upper triangle to avoid duplicates)
    for i in range(K):
        for j in range(i + 1, K):
            cap = max(M[i, j], M[j, i])
            if cap > 0:
                G.add_edge(f"MEC_{i}", f"MEC_{j}", weight=cap)

    # Positions: circular for MEC
    pos = nx.circular_layout([f"MEC_{i}" for i in range(K)])

    # Optionally add cloud as a separate node
    if with_cloud:
        G.add_node("CLOUD", layer="cloud")
        # place cloud slightly above center
        pos["CLOUD"] = np.array([0.0, 1.25])
        for i in range(K):
            cap_cloud = M[i, K]
            if cap_cloud > 0:
                G.add_edge(f"MEC_{i}", "CLOUD", weight=cap_cloud)

    # Draw
    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n, d in G.nodes(data=True) if d.get("layer")=="mec"])
    if with_cloud:
        nx.draw_networkx_nodes(G, pos, nodelist=["CLOUD"], node_shape="s")

    # MEC↔MEC edges
    edges_mm = [(u,v) for u,v in G.edges() if "CLOUD" not in (u,v)]
    nx.draw_networkx_edges(G, pos, edgelist=edges_mm)

    # MEC→Cloud edges
    edges_mc = [(u,v) for u,v in G.edges() if "CLOUD" in (u,v)]
    nx.draw_networkx_edges(G, pos, edgelist=edges_mc, style="dashed")

    # labels
    nx.draw_networkx_labels(G, pos, font_size=9)

    # Edge labels with capacities
    edge_labels = {(u,v): f"{G[u][v]['weight']:.1f}" for u,v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png

# ----------------------------
# Report (Markdown)
# ----------------------------
def _write_markdown_report(topo: dict, meta: dict, graph_png: Optional[str], out_md: str):
    K = topo["number_of_servers"]
    units = meta.get("units", {})
    compute_unit = units.get("compute", "CPU cycles per slot")
    link_unit = units.get("links", "MB per slot")
    time_unit = units.get("time_step", "seconds")

    priv = topo["private_cpu_capacities"]
    pub  = topo["public_cpu_capacities"]
    cloud = topo["cloud_computational_capacity"]

    # Simple summaries
    def s(lst): 
        if not lst: return "n/a"
        arr = np.array(lst, dtype=float)
        return f"min={arr.min():.3g}, mean={arr.mean():.3g}, max={arr.max():.3g}"

    M = np.array(topo["connection_matrix"], dtype=float)
    horiz = M[:, :K]
    vert  = M[:, K]

    md = []
    md.append(f"# Topology Report\n")
    md.append(f"- **Servers (MEC)**: {K}")
    md.append(f"- **Time step (Δ)**: {topo['time_step']} {time_unit}")
    md.append(f"- **Topology type**: {topo.get('topology_type','n/a')}, **skip_k**: {topo.get('skip_k','-')}, **symmetric**: {topo.get('symmetric','-')}")
    md.append("")
    md.append(f"## Compute Capacities ({compute_unit})")
    md.append(f"- Private (per MEC): {s(priv)}")
    md.append(f"- Public  (per MEC): {s(pub)}")
    md.append(f"- Cloud (single): {cloud:.3g}")
    md.append("")
    md.append(f"## Link Capacities ({link_unit})")
    md.append(f"- Horizontal MEC↔MEC (non-zero entries): {int((horiz>0).sum())}")
    md.append(f"- MEC→Cloud (length K): min={vert.min():.3g}, mean={vert.mean():.3g}, max={vert.max():.3g}")
    md.append("")
    if graph_png:
        md.append(f"## Graph")
        md.append(f"![Topology Graph]({os.path.basename(graph_png)})")
        md.append("")
    md.append("## Notes")
    md.append("- Values are per slot; per-slot = per-second × Δ.")
    md.append(f"- Units: compute={compute_unit}, links={link_unit}, time_step={time_unit}.")
    md_txt = "\n".join(md)
    _save_text(md_txt, out_md)
    return out_md

# ----------------------------
# Main builder
# ----------------------------
def build_topology(h: TopologyHyper,
                   out_topology: str = "./topology/topology.json",
                   out_meta: str = "./topology/topology_meta.json") -> Dict[str, str]:
    rng = np.random.default_rng(h.seed)

    # compute
    private_caps, public_caps = _build_compute_caps(h, rng)
    cloud_cap_sec = _sample_cloud_capacity(h, rng)
    cloud_cap = float(cloud_cap_sec * h.time_step)  # per-slot

    # links
    M = _build_connection_matrix(h, rng)

    # HOODIE-compatible payload
    topo = {
        "number_of_servers": h.number_of_servers,
        "private_cpu_capacities": private_caps,     # cycles/slot
        "public_cpu_capacities": public_caps,       # cycles/slot
        "cloud_computational_capacity": cloud_cap,  # cycles/slot
        "connection_matrix": M.tolist(),            # MB/slot
        "time_step": h.time_step,
        "topology_type": h.topology_type,
        "skip_k": h.skip_k,
        "symmetric": h.symmetric
    }
    meta = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fingerprint": _fp(topo),
        "env": {"python": platform.python_version(), "user": getpass.getuser()},
        "units": {
            "compute": "CPU cycles per slot",
            "links": "MB per slot",
            "time_step": "seconds"
        },
        "notes": {
            "inputs_unit": {"compute": "CPU cycles per second", "links": "MB per second"},
            "conversion": "per_slot = per_second * time_step"
        }
    }

    # Save core outputs
    _save_json(topo, out_topology)
    _save_json(meta, out_meta)

    # Extras (do NOT change core structure)
    out_dir = os.path.dirname(out_topology) or "."
    cm_csv = os.path.join(out_dir, "connection_matrix.csv")
    _save_matrix_csv(M, cm_csv)

    graph_png = None
    if _GRAPH_OK:
        graph_png = os.path.join(out_dir, "topology_graph.png")
        _draw_graph_png(M, graph_png, title="MEC Graph (MB/slot)", with_cloud=True)

    report_md = os.path.join(out_dir, "topology_report.md")
    _write_markdown_report(topo, meta, graph_png, report_md)

    return {
        "topology_json": out_topology,
        "meta_json": out_meta,
        "connection_matrix_csv": cm_csv,
        "graph_png": graph_png if graph_png else "",
        "report_md": report_md
    }

# ----------------------------
# Quick CLI example
# ----------------------------
def build_from_hyperparameters_json(hparams_path: str,
                                    out_dir: str = "./topology") -> Dict[str, str]:
    with open(hparams_path, "r", encoding="utf-8") as f:
        hp = json.load(f)

    th = TopologyHyper(
        number_of_servers = int(hp.get("number_of_servers", 18)),
        time_step         = float(hp.get("time_step", 1.0)),
        private_cpu_min   = hp.get("private_cpu_min"),
        private_cpu_max   = hp.get("private_cpu_max"),
        public_cpu_min    = hp.get("public_cpu_min"),
        public_cpu_max    = hp.get("public_cpu_max"),
        cpu_total_min     = hp.get("cpu_total_min"),
        cpu_total_max     = hp.get("cpu_total_max"),
        public_share      = hp.get("public_share"),
        cloud_capacity    = hp.get("cloud_capacity"),
        cloud_capacity_min= hp.get("cloud_capacity_min"),
        cloud_capacity_max= hp.get("cloud_capacity_max"),
        horiz_cap_min     = float(hp.get("horizontal_capacities_min", 8.0)),
        horiz_cap_max     = float(hp.get("horizontal_capacities_max", 12.0)),
        cloud_cap_min     = float(hp.get("cloud_capacities_min", 50.0)),
        cloud_cap_max     = float(hp.get("cloud_capacities_max", 200.0)),
        topology_type     = hp.get("topology_type", "skip_connections"),
        skip_k            = int(hp.get("skip_k", 5)),
        symmetric         = bool(hp.get("symmetric", True)),
        seed              = int(hp.get("seed", 2025))
    )

    os.makedirs(out_dir, exist_ok=True)
    return build_topology(
        th,
        out_topology=os.path.join(out_dir, "topology.json"),
        out_meta=os.path.join(out_dir, "topology_meta.json")
    )

if __name__ == "__main__":
    H = TopologyHyper(
        number_of_servers=18,
        time_step=1.0,
        private_cpu_min=1.2e9, private_cpu_max=1.8e9,   # cycles/s
        public_cpu_min=0.5e9,  public_cpu_max=0.9e9,    # cycles/s
        cloud_capacity=3.0e10,                          # cycles/s
        horiz_cap_min=8.0, horiz_cap_max=12.0,          # MB/s
        cloud_cap_min=80.0, cloud_cap_max=120.0,        # MB/s
        topology_type="skip_connections", skip_k=5, symmetric=True,
        seed=20251026
    )
    build_topology(H, out_topology="./topology/topology.json",
                      out_meta="./topology/topology_meta.json")

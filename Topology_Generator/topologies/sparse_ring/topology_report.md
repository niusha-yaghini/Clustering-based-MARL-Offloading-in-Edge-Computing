# Topology Report

- **Servers (MEC)**: 18
- **Time step (Δ)**: 1.0 seconds
- **Topology type**: skip_connections, **skip_k**: 1, **symmetric**: True, **num_clusters**: 3

## Compute Capacities (CPU cycles per slot)
- Private (per MEC): min=1.21e+09, mean=1.41e+09, max=1.76e+09
- Public  (per MEC): min=5.85e+08, mean=7.44e+08, max=8.96e+08
- Cloud (single): 3e+10

## Link Capacities (MB per slot)
- Horizontal MEC↔MEC (non-zero entries): 36
- MEC→Cloud (length K): min=82, mean=101, max=120

## Graph
![Topology Graph](topology_graph.png)

## Notes
- Values are per slot; per-slot = per-second × Δ.
- Units: compute=CPU cycles per slot, links=MB per slot, time_step=seconds.
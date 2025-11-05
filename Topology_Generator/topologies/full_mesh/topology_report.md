# Topology Report

- **Servers (MEC)**: 18
- **Time step (Δ)**: 1.0 seconds
- **Topology type**: fully_connected, **skip_k**: 1, **symmetric**: True, **num_clusters**: 3

## Compute Capacities (CPU cycles per slot)
- Private (per MEC): min=1.23e+09, mean=1.49e+09, max=1.79e+09
- Public  (per MEC): min=5.19e+08, mean=7.17e+08, max=8.66e+08
- Cloud (single): 3e+10

## Link Capacities (MB per slot)
- Horizontal MEC↔MEC (non-zero entries): 306 (density=1.000)
- MEC→Cloud (length K): min=80.3, mean=99.7, max=117

## Graph
![Topology Graph](topology_graph.png)

## Notes
- Values are per slot; per-slot = per-second × Δ.
- Units: compute=CPU cycles per slot, links=MB per slot, time_step=seconds.
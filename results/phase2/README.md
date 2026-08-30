# Phase 2: Bottleneck Mechanism Analysis

## Goal

Explain why continuous layer partition changes sampled SLA-safe capacity under compute heterogeneity. Phase 2 does **not** introduce a search heuristic. It reuses the Phase-1 synthetic event-driven evaluator and inspects the chain:

`layer allocation -> compute cost -> residual TPOT budget -> network commitment -> sampled SLA-safe capacity`.

## Online validation

GitHub Actions workflow: `phase2-bottleneck-mechanism`.

- Run 1 (`33298373911`) completed successfully and exposed a comparison-design issue: each partition was inspected at its own first unsafe intensity, so raw network commitment values were not comparable across placements with different capacity edges.
- Run 2 (`33298503048`) completed successfully and corrected the experiment by comparing worst / uniform / best placements at the **same workload intensity**: the uniform partition's first sampled unsafe intensity.
- Complete CSV / JSON / logs from Run 2 are preserved in the workflow artifact `phase2-bottleneck-mechanism-results`.

## Common-load results

| Scenario | Reference lambda | Uniform safe? | Best safe? | Uniform weighted layers | Best weighted layers | Uniform peak normalized commitment | Best peak normalized commitment | Uniform min link headroom (MB/s) | Best min link headroom (MB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| homogeneous control | 0.0095 | no | no | 80.000 | 80.000 | 1.1079 | 1.1079 | -134.72 | -134.72 |
| mild alternating | 0.0095 | no | no | 81.841 | 81.841 | 1.1439 | 1.1439 | -179.84 | -179.84 |
| single slow stage | 0.0070 | no | **yes** | 88.182 | 84.909 | 1.0453 | **0.9839** | -56.62 | **+20.16** |
| graded | 0.0065 | no | **yes** | 89.680 | 83.512 | 1.0762 | **0.7557** | -95.26 | **+305.41** |
| shuffled severe | 0.0065 | no | **yes** | 90.266 | 83.193 | 1.0887 | **0.7516** | -110.94 | **+310.49** |

For the three scenarios with a material Phase-1 capacity gap, the best observed placement also reduces the weighted compute cost, increases the minimum Decode residual budget, lowers peak network commitment at a common load, and changes the uniform partition's unsafe verdict to safe.

Representative Decode diagnostics:

- **single slow stage**: max Decode compute 0.05154 -> 0.04963 s; min Decode residual 0.05222 -> 0.05566 s.
- **graded**: max Decode compute 0.05242 -> 0.04814 s; min Decode residual 0.05064 -> 0.05835 s.
- **shuffled severe**: max Decode compute 0.05276 -> 0.04795 s; min Decode residual 0.05003 -> 0.05868 s.

The homogeneous control remains invariant, as expected. Mild alternating has only a small Phase-1 placement gap and the uniform partition is already among the best sampled placements, so no stronger mechanism contrast is expected there.

## Phase-2 conclusion

**Supported in 4/4 heterogeneous scenarios under the common-load criterion**, with the practically important separation visible in the three scenarios that had material Phase-1 capacity gaps.

The mechanism supported by the current synthetic model is:

1. assigning fewer layers to slower stages reduces the aggregate profiled compute cost;
2. lower Decode compute cost leaves a larger residual TPOT budget;
3. the evaluator therefore requires a smaller per-request network commitment;
4. aggregate link commitment falls below capacity at workload intensities where the uniform placement is already unsafe;
5. sampled SLA-safe capacity increases.

## Scope / limitation

These results are a **mechanism and feasibility screen**, not a HELIX numerical reproduction. Phase 1 and Phase 2 use deterministic synthetic normalized compute profiles because the complete HELIX per-layer profile and exact workload trace were not available in this repository. The next stage should test whether a simple SLA-aware partition search can outperform uniform, random, and compute-balanced baselines before broader validation is attempted.

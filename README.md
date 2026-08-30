# SLA Layer-Partition Sensitivity

This repository is an independent, self-contained phase-1 experiment answering one narrow question:

> Does changing the continuous 80-layer partition across eight compute-heterogeneous pipeline stages materially change sampled SLA-safe capacity?

It was created from scratch. It does not contain, modify, branch from, or write to any prior repository.

## Scope

The experiment keeps the public setup reported in *An SLA-Safe Capacity Evaluator for Layer-Level LLM Pipelines*: an 80-layer LLaMA-2-70B-shaped model, eight 40 GB stages, continuous ordered layer intervals, TTFT = 2.0 s, TPOT = 0.150 s, 5 ms fixed overhead, 59.3772 microseconds/token accounting-only intrinsic Prefill overhead, 16-token Decode blocks, and sampled workload intensities from 0.006 to 0.022.

The evaluator preserves the paper's relevant semantics:

- atomic Prefill and block-granular Decode events;
- state-dependent profiled compute;
- TTFT/TPOT residual-budget checks;
- proportional per-link latency-budget allocation and aggregate bandwidth commitments;
- event-level weight, workspace, reserve, KV, and activation memory checks;
- largest-safe / nearest-unsafe sampled capacity reporting with monotonicity checks.

## Important limitation

The uploaded paper does not publish the underlying HELIX per-layer profile tables or the exact finite request traces. This phase therefore uses deterministic, normalized synthetic compute profiles and a deterministic finite workload, both fully specified in `config/phase1.json`. Results are a placement-sensitivity screening result, not a numerical reproduction of the paper or a claim about production A100 performance.

## Experiment matrix

Five compute scenarios are tested: a homogeneous control plus four heterogeneous patterns (mild alternating, one slow stage, graded speeds, and shuffled severe speeds). Each scenario receives exactly 20 unique legal continuous partitions. The set always includes uniform placement; heterogeneous cases also include a compute-balanced partition, with the remainder generated deterministically by boundary-preserving layer transfers.

A scenario is called materially sensitive only when both conditions hold:

- best-to-worst sampled safe-capacity gap is at least 10%;
- absolute sampled gap is at least 0.001 intensity units.

## Reproduce locally

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m sla_partition_sensitivity.experiment \
  --config config/phase1.json \
  --output results/phase1
```

GitHub Actions runs the same validation and experiment on every push to `main`, then uploads the complete `results/phase1` directory as an artifact.

## Output contract

- `capacity_results.csv`: one row per scenario/partition, including the sampled transition bracket and first limiting constraint.
- `sampled_trials.csv`: all 3,300 individual sampled verdicts and diagnostics.
- `summary.json`: scenario gaps, best/worst partitions, violation counts, and the phase-1 yes/no conclusion.
- `experiment.log`: complete human-readable experiment trace.

## Local preflight result

The deterministic local preflight produced the following placement gaps. These values remain provisional until the identical GitHub Actions run succeeds.

| Scenario | Uniform safe lambda | Best safe lambda | Best-worst gap | Material by preset rule? |
|---|---:|---:|---:|:---:|
| Homogeneous control | 0.0090 | 0.0090 | 0.0% | No |
| Mild alternating | 0.0090 | 0.0090 | 5.9% | No |
| Single slow stage | 0.0065 | 0.0085 | 41.7% | Yes |
| Graded | 0.0060 | 0.0085 | 41.7% | Yes |
| Shuffled severe | 0.0060 | 0.0085 | 41.7% | Yes |

The homogeneous control's zero gap is an important sanity check: when all stage speeds are equal, redistributing layers does not change the evaluator's weighted compute path. Under stronger heterogeneity, moving fewer layers to slow stages changes residual TPOT budget and therefore aggregate network commitment. The first sampled unsafe constraint is the network commitment in all 100 placement cases.

The result files are kept in the repository so it remains auditable without downloading a CI artifact.

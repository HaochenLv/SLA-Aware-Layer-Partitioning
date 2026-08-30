from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .model import Request, evaluate, validate_partition


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lambda_grid(cfg: dict[str, Any]) -> list[float]:
    sampling = cfg["sampling"]
    start = sampling["lambda_start"]
    stop = sampling["lambda_stop"]
    step = sampling["lambda_step"]
    count = round((stop - start) / step)
    return [round(start + i * step, 10) for i in range(count + 1)]


def build_workload(cfg: dict[str, Any], intensity: float) -> list[Request]:
    spec = cfg["workload"]
    rng = random.Random(spec["seed"])
    rate = spec["normalized_source_rate_rps"] * intensity
    requests = []
    for request_id in range(spec["request_count"]):
        # Finite, ordered, evenly spaced arrivals; only their relative times
        # change with intensity, matching Eq. (27)'s intent.
        arrival = request_id / rate
        requests.append(
            Request(
                request_id=request_id,
                arrival_s=arrival,
                input_tokens=rng.choice(spec["input_token_choices"]),
                output_tokens=rng.choice(spec["output_token_choices"]),
            )
        )
    return requests


def _compute_balanced_partition(speeds: list[float], cfg: dict[str, Any]) -> list[int]:
    layers = cfg["model"]["layers"]
    lo = cfg["partitions"]["min_layers_per_stage"]
    hi = cfg["partitions"]["max_layers_per_stage"]
    raw = [layers * speed / sum(speeds) for speed in speeds]
    result = [max(lo, min(hi, int(value))) for value in raw]
    while sum(result) < layers:
        candidates = [i for i in range(len(result)) if result[i] < hi]
        best = max(candidates, key=lambda i: raw[i] - result[i])
        result[best] += 1
    while sum(result) > layers:
        candidates = [i for i in range(len(result)) if result[i] > lo]
        best = max(candidates, key=lambda i: result[i] - raw[i])
        result[best] -= 1
    return result


def generate_partitions(speeds: list[float], cfg: dict[str, Any]) -> list[tuple[str, list[int]]]:
    spec = cfg["partitions"]
    stages = cfg["model"]["stages"]
    uniform = [cfg["model"]["layers"] // stages] * stages
    candidates: list[tuple[str, list[int]]] = [("uniform", uniform)]
    balanced = _compute_balanced_partition(speeds, cfg)
    if balanced != uniform:
        candidates.append(("compute_balanced", balanced))

    rng = random.Random(spec["seed"])
    seen = {tuple(partition) for _, partition in candidates}
    random_index = 0
    while len(candidates) < spec["count_per_scenario"]:
        partition = uniform.copy()
        for _ in range(rng.randint(8, 40)):
            source, target = rng.sample(range(stages), 2)
            if partition[source] > spec["min_layers_per_stage"] and partition[target] < spec["max_layers_per_stage"]:
                partition[source] -= 1
                partition[target] += 1
        key = tuple(partition)
        if key in seen:
            continue
        validate_partition(partition, cfg)
        seen.add(key)
        candidates.append((f"random_{random_index:02d}", partition))
        random_index += 1
    return candidates


def run_experiment(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    log_lines = [f"experiment_id={cfg['experiment_id']}"]
    grid = lambda_grid(cfg)
    scenario_summaries: dict[str, Any] = {}

    for scenario, speeds in cfg["scenarios"].items():
        partitions = generate_partitions(speeds, cfg)
        scenario_rows = []
        log_lines.append(f"scenario={scenario} speeds={speeds} partitions={len(partitions)}")
        for partition_id, partition in partitions:
            trials = []
            for intensity in grid:
                verdict = evaluate(build_workload(cfg, intensity), partition, speeds, cfg)
                trials.append((intensity, verdict))
                trial_row = {
                    "scenario": scenario,
                    "partition_id": partition_id,
                    "partition": "-".join(map(str, partition)),
                    "lambda": intensity,
                    "safe": verdict.safe,
                    "first_violation": verdict.first_violation,
                    "event_count": verdict.event_count,
                    "peak_prefill": verdict.peak_prefill,
                    "peak_decode": verdict.peak_decode,
                    "minimum_link_headroom_mb_s": round(verdict.minimum_link_headroom_mb_s, 6),
                    "peak_memory_gb": round(verdict.peak_memory_gb, 6),
                }
                trial_rows.append(trial_row)
                log_lines.append(
                    "  trial "
                    + " ".join(f"{key}={value}" for key, value in trial_row.items())
                )
            verdict_bits = [verdict.safe for _, verdict in trials]
            monotonic = all(not verdict_bits[i] or verdict_bits[i - 1] for i in range(1, len(verdict_bits)))
            safe_values = [intensity for intensity, verdict in trials if verdict.safe]
            safe_lambda = max(safe_values) if safe_values else None
            unsafe_above = [
                intensity
                for intensity, verdict in trials
                if not verdict.safe and (safe_lambda is None or intensity > safe_lambda)
            ]
            unsafe_lambda = min(unsafe_above) if unsafe_above else None
            first_unsafe = next((verdict for _, verdict in trials if not verdict.safe), None)
            row = {
                "scenario": scenario,
                "partition_id": partition_id,
                "partition": "-".join(map(str, partition)),
                "weighted_layers": round(sum(n / speed for n, speed in zip(partition, speeds)), 6),
                "safe_lambda": safe_lambda,
                "unsafe_lambda": unsafe_lambda,
                "right_censored": unsafe_lambda is None,
                "sampled_monotonic": monotonic,
                "first_violation": first_unsafe.first_violation if first_unsafe else None,
                "peak_decode_at_first_unsafe": first_unsafe.peak_decode if first_unsafe else None,
            }
            rows.append(row)
            scenario_rows.append(row)
            log_lines.append(
                f"  {partition_id:18s} partition={partition} safe={safe_lambda} "
                f"unsafe={unsafe_lambda} violation={row['first_violation']} monotonic={monotonic}"
            )

        observed = [row["safe_lambda"] for row in scenario_rows if row["safe_lambda"] is not None]
        minimum = min(observed)
        maximum = max(observed)
        absolute_gap = round(maximum - minimum, 10)
        relative_gap = round((maximum / minimum - 1.0) * 100.0, 3)
        significant = (
            relative_gap >= cfg["sampling"]["significant_relative_gap_pct"]
            and absolute_gap >= cfg["sampling"]["significant_absolute_gap"]
        )
        scenario_summaries[scenario] = {
            "partition_count": len(scenario_rows),
            "minimum_safe_lambda": minimum,
            "maximum_safe_lambda": maximum,
            "absolute_gap": absolute_gap,
            "relative_gap_pct": relative_gap,
            "unique_capacity_count": len(set(observed)),
            "significant": significant,
            "best_partitions": [row["partition"] for row in scenario_rows if row["safe_lambda"] == maximum],
            "worst_partitions": [row["partition"] for row in scenario_rows if row["safe_lambda"] == minimum],
            "violation_counts": dict(Counter(row["first_violation"] or "right_censored" for row in scenario_rows)),
        }
        log_lines.append(
            f"summary scenario={scenario} min={minimum} max={maximum} absolute_gap={absolute_gap} "
            f"relative_gap_pct={relative_gap} significant={significant}"
        )

    heterogeneous = [value for key, value in scenario_summaries.items() if key != "homogeneous_control"]
    conclusion = {
        "question": "Does continuous layer partition materially affect sampled SLA-safe capacity under compute heterogeneity?",
        "criterion": {
            "relative_gap_pct_at_least": cfg["sampling"]["significant_relative_gap_pct"],
            "absolute_gap_at_least": cfg["sampling"]["significant_absolute_gap"],
        },
        "heterogeneous_scenarios_tested": len(heterogeneous),
        "heterogeneous_scenarios_significant": sum(item["significant"] for item in heterogeneous),
        "answer": "yes" if any(item["significant"] for item in heterogeneous) else "no",
    }
    summary = {
        "experiment_id": cfg["experiment_id"],
        "semantics": "paper-aligned event-driven screening with synthetic normalized compute profiles",
        "scenario_summaries": scenario_summaries,
        "conclusion": conclusion,
    }
    log_lines.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, trial_rows, summary, "\n".join(log_lines) + "\n"


def write_outputs(
    rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    log: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "capacity_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "sampled_trials.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase1"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, trial_rows, summary, log = run_experiment(cfg)
    write_outputs(rows, trial_rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

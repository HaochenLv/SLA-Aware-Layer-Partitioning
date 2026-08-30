from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from .experiment import _compute_balanced_partition, build_workload, lambda_grid, load_config
from .model import evaluate, validate_partition


def uniform_partition(cfg: dict[str, Any]) -> list[int]:
    stages = cfg["model"]["stages"]
    layers = cfg["model"]["layers"]
    if layers % stages != 0:
        raise ValueError("phase-3 uniform baseline requires an even layer split")
    return [layers // stages] * stages


def adjacent_boundary_neighbors(partition: list[int], cfg: dict[str, Any]) -> list[list[int]]:
    """Move one layer across exactly one adjacent stage boundary."""
    lo = cfg["partitions"]["min_layers_per_stage"]
    hi = cfg["partitions"]["max_layers_per_stage"]
    neighbors: list[list[int]] = []
    for boundary in range(len(partition) - 1):
        if partition[boundary] > lo and partition[boundary + 1] < hi:
            candidate = partition.copy()
            candidate[boundary] -= 1
            candidate[boundary + 1] += 1
            validate_partition(candidate, cfg)
            neighbors.append(candidate)
        if partition[boundary] < hi and partition[boundary + 1] > lo:
            candidate = partition.copy()
            candidate[boundary] += 1
            candidate[boundary + 1] -= 1
            validate_partition(candidate, cfg)
            neighbors.append(candidate)
    # Preserve deterministic order while removing accidental duplicates.
    seen: set[tuple[int, ...]] = set()
    unique: list[list[int]] = []
    for candidate in neighbors:
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _evaluate_partition(
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
    cache: dict[tuple[int, ...], dict[str, Any]],
) -> dict[str, Any]:
    key = tuple(partition)
    if key in cache:
        return cache[key]

    verdicts = {}
    for intensity in lambda_grid(cfg):
        verdicts[intensity] = evaluate(build_workload(cfg, intensity), partition, speeds, cfg)
    safe_values = [lam for lam, verdict in verdicts.items() if verdict.safe]
    safe_lambda = max(safe_values) if safe_values else None
    unsafe_values = [
        lam
        for lam, verdict in verdicts.items()
        if not verdict.safe and (safe_lambda is None or lam > safe_lambda)
    ]
    unsafe_lambda = min(unsafe_values) if unsafe_values else None
    record = {
        "partition": partition.copy(),
        "partition_text": "-".join(map(str, partition)),
        "weighted_layers": sum(n / speed for n, speed in zip(partition, speeds)),
        "safe_lambda": safe_lambda,
        "unsafe_lambda": unsafe_lambda,
        "verdicts": verdicts,
    }
    cache[key] = record
    return record


def _score(record: dict[str, Any], reference_lambda: float) -> tuple[float, int, float]:
    """SLA-first score: capacity, survival at stress load, then network margin."""
    verdict = record["verdicts"][reference_lambda]
    capacity = record["safe_lambda"] if record["safe_lambda"] is not None else float("-inf")
    return (
        capacity,
        int(verdict.safe),
        verdict.minimum_link_headroom_mb_s,
    )


def _public_record(record: dict[str, Any], reference_lambda: float) -> dict[str, Any]:
    verdict = record["verdicts"][reference_lambda]
    return {
        "partition": record["partition_text"],
        "weighted_layers": round(record["weighted_layers"], 6),
        "safe_lambda": record["safe_lambda"],
        "unsafe_lambda": record["unsafe_lambda"],
        "safe_at_reference": verdict.safe,
        "first_violation_at_reference": verdict.first_violation,
        "minimum_link_headroom_mb_s_at_reference": round(verdict.minimum_link_headroom_mb_s, 6),
    }


def boundary_local_search(
    speeds: list[float],
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], int, float]:
    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    current = _evaluate_partition(uniform_partition(cfg), speeds, cfg, cache)
    reference_lambda = current["unsafe_lambda"]
    if reference_lambda is None:
        reference_lambda = lambda_grid(cfg)[-1]

    history: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "partition": current["partition_text"],
            "safe_lambda": current["safe_lambda"],
            "safe_at_reference": current["verdicts"][reference_lambda].safe,
            "minimum_link_headroom_mb_s_at_reference": round(
                current["verdicts"][reference_lambda].minimum_link_headroom_mb_s, 6
            ),
        }
    ]
    visited = {tuple(current["partition"])}
    max_iterations = cfg.get("phase3", {}).get("max_iterations", 40)

    for iteration in range(1, max_iterations + 1):
        candidates = []
        for neighbor in adjacent_boundary_neighbors(current["partition"], cfg):
            key = tuple(neighbor)
            if key in visited:
                continue
            visited.add(key)
            candidates.append(_evaluate_partition(neighbor, speeds, cfg, cache))
        if not candidates:
            break

        best = max(candidates, key=lambda item: (_score(item, reference_lambda), tuple(item["partition"])))
        if _score(best, reference_lambda) <= _score(current, reference_lambda):
            break
        current = best
        history.append(
            {
                "iteration": iteration,
                "partition": current["partition_text"],
                "safe_lambda": current["safe_lambda"],
                "safe_at_reference": current["verdicts"][reference_lambda].safe,
                "minimum_link_headroom_mb_s_at_reference": round(
                    current["verdicts"][reference_lambda].minimum_link_headroom_mb_s, 6
                ),
            }
        )

    return current, history, len(cache), reference_lambda


def _random_partition(rng: random.Random, cfg: dict[str, Any]) -> list[int]:
    partition = uniform_partition(cfg)
    stages = len(partition)
    lo = cfg["partitions"]["min_layers_per_stage"]
    hi = cfg["partitions"]["max_layers_per_stage"]
    for _ in range(rng.randint(8, 50)):
        source, target = rng.sample(range(stages), 2)
        if partition[source] > lo and partition[target] < hi:
            partition[source] -= 1
            partition[target] += 1
    validate_partition(partition, cfg)
    return partition


def equal_budget_random_search(
    speeds: list[float],
    cfg: dict[str, Any],
    budget: int,
    reference_lambda: float,
    scenario_index: int,
) -> dict[str, Any]:
    cache: dict[tuple[int, ...], dict[str, Any]] = {}
    uniform = uniform_partition(cfg)
    best = _evaluate_partition(uniform, speeds, cfg, cache)
    rng = random.Random(cfg["partitions"]["seed"] + 1000 + scenario_index)
    attempts = 0
    while len(cache) < budget and attempts < budget * 100:
        attempts += 1
        candidate = _random_partition(rng, cfg)
        record = _evaluate_partition(candidate, speeds, cfg, cache)
        if _score(record, reference_lambda) > _score(best, reference_lambda):
            best = record
    return best


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    comparison_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    scenario_summaries: dict[str, Any] = {}
    log = ["experiment_id=phase3-sla-first-boundary-search-v1"]

    phase3_cfg = cfg.get("phase3", {})
    target_scenarios = set(
        phase3_cfg.get("target_scenarios", ["single_slow_stage", "graded", "shuffled_severe"])
    )

    for scenario_index, (scenario, speeds) in enumerate(cfg["scenarios"].items()):
        boundary, history, budget, reference_lambda = boundary_local_search(speeds, cfg)
        cache: dict[tuple[int, ...], dict[str, Any]] = {}
        uniform = _evaluate_partition(uniform_partition(cfg), speeds, cfg, cache)
        balanced = _evaluate_partition(_compute_balanced_partition(speeds, cfg), speeds, cfg, cache)
        random_best = equal_budget_random_search(
            speeds, cfg, budget, reference_lambda, scenario_index
        )

        methods = {
            "uniform": uniform,
            "compute_balanced": balanced,
            "equal_budget_random": random_best,
            "sla_boundary_search": boundary,
        }
        for method, record in methods.items():
            public = _public_record(record, reference_lambda)
            comparison_rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "reference_lambda": reference_lambda,
                    "partition_evaluation_budget": budget if method in {"equal_budget_random", "sla_boundary_search"} else 1,
                    **public,
                }
            )

        for item in history:
            history_rows.append({"scenario": scenario, "reference_lambda": reference_lambda, **item})

        uniform_capacity = uniform["safe_lambda"]
        boundary_capacity = boundary["safe_lambda"]
        balanced_capacity = balanced["safe_lambda"]
        random_capacity = random_best["safe_lambda"]
        summary = {
            "reference_lambda": reference_lambda,
            "search_partition_evaluations": budget,
            "search_iterations": len(history) - 1,
            "uniform": _public_record(uniform, reference_lambda),
            "compute_balanced": _public_record(balanced, reference_lambda),
            "equal_budget_random": _public_record(random_best, reference_lambda),
            "sla_boundary_search": _public_record(boundary, reference_lambda),
            "checks": {
                "boundary_not_worse_than_uniform": boundary_capacity >= uniform_capacity,
                "boundary_matches_or_beats_compute_balanced": boundary_capacity >= balanced_capacity,
                "boundary_matches_or_beats_equal_budget_random": boundary_capacity >= random_capacity,
                "boundary_improves_uniform": boundary_capacity > uniform_capacity,
            },
        }
        scenario_summaries[scenario] = summary
        log.append(json.dumps({"scenario": scenario, **summary}, sort_keys=True))

    target = [scenario_summaries[name] for name in target_scenarios]
    homogeneous = scenario_summaries["homogeneous_control"]
    conclusion = {
        "question": "Can a lightweight adjacent-boundary search recover higher sampled SLA-safe capacity than uniform partitioning without changing evaluator semantics?",
        "target_scenarios": sorted(target_scenarios),
        "target_scenarios_tested": len(target),
        "target_scenarios_improving_uniform": sum(x["checks"]["boundary_improves_uniform"] for x in target),
        "target_scenarios_matching_or_beating_compute_balanced": sum(
            x["checks"]["boundary_matches_or_beats_compute_balanced"] for x in target
        ),
        "target_scenarios_matching_or_beating_equal_budget_random": sum(
            x["checks"]["boundary_matches_or_beats_equal_budget_random"] for x in target
        ),
        "homogeneous_control_unchanged": not homogeneous["checks"]["boundary_improves_uniform"],
    }
    conclusion["answer"] = (
        "yes"
        if conclusion["target_scenarios_improving_uniform"] >= 2
        and conclusion["homogeneous_control_unchanged"]
        else "not_yet"
    )
    summary = {
        "experiment_id": "phase3-sla-first-boundary-search-v1",
        "semantics": "same synthetic event-driven evaluator as phases 1-2; search changes only contiguous layer boundaries; no routing, replication, migration, or scheduling optimization",
        "search_rule": "capacity first; at the uniform first-unsafe workload, prefer survival and then larger minimum network headroom; move one layer across one adjacent boundary per accepted step",
        "scenario_summaries": scenario_summaries,
        "conclusion": conclusion,
    }
    log.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return comparison_rows, history_rows, summary, "\n".join(log) + "\n"


def write_outputs(
    comparison_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    log: str,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "method_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    with (output / "search_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase3.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase3"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, history, summary, log = run(cfg)
    write_outputs(rows, history, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

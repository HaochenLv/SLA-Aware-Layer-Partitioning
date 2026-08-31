from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from .experiment import load_config
from .helix_profile import derive_phase_speed_factors
from .phase10_method_compare import static_baseline_shifts
from .phase11_trace_validation import adaptive_capacity, build_helix_workload, verify_trace_inputs
from .phase14_revision_experiments import _configured, _mean_std_ci95, _oracle


def _value(record: dict[str, Any]) -> float:
    value = record.get("safe_intensity")
    return -math.inf if value is None else float(value)


def _best(probe_order: list[int], capacity_for: Callable[[int], dict[str, Any]]) -> int:
    return int(
        max(
            probe_order,
            key=lambda shift: (_value(capacity_for(shift)), -abs(shift), -shift),
        )
    )


def single_direction_adaptive_search(
    capacity_for: Callable[[int], dict[str, Any]], radius: int
) -> tuple[int, list[int]]:
    """Generalize Phase 13 by continuing only along its chosen nondecreasing direction.

    At radius=2 this is exactly the Phase-13 four-probe rule. For larger radii,
    expansion continues while sampled SLA-safe capacity does not decrease and
    stops after the first decrease or the candidate boundary.
    """
    if radius < 2:
        raise ValueError("radius must be at least 2")
    probe_order: list[int] = []

    def probe(shift: int) -> dict[str, Any]:
        if shift not in probe_order:
            probe_order.append(shift)
        return capacity_for(shift)

    initial = {shift: probe(shift) for shift in (-1, 0, 1)}
    uniform_value = _value(initial[0])
    best_neighbor = max(
        (-1, 1),
        key=lambda shift: (_value(initial[shift]), -abs(shift), -shift),
    )
    previous_value = _value(initial[best_neighbor])
    if previous_value >= uniform_value:
        direction = 1 if best_neighbor > 0 else -1
        for distance in range(2, radius + 1):
            shift = direction * distance
            current_value = _value(probe(shift))
            if current_value < previous_value:
                break
            previous_value = current_value

    return _best(probe_order, capacity_for), probe_order


def bidirectional_adaptive_search(
    capacity_for: Callable[[int], dict[str, Any]], radius: int
) -> tuple[int, list[int]]:
    """Expand every direction that is nondecreasing from uniform, stopping on decline."""
    if radius < 2:
        raise ValueError("radius must be at least 2")
    probe_order: list[int] = []

    def probe(shift: int) -> dict[str, Any]:
        if shift not in probe_order:
            probe_order.append(shift)
        return capacity_for(shift)

    initial = {shift: probe(shift) for shift in (-1, 0, 1)}
    uniform_value = _value(initial[0])
    for direction in (-1, 1):
        previous_value = _value(initial[direction])
        if previous_value < uniform_value:
            continue
        for distance in range(2, radius + 1):
            shift = direction * distance
            current_value = _value(probe(shift))
            if current_value < previous_value:
                break
            previous_value = current_value

    return _best(probe_order, capacity_for), probe_order


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {
            (str(row["condition"]), int(row["candidate_radius"]), str(row["method"]), str(row["regime"]))
            for row in rows
        }
    )
    for condition, radius, method, regime in keys:
        subset = [
            row
            for row in rows
            if row["condition"] == condition
            and int(row["candidate_radius"]) == radius
            and row["method"] == method
            and row["regime"] == regime
        ]
        output.append(
            {
                "condition": condition,
                "candidate_radius": radius,
                "candidate_count": 2 * radius + 1,
                "method": method,
                "regime": regime,
                "trials": len(subset),
                "exact_oracle_shift_matches": sum(bool(row["matches_oracle_shift"]) for row in subset),
                "near_oracle_trials": sum(bool(row["near_oracle"]) for row in subset),
                "not_worse_than_uniform": sum(bool(row["not_worse_than_uniform"]) for row in subset),
                "not_worse_than_joint_compute": sum(
                    bool(row["not_worse_than_joint_compute"]) for row in subset
                ),
                "oracle_ratio": _mean_std_ci95(
                    [float(row["oracle_ratio"]) for row in subset if row["oracle_ratio"] is not None]
                ),
                "oracle_gap_pct": _mean_std_ci95(
                    [float(row["oracle_gap_pct"]) for row in subset if row["oracle_gap_pct"] is not None]
                ),
                "gain_over_uniform_pct": _mean_std_ci95(
                    [
                        float(row["gain_over_uniform_pct"])
                        for row in subset
                        if row["gain_over_uniform_pct"] is not None
                    ]
                ),
                "candidate_probe_count": _mean_std_ci95(
                    [float(row["candidate_probe_count"]) for row in subset]
                ),
                "method_evaluator_calls": _mean_std_ci95(
                    [float(row["method_evaluator_calls"]) for row in subset]
                ),
                "exhaustive_evaluator_calls": _mean_std_ci95(
                    [float(row["exhaustive_evaluator_calls"]) for row in subset]
                ),
                "evaluator_call_reduction_pct": _mean_std_ci95(
                    [float(row["evaluator_call_reduction_pct"]) for row in subset]
                ),
                "method_runtime_s": _mean_std_ci95(
                    [float(row["method_runtime_s"]) for row in subset]
                ),
                "exhaustive_runtime_s": _mean_std_ci95(
                    [float(row["exhaustive_runtime_s"]) for row in subset]
                ),
            }
        )
    return output


def _run_condition(
    base_cfg: dict[str, Any],
    *,
    condition: str,
    seeds: list[int],
    bandwidth_multiplier: float,
    radii: list[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    max_radius = max(radii)
    cfg = _configured(
        base_cfg,
        seeds=seeds,
        radius=max_radius,
        bandwidth_multiplier=bandwidth_multiplier,
    )
    factors = derive_phase_speed_factors(cfg)
    machines = cfg["phase6"]["stage_machines"]
    prefill_speeds = [factors[machine]["prefill_speed"] for machine in machines]
    decode_speeds = [factors[machine]["decode_speed"] for machine in machines]
    static_shifts = static_baseline_shifts(prefill_speeds, decode_speeds, cfg)

    rows: list[dict[str, Any]] = []
    log_lines = [
        f"condition={condition}",
        f"seeds={seeds}",
        f"bandwidth_multiplier={bandwidth_multiplier}",
        f"radii={radii}",
    ]

    for seed in seeds:
        base_workload = build_helix_workload(cfg, int(seed))
        for regime_name, regime in cfg["phase11"]["regimes"].items():
            regime_cfg = copy.deepcopy(cfg)
            regime_cfg["sla"]["ttft_s"] = regime["ttft_s"]
            regime_cfg["sla"]["tpot_s"] = regime["tpot_s"]
            capacities: dict[int, dict[str, Any]] = {}
            elapsed_s: dict[int, float] = {}

            def capacity_for(shift: int) -> dict[str, Any]:
                shift = int(shift)
                if shift not in capacities:
                    start = time.perf_counter()
                    capacities[shift] = adaptive_capacity(
                        base_workload,
                        shift,
                        prefill_speeds,
                        decode_speeds,
                        regime_cfg,
                    )
                    elapsed_s[shift] = time.perf_counter() - start
                return capacities[shift]

            searches: dict[tuple[int, str], tuple[int, list[int]]] = {}
            for radius in radii:
                searches[(radius, "single_direction_adaptive")] = single_direction_adaptive_search(
                    capacity_for, radius
                )
                searches[(radius, "bidirectional_adaptive")] = bidirectional_adaptive_search(
                    capacity_for, radius
                )

            # Exhaustive fill is evaluation-only and occurs after every search decision.
            for shift in range(-max_radius, max_radius + 1):
                capacity_for(shift)

            uniform_capacity = capacities[0]["safe_intensity"]
            joint_shift = int(static_shifts["phase_agnostic_joint_compute"])
            joint_capacity = capacities[joint_shift]["safe_intensity"]

            for radius in radii:
                candidate_shifts = list(range(-radius, radius + 1))
                oracle_capacity, oracle_shifts = _oracle(capacities, candidate_shifts)
                exhaustive_calls = sum(
                    int(capacities[shift]["evaluations"]) for shift in candidate_shifts
                )
                exhaustive_runtime_s = sum(elapsed_s[shift] for shift in candidate_shifts)

                for method in ("single_direction_adaptive", "bidirectional_adaptive"):
                    proposed_shift, probe_order = searches[(radius, method)]
                    proposed_capacity = capacities[proposed_shift]["safe_intensity"]
                    method_calls = sum(int(capacities[shift]["evaluations"]) for shift in probe_order)
                    method_runtime_s = sum(elapsed_s[shift] for shift in probe_order)

                    ratio = None
                    gap_pct = None
                    gain_uniform = None
                    if proposed_capacity is not None and oracle_capacity not in (None, 0):
                        ratio = float(proposed_capacity) / float(oracle_capacity)
                        gap_pct = (1.0 - ratio) * 100.0
                    if proposed_capacity is not None and uniform_capacity not in (None, 0):
                        gain_uniform = (
                            float(proposed_capacity) / float(uniform_capacity) - 1.0
                        ) * 100.0

                    row = {
                        "condition": condition,
                        "seed": int(seed),
                        "regime": regime_name,
                        "bandwidth_multiplier": float(bandwidth_multiplier),
                        "candidate_radius": radius,
                        "candidate_count": len(candidate_shifts),
                        "method": method,
                        "proposed_shift": int(proposed_shift),
                        "probe_order": ",".join(map(str, probe_order)),
                        "candidate_probe_count": len(probe_order),
                        "proposed_safe_intensity": proposed_capacity,
                        "uniform_safe_intensity": uniform_capacity,
                        "joint_compute_shift": joint_shift,
                        "joint_compute_safe_intensity": joint_capacity,
                        "oracle_safe_intensity": oracle_capacity,
                        "oracle_shifts": ",".join(map(str, sorted(oracle_shifts))),
                        "oracle_ratio": round(ratio, 6) if ratio is not None else None,
                        "oracle_gap_pct": round(gap_pct, 6) if gap_pct is not None else None,
                        "gain_over_uniform_pct": round(gain_uniform, 6) if gain_uniform is not None else None,
                        "matches_oracle_shift": proposed_shift in oracle_shifts,
                        "near_oracle": (
                            ratio is not None
                            and ratio >= float(base_cfg["phase14"]["near_oracle_ratio"])
                        ),
                        "not_worse_than_uniform": (
                            proposed_capacity is not None
                            and uniform_capacity is not None
                            and proposed_capacity >= uniform_capacity
                        ),
                        "not_worse_than_joint_compute": (
                            proposed_capacity is not None
                            and joint_capacity is not None
                            and proposed_capacity >= joint_capacity
                        ),
                        "method_evaluator_calls": method_calls,
                        "exhaustive_evaluator_calls": exhaustive_calls,
                        "evaluator_call_reduction_pct": round(
                            (1.0 - method_calls / exhaustive_calls) * 100.0, 6
                        ) if exhaustive_calls else None,
                        "method_runtime_s": round(method_runtime_s, 6),
                        "exhaustive_runtime_s": round(exhaustive_runtime_s, 6),
                        "all_sampled_monotonic": all(
                            bool(capacities[shift]["sampled_monotonic"])
                            for shift in candidate_shifts
                        ),
                        "no_right_censoring": not any(
                            bool(capacities[shift]["right_censored"])
                            for shift in candidate_shifts
                        ),
                    }
                    rows.append(row)
                    log_lines.append(json.dumps(row, sort_keys=True))

    return rows, log_lines


def run(cfg: dict[str, Any], suite: str) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    checksums = verify_trace_inputs(cfg)
    spec = cfg["phase14"]
    radii = [int(value) for value in spec["search_space_radii"]]
    rows: list[dict[str, Any]] = []
    logs = [f"experiment_id=phase15-adaptive-directional-search-v1", f"suite={suite}"]

    if suite == "expanded":
        condition_rows, condition_logs = _run_condition(
            cfg,
            condition="expanded_20_seed",
            seeds=[int(seed) for seed in spec["expanded_workload_seeds"]],
            bandwidth_multiplier=1.0,
            radii=radii,
        )
        rows.extend(condition_rows)
        logs.extend(condition_logs)
    elif suite == "bandwidth":
        for multiplier in spec["bandwidth_multipliers"]:
            condition_rows, condition_logs = _run_condition(
                cfg,
                condition=f"bandwidth_{float(multiplier):g}x",
                seeds=[int(seed) for seed in spec["bandwidth_workload_seeds"]],
                bandwidth_multiplier=float(multiplier),
                radii=[max(radii)],
            )
            rows.extend(condition_rows)
            logs.extend(condition_logs)
    else:
        raise ValueError(f"unknown suite: {suite}")

    summary = {
        "experiment_id": "phase15-adaptive-directional-search-v1",
        "suite": suite,
        "provenance": {
            "source_repo_commit": cfg["phase11"]["source_repo_commit"],
            "upstream_helix_commit": cfg["phase11"]["upstream_helix_commit"],
            "artifact_sha256": checksums,
        },
        "unchanged": [
            "event-driven SLA-safe capacity evaluator",
            "HELIX-derived phase-specific speed factors",
            "Azure-derived finite workload construction",
            "TTFT/TPOT SLA regimes",
            "contiguous one-dimensional alternating-stage shift family",
        ],
        "search_variants": {
            "single_direction_adaptive": (
                "Exact Phase-13 behavior for radius 2; for larger radii, continue along the chosen "
                "nondecreasing direction until the first capacity decrease or boundary."
            ),
            "bidirectional_adaptive": (
                "Starting from {-1,0,+1}, continue each direction whose adjacent candidate is not "
                "worse than uniform, stopping that direction at the first capacity decrease or boundary."
            ),
        },
        "interpretation": (
            "The exhaustive candidate family is filled only after the search decisions. Search cost is "
            "reported as candidate probes, underlying evaluator calls, and measured candidate-evaluation "
            "wall time. This experiment tests whether the submitted fixed-radius rule has a natural "
            "scalable extension; it does not change evaluator equations or claim new hardware validation."
        ),
        "aggregate": _aggregate(rows),
        "all_sampled_monotonic": all(bool(row["all_sampled_monotonic"]) for row in rows),
        "no_right_censoring": all(bool(row["no_right_censoring"]) for row in rows),
    }
    logs.append("summary=" + json.dumps(summary, sort_keys=True))
    return rows, summary, "\n".join(logs) + "\n"


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], log: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trial_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase14.json"))
    parser.add_argument("--suite", choices=("expanded", "bandwidth"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg, args.suite)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

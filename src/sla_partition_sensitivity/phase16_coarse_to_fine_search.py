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


def coarse_to_fine_search(
    capacity_for: Callable[[int], dict[str, Any]],
    radius: int,
    top_k: int,
) -> tuple[int, list[int]]:
    """Probe a stride-2 coarse grid, then refine odd neighbors of the top-k coarse points."""
    if radius < 2 or radius % 2:
        raise ValueError("coarse-to-fine experiment expects an even radius >= 2")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    probe_order: list[int] = []

    def probe(shift: int) -> dict[str, Any]:
        if shift not in probe_order:
            probe_order.append(shift)
        return capacity_for(shift)

    coarse = list(range(-radius, radius + 1, 2))
    for shift in coarse:
        probe(shift)

    ranked = sorted(
        coarse,
        key=lambda shift: (_value(capacity_for(shift)), -abs(shift), -shift),
        reverse=True,
    )
    selected = ranked[: min(top_k, len(ranked))]
    for center in selected:
        for neighbor in (center - 1, center + 1):
            if -radius <= neighbor <= radius and neighbor not in coarse:
                probe(neighbor)

    best = max(
        probe_order,
        key=lambda shift: (_value(capacity_for(shift)), -abs(shift), -shift),
    )
    return int(best), probe_order


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted(
        {
            (str(row["condition"]), str(row["regime"]), str(row["method"]))
            for row in rows
        }
    )
    for condition, regime, method in keys:
        subset = [
            row
            for row in rows
            if row["condition"] == condition
            and row["regime"] == regime
            and row["method"] == method
        ]
        output.append(
            {
                "condition": condition,
                "regime": regime,
                "method": method,
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
    radius: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    cfg = _configured(
        base_cfg,
        seeds=seeds,
        radius=radius,
        bandwidth_multiplier=bandwidth_multiplier,
    )
    factors = derive_phase_speed_factors(cfg)
    machines = cfg["phase6"]["stage_machines"]
    prefill_speeds = [factors[machine]["prefill_speed"] for machine in machines]
    decode_speeds = [factors[machine]["decode_speed"] for machine in machines]
    static_shifts = static_baseline_shifts(prefill_speeds, decode_speeds, cfg)
    candidate_shifts = list(range(-radius, radius + 1))

    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    logs = [
        f"condition={condition}",
        f"seeds={seeds}",
        f"bandwidth_multiplier={bandwidth_multiplier}",
        f"radius={radius}",
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

            searches: dict[str, tuple[int, list[int]]] = {}
            for top_k in (1, 2, 3):
                searches[f"coarse_top{top_k}"] = coarse_to_fine_search(
                    capacity_for, radius, top_k
                )

            # Evaluation-only exhaustive fill after search decisions.
            for shift in candidate_shifts:
                capacity_for(shift)

            oracle_capacity, oracle_shifts = _oracle(capacities, candidate_shifts)
            uniform_capacity = capacities[0]["safe_intensity"]
            joint_shift = int(static_shifts["phase_agnostic_joint_compute"])
            joint_capacity = capacities[joint_shift]["safe_intensity"]
            exhaustive_calls = sum(
                int(capacities[shift]["evaluations"]) for shift in candidate_shifts
            )
            exhaustive_runtime_s = sum(elapsed_s[shift] for shift in candidate_shifts)

            for shift in candidate_shifts:
                record = capacities[shift]
                profiles.append(
                    {
                        "condition": condition,
                        "seed": int(seed),
                        "regime": regime_name,
                        "bandwidth_multiplier": float(bandwidth_multiplier),
                        "shift": int(shift),
                        "safe_intensity": record["safe_intensity"],
                        "unsafe_intensity": record["unsafe_intensity"],
                        "evaluations": int(record["evaluations"]),
                        "elapsed_s": round(elapsed_s[shift], 6),
                        "sampled_monotonic": bool(record["sampled_monotonic"]),
                        "right_censored": bool(record["right_censored"]),
                        "is_oracle_shift": shift in oracle_shifts,
                    }
                )

            for method, (proposed_shift, probe_order) in searches.items():
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
                }
                rows.append(row)
                logs.append(json.dumps(row, sort_keys=True))

    return rows, profiles, logs


def run(cfg: dict[str, Any], suite: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    checksums = verify_trace_inputs(cfg)
    spec = cfg["phase14"]
    radius = int(spec["expanded_candidate_radius"])
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    logs = ["experiment_id=phase16-coarse-to-fine-search-v1", f"suite={suite}"]

    if suite == "expanded":
        r, p, l = _run_condition(
            cfg,
            condition="expanded_20_seed",
            seeds=[int(seed) for seed in spec["expanded_workload_seeds"]],
            bandwidth_multiplier=1.0,
            radius=radius,
        )
        rows.extend(r)
        profiles.extend(p)
        logs.extend(l)
    elif suite == "bandwidth":
        for multiplier in spec["bandwidth_multipliers"]:
            r, p, l = _run_condition(
                cfg,
                condition=f"bandwidth_{float(multiplier):g}x",
                seeds=[int(seed) for seed in spec["bandwidth_workload_seeds"]],
                bandwidth_multiplier=float(multiplier),
                radius=radius,
            )
            rows.extend(r)
            profiles.extend(p)
            logs.extend(l)
    else:
        raise ValueError(f"unknown suite: {suite}")

    summary = {
        "experiment_id": "phase16-coarse-to-fine-search-v1",
        "suite": suite,
        "provenance": {
            "source_repo_commit": cfg["phase11"]["source_repo_commit"],
            "upstream_helix_commit": cfg["phase11"]["upstream_helix_commit"],
            "artifact_sha256": checksums,
        },
        "method": (
            "Evaluate the stride-2 coarse grid {-4,-2,0,2,4}, rank coarse candidates by sampled "
            "SLA-safe capacity, then evaluate odd neighbors of the best one, two, or three coarse "
            "candidates. The exhaustive nine-shift family is filled only after method decisions."
        ),
        "reason": (
            "Phase-15 showed that Decode capacity is directionally well behaved but Prefill can contain "
            "a valley followed by a better farther shift, invalidating a global unimodality assumption."
        ),
        "aggregate": _aggregate(rows),
        "capacity_profile_rows": len(profiles),
    }
    logs.append("summary=" + json.dumps(summary, sort_keys=True))
    return rows, profiles, summary, "\n".join(logs) + "\n"


def write_outputs(
    rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    summary: dict[str, Any],
    log: str,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trial_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "capacity_profiles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiles[0]))
        writer.writeheader()
        writer.writerows(profiles)
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
    rows, profiles, summary, log = run(cfg, args.suite)
    write_outputs(rows, profiles, summary, log, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

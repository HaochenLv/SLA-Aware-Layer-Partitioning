from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from .experiment import load_config
from .helix_profile import derive_phase_speed_factors
from .phase10_method_compare import static_baseline_shifts
from .phase11_trace_validation import adaptive_capacity, build_helix_workload, verify_trace_inputs
from .phase13_evaluator_guided_search import evaluator_guided_local_search
from .phase6_profiled import shifted_partition


def _capacity_value(record: dict[str, Any]) -> float:
    value = record.get("safe_intensity")
    return -math.inf if value is None else float(value)


def _mean_std_ci95(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.mean(values)
    if len(values) == 1:
        return {
            "n": 1,
            "mean": round(mean, 6),
            "std": 0.0,
            "ci95_low": round(mean, 6),
            "ci95_high": round(mean, 6),
        }
    std = statistics.stdev(values)
    margin = 1.96 * std / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci95_low": round(mean - margin, 6),
        "ci95_high": round(mean + margin, 6),
    }


def _configured(
    base_cfg: dict[str, Any],
    *,
    seeds: list[int],
    radius: int,
    bandwidth_multiplier: float,
) -> dict[str, Any]:
    """Return a controlled experiment configuration without changing evaluator semantics."""
    if radius < 2 or radius > 9:
        raise ValueError("candidate radius must be in [2, 9] for the 80-layer, 8-stage family")
    if bandwidth_multiplier <= 0:
        raise ValueError("bandwidth multiplier must be positive")

    cfg = copy.deepcopy(base_cfg)
    shifts = list(range(-radius, radius + 1))
    cfg["partitions"]["min_layers_per_stage"] = 10 - radius
    cfg["partitions"]["max_layers_per_stage"] = 10 + radius
    cfg["phase10"]["candidate_shifts"] = shifts
    cfg["phase11"]["candidate_shifts"] = shifts
    cfg["phase11"]["workload_seeds"] = [int(seed) for seed in seeds]
    cfg["network"]["link_capacity_mb_s"] = (
        float(base_cfg["network"]["link_capacity_mb_s"]) * float(bandwidth_multiplier)
    )
    return cfg


def _oracle(capacities: dict[int, dict[str, Any]], shifts: list[int]) -> tuple[float | None, list[int]]:
    observed = [
        capacities[shift]["safe_intensity"]
        for shift in shifts
        if capacities[shift]["safe_intensity"] is not None
    ]
    if not observed:
        return None, []
    best = max(observed)
    return best, [shift for shift in shifts if capacities[shift]["safe_intensity"] == best]


def _aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime: dict[str, Any] = {}
    for regime in sorted({str(item["regime"]) for item in trials}):
        subset = [item for item in trials if item["regime"] == regime]
        ratios = [float(item["oracle_ratio"]) for item in subset if item["oracle_ratio"] is not None]
        gaps = [float(item["oracle_gap_pct"]) for item in subset if item["oracle_gap_pct"] is not None]
        gains_uniform = [
            float(item["gain_over_uniform_pct"])
            for item in subset
            if item["gain_over_uniform_pct"] is not None
        ]
        gains_joint = [
            float(item["gain_over_joint_compute_pct"])
            for item in subset
            if item["gain_over_joint_compute_pct"] is not None
        ]
        by_regime[regime] = {
            "trials": len(subset),
            "exact_oracle_shift_matches": sum(bool(item["matches_oracle_shift"]) for item in subset),
            "near_oracle_trials": sum(bool(item["near_oracle"]) for item in subset),
            "not_worse_than_uniform": sum(bool(item["not_worse_than_uniform"]) for item in subset),
            "not_worse_than_joint_compute": sum(
                bool(item["not_worse_than_joint_compute"]) for item in subset
            ),
            "oracle_ratio": _mean_std_ci95(ratios),
            "oracle_gap_pct": _mean_std_ci95(gaps),
            "gain_over_uniform_pct": _mean_std_ci95(gains_uniform),
            "gain_over_joint_compute_pct": _mean_std_ci95(gains_joint),
            "candidate_probe_count": _mean_std_ci95(
                [float(item["candidate_probe_count"]) for item in subset]
            ),
            "method_evaluator_calls": _mean_std_ci95(
                [float(item["method_evaluator_calls"]) for item in subset]
            ),
        }
    return by_regime


def _run_condition(
    base_cfg: dict[str, Any],
    *,
    condition: str,
    seeds: list[int],
    radius: int,
    bandwidth_multiplier: float,
    search_space_radii: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[str]]:
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
    candidate_shifts = [int(value) for value in cfg["phase11"]["candidate_shifts"]]

    trials: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    log_lines = [
        f"condition={condition}",
        f"candidate_shifts={candidate_shifts}",
        f"bandwidth_multiplier={bandwidth_multiplier}",
        "static_baseline_shifts=" + json.dumps(static_shifts, sort_keys=True),
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

            # IMPORTANT: this is the original Phase-13 four-probe rule. Enlarging
            # the oracle family does not grant the method access to +/-3 or +/-4.
            proposed_shift, probe_order = evaluator_guided_local_search(capacity_for)
            method_probe_order = list(probe_order)
            method_evaluator_calls = sum(
                int(capacities[shift]["evaluations"]) for shift in method_probe_order
            )
            method_runtime_s = sum(elapsed_s[shift] for shift in method_probe_order)

            # Fill the enlarged local family only after the method has committed.
            for shift in candidate_shifts:
                capacity_for(shift)

            oracle_capacity, oracle_shifts = _oracle(capacities, candidate_shifts)
            proposed_capacity = capacities[proposed_shift]["safe_intensity"]
            uniform_capacity = capacities[0]["safe_intensity"]
            joint_shift = int(static_shifts["phase_agnostic_joint_compute"])
            joint_capacity = capacities[joint_shift]["safe_intensity"]

            oracle_ratio = None
            oracle_gap_pct = None
            gain_uniform = None
            gain_joint = None
            if proposed_capacity is not None and oracle_capacity not in (None, 0):
                oracle_ratio = float(proposed_capacity) / float(oracle_capacity)
                oracle_gap_pct = (1.0 - oracle_ratio) * 100.0
            if proposed_capacity is not None and uniform_capacity not in (None, 0):
                gain_uniform = (float(proposed_capacity) / float(uniform_capacity) - 1.0) * 100.0
            if proposed_capacity is not None and joint_capacity not in (None, 0):
                gain_joint = (float(proposed_capacity) / float(joint_capacity) - 1.0) * 100.0

            trial = {
                "condition": condition,
                "seed": int(seed),
                "regime": regime_name,
                "request_count": len(base_workload),
                "bandwidth_multiplier": float(bandwidth_multiplier),
                "link_capacity_mb_s": float(regime_cfg["network"]["link_capacity_mb_s"]),
                "candidate_count": len(candidate_shifts),
                "candidate_radius": radius,
                "proposed_shift": int(proposed_shift),
                "proposed_partition": "-".join(
                    map(str, shifted_partition(int(proposed_shift), regime_cfg))
                ),
                "proposed_safe_intensity": proposed_capacity,
                "uniform_safe_intensity": uniform_capacity,
                "joint_compute_shift": joint_shift,
                "joint_compute_safe_intensity": joint_capacity,
                "oracle_safe_intensity": oracle_capacity,
                "oracle_shifts": ",".join(map(str, sorted(oracle_shifts))),
                "oracle_ratio": round(oracle_ratio, 6) if oracle_ratio is not None else None,
                "oracle_gap_pct": round(oracle_gap_pct, 6) if oracle_gap_pct is not None else None,
                "gain_over_uniform_pct": round(gain_uniform, 6) if gain_uniform is not None else None,
                "gain_over_joint_compute_pct": round(gain_joint, 6) if gain_joint is not None else None,
                "matches_oracle_shift": proposed_shift in oracle_shifts,
                "near_oracle": (
                    oracle_ratio is not None
                    and oracle_ratio >= float(base_cfg["phase14"]["near_oracle_ratio"])
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
                "candidate_probe_count": len(method_probe_order),
                "probe_order": ",".join(map(str, method_probe_order)),
                "method_evaluator_calls": method_evaluator_calls,
                "method_runtime_s": round(method_runtime_s, 6),
                "all_sampled_monotonic": all(
                    bool(record["sampled_monotonic"]) for record in capacities.values()
                ),
                "no_right_censoring": not any(
                    bool(record["right_censored"]) for record in capacities.values()
                ),
            }
            trials.append(trial)
            log_lines.append(json.dumps(trial, sort_keys=True))

            for scaling_radius in search_space_radii:
                if scaling_radius > radius:
                    continue
                subset_shifts = list(range(-scaling_radius, scaling_radius + 1))
                subset_oracle, subset_oracle_shifts = _oracle(capacities, subset_shifts)
                subset_ratio = None
                if proposed_capacity is not None and subset_oracle not in (None, 0):
                    subset_ratio = float(proposed_capacity) / float(subset_oracle)
                exhaustive_calls = sum(
                    int(capacities[shift]["evaluations"]) for shift in subset_shifts
                )
                exhaustive_runtime_s = sum(elapsed_s[shift] for shift in subset_shifts)
                scaling_rows.append(
                    {
                        "condition": condition,
                        "seed": int(seed),
                        "regime": regime_name,
                        "candidate_radius": int(scaling_radius),
                        "candidate_count": len(subset_shifts),
                        "method_candidate_probes": len(method_probe_order),
                        "method_evaluator_calls": method_evaluator_calls,
                        "exhaustive_evaluator_calls": exhaustive_calls,
                        "evaluator_call_reduction_pct": round(
                            (1.0 - method_evaluator_calls / exhaustive_calls) * 100.0, 6
                        ) if exhaustive_calls else None,
                        "method_runtime_s": round(method_runtime_s, 6),
                        "exhaustive_runtime_s": round(exhaustive_runtime_s, 6),
                        "runtime_reduction_pct": round(
                            (1.0 - method_runtime_s / exhaustive_runtime_s) * 100.0, 6
                        ) if exhaustive_runtime_s else None,
                        "proposed_shift": int(proposed_shift),
                        "oracle_shifts": ",".join(map(str, sorted(subset_oracle_shifts))),
                        "oracle_ratio": round(subset_ratio, 6) if subset_ratio is not None else None,
                        "matches_oracle_shift": proposed_shift in subset_oracle_shifts,
                    }
                )

    summary = {
        "condition": condition,
        "seeds": [int(seed) for seed in seeds],
        "candidate_radius": radius,
        "candidate_count": len(candidate_shifts),
        "bandwidth_multiplier": float(bandwidth_multiplier),
        "link_capacity_mb_s": float(cfg["network"]["link_capacity_mb_s"]),
        "trial_count": len(trials),
        "by_regime": _aggregate_trials(trials),
        "all_sampled_monotonic": all(bool(item["all_sampled_monotonic"]) for item in trials),
        "no_right_censoring": all(bool(item["no_right_censoring"]) for item in trials),
    }
    return trials, scaling_rows, summary, log_lines


def _aggregate_scaling(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(int(row["candidate_radius"]), str(row["regime"])) for row in rows})
    for radius, regime in keys:
        subset = [
            row for row in rows
            if int(row["candidate_radius"]) == radius and str(row["regime"]) == regime
        ]
        output.append(
            {
                "candidate_radius": radius,
                "candidate_count": 2 * radius + 1,
                "regime": regime,
                "trials": len(subset),
                "mean_method_candidate_probes": round(
                    statistics.mean(float(row["method_candidate_probes"]) for row in subset), 6
                ),
                "mean_method_evaluator_calls": round(
                    statistics.mean(float(row["method_evaluator_calls"]) for row in subset), 6
                ),
                "mean_exhaustive_evaluator_calls": round(
                    statistics.mean(float(row["exhaustive_evaluator_calls"]) for row in subset), 6
                ),
                "mean_evaluator_call_reduction_pct": round(
                    statistics.mean(float(row["evaluator_call_reduction_pct"]) for row in subset), 6
                ),
                "mean_method_runtime_s": round(
                    statistics.mean(float(row["method_runtime_s"]) for row in subset), 6
                ),
                "mean_exhaustive_runtime_s": round(
                    statistics.mean(float(row["exhaustive_runtime_s"]) for row in subset), 6
                ),
                "mean_runtime_reduction_pct": round(
                    statistics.mean(float(row["runtime_reduction_pct"]) for row in subset), 6
                ),
                "exact_oracle_shift_matches": sum(bool(row["matches_oracle_shift"]) for row in subset),
                "oracle_ratio": _mean_std_ci95(
                    [float(row["oracle_ratio"]) for row in subset if row["oracle_ratio"] is not None]
                ),
            }
        )
    return output


def run(cfg: dict[str, Any], suite: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    checksums = verify_trace_inputs(cfg)
    phase14 = cfg["phase14"]
    all_trials: list[dict[str, Any]] = []
    all_scaling: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    log_lines = [f"experiment_id={cfg['experiment_id']}", f"suite={suite}"]

    if suite == "expanded":
        trials, scaling, summary, lines = _run_condition(
            cfg,
            condition="expanded_20_seed_9_candidate",
            seeds=[int(seed) for seed in phase14["expanded_workload_seeds"]],
            radius=int(phase14["expanded_candidate_radius"]),
            bandwidth_multiplier=1.0,
            search_space_radii=[int(value) for value in phase14["search_space_radii"]],
        )
        all_trials.extend(trials)
        all_scaling.extend(scaling)
        conditions.append(summary)
        log_lines.extend(lines)
    elif suite == "bandwidth":
        for multiplier in phase14["bandwidth_multipliers"]:
            trials, scaling, summary, lines = _run_condition(
                cfg,
                condition=f"bandwidth_{float(multiplier):g}x",
                seeds=[int(seed) for seed in phase14["bandwidth_workload_seeds"]],
                radius=int(phase14["expanded_candidate_radius"]),
                bandwidth_multiplier=float(multiplier),
                search_space_radii=[],
            )
            all_trials.extend(trials)
            all_scaling.extend(scaling)
            conditions.append(summary)
            log_lines.extend(lines)
    else:
        raise ValueError(f"unknown suite: {suite}")

    summary = {
        "experiment_id": cfg["experiment_id"],
        "suite": suite,
        "provenance": {
            "source_repo_commit": cfg["phase11"]["source_repo_commit"],
            "upstream_helix_commit": cfg["phase11"]["upstream_helix_commit"],
            "artifact_sha256": checksums,
        },
        "controlled_changes": {
            "unchanged": [
                "event-driven evaluator equations and feasibility rules",
                "HELIX-derived Prefill and Decode speed factors",
                "120-second Azure-derived finite workload construction",
                "two TTFT/TPOT SLA regimes",
                "Phase-13 evaluator-guided four-probe decision rule",
            ],
            "expanded_suite": [
                "workload length-sampling seeds increase from 6 to 20",
                "post-decision exhaustive local oracle expands from five shifts [-2,2] to nine shifts [-4,4]",
                "nested 5/7/9-candidate oracle families are used to measure evaluator-call and wall-clock scaling",
            ],
            "bandwidth_suite": [
                "link capacity is multiplied by 0.5x, 1.0x, or 2.0x while all other evaluator inputs are held fixed",
                "the bandwidth sweep is a controlled sensitivity study, not additional hardware validation",
            ],
        },
        "important_interpretation": (
            "The enlarged oracle is evaluation-only. The proposed Phase-13 method still probes only shifts "
            "-1, 0, +1 and at most one of -2/+2 before committing, so any agreement with a nine-candidate "
            "oracle is out-of-search robustness rather than access to the extra candidates. Runtime scaling "
            "uses measured first-evaluation times for the nested candidate subsets in the same trial."
        ),
        "confidence_interval": "normal-approximation 95% CI for the sample mean: mean +/- 1.96 * sample_std / sqrt(n)",
        "conditions": conditions,
        "search_cost_summary": _aggregate_scaling(all_scaling) if all_scaling else [],
    }
    log_lines.append("summary=" + json.dumps(summary, sort_keys=True))
    return all_trials, all_scaling, summary, "\n".join(log_lines) + "\n"


def write_outputs(
    trials: list[dict[str, Any]],
    scaling: list[dict[str, Any]],
    summary: dict[str, Any],
    log: str,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if trials:
        with (output / "trial_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
    if scaling:
        with (output / "search_cost.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scaling[0]))
            writer.writeheader()
            writer.writerows(scaling)
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
    trials, scaling, summary, log = run(cfg, args.suite)
    write_outputs(trials, scaling, summary, log, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

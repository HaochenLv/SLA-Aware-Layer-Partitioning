from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

from .experiment import load_config
from .helix_profile import derive_phase_speed_factors
from .phase10_method_compare import static_baseline_shifts
from .phase11_trace_validation import adaptive_capacity, build_helix_workload, verify_trace_inputs
from .phase6_profiled import shifted_partition


def _capacity_value(record: dict[str, Any]) -> float:
    value = record.get("safe_intensity")
    return -math.inf if value is None else float(value)


def evaluator_guided_local_search(
    capacity_for: Callable[[int], dict[str, Any]],
) -> tuple[int, list[int]]:
    """Search the five-shift local family using at most four capacity probes.

    The search starts at the uniform partition and its two adjacent shifts.
    It then expands one more step only in the better non-decreasing direction.
    Candidate ranking uses evaluator-induced sampled SLA-safe capacity directly;
    no single-load headroom surrogate is used.
    """

    probe_order: list[int] = []

    def probe(shift: int) -> dict[str, Any]:
        if shift not in probe_order:
            probe_order.append(shift)
        return capacity_for(shift)

    initial = {shift: probe(shift) for shift in (-1, 0, 1)}
    uniform_value = _capacity_value(initial[0])

    best_neighbor = max(
        (-1, 1),
        key=lambda shift: (_capacity_value(initial[shift]), -abs(shift), -shift),
    )
    if _capacity_value(initial[best_neighbor]) >= uniform_value:
        probe(2 if best_neighbor > 0 else -2)

    best_shift = max(
        probe_order,
        key=lambda shift: (_capacity_value(capacity_for(shift)), -abs(shift), -shift),
    )
    return int(best_shift), probe_order


def _public_probe_trace(
    probe_order: list[int],
    capacities: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "shift": int(shift),
            "safe_intensity": capacities[shift]["safe_intensity"],
            "unsafe_intensity": capacities[shift]["unsafe_intensity"],
            "first_violation": capacities[shift]["first_violation"],
            "right_censored": capacities[shift]["right_censored"],
            "sampled_monotonic": capacities[shift]["sampled_monotonic"],
            "evaluator_calls": capacities[shift]["evaluations"],
        }
        for shift in probe_order
    ]


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    checksums = verify_trace_inputs(cfg)
    factors = derive_phase_speed_factors(cfg)
    machines = cfg["phase6"]["stage_machines"]
    prefill_speeds = [factors[machine]["prefill_speed"] for machine in machines]
    decode_speeds = [factors[machine]["decode_speed"] for machine in machines]
    static_shifts = static_baseline_shifts(prefill_speeds, decode_speeds, cfg)

    rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    log_lines = [f"experiment_id={cfg['experiment_id']}"]

    for seed in cfg["phase11"]["workload_seeds"]:
        base = build_helix_workload(cfg, int(seed))
        for regime_name, regime in cfg["phase11"]["regimes"].items():
            regime_cfg = copy.deepcopy(cfg)
            regime_cfg["sla"]["ttft_s"] = regime["ttft_s"]
            regime_cfg["sla"]["tpot_s"] = regime["tpot_s"]

            capacities: dict[int, dict[str, Any]] = {}

            def capacity_for(shift: int) -> dict[str, Any]:
                shift = int(shift)
                if shift not in capacities:
                    capacities[shift] = adaptive_capacity(
                        base,
                        shift,
                        prefill_speeds,
                        decode_speeds,
                        regime_cfg,
                    )
                return capacities[shift]

            proposed_shift, probe_order = evaluator_guided_local_search(capacity_for)
            method_probe_order = list(probe_order)
            method_probe_trace = _public_probe_trace(method_probe_order, capacities)
            method_evaluator_calls = sum(
                capacities[shift]["evaluations"] for shift in method_probe_order
            )

            # Fill only the still-missing candidates after the method decision so
            # the exhaustive five-shift oracle is used for evaluation, not search.
            for shift in cfg["phase11"]["candidate_shifts"]:
                capacity_for(int(shift))

            observed = [
                record["safe_intensity"]
                for record in capacities.values()
                if record["safe_intensity"] is not None
            ]
            oracle_capacity = max(observed) if observed else None
            oracle_shifts = [
                shift
                for shift, record in capacities.items()
                if record["safe_intensity"] == oracle_capacity
            ] if oracle_capacity is not None else []

            methods = dict(static_shifts)
            methods["sla_evaluator_guided_local_search"] = proposed_shift
            for method, shift in methods.items():
                record = capacities[int(shift)]
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime_name,
                        "method": method,
                        "shift": int(shift),
                        "partition": "-".join(map(str, shifted_partition(int(shift), regime_cfg))),
                        "safe_intensity": record["safe_intensity"],
                        "unsafe_intensity": record["unsafe_intensity"],
                        "first_violation": record["first_violation"],
                        "sampled_monotonic": record["sampled_monotonic"],
                        "capacity_evaluations": record["evaluations"],
                        "oracle_safe_intensity": oracle_capacity,
                        "oracle_shifts": ",".join(map(str, sorted(oracle_shifts))),
                    }
                )

            uniform_capacity = capacities[0]["safe_intensity"]
            proposed_capacity = capacities[proposed_shift]["safe_intensity"]
            joint_shift = int(static_shifts["phase_agnostic_joint_compute"])
            joint_capacity = capacities[joint_shift]["safe_intensity"]

            oracle_ratio = None
            gain_uniform = None
            gain_joint = None
            if proposed_capacity is not None and oracle_capacity not in (None, 0):
                oracle_ratio = proposed_capacity / oracle_capacity
            if proposed_capacity is not None and uniform_capacity not in (None, 0):
                gain_uniform = (proposed_capacity / uniform_capacity - 1.0) * 100.0
            if proposed_capacity is not None and joint_capacity not in (None, 0):
                gain_joint = (proposed_capacity / joint_capacity - 1.0) * 100.0

            trial = {
                "seed": seed,
                "regime": regime_name,
                "request_count": len(base),
                "proposed_shift": proposed_shift,
                "proposed_safe_intensity": proposed_capacity,
                "uniform_safe_intensity": uniform_capacity,
                "joint_compute_shift": joint_shift,
                "joint_compute_safe_intensity": joint_capacity,
                "oracle_safe_intensity": oracle_capacity,
                "oracle_shifts": sorted(oracle_shifts),
                "oracle_ratio": round(oracle_ratio, 6) if oracle_ratio is not None else None,
                "gain_over_uniform_pct": round(gain_uniform, 3) if gain_uniform is not None else None,
                "gain_over_joint_compute_pct": round(gain_joint, 3) if gain_joint is not None else None,
                "matches_oracle_shift": proposed_shift in oracle_shifts,
                "near_oracle": oracle_ratio is not None and oracle_ratio >= cfg["phase13"]["near_oracle_ratio"],
                "not_worse_than_uniform": proposed_capacity is not None and uniform_capacity is not None and proposed_capacity >= uniform_capacity,
                "not_worse_than_joint_compute": proposed_capacity is not None and joint_capacity is not None and proposed_capacity >= joint_capacity,
                "candidate_probe_count": len(method_probe_order),
                "method_evaluator_calls": method_evaluator_calls,
                "probe_trace": method_probe_trace,
                "all_sampled_monotonic": all(record["sampled_monotonic"] for record in capacities.values()),
                "no_right_censoring": not any(record["right_censored"] for record in capacities.values()),
            }
            trials.append(trial)
            log_lines.append(json.dumps(trial, sort_keys=True))

    gains = [item["gain_over_uniform_pct"] for item in trials if item["gain_over_uniform_pct"] is not None]
    conclusion = {
        "question": "Does evaluator-guided local capacity search remove the single-load surrogate-ranking misses while retaining at most four candidate capacity probes on the 120-second HELIX Azure-derived workloads?",
        "trials": len(trials),
        "near_oracle_trials": sum(item["near_oracle"] for item in trials),
        "exact_oracle_shift_matches": sum(item["matches_oracle_shift"] for item in trials),
        "not_worse_than_uniform": sum(item["not_worse_than_uniform"] for item in trials),
        "not_worse_than_phase_agnostic_joint_compute": sum(item["not_worse_than_joint_compute"] for item in trials),
        "median_gain_over_uniform_pct": round(statistics.median(gains), 3) if gains else None,
        "minimum_gain_over_uniform_pct": round(min(gains), 3) if gains else None,
        "max_candidate_probe_count": max(item["candidate_probe_count"] for item in trials),
        "max_method_evaluator_calls": max(item["method_evaluator_calls"] for item in trials),
        "all_sampled_monotonic": all(item["all_sampled_monotonic"] for item in trials),
        "no_right_censoring": all(item["no_right_censoring"] for item in trials),
        "minimum_near_oracle_trials": cfg["phase13"]["minimum_near_oracle_trials"],
        "minimum_not_worse_than_uniform": cfg["phase13"]["minimum_not_worse_than_uniform"],
        "minimum_not_worse_than_joint_compute": cfg["phase13"]["minimum_not_worse_than_joint_compute"],
        "maximum_candidate_probes": cfg["phase13"]["maximum_candidate_probes"],
    }
    conclusion["answer"] = (
        "yes"
        if conclusion["near_oracle_trials"] >= cfg["phase13"]["minimum_near_oracle_trials"]
        and conclusion["not_worse_than_uniform"] >= cfg["phase13"]["minimum_not_worse_than_uniform"]
        and conclusion["not_worse_than_phase_agnostic_joint_compute"] >= cfg["phase13"]["minimum_not_worse_than_joint_compute"]
        and conclusion["max_candidate_probe_count"] <= cfg["phase13"]["maximum_candidate_probes"]
        and conclusion["all_sampled_monotonic"]
        and conclusion["no_right_censoring"]
        else "not_yet"
    )
    conclusion["experimental_freeze_ready"] = conclusion["answer"] == "yes"

    summary = {
        "experiment_id": cfg["experiment_id"],
        "provenance": {
            "source_repository": "HaochenLv/sla-aware-evaluator",
            "source_repo_commit": cfg["phase11"]["source_repo_commit"],
            "upstream_helix_commit": cfg["phase11"]["upstream_helix_commit"],
            "artifact_sha256": checksums,
            "workload_duration_s": cfg["phase11"]["duration_s"],
            "workload_kind": "HELIX_AzureConversation_derived_generated_workload",
        },
        "semantics": "The evaluator, HELIX-derived phase-specific speed vectors, 120-second Azure-derived workload construction, candidate partition family, SLA regimes, and evaluator-induced sampled capacity definition are unchanged from Phase 12. Only the search ranking signal changes: the method directly compares sampled SLA-safe capacity for the uniform partition and adjacent boundary shifts, then evaluates one farther shift in the better non-decreasing direction. The exhaustive five-shift oracle is filled only after the method decision and is used solely for evaluation.",
        "static_baseline_shifts": static_shifts,
        "trial_summaries": trials,
        "conclusion": conclusion,
    }
    log_lines.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, summary, "\n".join(log_lines) + "\n"


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], log: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "method_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase13.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase13"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

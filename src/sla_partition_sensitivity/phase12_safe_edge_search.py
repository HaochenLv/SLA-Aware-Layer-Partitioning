from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .experiment import load_config
from .helix_profile import derive_phase_speed_factors
from .phase10_method_compare import static_baseline_shifts
from .phase11_trace_validation import (
    adaptive_capacity,
    build_helix_workload,
    scale_workload,
    verify_trace_inputs,
)
from .phase6_profiled import shifted_partition
from .profiled_model import evaluate_profiled


def _evaluate_at(
    base_workload,
    intensity: float,
    shift: int,
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
):
    return evaluate_profiled(
        scale_workload(base_workload, intensity),
        shifted_partition(shift, cfg),
        prefill_speeds,
        decode_speeds,
        cfg,
    )


def _score(verdict: Any) -> tuple[int, float]:
    return (int(verdict.safe), verdict.minimum_link_headroom_mb_s)


def safe_edge_search(
    base_workload,
    uniform_capacity: dict[str, Any],
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
) -> tuple[int, float, list[dict[str, Any]]]:
    reference = uniform_capacity["safe_intensity"]
    if reference is None:
        raise ValueError("uniform partition has no safe capacity edge")

    probes: list[tuple[int, Any]] = []
    for shift in (-1, 0, 1):
        probes.append(
            (
                shift,
                _evaluate_at(
                    base_workload,
                    reference,
                    shift,
                    prefill_speeds,
                    decode_speeds,
                    cfg,
                ),
            )
        )
    uniform_verdict = next(verdict for shift, verdict in probes if shift == 0)
    if not uniform_verdict.safe:
        raise RuntimeError("safe-edge reference must keep the uniform partition feasible")

    best_shift, best_verdict = max(
        probes,
        key=lambda item: (_score(item[1]), -abs(item[0]), -item[0]),
    )
    if best_shift != 0 and _score(best_verdict) > _score(uniform_verdict):
        extreme = 2 if best_shift > 0 else -2
        extreme_verdict = _evaluate_at(
            base_workload,
            reference,
            extreme,
            prefill_speeds,
            decode_speeds,
            cfg,
        )
        probes.append((extreme, extreme_verdict))
        if _score(extreme_verdict) > _score(best_verdict):
            best_shift, best_verdict = extreme, extreme_verdict

    public = [
        {
            "shift": shift,
            "safe": verdict.safe,
            "first_violation": verdict.first_violation,
            "minimum_link_headroom_mb_s": round(verdict.minimum_link_headroom_mb_s, 6),
        }
        for shift, verdict in probes
    ]
    return int(best_shift), reference, public


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

            capacities = {
                int(shift): adaptive_capacity(
                    base,
                    int(shift),
                    prefill_speeds,
                    decode_speeds,
                    regime_cfg,
                )
                for shift in cfg["phase11"]["candidate_shifts"]
            }
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

            proposed_shift, reference, probes = safe_edge_search(
                base,
                capacities[0],
                prefill_speeds,
                decode_speeds,
                regime_cfg,
            )
            methods = dict(static_shifts)
            methods["sla_safe_edge_search"] = proposed_shift

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
                        "oracle_safe_intensity": oracle_capacity,
                        "oracle_shifts": ",".join(map(str, oracle_shifts)),
                    }
                )

            uniform_capacity = capacities[0]["safe_intensity"]
            proposed_capacity = capacities[proposed_shift]["safe_intensity"]
            joint_capacity = capacities[static_shifts["phase_agnostic_joint_compute"]]["safe_intensity"]
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
                "joint_compute_safe_intensity": joint_capacity,
                "oracle_safe_intensity": oracle_capacity,
                "oracle_shifts": oracle_shifts,
                "oracle_ratio": round(oracle_ratio, 6) if oracle_ratio is not None else None,
                "gain_over_uniform_pct": round(gain_uniform, 3) if gain_uniform is not None else None,
                "gain_over_joint_compute_pct": round(gain_joint, 3) if gain_joint is not None else None,
                "matches_oracle_shift": proposed_shift in oracle_shifts,
                "near_oracle": oracle_ratio is not None and oracle_ratio >= cfg["phase12"]["near_oracle_ratio"],
                "not_worse_than_uniform": proposed_capacity is not None and uniform_capacity is not None and proposed_capacity >= uniform_capacity,
                "not_worse_than_joint_compute": proposed_capacity is not None and joint_capacity is not None and proposed_capacity >= joint_capacity,
                "reference_safe_intensity": reference,
                "probe_count": len(probes),
                "probe_trace": probes,
                "all_sampled_monotonic": all(record["sampled_monotonic"] for record in capacities.values()),
                "no_right_censoring": not any(record["right_censored"] for record in capacities.values()),
            }
            trials.append(trial)
            log_lines.append(json.dumps(trial, sort_keys=True))

    gains = [item["gain_over_uniform_pct"] for item in trials if item["gain_over_uniform_pct"] is not None]
    conclusion = {
        "question": "Does probing candidate boundaries at the uniform partition's last safe workload edge avoid the overload-severity failure of the first-unsafe stress probe while retaining a low evaluation cost on the 120-second HELIX Azure-derived workloads?",
        "trials": len(trials),
        "near_oracle_trials": sum(item["near_oracle"] for item in trials),
        "exact_oracle_shift_matches": sum(item["matches_oracle_shift"] for item in trials),
        "not_worse_than_uniform": sum(item["not_worse_than_uniform"] for item in trials),
        "not_worse_than_phase_agnostic_joint_compute": sum(item["not_worse_than_joint_compute"] for item in trials),
        "median_gain_over_uniform_pct": round(statistics.median(gains), 3) if gains else None,
        "minimum_gain_over_uniform_pct": round(min(gains), 3) if gains else None,
        "max_probe_count": max(item["probe_count"] for item in trials),
        "all_sampled_monotonic": all(item["all_sampled_monotonic"] for item in trials),
        "no_right_censoring": all(item["no_right_censoring"] for item in trials),
        "minimum_near_oracle_trials": cfg["phase12"]["minimum_near_oracle_trials"],
        "minimum_not_worse_than_uniform": cfg["phase12"]["minimum_not_worse_than_uniform"],
    }
    conclusion["answer"] = (
        "yes"
        if conclusion["near_oracle_trials"] >= cfg["phase12"]["minimum_near_oracle_trials"]
        and conclusion["not_worse_than_uniform"] >= cfg["phase12"]["minimum_not_worse_than_uniform"]
        and conclusion["all_sampled_monotonic"]
        and conclusion["no_right_censoring"]
        else "not_yet"
    )

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
        "semantics": "The evaluator, HELIX-derived phase-specific speed vectors, 120-second Azure-derived workload construction, and capacity definition are unchanged from Phase 11. Only the search probe location changes: candidates are compared at the uniform partition's last safe capacity edge rather than at its first unsafe edge. Because uniform is feasible at this common load, the search never ranks candidates solely by how severely they violate constraints beyond the feasible region.",
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
    parser.add_argument("--config", type=Path, default=Path("config/phase12.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase12"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

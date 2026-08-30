from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

from .experiment import load_config
from .helix_profile import derive_phase_speed_factors
from .phase6_profiled import capacity_record, shifted_partition


def _grid(cfg: dict[str, Any]) -> list[float]:
    spec = cfg["phase8"]
    start = spec["lambda_start"]
    stop = spec["lambda_stop"]
    step = spec["lambda_step"]
    count = round((stop - start) / step)
    return [round(start + i * step, 10) for i in range(count + 1)]


def _inverse_interpolate(left: float, right: float, alpha: float) -> float:
    return 1.0 / ((1.0 - alpha) / left + alpha / right)


def sla_point(alpha: float, cfg: dict[str, Any]) -> tuple[float, float]:
    spec = cfg["phase8"]
    left = spec["prefill_endpoint"]
    right = spec["decode_endpoint"]
    mode = spec.get("interpolation", "inverse_budget")
    if mode == "inverse_budget":
        ttft = _inverse_interpolate(left["ttft_s"], right["ttft_s"], alpha)
        tpot = _inverse_interpolate(left["tpot_s"], right["tpot_s"], alpha)
    elif mode == "linear":
        ttft = left["ttft_s"] + alpha * (right["ttft_s"] - left["ttft_s"])
        tpot = left["tpot_s"] + alpha * (right["tpot_s"] - left["tpot_s"])
    else:
        raise ValueError(f"unknown phase-8 interpolation mode: {mode}")
    return round(ttft, 6), round(tpot, 6)


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    factors = derive_phase_speed_factors(cfg)
    stage_machines = cfg["phase6"]["stage_machines"]
    prefill_speeds = [factors[machine]["prefill_speed"] for machine in stage_machines]
    decode_speeds = [factors[machine]["decode_speed"] for machine in stage_machines]
    grid = _grid(cfg)

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    log_lines = [f"experiment_id={cfg['experiment_id']}"]

    for alpha in cfg["phase8"]["alpha_values"]:
        point_cfg = copy.deepcopy(cfg)
        ttft_s, tpot_s = sla_point(float(alpha), cfg)
        point_cfg["sla"]["ttft_s"] = ttft_s
        point_cfg["sla"]["tpot_s"] = tpot_s
        point_rows: list[dict[str, Any]] = []

        for shift in cfg["phase8"]["boundary_shifts"]:
            partition = shifted_partition(int(shift), point_cfg)
            record = capacity_record(
                partition,
                prefill_speeds,
                decode_speeds,
                point_cfg,
                grid,
            )
            row = {
                "alpha": float(alpha),
                "ttft_s": ttft_s,
                "tpot_s": tpot_s,
                "shift": int(shift),
                "partition": "-".join(map(str, partition)),
                **record,
            }
            rows.append(row)
            point_rows.append(row)
            log_lines.append(json.dumps(row, sort_keys=True))

        observed = [row["safe_lambda"] for row in point_rows if row["safe_lambda"] is not None]
        best_capacity = max(observed) if observed else None
        best_shifts = [
            row["shift"] for row in point_rows if row["safe_lambda"] == best_capacity
        ] if best_capacity is not None else []
        uniform = next(row for row in point_rows if row["shift"] == 0)
        gain_pct = None
        if best_capacity is not None and uniform["safe_lambda"] not in (None, 0):
            gain_pct = round((best_capacity / uniform["safe_lambda"] - 1.0) * 100.0, 3)
        summary = {
            "alpha": float(alpha),
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "best_safe_lambda": best_capacity,
            "best_shifts": best_shifts,
            "uniform_safe_lambda": uniform["safe_lambda"],
            "best_gain_over_uniform_pct": gain_pct,
            "uniform_first_violation": uniform["first_violation"],
            "all_sampled_monotonic": all(row["sampled_monotonic"] for row in point_rows),
            "any_right_censored": any(row["right_censored"] for row in point_rows),
        }
        summaries.append(summary)
        log_lines.append("sla_summary=" + json.dumps(summary, sort_keys=True))

    representative = [
        sum(item["best_shifts"]) / len(item["best_shifts"])
        if item["best_shifts"] else 0.0
        for item in summaries
    ]
    nonincreasing = all(
        representative[index] <= representative[index - 1] + 1e-12
        for index in range(1, len(representative))
    )
    strict_positive_points = sum(
        bool(item["best_shifts"]) and all(shift > 0 for shift in item["best_shifts"])
        for item in summaries
    )
    strict_negative_points = sum(
        bool(item["best_shifts"]) and all(shift < 0 for shift in item["best_shifts"])
        for item in summaries
    )
    start_positive = bool(summaries[0]["best_shifts"]) and all(
        shift > 0 for shift in summaries[0]["best_shifts"]
    )
    end_negative = bool(summaries[-1]["best_shifts"]) and all(
        shift < 0 for shift in summaries[-1]["best_shifts"]
    )
    no_right_censoring = not any(item["any_right_censored"] for item in summaries)
    distinct_best_sets = len({tuple(item["best_shifts"]) for item in summaries})
    conclusion = {
        "question": "Along a controlled TTFT-to-TPOT SLA path, does the preferred layer-boundary direction change from L4x2-heavy to T4x4-heavy rather than remaining one static compute-balanced choice?",
        "start_prefers_more_l4x2_layers": start_positive,
        "end_prefers_more_t4x4_layers": end_negative,
        "representative_best_shift_nonincreasing": nonincreasing,
        "strict_positive_best_points": strict_positive_points,
        "strict_negative_best_points": strict_negative_points,
        "distinct_best_shift_sets": distinct_best_sets,
        "no_right_censoring": no_right_censoring,
        "all_sampled_monotonic": all(item["all_sampled_monotonic"] for item in summaries),
    }
    conclusion["answer"] = (
        "yes"
        if start_positive
        and end_negative
        and strict_positive_points > 0
        and strict_negative_points > 0
        and distinct_best_sets >= 2
        and no_right_censoring
        and conclusion["all_sampled_monotonic"]
        else "not_yet"
    )

    summary = {
        "experiment_id": cfg["experiment_id"],
        "provenance": {
            "helix_commit": "8639497a4aaf1eb3b7594614cb0bbd376c1342b3",
            "reference_machine": cfg["phase6"]["reference_machine"],
            "stage_machines": stage_machines,
            "workload_seed": cfg["workload"]["seed"],
        },
        "semantics": "Phase-8 preserves the Phase-6/7 profiled evaluator and the same five contiguous boundary shifts. Only the TTFT/TPOT pair changes. The path interpolates inverse SLA budgets between the validated TTFT-tight and TPOT-tight endpoints so the middle of the sweep does not become trivially loose; this is a controlled sensitivity construction, not online adaptation or a new evaluator objective.",
        "sla_points": summaries,
        "conclusion": conclusion,
    }
    log_lines.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, summary, "\n".join(log_lines) + "\n"


def write_outputs(
    rows: list[dict[str, Any]], summary: dict[str, Any], log: str, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sla_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase8_sla_sweep.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase8-sla-sweep"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

from .experiment import build_workload, load_config
from .helix_profile import derive_phase_speed_factors
from .phase6_profiled import shifted_partition
from .profiled_model import evaluate_profiled


def _inverse_interpolate(left: float, right: float, alpha: float) -> float:
    return 1.0 / ((1.0 - alpha) / left + alpha / right)


def sla_point(alpha: float, cfg: dict[str, Any]) -> tuple[float, float]:
    spec = cfg["phase9"]
    left = spec["prefill_endpoint"]
    right = spec["decode_endpoint"]
    return (
        round(_inverse_interpolate(left["ttft_s"], right["ttft_s"], alpha), 6),
        round(_inverse_interpolate(left["tpot_s"], right["tpot_s"], alpha), 6),
    )


def _frange(start: float, stop: float, step: float) -> list[float]:
    count = round((stop - start) / step)
    return [round(start + index * step, 10) for index in range(count + 1)]


def refined_capacity(
    partition: list[int],
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    spec = cfg["phase9"]
    evaluated: list[tuple[float, Any]] = []
    previous_safe: float | None = None
    first_unsafe_lambda: float | None = None
    first_unsafe_verdict = None

    for intensity in _frange(
        spec["coarse_lambda_start"],
        spec["coarse_lambda_stop"],
        spec["coarse_lambda_step"],
    ):
        verdict = evaluate_profiled(
            build_workload(cfg, intensity),
            partition,
            prefill_speeds,
            decode_speeds,
            cfg,
        )
        evaluated.append((intensity, verdict))
        if verdict.safe:
            previous_safe = intensity
            continue
        first_unsafe_lambda = intensity
        first_unsafe_verdict = verdict
        break

    if first_unsafe_lambda is None:
        return {
            "safe_lambda": previous_safe,
            "unsafe_lambda": None,
            "right_censored": True,
            "sampled_monotonic": True,
            "first_violation": None,
            "peak_decode_at_first_unsafe": None,
            "evaluations": len(evaluated),
        }

    lower = previous_safe
    if lower is not None:
        fine_start = round(lower + spec["fine_lambda_step"], 10)
        fine_stop = round(first_unsafe_lambda - spec["fine_lambda_step"], 10)
        if fine_start <= fine_stop + 1e-12:
            for intensity in _frange(fine_start, fine_stop, spec["fine_lambda_step"]):
                verdict = evaluate_profiled(
                    build_workload(cfg, intensity),
                    partition,
                    prefill_speeds,
                    decode_speeds,
                    cfg,
                )
                evaluated.append((intensity, verdict))
                if verdict.safe:
                    lower = intensity
                else:
                    first_unsafe_lambda = intensity
                    first_unsafe_verdict = verdict
                    break

    evaluated.sort(key=lambda item: item[0])
    verdict_bits = [verdict.safe for _, verdict in evaluated]
    monotonic = all(
        not verdict_bits[index] or verdict_bits[index - 1]
        for index in range(1, len(verdict_bits))
    )
    return {
        "safe_lambda": lower,
        "unsafe_lambda": first_unsafe_lambda,
        "right_censored": False,
        "sampled_monotonic": monotonic,
        "first_violation": first_unsafe_verdict.first_violation if first_unsafe_verdict else None,
        "peak_decode_at_first_unsafe": (
            first_unsafe_verdict.peak_decode if first_unsafe_verdict else None
        ),
        "evaluations": len(evaluated),
    }


def _direction(best_shifts: list[int]) -> str:
    if best_shifts and all(shift > 0 for shift in best_shifts):
        return "l4x2_heavy"
    if best_shifts and all(shift < 0 for shift in best_shifts):
        return "t4x4_heavy"
    if best_shifts == [0]:
        return "uniform"
    return "tie_or_mixed"


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    factors = derive_phase_speed_factors(cfg)
    stage_machines = cfg["phase6"]["stage_machines"]
    prefill_speeds = [factors[machine]["prefill_speed"] for machine in stage_machines]
    decode_speeds = [factors[machine]["decode_speed"] for machine in stage_machines]

    rows: list[dict[str, Any]] = []
    point_summaries: list[dict[str, Any]] = []
    log_lines = [f"experiment_id={cfg['experiment_id']}"]

    for alpha in cfg["phase9"]["alpha_values"]:
        point_cfg = copy.deepcopy(cfg)
        ttft_s, tpot_s = sla_point(float(alpha), cfg)
        point_cfg["sla"]["ttft_s"] = ttft_s
        point_cfg["sla"]["tpot_s"] = tpot_s
        point_rows: list[dict[str, Any]] = []

        for shift in cfg["phase9"]["boundary_shifts"]:
            partition = shifted_partition(int(shift), point_cfg)
            record = refined_capacity(
                partition,
                prefill_speeds,
                decode_speeds,
                point_cfg,
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

        capacities = [row["safe_lambda"] for row in point_rows if row["safe_lambda"] is not None]
        best_capacity = max(capacities) if capacities else None
        best_shifts = [
            row["shift"] for row in point_rows if row["safe_lambda"] == best_capacity
        ] if best_capacity is not None else []
        uniform = next(row for row in point_rows if row["shift"] == 0)
        gain_pct = None
        if best_capacity is not None and uniform["safe_lambda"] not in (None, 0):
            gain_pct = round((best_capacity / uniform["safe_lambda"] - 1.0) * 100.0, 3)
        point_summary = {
            "alpha": float(alpha),
            "ttft_s": ttft_s,
            "tpot_s": tpot_s,
            "best_safe_lambda": best_capacity,
            "best_shifts": best_shifts,
            "direction": _direction(best_shifts),
            "uniform_safe_lambda": uniform["safe_lambda"],
            "best_gain_over_uniform_pct": gain_pct,
            "all_sampled_monotonic": all(row["sampled_monotonic"] for row in point_rows),
            "any_right_censored": any(row["right_censored"] for row in point_rows),
            "total_evaluations": sum(row["evaluations"] for row in point_rows),
        }
        point_summaries.append(point_summary)
        log_lines.append("point_summary=" + json.dumps(point_summary, sort_keys=True))

    positive = [item for item in point_summaries if item["direction"] == "l4x2_heavy"]
    negative = [item for item in point_summaries if item["direction"] == "t4x4_heavy"]
    last_positive = max((item["alpha"] for item in positive), default=None)
    first_negative = min((item["alpha"] for item in negative), default=None)
    crossover_bracket = None
    if last_positive is not None and first_negative is not None and last_positive < first_negative:
        crossover_bracket = [last_positive, first_negative]

    conclusion = {
        "question": "Can the SLA-dependent partition preference crossover be localized without capacity right-censoring?",
        "start_direction": point_summaries[0]["direction"],
        "end_direction": point_summaries[-1]["direction"],
        "last_l4x2_heavy_alpha": last_positive,
        "first_t4x4_heavy_alpha": first_negative,
        "crossover_bracket": crossover_bracket,
        "no_right_censoring": not any(item["any_right_censored"] for item in point_summaries),
        "all_sampled_monotonic": all(item["all_sampled_monotonic"] for item in point_summaries),
        "max_gain_over_uniform_pct": max(
            (item["best_gain_over_uniform_pct"] or 0.0 for item in point_summaries),
            default=0.0,
        ),
    }
    conclusion["answer"] = (
        "yes"
        if conclusion["start_direction"] == "l4x2_heavy"
        and conclusion["end_direction"] == "t4x4_heavy"
        and crossover_bracket is not None
        and conclusion["no_right_censoring"]
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
        "semantics": "Phase-9 keeps the Phase-6/7 evaluator and HELIX-derived Prefill/Decode speed vectors unchanged. It refines only the sampled-capacity search: a coarse grid brackets the first unsafe load and a fine grid refines that bracket. The SLA path uses inverse-budget interpolation solely to localize the preference crossover under nontrivial load.",
        "points": point_summaries,
        "conclusion": conclusion,
    }
    log_lines.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, summary, "\n".join(log_lines) + "\n"


def write_outputs(
    rows: list[dict[str, Any]], summary: dict[str, Any], log: str, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "crossover.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase9_crossover.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase9-crossover"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

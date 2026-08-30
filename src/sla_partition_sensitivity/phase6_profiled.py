from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

from .experiment import build_workload, load_config
from .helix_profile import derive_phase_speed_factors
from .profiled_model import evaluate_profiled


def regime_grid(regime: dict[str, Any]) -> list[float]:
    start = regime["lambda_start"]
    stop = regime["lambda_stop"]
    step = regime["lambda_step"]
    count = round((stop - start) / step)
    return [round(start + i * step, 10) for i in range(count + 1)]


def shifted_partition(shift: int, cfg: dict[str, Any]) -> list[int]:
    if cfg["model"]["stages"] != 8 or cfg["model"]["layers"] != 80:
        raise ValueError("phase-6 controlled family expects the 80-layer, 8-stage setup")
    partition = []
    for stage in range(8):
        partition.append(10 + shift if stage % 2 == 0 else 10 - shift)
    lo = cfg["partitions"]["min_layers_per_stage"]
    hi = cfg["partitions"]["max_layers_per_stage"]
    if any(value < lo or value > hi for value in partition):
        raise ValueError(f"shift {shift} violates phase-6 partition bounds")
    return partition


def capacity_record(
    partition: list[int],
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
    grid: list[float],
) -> dict[str, Any]:
    trials = []
    for intensity in grid:
        verdict = evaluate_profiled(
            build_workload(cfg, intensity),
            partition,
            prefill_speeds,
            decode_speeds,
            cfg,
        )
        trials.append((intensity, verdict))

    safe_values = [intensity for intensity, verdict in trials if verdict.safe]
    safe_lambda = max(safe_values) if safe_values else None
    unsafe_values = [
        intensity
        for intensity, verdict in trials
        if not verdict.safe and (safe_lambda is None or intensity > safe_lambda)
    ]
    unsafe_lambda = min(unsafe_values) if unsafe_values else None
    first_unsafe = next((verdict for _, verdict in trials if not verdict.safe), None)
    verdict_bits = [verdict.safe for _, verdict in trials]
    monotonic = all(
        not verdict_bits[index] or verdict_bits[index - 1]
        for index in range(1, len(verdict_bits))
    )
    return {
        "safe_lambda": safe_lambda,
        "unsafe_lambda": unsafe_lambda,
        "right_censored": unsafe_lambda is None,
        "sampled_monotonic": monotonic,
        "first_violation": first_unsafe.first_violation if first_unsafe else None,
        "peak_decode_at_first_unsafe": first_unsafe.peak_decode if first_unsafe else None,
    }


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    factors = derive_phase_speed_factors(cfg)
    stage_machines = cfg["phase6"]["stage_machines"]
    prefill_speeds = [factors[machine]["prefill_speed"] for machine in stage_machines]
    decode_speeds = [factors[machine]["decode_speed"] for machine in stage_machines]

    rows: list[dict[str, Any]] = []
    log_lines = [f"experiment_id={cfg['experiment_id']}"]
    log_lines.append("stage_machines=" + json.dumps(stage_machines))
    log_lines.append("profile_factors=" + json.dumps(factors, sort_keys=True))
    regime_summaries: dict[str, Any] = {}

    for regime_name, regime in cfg["phase6"]["regimes"].items():
        regime_cfg = copy.deepcopy(cfg)
        regime_cfg["sla"]["ttft_s"] = regime["ttft_s"]
        regime_cfg["sla"]["tpot_s"] = regime["tpot_s"]
        grid = regime_grid(regime)
        regime_rows: list[dict[str, Any]] = []

        for shift in cfg["phase6"]["boundary_shifts"]:
            partition = shifted_partition(shift, regime_cfg)
            record = capacity_record(
                partition,
                prefill_speeds,
                decode_speeds,
                regime_cfg,
                grid,
            )
            row = {
                "regime": regime_name,
                "ttft_s": regime["ttft_s"],
                "tpot_s": regime["tpot_s"],
                "shift": shift,
                "partition": "-".join(map(str, partition)),
                "l4x2_layers_per_stage": 10 + shift,
                "t4x4_layers_per_stage": 10 - shift,
                "weighted_prefill_layers": round(
                    sum(n / speed for n, speed in zip(partition, prefill_speeds)), 6
                ),
                "weighted_decode_layers": round(
                    sum(n / speed for n, speed in zip(partition, decode_speeds)), 6
                ),
                **record,
            }
            rows.append(row)
            regime_rows.append(row)
            log_lines.append(json.dumps(row, sort_keys=True))

        observed = [row["safe_lambda"] for row in regime_rows if row["safe_lambda"] is not None]
        if not observed:
            best_capacity = None
            best_shifts: list[int] = []
        else:
            best_capacity = max(observed)
            best_shifts = [
                row["shift"] for row in regime_rows if row["safe_lambda"] == best_capacity
            ]
        uniform = next(row for row in regime_rows if row["shift"] == 0)
        best_gain_pct = None
        if best_capacity is not None and uniform["safe_lambda"] not in (None, 0):
            best_gain_pct = round((best_capacity / uniform["safe_lambda"] - 1.0) * 100.0, 3)
        regime_summaries[regime_name] = {
            "ttft_s": regime["ttft_s"],
            "tpot_s": regime["tpot_s"],
            "best_safe_lambda": best_capacity,
            "best_shifts": best_shifts,
            "uniform_safe_lambda": uniform["safe_lambda"],
            "best_gain_over_uniform_pct": best_gain_pct,
            "first_violation_at_uniform": uniform["first_violation"],
        }

    decode_best = regime_summaries["decode_constrained"]["best_shifts"]
    prefill_best = regime_summaries["prefill_constrained"]["best_shifts"]
    conclusion = {
        "question": "Do HELIX-derived phase-specific machine profiles make the preferred layer-boundary direction depend on whether TPOT or TTFT is the active SLA constraint?",
        "decode_constrained_prefers_more_t4x4_layers": bool(decode_best) and all(shift < 0 for shift in decode_best),
        "prefill_constrained_prefers_more_l4x2_layers": bool(prefill_best) and all(shift > 0 for shift in prefill_best),
    }
    conclusion["direction_flip"] = (
        conclusion["decode_constrained_prefers_more_t4x4_layers"]
        and conclusion["prefill_constrained_prefers_more_l4x2_layers"]
    )
    conclusion["answer"] = "yes" if conclusion["direction_flip"] else "not_yet"

    summary = {
        "experiment_id": cfg["experiment_id"],
        "provenance": {
            "helix_commit": "8639497a4aaf1eb3b7594614cb0bbd376c1342b3",
            "reference_machine": cfg["phase6"]["reference_machine"],
            "stage_machines": stage_machines,
            "profile_points": {
                "prefill": cfg["phase6"]["prefill_profile_points"],
                "decode": cfg["phase6"]["decode_profile_points"],
            },
        },
        "profile_factors": factors,
        "semantics": "The phase-1-to-5 event-driven evaluator and resource equations are preserved. Only the single scalar stage-speed assumption is replaced by separate Prefill and Decode speed vectors. Those vectors are median A100/machine per-layer time ratios from pinned HELIX tables at workload-relevant profile points. This is a profile-derived heterogeneity sensitivity experiment, not a HELIX scheduler replay or direct hardware execution result.",
        "shift_convention": "positive shift moves one or two layers from each T4x4 stage to the adjacent L4x2 stage; negative shift moves layers toward T4x4",
        "regime_summaries": regime_summaries,
        "conclusion": conclusion,
    }
    log_lines.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, summary, "\n".join(log_lines) + "\n"


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    log: str,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "capacity_by_shift.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase6.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase6"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

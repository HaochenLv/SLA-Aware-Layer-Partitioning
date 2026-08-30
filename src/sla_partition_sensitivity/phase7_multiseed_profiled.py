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
from .phase6_profiled import capacity_record, regime_grid, shifted_partition


def _best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed = [row["safe_lambda"] for row in rows if row["safe_lambda"] is not None]
    if not observed:
        return []
    best = max(observed)
    return [row for row in rows if row["safe_lambda"] == best]


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    factors = derive_phase_speed_factors(cfg)
    stage_machines = cfg["phase6"]["stage_machines"]
    prefill_speeds = [factors[machine]["prefill_speed"] for machine in stage_machines]
    decode_speeds = [factors[machine]["decode_speed"] for machine in stage_machines]

    capacity_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    log_lines = [f"experiment_id={cfg['experiment_id']}"]

    for seed in cfg["phase7"]["workload_seeds"]:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["workload"]["seed"] = seed
        per_regime: dict[str, dict[str, Any]] = {}

        for regime_name, regime in cfg["phase6"]["regimes"].items():
            regime_cfg = copy.deepcopy(seed_cfg)
            regime_cfg["sla"]["ttft_s"] = regime["ttft_s"]
            regime_cfg["sla"]["tpot_s"] = regime["tpot_s"]
            grid = regime_grid(regime)
            local_rows: list[dict[str, Any]] = []

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
                    "seed": seed,
                    "regime": regime_name,
                    "shift": shift,
                    "partition": "-".join(map(str, partition)),
                    "safe_lambda": record["safe_lambda"],
                    "unsafe_lambda": record["unsafe_lambda"],
                    "sampled_monotonic": record["sampled_monotonic"],
                    "first_violation": record["first_violation"],
                }
                capacity_rows.append(row)
                local_rows.append(row)

            best = _best_rows(local_rows)
            uniform = next(row for row in local_rows if row["shift"] == 0)
            best_capacity = best[0]["safe_lambda"] if best else None
            gain_pct = None
            if best_capacity is not None and uniform["safe_lambda"] not in (None, 0):
                gain_pct = (best_capacity / uniform["safe_lambda"] - 1.0) * 100.0
            per_regime[regime_name] = {
                "best_shifts": [row["shift"] for row in best],
                "best_safe_lambda": best_capacity,
                "uniform_safe_lambda": uniform["safe_lambda"],
                "gain_over_uniform_pct": gain_pct,
                "uniform_first_violation": uniform["first_violation"],
                "all_sampled_monotonic": all(row["sampled_monotonic"] for row in local_rows),
            }

        decode = per_regime["decode_constrained"]
        prefill = per_regime["prefill_constrained"]
        decode_prefers_t4x4 = bool(decode["best_shifts"]) and all(
            shift < 0 for shift in decode["best_shifts"]
        )
        prefill_prefers_l4x2 = bool(prefill["best_shifts"]) and all(
            shift > 0 for shift in prefill["best_shifts"]
        )
        direction_flip = decode_prefers_t4x4 and prefill_prefers_l4x2
        seed_row = {
            "seed": seed,
            "decode_best_shifts": ",".join(map(str, decode["best_shifts"])),
            "decode_uniform_safe_lambda": decode["uniform_safe_lambda"],
            "decode_best_safe_lambda": decode["best_safe_lambda"],
            "decode_gain_over_uniform_pct": round(decode["gain_over_uniform_pct"], 3) if decode["gain_over_uniform_pct"] is not None else None,
            "prefill_best_shifts": ",".join(map(str, prefill["best_shifts"])),
            "prefill_uniform_safe_lambda": prefill["uniform_safe_lambda"],
            "prefill_best_safe_lambda": prefill["best_safe_lambda"],
            "prefill_gain_over_uniform_pct": round(prefill["gain_over_uniform_pct"], 3) if prefill["gain_over_uniform_pct"] is not None else None,
            "decode_prefers_more_t4x4_layers": decode_prefers_t4x4,
            "prefill_prefers_more_l4x2_layers": prefill_prefers_l4x2,
            "direction_flip": direction_flip,
            "all_sampled_monotonic": decode["all_sampled_monotonic"] and prefill["all_sampled_monotonic"],
        }
        seed_rows.append(seed_row)
        log_lines.append(json.dumps(seed_row, sort_keys=True))

    flip_count = sum(row["direction_flip"] for row in seed_rows)
    decode_direction_count = sum(row["decode_prefers_more_t4x4_layers"] for row in seed_rows)
    prefill_direction_count = sum(row["prefill_prefers_more_l4x2_layers"] for row in seed_rows)
    decode_gains = [
        row["decode_gain_over_uniform_pct"]
        for row in seed_rows
        if row["decode_gain_over_uniform_pct"] is not None
    ]
    prefill_gains = [
        row["prefill_gain_over_uniform_pct"]
        for row in seed_rows
        if row["prefill_gain_over_uniform_pct"] is not None
    ]
    threshold = cfg["phase7"]["minimum_direction_flips_for_robustness"]
    conclusion = {
        "question": "Does the HELIX-derived SLA-dependent partition-direction flip persist across the six fixed workload seeds?",
        "seeds_tested": len(seed_rows),
        "decode_direction_count": decode_direction_count,
        "prefill_direction_count": prefill_direction_count,
        "direction_flip_count": flip_count,
        "minimum_required": threshold,
        "all_sampled_monotonic": all(row["all_sampled_monotonic"] for row in seed_rows),
        "median_decode_gain_over_uniform_pct": round(statistics.median(decode_gains), 3),
        "median_prefill_gain_over_uniform_pct": round(statistics.median(prefill_gains), 3),
    }
    conclusion["answer"] = (
        "yes"
        if flip_count >= threshold and conclusion["all_sampled_monotonic"]
        else "not_yet"
    )

    summary = {
        "experiment_id": cfg["experiment_id"],
        "semantics": "Same Phase-6 pinned HELIX profile-derived Prefill/Decode speed vectors and controlled TTFT/TPOT regimes; only workload length-sampling seed changes across [0,1,2,3,7,19].",
        "profile_factors": factors,
        "seed_results": seed_rows,
        "conclusion": conclusion,
    }
    log_lines.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return capacity_rows, seed_rows, summary, "\n".join(log_lines) + "\n"


def write_outputs(
    capacity_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    log: str,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "capacity_by_seed_shift.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(capacity_rows[0]))
        writer.writeheader()
        writer.writerows(capacity_rows)
    with (output / "seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase7.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase7"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    capacity_rows, seed_rows, summary, log = run(cfg)
    write_outputs(capacity_rows, seed_rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

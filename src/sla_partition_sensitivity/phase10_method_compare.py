from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .experiment import build_workload, load_config
from .helix_profile import derive_phase_speed_factors
from .phase6_profiled import capacity_record, regime_grid, shifted_partition
from .profiled_model import evaluate_profiled


def weighted_layers(partition: list[int], speeds: list[float]) -> float:
    return sum(n / speed for n, speed in zip(partition, speeds))


def static_baseline_shifts(
    prefill_speeds: list[float], decode_speeds: list[float], cfg: dict[str, Any]
) -> dict[str, int]:
    shifts = cfg["phase10"]["candidate_shifts"]
    records = []
    for shift in shifts:
        partition = shifted_partition(int(shift), cfg)
        records.append(
            {
                "shift": int(shift),
                "prefill": weighted_layers(partition, prefill_speeds),
                "decode": weighted_layers(partition, decode_speeds),
            }
        )
    best_prefill = min(item["prefill"] for item in records)
    best_decode = min(item["decode"] for item in records)
    prefill_shift = min(
        (item for item in records if item["prefill"] == best_prefill),
        key=lambda item: (abs(item["shift"]), item["shift"]),
    )["shift"]
    decode_shift = min(
        (item for item in records if item["decode"] == best_decode),
        key=lambda item: (abs(item["shift"]), item["shift"]),
    )["shift"]
    for item in records:
        item["joint_score"] = max(
            item["prefill"] / best_prefill,
            item["decode"] / best_decode,
        )
    joint_shift = min(
        records,
        key=lambda item: (item["joint_score"], abs(item["shift"]), item["shift"]),
    )["shift"]
    return {
        "uniform": 0,
        "prefill_balanced": prefill_shift,
        "decode_balanced": decode_shift,
        "phase_agnostic_joint_compute": joint_shift,
    }


def probe_shift(
    shift: int,
    reference_lambda: float,
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    partition = shifted_partition(shift, cfg)
    verdict = evaluate_profiled(
        build_workload(cfg, reference_lambda),
        partition,
        prefill_speeds,
        decode_speeds,
        cfg,
    )
    return {
        "shift": shift,
        "safe": verdict.safe,
        "headroom": verdict.minimum_link_headroom_mb_s,
        "first_violation": verdict.first_violation,
    }


def probe_score(record: dict[str, Any]) -> tuple[int, float]:
    return (int(record["safe"]), record["headroom"])


def sla_stress_search(
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
    grid: list[float],
) -> tuple[int, float, list[dict[str, Any]]]:
    uniform = capacity_record(
        shifted_partition(0, cfg), prefill_speeds, decode_speeds, cfg, grid
    )
    reference_lambda = uniform["unsafe_lambda"] or grid[-1]
    probes = [
        probe_shift(shift, reference_lambda, prefill_speeds, decode_speeds, cfg)
        for shift in (-1, 0, 1)
    ]
    best = max(probes, key=lambda item: (probe_score(item), -abs(item["shift"]), -item["shift"]))
    if best["shift"] != 0 and probe_score(best) > probe_score(next(x for x in probes if x["shift"] == 0)):
        extreme = 2 if best["shift"] > 0 else -2
        extra = probe_shift(extreme, reference_lambda, prefill_speeds, decode_speeds, cfg)
        probes.append(extra)
        if probe_score(extra) > probe_score(best):
            best = extra
    return int(best["shift"]), reference_lambda, probes


def _capacity(
    shift: int,
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
    grid: list[float],
) -> dict[str, Any]:
    return capacity_record(
        shifted_partition(shift, cfg), prefill_speeds, decode_speeds, cfg, grid
    )


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    factors = derive_phase_speed_factors(cfg)
    machines = cfg["phase6"]["stage_machines"]
    prefill_speeds = [factors[machine]["prefill_speed"] for machine in machines]
    decode_speeds = [factors[machine]["decode_speed"] for machine in machines]
    baselines = static_baseline_shifts(prefill_speeds, decode_speeds, cfg)

    rows: list[dict[str, Any]] = []
    trial_summaries: list[dict[str, Any]] = []
    log_lines = [f"experiment_id={cfg['experiment_id']}"]
    log_lines.append("static_baselines=" + json.dumps(baselines, sort_keys=True))

    for seed in cfg["phase10"]["workload_seeds"]:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["workload"]["seed"] = seed
        for regime_name, regime in cfg["phase6"]["regimes"].items():
            regime_cfg = copy.deepcopy(seed_cfg)
            regime_cfg["sla"]["ttft_s"] = regime["ttft_s"]
            regime_cfg["sla"]["tpot_s"] = regime["tpot_s"]
            grid = regime_grid(regime)

            all_shift_records: dict[int, dict[str, Any]] = {
                int(shift): _capacity(
                    int(shift), prefill_speeds, decode_speeds, regime_cfg, grid
                )
                for shift in cfg["phase10"]["candidate_shifts"]
            }
            observed = [
                record["safe_lambda"]
                for record in all_shift_records.values()
                if record["safe_lambda"] is not None
            ]
            oracle_capacity = max(observed) if observed else None
            oracle_shifts = [
                shift
                for shift, record in all_shift_records.items()
                if record["safe_lambda"] == oracle_capacity
            ] if oracle_capacity is not None else []

            proposed_shift, reference_lambda, probes = sla_stress_search(
                prefill_speeds, decode_speeds, regime_cfg, grid
            )
            methods = dict(baselines)
            methods["sla_stress_search"] = proposed_shift

            for method, shift in methods.items():
                record = all_shift_records[int(shift)]
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime_name,
                        "method": method,
                        "shift": int(shift),
                        "partition": "-".join(map(str, shifted_partition(int(shift), regime_cfg))),
                        "safe_lambda": record["safe_lambda"],
                        "unsafe_lambda": record["unsafe_lambda"],
                        "sampled_monotonic": record["sampled_monotonic"],
                        "first_violation": record["first_violation"],
                        "reference_lambda": reference_lambda,
                        "oracle_safe_lambda": oracle_capacity,
                        "oracle_shifts": ",".join(map(str, oracle_shifts)),
                    }
                )

            uniform_capacity = all_shift_records[0]["safe_lambda"]
            proposed_capacity = all_shift_records[proposed_shift]["safe_lambda"]
            joint_capacity = all_shift_records[baselines["phase_agnostic_joint_compute"]]["safe_lambda"]
            gain_uniform = None
            gain_joint = None
            if proposed_capacity is not None and uniform_capacity not in (None, 0):
                gain_uniform = (proposed_capacity / uniform_capacity - 1.0) * 100.0
            if proposed_capacity is not None and joint_capacity not in (None, 0):
                gain_joint = (proposed_capacity / joint_capacity - 1.0) * 100.0
            trial = {
                "seed": seed,
                "regime": regime_name,
                "proposed_shift": proposed_shift,
                "oracle_shifts": oracle_shifts,
                "proposed_safe_lambda": proposed_capacity,
                "uniform_safe_lambda": uniform_capacity,
                "joint_compute_safe_lambda": joint_capacity,
                "oracle_safe_lambda": oracle_capacity,
                "gain_over_uniform_pct": round(gain_uniform, 3) if gain_uniform is not None else None,
                "gain_over_joint_compute_pct": round(gain_joint, 3) if gain_joint is not None else None,
                "matches_oracle": proposed_shift in oracle_shifts,
                "improves_uniform": proposed_capacity is not None and uniform_capacity is not None and proposed_capacity > uniform_capacity,
                "not_worse_than_joint_compute": proposed_capacity is not None and joint_capacity is not None and proposed_capacity >= joint_capacity,
                "probe_count": len(probes),
                "probe_trace": probes,
                "all_sampled_monotonic": all(record["sampled_monotonic"] for record in all_shift_records.values()),
            }
            trial_summaries.append(trial)
            log_lines.append(json.dumps(trial, sort_keys=True))

    gains = [
        item["gain_over_uniform_pct"]
        for item in trial_summaries
        if item["gain_over_uniform_pct"] is not None
    ]
    oracle_matches = sum(item["matches_oracle"] for item in trial_summaries)
    uniform_improvements = sum(item["improves_uniform"] for item in trial_summaries)
    joint_not_worse = sum(item["not_worse_than_joint_compute"] for item in trial_summaries)
    conclusion = {
        "question": "Can a four-probe SLA-stress boundary search adapt to HELIX-derived phase-specific heterogeneity and recover near-oracle sampled SLA-safe capacity across both TTFT- and TPOT-sensitive regimes?",
        "trials": len(trial_summaries),
        "oracle_matches": oracle_matches,
        "uniform_improvements": uniform_improvements,
        "not_worse_than_phase_agnostic_joint_compute": joint_not_worse,
        "median_gain_over_uniform_pct": round(statistics.median(gains), 3) if gains else None,
        "max_probe_count": max(item["probe_count"] for item in trial_summaries),
        "all_sampled_monotonic": all(item["all_sampled_monotonic"] for item in trial_summaries),
        "minimum_oracle_matches": cfg["phase10"]["minimum_oracle_matches"],
        "minimum_uniform_improvements": cfg["phase10"]["minimum_uniform_improvements"],
    }
    conclusion["answer"] = (
        "yes"
        if oracle_matches >= cfg["phase10"]["minimum_oracle_matches"]
        and uniform_improvements >= cfg["phase10"]["minimum_uniform_improvements"]
        and joint_not_worse >= cfg["phase10"]["minimum_oracle_matches"]
        and conclusion["all_sampled_monotonic"]
        else "not_yet"
    )
    summary = {
        "experiment_id": cfg["experiment_id"],
        "provenance": {
            "helix_commit": "8639497a4aaf1eb3b7594614cb0bbd376c1342b3",
            "stage_machines": machines,
            "workload_seeds": cfg["phase10"]["workload_seeds"],
        },
        "semantics": "The event-driven evaluator and HELIX-derived phase-specific speed vectors are unchanged. Static compute baselines ignore SLA. The proposed search starts at uniform, probes shifts -1/0/+1 only at the uniform partition's first sampled unsafe workload, then probes one extreme shift in the improving direction. Full capacity is used only for final evaluation, not during the search.",
        "static_baseline_shifts": baselines,
        "trial_summaries": trial_summaries,
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
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase10.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase10"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

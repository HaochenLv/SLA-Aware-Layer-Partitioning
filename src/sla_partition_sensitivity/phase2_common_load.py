from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .diagnostics import evaluate_with_diagnostics
from .experiment import build_workload, generate_partitions, lambda_grid, load_config
from .model import evaluate


def _capacity(
    cfg: dict[str, Any],
    partition: list[int],
    speeds: list[float],
) -> tuple[float | None, float | None]:
    trials = [
        (lam, evaluate(build_workload(cfg, lam), partition, speeds, cfg))
        for lam in lambda_grid(cfg)
    ]
    safe = [lam for lam, verdict in trials if verdict.safe]
    safe_lambda = max(safe) if safe else None
    unsafe = [
        lam
        for lam, verdict in trials
        if not verdict.safe and (safe_lambda is None or lam > safe_lambda)
    ]
    return safe_lambda, min(unsafe) if unsafe else None


def _weighted_layers(partition: list[int], speeds: list[float]) -> float:
    return sum(n / speed for n, speed in zip(partition, speeds))


def _select_representatives(
    cfg: dict[str, Any],
    speeds: list[float],
) -> dict[str, tuple[str, list[int], float | None, float | None, float]]:
    scored = []
    for partition_id, partition in generate_partitions(speeds, cfg):
        safe_lambda, unsafe_lambda = _capacity(cfg, partition, speeds)
        scored.append(
            (
                partition_id,
                partition,
                safe_lambda,
                unsafe_lambda,
                _weighted_layers(partition, speeds),
            )
        )

    observed = [item[2] for item in scored if item[2] is not None]
    best_lambda = max(observed)
    worst_lambda = min(observed)
    uniform = next(item for item in scored if item[0] == "uniform")

    best_candidates = [item for item in scored if item[2] == best_lambda]
    worst_candidates = [item for item in scored if item[2] == worst_lambda]
    best = min(best_candidates, key=lambda item: (item[4], tuple(item[1])))
    worst = max(worst_candidates, key=lambda item: (item[4], tuple(item[1])))

    return {"worst": worst, "uniform": uniform, "best": best}


def _snapshot_for_run(run) -> dict[str, Any]:
    if run.verdict.safe:
        return run.tightest_snapshot or {}
    return run.first_violation_snapshot or run.tightest_snapshot or {}


def _round_value(value: Any, digits: int = 9) -> Any:
    return round(value, digits) if isinstance(value, float) else value


def _make_row(
    scenario: str,
    role: str,
    item: tuple[str, list[int], float | None, float | None, float],
    speeds: list[float],
    reference_lambda: float,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    partition_id, partition, safe_lambda, unsafe_lambda, weighted_layers = item
    workload = build_workload(cfg, reference_lambda)
    run = evaluate_with_diagnostics(workload, partition, speeds, cfg)
    snapshot = _snapshot_for_run(run)

    row: dict[str, Any] = {
        "scenario": scenario,
        "role": role,
        "partition_id": partition_id,
        "partition": "-".join(map(str, partition)),
        "weighted_layers": round(weighted_layers, 6),
        "safe_lambda": safe_lambda,
        "unsafe_lambda": unsafe_lambda,
        "reference_lambda": reference_lambda,
        "safe_at_reference": run.verdict.safe,
        "first_violation_at_reference": run.verdict.first_violation,
        "event_count_at_reference": run.verdict.event_count,
        "minimum_link_headroom_mb_s_at_reference": round(
            run.verdict.minimum_link_headroom_mb_s, 6
        ),
    }
    for key in (
        "event_time_s",
        "event_position",
        "active_requests",
        "active_prefill",
        "active_decode",
        "critical_stage",
        "critical_stage_layers",
        "critical_stage_speed",
        "critical_stage_weighted_layers",
        "max_prefill_compute_s",
        "max_decode_compute_s",
        "min_prefill_residual_s",
        "min_decode_residual_s",
        "min_residual_s",
        "critical_request_id",
        "critical_request_phase",
        "aggregate_link_commitment_mb_s",
        "max_request_commitment_mb_s",
        "network_utilization",
        "link_headroom_mb_s",
    ):
        row[key] = _round_value(snapshot.get(key))
    return row


def _scenario_significant(
    rows: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> bool:
    best = rows["best"]["safe_lambda"]
    worst = rows["worst"]["safe_lambda"]
    if best is None or worst is None or worst <= 0:
        return False
    absolute_gap = best - worst
    relative_gap = (best / worst - 1.0) * 100.0
    return (
        absolute_gap >= cfg["sampling"]["significant_absolute_gap"]
        and relative_gap >= cfg["sampling"]["significant_relative_gap_pct"]
    )


def _greater(a: float | None, b: float | None, tol: float = 1e-12) -> bool:
    return a is not None and b is not None and a > b + tol


def _less(a: float | None, b: float | None, tol: float = 1e-12) -> bool:
    return a is not None and b is not None and a < b - tol


def _equal(a: float | None, b: float | None, tol: float = 1e-12) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def run(
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    log = [f"experiment_id={cfg['experiment_id']}:common-load-v3"]

    scenario_names = cfg.get("phase2", {}).get(
        "scenarios", list(cfg["scenarios"])
    )

    for scenario in scenario_names:
        speeds = cfg["scenarios"][scenario]
        reps = _select_representatives(cfg, speeds)
        uniform = reps["uniform"]
        reference_lambda = (
            uniform[3] if uniform[3] is not None else uniform[2]
        )
        if reference_lambda is None:
            raise RuntimeError(
                f"no sampled reference intensity available for {scenario}"
            )

        scenario_rows: dict[str, dict[str, Any]] = {}
        for role in ("worst", "uniform", "best"):
            row = _make_row(
                scenario,
                role,
                reps[role],
                speeds,
                reference_lambda,
                cfg,
            )
            rows.append(row)
            scenario_rows[role] = row
            log.append(json.dumps(row, sort_keys=True))

        w = scenario_rows["worst"]
        u = scenario_rows["uniform"]
        b = scenario_rows["best"]
        significant = _scenario_significant(scenario_rows, cfg)

        if scenario == "homogeneous_control":
            chain_checks = {
                "capacity_equal": b["safe_lambda"] == u["safe_lambda"],
                "weighted_compute_equal": _equal(
                    b["weighted_layers"], u["weighted_layers"]
                ),
                "network_utilization_equal": _equal(
                    b["network_utilization"], u["network_utilization"]
                ),
            }
            mechanism_supported = all(chain_checks.values())
        else:
            chain_checks = {
                "capacity_higher": (
                    b["safe_lambda"] is not None
                    and u["safe_lambda"] is not None
                    and b["safe_lambda"] > u["safe_lambda"]
                ),
                "decode_compute_lower": _less(
                    b["max_decode_compute_s"], u["max_decode_compute_s"]
                ),
                "decode_residual_higher": _greater(
                    b["min_decode_residual_s"], u["min_decode_residual_s"]
                ),
                "network_utilization_lower": _less(
                    b["network_utilization"], u["network_utilization"]
                ),
                "best_survives_uniform_first_unsafe": (
                    b["safe_at_reference"] and not u["safe_at_reference"]
                ),
            }
            mechanism_supported = all(chain_checks.values())

        def compact(row: dict[str, Any]) -> dict[str, Any]:
            keys = [
                "partition",
                "weighted_layers",
                "safe_lambda",
                "unsafe_lambda",
                "safe_at_reference",
                "first_violation_at_reference",
                "max_decode_compute_s",
                "min_decode_residual_s",
                "network_utilization",
                "minimum_link_headroom_mb_s_at_reference",
                "active_prefill",
                "active_decode",
            ]
            return {key: row.get(key) for key in keys}

        summaries[scenario] = {
            "reference_lambda": reference_lambda,
            "phase1_significant": significant,
            "worst": compact(w),
            "uniform": compact(u),
            "best": compact(b),
            "chain_checks": chain_checks,
            "mechanism_supported": mechanism_supported,
        }

    significant_heterogeneous = [
        scenario
        for scenario, summary in summaries.items()
        if scenario != "homogeneous_control"
        and summary["phase1_significant"]
    ]
    supported = [
        scenario
        for scenario in significant_heterogeneous
        if summaries[scenario]["mechanism_supported"]
    ]
    control_supported = summaries.get(
        "homogeneous_control", {}
    ).get("mechanism_supported", True)
    minimum_supported = cfg.get("phase2", {}).get(
        "minimum_supported_significant_scenarios", 2
    )
    conclusion = {
        "question": (
            "At the same workload intensity, is the phase-1 capacity gain "
            "explained by lower Decode compute cost -> larger residual TPOT "
            "budget -> lower network commitment?"
        ),
        "comparison_rule": (
            "worst, uniform, and best are compared at the uniform partition's "
            "first sampled unsafe intensity; unsafe runs use their first "
            "violation snapshot and safe runs use their tightest network snapshot"
        ),
        "phase1_significant_heterogeneous_scenarios": significant_heterogeneous,
        "mechanism_chain_supported_scenarios": supported,
        "supported_count": len(supported),
        "tested_significant_count": len(significant_heterogeneous),
        "homogeneous_control_consistent": control_supported,
        "answer": (
            "yes"
            if len(supported) >= minimum_supported and control_supported
            else "not_yet"
        ),
    }
    summary = {
        "experiment_id": f"{cfg['experiment_id']}:common-load-v3",
        "semantics": (
            "phase-1 synthetic evaluator with diagnostic-only replay; "
            "no search heuristic and no change to phase-1 verdict semantics"
        ),
        "representative_selection": {
            "uniform": "10 layers per stage",
            "best": (
                "maximum sampled safe lambda; ties choose minimum weighted "
                "compute cost then lexicographic partition"
            ),
            "worst": (
                "minimum sampled safe lambda; ties choose maximum weighted "
                "compute cost then lexicographic partition"
            ),
        },
        "scenario_summaries": summaries,
        "conclusion": conclusion,
    }
    log.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, summary, "\n".join(log) + "\n"


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    log: str,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "common_load_diagnostics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("config/phase2.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/phase2-common-load")
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

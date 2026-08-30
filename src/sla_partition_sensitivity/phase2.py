from __future__ import annotations

import argparse
import csv
import heapq
import json
from pathlib import Path
from typing import Any

from .experiment import build_workload, generate_partitions, lambda_grid, load_config
from .model import ActiveRequest, EPS, Request, _decode_compute_s, _prefill_compute_s, evaluate


def _state_diagnostics(
    active: dict[int, ActiveRequest],
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    np = sum(item.phase == "prefill" for item in active.values())
    nd = sum(item.phase == "decode" for item in active.values())
    sla = cfg["sla"]
    profile = cfg["profile"]
    network = cfg["network"]
    hop_count = cfg["model"]["stages"] - 1

    residuals: list[float] = []
    compute_times: list[float] = []
    request_commitments: list[float] = []
    phase_residuals = {"prefill": [], "decode": []}
    phase_compute = {"prefill": [], "decode": []}

    for item in active.values():
        if item.phase == "prefill":
            compute = item.prefill_compute_s
            intrinsic = sla["intrinsic_prefill_us_per_token"] * item.request.input_tokens / 1_000_000.0
            residual = sla["ttft_s"] - compute - intrinsic - sla["fixed_overhead_s"]
        else:
            compute = _decode_compute_s(
                item.request,
                item.decoded_tokens,
                partition,
                speeds,
                cfg,
                np,
                nd,
            )
            block_delay = (
                profile["decode_block_tokens"]
                * compute
                * profile["prefill_interference_penalty"]
                * np
            )
            residual = sla["tpot_s"] - compute - block_delay - sla["fixed_overhead_s"]

        residuals.append(residual)
        compute_times.append(compute)
        phase_residuals[item.phase].append(residual)
        phase_compute[item.phase].append(compute)
        if residual > EPS:
            per_link_budget = residual / hop_count
            request_commitments.append(network["activation_mb"] / per_link_budget)
        else:
            request_commitments.append(float("inf"))

    aggregate_commitment = sum(request_commitments)
    headroom = network["link_capacity_mb_s"] - aggregate_commitment
    violation = None
    if any(value <= EPS for value in residuals):
        violation = "sla_budget"
    elif aggregate_commitment > network["link_capacity_mb_s"] + EPS:
        violation = "network"

    return {
        "active_requests": len(active),
        "active_prefill": np,
        "active_decode": nd,
        "minimum_residual_s": min(residuals) if residuals else None,
        "mean_residual_s": sum(residuals) / len(residuals) if residuals else None,
        "minimum_prefill_residual_s": min(phase_residuals["prefill"]) if phase_residuals["prefill"] else None,
        "minimum_decode_residual_s": min(phase_residuals["decode"]) if phase_residuals["decode"] else None,
        "maximum_prefill_compute_s": max(phase_compute["prefill"]) if phase_compute["prefill"] else None,
        "maximum_decode_compute_s": max(phase_compute["decode"]) if phase_compute["decode"] else None,
        "aggregate_link_commitment_mb_s": aggregate_commitment,
        "link_capacity_mb_s": network["link_capacity_mb_s"],
        "normalized_link_commitment": aggregate_commitment / network["link_capacity_mb_s"],
        "link_headroom_mb_s": headroom,
        "violation": violation,
    }


def diagnose_first_violation(
    workload: list[Request],
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    queue: list[tuple[float, int, str, int]] = []
    serial = 0
    for request in workload:
        heapq.heappush(queue, (request.arrival_s, serial, "arrival", request.request_id))
        serial += 1
    by_id = {request.request_id: request for request in workload}
    active: dict[int, ActiveRequest] = {}
    event_count = 0

    while queue:
        event_time = queue[0][0]
        tied: list[tuple[float, int, str, int]] = []
        while queue and abs(queue[0][0] - event_time) <= EPS:
            tied.append(heapq.heappop(queue))

        pre = _state_diagnostics(active, partition, speeds, cfg)
        if pre["violation"]:
            return {"event_time_s": event_time, "event_count": event_count, "check_position": "pre", **pre}

        for _, _, kind, request_id in tied:
            if kind == "arrival":
                request = by_id[request_id]
                np = sum(item.phase == "prefill" for item in active.values()) + 1
                nd = sum(item.phase == "decode" for item in active.values())
                compute = _prefill_compute_s(request, partition, speeds, cfg, np, nd)
                active[request_id] = ActiveRequest(request=request, prefill_compute_s=compute)
                heapq.heappush(queue, (event_time + compute, serial, "prefill_done", request_id))
                serial += 1
            elif kind == "prefill_done" and request_id in active:
                active[request_id].phase = "decode"
                np = sum(item.phase == "prefill" for item in active.values())
                nd = sum(item.phase == "decode" for item in active.values())
                token_time = _decode_compute_s(active[request_id].request, 0, partition, speeds, cfg, np, nd)
                block = min(cfg["profile"]["decode_block_tokens"], active[request_id].request.output_tokens)
                heapq.heappush(queue, (event_time + block * token_time, serial, "decode_block", request_id))
                serial += 1
            elif kind == "decode_block" and request_id in active:
                item = active[request_id]
                block = min(cfg["profile"]["decode_block_tokens"], item.request.output_tokens - item.decoded_tokens)
                item.decoded_tokens += block
                if item.decoded_tokens >= item.request.output_tokens:
                    del active[request_id]
                else:
                    np = sum(other.phase == "prefill" for other in active.values())
                    nd = sum(other.phase == "decode" for other in active.values())
                    token_time = _decode_compute_s(item.request, item.decoded_tokens, partition, speeds, cfg, np, nd)
                    next_block = min(cfg["profile"]["decode_block_tokens"], item.request.output_tokens - item.decoded_tokens)
                    heapq.heappush(queue, (event_time + next_block * token_time, serial, "decode_block", request_id))
                    serial += 1

        event_count += 1
        post = _state_diagnostics(active, partition, speeds, cfg)
        if post["violation"]:
            return {"event_time_s": event_time, "event_count": event_count, "check_position": "post", **post}

    return {"event_count": event_count, "violation": None}


def _capacity_for_partition(
    cfg: dict[str, Any], partition: list[int], speeds: list[float]
) -> tuple[float | None, float | None]:
    trials = []
    for intensity in lambda_grid(cfg):
        trials.append((intensity, evaluate(build_workload(cfg, intensity), partition, speeds, cfg)))
    safe_values = [intensity for intensity, verdict in trials if verdict.safe]
    safe_lambda = max(safe_values) if safe_values else None
    unsafe_values = [intensity for intensity, verdict in trials if not verdict.safe and (safe_lambda is None or intensity > safe_lambda)]
    return safe_lambda, min(unsafe_values) if unsafe_values else None


def run_phase2(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    scenario_summaries: dict[str, Any] = {}
    log_lines = ["experiment_id=phase2-bottleneck-mechanism-v1"]

    for scenario, speeds in cfg["scenarios"].items():
        partitions = generate_partitions(speeds, cfg)
        scored = []
        for partition_id, partition in partitions:
            safe_lambda, unsafe_lambda = _capacity_for_partition(cfg, partition, speeds)
            scored.append((partition_id, partition, safe_lambda, unsafe_lambda))
        observed = [item[2] for item in scored if item[2] is not None]
        best_lambda = max(observed)
        worst_lambda = min(observed)
        uniform = next(item for item in scored if item[0] == "uniform")
        best = next(item for item in scored if item[2] == best_lambda)
        worst = next(item for item in scored if item[2] == worst_lambda)
        selections = [("worst", worst), ("uniform", uniform), ("best", best)]

        scenario_rows = []
        for role, (partition_id, partition, safe_lambda, unsafe_lambda) in selections:
            intensity = unsafe_lambda if unsafe_lambda is not None else safe_lambda
            diag = diagnose_first_violation(build_workload(cfg, intensity), partition, speeds, cfg)
            weighted_layers = sum(n / speed for n, speed in zip(partition, speeds))
            row = {
                "scenario": scenario,
                "role": role,
                "partition_id": partition_id,
                "partition": "-".join(map(str, partition)),
                "weighted_layers": round(weighted_layers, 6),
                "safe_lambda": safe_lambda,
                "unsafe_lambda": unsafe_lambda,
                "diagnostic_lambda": intensity,
                **diag,
            }
            rows.append(row)
            scenario_rows.append(row)
            log_lines.append(json.dumps(row, sort_keys=True))

        def role_row(name: str) -> dict[str, Any]:
            return next(row for row in scenario_rows if row["role"] == name)

        worst_row = role_row("worst")
        uniform_row = role_row("uniform")
        best_row = role_row("best")
        mechanism_supported = (
            scenario == "homogeneous_control"
            or (
                best_row["weighted_layers"] <= uniform_row["weighted_layers"] + 1e-12
                and best_row["safe_lambda"] >= uniform_row["safe_lambda"]
                and best_row.get("normalized_link_commitment", float("inf")) <= uniform_row.get("normalized_link_commitment", float("inf")) + 1e-12
            )
        )
        scenario_summaries[scenario] = {
            "worst": {key: worst_row.get(key) for key in ["partition", "weighted_layers", "safe_lambda", "unsafe_lambda", "minimum_residual_s", "maximum_decode_compute_s", "normalized_link_commitment", "violation"]},
            "uniform": {key: uniform_row.get(key) for key in ["partition", "weighted_layers", "safe_lambda", "unsafe_lambda", "minimum_residual_s", "maximum_decode_compute_s", "normalized_link_commitment", "violation"]},
            "best": {key: best_row.get(key) for key in ["partition", "weighted_layers", "safe_lambda", "unsafe_lambda", "minimum_residual_s", "maximum_decode_compute_s", "normalized_link_commitment", "violation"]},
            "mechanism_supported": mechanism_supported,
        }

    heterogeneous = [value for key, value in scenario_summaries.items() if key != "homogeneous_control"]
    conclusion = {
        "question": "Does improved layer allocation raise SLA-safe capacity through lower compute cost, larger residual SLA budget, and lower network commitment?",
        "heterogeneous_scenarios_tested": len(heterogeneous),
        "heterogeneous_scenarios_supporting_mechanism": sum(item["mechanism_supported"] for item in heterogeneous),
        "answer": "yes" if sum(item["mechanism_supported"] for item in heterogeneous) >= 2 else "not_yet",
    }
    summary = {
        "experiment_id": "phase2-bottleneck-mechanism-v1",
        "semantics": "same phase-1 synthetic event-driven evaluator; diagnostics only, no search heuristic",
        "scenario_summaries": scenario_summaries,
        "conclusion": conclusion,
    }
    log_lines.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, summary, "\n".join(log_lines) + "\n"


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], log: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mechanism_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run_phase2(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

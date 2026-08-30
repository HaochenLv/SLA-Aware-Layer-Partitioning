from __future__ import annotations

import argparse
import csv
import heapq
import json
from pathlib import Path
from typing import Any

from .experiment import build_workload, generate_partitions, lambda_grid, load_config
from .model import ActiveRequest, EPS, Request, _decode_compute_s, _prefill_compute_s, evaluate


def _snapshot(active: dict[int, ActiveRequest], partition: list[int], speeds: list[float], cfg: dict[str, Any]) -> dict[str, Any]:
    np = sum(item.phase == "prefill" for item in active.values())
    nd = sum(item.phase == "decode" for item in active.values())
    sla = cfg["sla"]
    profile = cfg["profile"]
    network = cfg["network"]
    hops = cfg["model"]["stages"] - 1
    residuals: list[float] = []
    decode_residuals: list[float] = []
    decode_compute: list[float] = []
    commitments: list[float] = []

    for item in active.values():
        if item.phase == "prefill":
            compute = item.prefill_compute_s
            intrinsic = sla["intrinsic_prefill_us_per_token"] * item.request.input_tokens / 1_000_000.0
            residual = sla["ttft_s"] - compute - intrinsic - sla["fixed_overhead_s"]
        else:
            compute = _decode_compute_s(item.request, item.decoded_tokens, partition, speeds, cfg, np, nd)
            block_delay = profile["decode_block_tokens"] * compute * profile["prefill_interference_penalty"] * np
            residual = sla["tpot_s"] - compute - block_delay - sla["fixed_overhead_s"]
            decode_residuals.append(residual)
            decode_compute.append(compute)
        residuals.append(residual)
        commitments.append(float("inf") if residual <= EPS else network["activation_mb"] / (residual / hops))

    aggregate = sum(commitments)
    return {
        "active_requests": len(active),
        "active_prefill": np,
        "active_decode": nd,
        "minimum_residual_s": min(residuals) if residuals else None,
        "minimum_decode_residual_s": min(decode_residuals) if decode_residuals else None,
        "maximum_decode_compute_s": max(decode_compute) if decode_compute else None,
        "aggregate_link_commitment_mb_s": aggregate,
        "normalized_link_commitment": aggregate / network["link_capacity_mb_s"],
        "link_headroom_mb_s": network["link_capacity_mb_s"] - aggregate,
    }


def diagnose_peak_network(workload: list[Request], partition: list[int], speeds: list[float], cfg: dict[str, Any]) -> dict[str, Any]:
    queue: list[tuple[float, int, str, int]] = []
    serial = 0
    for request in workload:
        heapq.heappush(queue, (request.arrival_s, serial, "arrival", request.request_id))
        serial += 1
    by_id = {request.request_id: request for request in workload}
    active: dict[int, ActiveRequest] = {}
    event_count = 0
    peak: dict[str, Any] | None = None

    def consider(event_time: float, position: str) -> None:
        nonlocal peak
        snap = _snapshot(active, partition, speeds, cfg)
        candidate = {"event_time_s": event_time, "event_count": event_count, "check_position": position, **snap}
        if peak is None or candidate["normalized_link_commitment"] > peak["normalized_link_commitment"]:
            peak = candidate

    while queue:
        event_time = queue[0][0]
        tied: list[tuple[float, int, str, int]] = []
        while queue and abs(queue[0][0] - event_time) <= EPS:
            tied.append(heapq.heappop(queue))
        consider(event_time, "pre")

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
        consider(event_time, "post")

    return peak or {"normalized_link_commitment": 0.0, "link_headroom_mb_s": cfg["network"]["link_capacity_mb_s"]}


def _capacity(cfg: dict[str, Any], partition: list[int], speeds: list[float]) -> tuple[float | None, float | None]:
    trials = [(lam, evaluate(build_workload(cfg, lam), partition, speeds, cfg)) for lam in lambda_grid(cfg)]
    safe = [lam for lam, verdict in trials if verdict.safe]
    safe_lambda = max(safe) if safe else None
    unsafe = [lam for lam, verdict in trials if not verdict.safe and (safe_lambda is None or lam > safe_lambda)]
    return safe_lambda, min(unsafe) if unsafe else None


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    log = ["experiment_id=phase2-common-load-mechanism-v2"]

    for scenario, speeds in cfg["scenarios"].items():
        scored = []
        for partition_id, partition in generate_partitions(speeds, cfg):
            safe_lambda, unsafe_lambda = _capacity(cfg, partition, speeds)
            scored.append((partition_id, partition, safe_lambda, unsafe_lambda))
        observed = [x[2] for x in scored if x[2] is not None]
        uniform = next(x for x in scored if x[0] == "uniform")
        best = next(x for x in scored if x[2] == max(observed))
        worst = next(x for x in scored if x[2] == min(observed))
        reference_lambda = uniform[3] if uniform[3] is not None else uniform[2]
        scenario_rows = []

        for role, item in [("worst", worst), ("uniform", uniform), ("best", best)]:
            partition_id, partition, safe_lambda, unsafe_lambda = item
            workload = build_workload(cfg, reference_lambda)
            verdict = evaluate(workload, partition, speeds, cfg)
            peak = diagnose_peak_network(workload, partition, speeds, cfg)
            row = {
                "scenario": scenario,
                "role": role,
                "partition_id": partition_id,
                "partition": "-".join(map(str, partition)),
                "weighted_layers": round(sum(n / speed for n, speed in zip(partition, speeds)), 6),
                "safe_lambda": safe_lambda,
                "unsafe_lambda": unsafe_lambda,
                "reference_lambda": reference_lambda,
                "safe_at_reference": verdict.safe,
                "first_violation_at_reference": verdict.first_violation,
                "minimum_link_headroom_mb_s_at_reference": round(verdict.minimum_link_headroom_mb_s, 6),
                **peak,
            }
            rows.append(row)
            scenario_rows.append(row)
            log.append(json.dumps(row, sort_keys=True))

        by_role = {row["role"]: row for row in scenario_rows}
        u = by_role["uniform"]
        b = by_role["best"]
        w = by_role["worst"]
        if scenario == "homogeneous_control":
            supported = (
                abs(b["weighted_layers"] - u["weighted_layers"]) <= 1e-12
                and abs(b["normalized_link_commitment"] - u["normalized_link_commitment"]) <= 1e-12
            )
        else:
            supported = (
                b["weighted_layers"] <= u["weighted_layers"] + 1e-12
                and b["normalized_link_commitment"] <= u["normalized_link_commitment"] + 1e-12
                and b["minimum_link_headroom_mb_s_at_reference"] >= u["minimum_link_headroom_mb_s_at_reference"] - 1e-12
                and b["safe_lambda"] >= u["safe_lambda"]
            )
        summaries[scenario] = {
            "reference_lambda": reference_lambda,
            "worst": {k: w.get(k) for k in ["partition", "weighted_layers", "safe_lambda", "safe_at_reference", "normalized_link_commitment", "minimum_decode_residual_s", "maximum_decode_compute_s", "minimum_link_headroom_mb_s_at_reference"]},
            "uniform": {k: u.get(k) for k in ["partition", "weighted_layers", "safe_lambda", "safe_at_reference", "normalized_link_commitment", "minimum_decode_residual_s", "maximum_decode_compute_s", "minimum_link_headroom_mb_s_at_reference"]},
            "best": {k: b.get(k) for k in ["partition", "weighted_layers", "safe_lambda", "safe_at_reference", "normalized_link_commitment", "minimum_decode_residual_s", "maximum_decode_compute_s", "minimum_link_headroom_mb_s_at_reference"]},
            "mechanism_supported": supported,
        }

    hetero = [value for key, value in summaries.items() if key != "homogeneous_control"]
    count = sum(x["mechanism_supported"] for x in hetero)
    conclusion = {
        "question": "At the same workload intensity, does a better partition reduce compute cost and peak network commitment while increasing SLA-safe capacity?",
        "heterogeneous_scenarios_tested": len(hetero),
        "heterogeneous_scenarios_supporting_mechanism": count,
        "answer": "yes" if count >= 3 else "not_yet",
        "comparison_rule": "all roles are compared at the uniform partition's first sampled unsafe intensity",
    }
    summary = {
        "experiment_id": "phase2-common-load-mechanism-v2",
        "semantics": "phase-1 synthetic evaluator; common-load mechanism diagnostic; no search heuristic",
        "scenario_summaries": summaries,
        "conclusion": conclusion,
    }
    log.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, summary, "\n".join(log) + "\n"


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], log: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "common_load_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase2-common-load"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

from .model import (
    EPS,
    ActiveRequest,
    Request,
    Verdict,
    _check_state,
    _decode_compute_s,
    _prefill_compute_s,
    validate_partition,
)


@dataclass(frozen=True)
class DiagnosticRun:
    verdict: Verdict
    tightest_snapshot: dict[str, Any] | None
    first_violation_snapshot: dict[str, Any] | None


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def measure_state(
    active: dict[int, ActiveRequest],
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Measure the compute -> residual-budget -> network-commitment chain.

    This is diagnostic-only. It does not alter the phase-1 evaluator verdict.
    """
    np = sum(item.phase == "prefill" for item in active.values())
    nd = sum(item.phase == "decode" for item in active.values())
    sla = cfg["sla"]
    profile = cfg["profile"]
    network = cfg["network"]

    weighted_by_stage = [n / speed for n, speed in zip(partition, speeds)]
    critical_stage = max(range(len(weighted_by_stage)), key=weighted_by_stage.__getitem__)

    min_prefill_residual = math.inf
    min_decode_residual = math.inf
    max_prefill_compute = 0.0
    max_decode_compute = 0.0
    aggregate_commitment = 0.0
    max_request_commitment = 0.0
    critical_request_id: int | None = None
    critical_request_phase: str | None = None
    critical_residual = math.inf

    hop_count = cfg["model"]["stages"] - 1
    for item in active.values():
        if item.phase == "prefill":
            compute = item.prefill_compute_s
            intrinsic = (
                sla["intrinsic_prefill_us_per_token"]
                * item.request.input_tokens
                / 1_000_000.0
            )
            residual = (
                sla["ttft_s"]
                - compute
                - intrinsic
                - sla["fixed_overhead_s"]
            )
            min_prefill_residual = min(min_prefill_residual, residual)
            max_prefill_compute = max(max_prefill_compute, compute)
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
            residual = (
                sla["tpot_s"]
                - compute
                - block_delay
                - sla["fixed_overhead_s"]
            )
            min_decode_residual = min(min_decode_residual, residual)
            max_decode_compute = max(max_decode_compute, compute)

        if residual < critical_residual:
            critical_residual = residual
            critical_request_id = item.request.request_id
            critical_request_phase = item.phase

        if residual <= EPS:
            request_commitment = math.inf
        else:
            # Phase-1 synthetic topology has equal activation size/capacity on
            # every pipeline hop, so Eq. (22)-(24) gives equal per-link budgets.
            request_commitment = network["activation_mb"] * hop_count / residual
        aggregate_commitment += request_commitment
        max_request_commitment = max(max_request_commitment, request_commitment)

    capacity = network["link_capacity_mb_s"]
    network_utilization = aggregate_commitment / capacity
    headroom = capacity - aggregate_commitment

    return {
        "active_requests": len(active),
        "active_prefill": np,
        "active_decode": nd,
        "weighted_layers": sum(weighted_by_stage),
        "critical_stage": critical_stage,
        "critical_stage_layers": partition[critical_stage],
        "critical_stage_speed": speeds[critical_stage],
        "critical_stage_weighted_layers": weighted_by_stage[critical_stage],
        "max_prefill_compute_s": max_prefill_compute if np else None,
        "max_decode_compute_s": max_decode_compute if nd else None,
        "min_prefill_residual_s": _finite_or_none(min_prefill_residual),
        "min_decode_residual_s": _finite_or_none(min_decode_residual),
        "min_residual_s": _finite_or_none(critical_residual),
        "critical_request_id": critical_request_id,
        "critical_request_phase": critical_request_phase,
        "aggregate_link_commitment_mb_s": _finite_or_none(aggregate_commitment),
        "max_request_commitment_mb_s": _finite_or_none(max_request_commitment),
        "network_utilization": _finite_or_none(network_utilization),
        "link_headroom_mb_s": _finite_or_none(headroom),
    }


def _annotated_snapshot(
    event_time: float,
    event_position: str,
    active: dict[int, ActiveRequest],
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    snapshot = measure_state(active, partition, speeds, cfg)
    snapshot["event_time_s"] = event_time
    snapshot["event_position"] = event_position
    return snapshot


def _is_tighter(candidate: dict[str, Any], current: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    candidate_util = candidate["network_utilization"]
    current_util = current["network_utilization"]
    if candidate_util is None:
        return False
    if current_util is None:
        return True
    return candidate_util > current_util + EPS


def evaluate_with_diagnostics(
    workload: list[Request],
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
) -> DiagnosticRun:
    """Replay phase-1 event semantics while retaining diagnostic snapshots."""
    validate_partition(partition, cfg)
    if len(speeds) != cfg["model"]["stages"] or any(speed <= 0 for speed in speeds):
        raise ValueError("every stage must have a positive compute speed")

    queue: list[tuple[float, int, str, int]] = []
    serial = 0
    for request in workload:
        heapq.heappush(queue, (request.arrival_s, serial, "arrival", request.request_id))
        serial += 1

    by_id = {request.request_id: request for request in workload}
    active: dict[int, ActiveRequest] = {}
    event_count = 0
    peak_prefill = 0
    peak_decode = 0
    minimum_headroom = cfg["network"]["link_capacity_mb_s"]
    peak_memory = 0.0
    tightest_snapshot: dict[str, Any] | None = None

    while queue:
        event_time = queue[0][0]
        tied: list[tuple[float, int, str, int]] = []
        while queue and abs(queue[0][0] - event_time) <= EPS:
            tied.append(heapq.heappop(queue))

        snapshot = _annotated_snapshot(
            event_time, "pre", active, partition, speeds, cfg
        )
        if _is_tighter(snapshot, tightest_snapshot):
            tightest_snapshot = snapshot

        violation, headroom, memory = _check_state(active, partition, speeds, cfg)
        minimum_headroom = min(minimum_headroom, headroom)
        peak_memory = max(peak_memory, memory)
        if violation:
            verdict = Verdict(
                False,
                violation,
                event_count,
                peak_prefill,
                peak_decode,
                minimum_headroom,
                peak_memory,
            )
            return DiagnosticRun(verdict, tightest_snapshot, snapshot)

        for _, _, kind, request_id in tied:
            if kind == "arrival":
                request = by_id[request_id]
                np = sum(item.phase == "prefill" for item in active.values()) + 1
                nd = sum(item.phase == "decode" for item in active.values())
                compute = _prefill_compute_s(
                    request, partition, speeds, cfg, np, nd
                )
                active[request_id] = ActiveRequest(
                    request=request, prefill_compute_s=compute
                )
                heapq.heappush(
                    queue,
                    (event_time + compute, serial, "prefill_done", request_id),
                )
                serial += 1
            elif kind == "prefill_done" and request_id in active:
                active[request_id].phase = "decode"
                np = sum(item.phase == "prefill" for item in active.values())
                nd = sum(item.phase == "decode" for item in active.values())
                token_time = _decode_compute_s(
                    active[request_id].request,
                    0,
                    partition,
                    speeds,
                    cfg,
                    np,
                    nd,
                )
                block = min(
                    cfg["profile"]["decode_block_tokens"],
                    active[request_id].request.output_tokens,
                )
                heapq.heappush(
                    queue,
                    (event_time + block * token_time, serial, "decode_block", request_id),
                )
                serial += 1
            elif kind == "decode_block" and request_id in active:
                item = active[request_id]
                block = min(
                    cfg["profile"]["decode_block_tokens"],
                    item.request.output_tokens - item.decoded_tokens,
                )
                item.decoded_tokens += block
                if item.decoded_tokens >= item.request.output_tokens:
                    del active[request_id]
                else:
                    np = sum(other.phase == "prefill" for other in active.values())
                    nd = sum(other.phase == "decode" for other in active.values())
                    token_time = _decode_compute_s(
                        item.request,
                        item.decoded_tokens,
                        partition,
                        speeds,
                        cfg,
                        np,
                        nd,
                    )
                    next_block = min(
                        cfg["profile"]["decode_block_tokens"],
                        item.request.output_tokens - item.decoded_tokens,
                    )
                    heapq.heappush(
                        queue,
                        (
                            event_time + next_block * token_time,
                            serial,
                            "decode_block",
                            request_id,
                        ),
                    )
                    serial += 1

        event_count += 1
        np = sum(item.phase == "prefill" for item in active.values())
        nd = sum(item.phase == "decode" for item in active.values())
        peak_prefill = max(peak_prefill, np)
        peak_decode = max(peak_decode, nd)

        snapshot = _annotated_snapshot(
            event_time, "post", active, partition, speeds, cfg
        )
        if _is_tighter(snapshot, tightest_snapshot):
            tightest_snapshot = snapshot

        violation, headroom, memory = _check_state(active, partition, speeds, cfg)
        minimum_headroom = min(minimum_headroom, headroom)
        peak_memory = max(peak_memory, memory)
        if violation:
            verdict = Verdict(
                False,
                violation,
                event_count,
                peak_prefill,
                peak_decode,
                minimum_headroom,
                peak_memory,
            )
            return DiagnosticRun(verdict, tightest_snapshot, snapshot)

    verdict = Verdict(
        True,
        None,
        event_count,
        peak_prefill,
        peak_decode,
        minimum_headroom,
        peak_memory,
    )
    return DiagnosticRun(verdict, tightest_snapshot, None)

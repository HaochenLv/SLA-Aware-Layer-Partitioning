from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any


EPS = 1e-12


@dataclass(frozen=True)
class Request:
    request_id: int
    arrival_s: float
    input_tokens: int
    output_tokens: int


@dataclass
class ActiveRequest:
    request: Request
    phase: str = "prefill"
    decoded_tokens: int = 0
    prefill_compute_s: float = 0.0


@dataclass(frozen=True)
class Verdict:
    safe: bool
    first_violation: str | None
    event_count: int
    peak_prefill: int
    peak_decode: int
    minimum_link_headroom_mb_s: float
    peak_memory_gb: float


def validate_partition(partition: list[int], cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    bounds = cfg["partitions"]
    if len(partition) != model["stages"]:
        raise ValueError("partition length must equal stage count")
    if sum(partition) != model["layers"]:
        raise ValueError("partition must assign every layer exactly once")
    if any(n < bounds["min_layers_per_stage"] for n in partition):
        raise ValueError("partition violates the minimum stage size")
    if any(n > bounds["max_layers_per_stage"] for n in partition):
        raise ValueError("partition violates the maximum stage size")
    for n in partition:
        static_gb = (
            n * model["weight_gb_per_layer"]
            + model["workspace_gb"]
            + model["reserve_gb"]
        )
        if static_gb > model["memory_gb_per_stage"] + EPS:
            raise ValueError("partition is statically memory-infeasible")


def _prefill_compute_s(
    request: Request,
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
    active_prefill: int,
    active_decode: int,
) -> float:
    profile = cfg["profile"]
    weighted_layers = sum(n / speed for n, speed in zip(partition, speeds))
    per_layer = (
        profile["prefill_fixed_s_per_layer"]
        + profile["prefill_s_per_input_token_layer"] * request.input_tokens
    )
    concurrency = 1.0 + 0.03 * max(0, active_prefill - 1) + 0.01 * active_decode
    return weighted_layers * per_layer * concurrency


def _decode_compute_s(
    request: Request,
    decoded_tokens: int,
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
    active_prefill: int,
    active_decode: int,
) -> float:
    profile = cfg["profile"]
    weighted_layers = sum(n / speed for n, speed in zip(partition, speeds))
    context = request.input_tokens + decoded_tokens + profile["decode_block_tokens"]
    context_multiplier = 1.0 + profile["decode_context_growth_at_8192"] * context / 8192.0
    concurrency_multiplier = (
        1.0
        + profile["decode_concurrency_penalty"] * max(0, active_decode - 1)
        + profile["prefill_interference_penalty"] * active_prefill
    )
    return (
        weighted_layers
        * profile["decode_s_per_token_layer"]
        * context_multiplier
        * concurrency_multiplier
    )


def _check_state(
    active: dict[int, ActiveRequest],
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
) -> tuple[str | None, float, float]:
    np = sum(item.phase == "prefill" for item in active.values())
    nd = sum(item.phase == "decode" for item in active.values())
    sla = cfg["sla"]
    profile = cfg["profile"]
    network = cfg["network"]

    link_commitments = [0.0] * (cfg["model"]["stages"] - 1)
    for item in active.values():
        if item.phase == "prefill":
            intrinsic = (
                sla["intrinsic_prefill_us_per_token"]
                * item.request.input_tokens
                / 1_000_000.0
            )
            residual = (
                sla["ttft_s"]
                - item.prefill_compute_s
                - intrinsic
                - sla["fixed_overhead_s"]
            )
        else:
            decode_compute = _decode_compute_s(
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
                * decode_compute
                * profile["prefill_interference_penalty"]
                * np
            )
            residual = (
                sla["tpot_s"]
                - decode_compute
                - block_delay
                - sla["fixed_overhead_s"]
            )
        if residual <= EPS:
            return "sla_budget", float("-inf"), 0.0

        # All links carry the same activation size. Eq. (22)-(24) therefore
        # splits the residual budget equally over the seven-link path.
        hop_count = len(link_commitments)
        per_link_budget_s = residual / hop_count
        commitment = network["activation_mb"] / per_link_budget_s
        for link in range(hop_count):
            link_commitments[link] += commitment

    min_headroom = min(
        (network["link_capacity_mb_s"] - value for value in link_commitments),
        default=network["link_capacity_mb_s"],
    )
    if min_headroom < -EPS:
        return "network", min_headroom, 0.0

    peak_memory = 0.0
    mem = cfg["memory"]
    model = cfg["model"]
    for stage, stage_layers in enumerate(partition):
        usage = (
            stage_layers * model["weight_gb_per_layer"]
            + model["workspace_gb"]
            + model["reserve_gb"]
        )
        for item in active.values():
            context = item.request.input_tokens + item.decoded_tokens
            usage += stage_layers * context * mem["kv_gb_per_token_layer"]
            if item.phase == "prefill":
                usage += stage_layers * mem["activation_gb_per_prefill_layer"]
        peak_memory = max(peak_memory, usage)
        if usage > model["memory_gb_per_stage"] + EPS:
            return f"memory_stage_{stage}", min_headroom, peak_memory
    return None, min_headroom, peak_memory


def evaluate(
    workload: list[Request],
    partition: list[int],
    speeds: list[float],
    cfg: dict[str, Any],
) -> Verdict:
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

    while queue:
        event_time = queue[0][0]
        tied: list[tuple[float, int, str, int]] = []
        while queue and abs(queue[0][0] - event_time) <= EPS:
            tied.append(heapq.heappop(queue))

        violation, headroom, memory = _check_state(active, partition, speeds, cfg)
        minimum_headroom = min(minimum_headroom, headroom)
        peak_memory = max(peak_memory, memory)
        if violation:
            return Verdict(False, violation, event_count, peak_prefill, peak_decode, minimum_headroom, peak_memory)

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
                token_time = _decode_compute_s(
                    active[request_id].request,
                    0,
                    partition,
                    speeds,
                    cfg,
                    np,
                    nd,
                )
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
                    token_time = _decode_compute_s(
                        item.request,
                        item.decoded_tokens,
                        partition,
                        speeds,
                        cfg,
                        np,
                        nd,
                    )
                    next_block = min(cfg["profile"]["decode_block_tokens"], item.request.output_tokens - item.decoded_tokens)
                    heapq.heappush(queue, (event_time + next_block * token_time, serial, "decode_block", request_id))
                    serial += 1

        event_count += 1
        np = sum(item.phase == "prefill" for item in active.values())
        nd = sum(item.phase == "decode" for item in active.values())
        peak_prefill = max(peak_prefill, np)
        peak_decode = max(peak_decode, nd)
        violation, headroom, memory = _check_state(active, partition, speeds, cfg)
        minimum_headroom = min(minimum_headroom, headroom)
        peak_memory = max(peak_memory, memory)
        if violation:
            return Verdict(False, violation, event_count, peak_prefill, peak_decode, minimum_headroom, peak_memory)

    return Verdict(True, None, event_count, peak_prefill, peak_decode, minimum_headroom, peak_memory)

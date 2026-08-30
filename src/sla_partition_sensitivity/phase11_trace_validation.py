from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import pickle
import random
import statistics
from pathlib import Path
from typing import Any

from .experiment import load_config
from .helix_profile import derive_phase_speed_factors
from .model import Request
from .phase6_profiled import shifted_partition
from .phase10_method_compare import static_baseline_shifts
from .profiled_model import evaluate_profiled


class PrimitiveOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"pickle global forbidden: {module}.{name}")


def _load_primitive_list(path: Path, expected_type: type) -> list[Any]:
    with path.open("rb") as handle:
        value = PrimitiveOnlyUnpickler(handle).load()
    if not isinstance(value, list) or not value:
        raise ValueError(f"expected non-empty primitive list: {path}")
    if any(type(item) is not expected_type for item in value):
        raise ValueError(f"unexpected element type in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_trace_inputs(cfg: dict[str, Any]) -> dict[str, str]:
    root = Path(cfg["phase11"]["runtime_trace_root"])
    observed: dict[str, str] = {}
    for name, expected in cfg["phase11"]["expected_sha256"].items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"missing runtime HELIX workload artifact: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {name}: {actual} != {expected}")
        observed[name] = actual
    return observed


def build_helix_workload(cfg: dict[str, Any], seed: int) -> list[Request]:
    spec = cfg["phase11"]
    duration_s = int(spec["duration_s"])
    if duration_s <= 0 or duration_s % 3:
        raise ValueError("duration_s must be a positive multiple of 3")
    target_rate = float(spec["target_request_rate"])
    root = Path(spec["runtime_trace_root"])
    arrivals = _load_primitive_list(root / "azure_conv_arrive_time.pkl", int)
    input_lengths = _load_primitive_list(root / "azure_conv_input.pkl", int)
    output_lengths = _load_primitive_list(root / "azure_conv_output.pkl", int)
    if len(arrivals) != 1200:
        raise ValueError("HELIX Azure arrival series must have 1200 three-second intervals")
    if len(input_lengths) != len(output_lengths):
        raise ValueError("HELIX Azure input/output length arrays must remain paired")

    interval_count = duration_s // 3
    offset = int(spec["interval_offset"])
    selected = [arrivals[(offset + i) % len(arrivals)] for i in range(interval_count)]
    source_mean = sum(arrivals) / len(arrivals)
    scale = 3.0 * target_rate / source_mean
    residual = 0.0
    rng = random.Random(seed)
    requests: list[Request] = []
    for interval_index, raw_count in enumerate(selected):
        scaled = raw_count * scale + residual
        count = max(round(scaled), 0)
        residual = scaled - count
        for position in range(count):
            arrival_s = interval_index * 3.0 + (position + 1) * 3.0 / (count + 1)
            index = rng.randrange(len(input_lengths))
            requests.append(
                Request(
                    request_id=len(requests),
                    arrival_s=arrival_s,
                    input_tokens=input_lengths[index],
                    output_tokens=output_lengths[index],
                )
            )
    if not requests:
        raise ValueError("selected HELIX workload window produced no requests")
    return requests


def scale_workload(workload: list[Request], intensity: float) -> list[Request]:
    if intensity <= 0:
        raise ValueError("intensity must be positive")
    origin = min(request.arrival_s for request in workload)
    return [
        Request(
            request_id=request.request_id,
            arrival_s=origin + (request.arrival_s - origin) / intensity,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
        )
        for request in workload
    ]


def _evaluate(
    base_workload: list[Request],
    intensity: float,
    shift: int,
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
):
    return evaluate_profiled(
        scale_workload(base_workload, intensity),
        shifted_partition(shift, cfg),
        prefill_speeds,
        decode_speeds,
        cfg,
    )


def adaptive_capacity(
    base_workload: list[Request],
    shift: int,
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    spec = cfg["phase11"]
    initial = float(spec["capacity_initial_intensity"])
    maximum = float(spec["capacity_max_intensity"])
    tolerance = float(spec["capacity_relative_tolerance"])
    cache: dict[float, Any] = {}

    def run(intensity: float):
        key = round(float(intensity), 12)
        if key not in cache:
            cache[key] = _evaluate(
                base_workload, key, shift, prefill_speeds, decode_speeds, cfg
            )
        return cache[key]

    high = initial
    high_verdict = run(high)
    if high_verdict.safe:
        low = high
        low_verdict = high_verdict
        while high < maximum:
            high = min(high * 2.0, maximum)
            high_verdict = run(high)
            if not high_verdict.safe:
                break
            low = high
            low_verdict = high_verdict
            if high >= maximum:
                ordered = sorted(cache.items())
                return {
                    "safe_intensity": low,
                    "unsafe_intensity": None,
                    "right_censored": True,
                    "sampled_monotonic": True,
                    "first_violation": None,
                    "evaluations": len(cache),
                }
    else:
        unsafe = high
        high_verdict = run(unsafe)
        low = high / 2.0
        low_verdict = run(low)
        while not low_verdict.safe and low > 1e-6:
            unsafe = low
            high_verdict = low_verdict
            low /= 2.0
            low_verdict = run(low)
        if not low_verdict.safe:
            return {
                "safe_intensity": None,
                "unsafe_intensity": unsafe,
                "right_censored": False,
                "sampled_monotonic": True,
                "first_violation": high_verdict.first_violation,
                "evaluations": len(cache),
            }
        high = unsafe

    while high - low > tolerance * max(high, 1.0):
        mid = (low + high) / 2.0
        verdict = run(mid)
        if verdict.safe:
            low = mid
        else:
            high = mid
            high_verdict = verdict

    ordered = sorted(cache.items())
    seen_unsafe = False
    monotonic = True
    for _, verdict in ordered:
        if verdict.safe and seen_unsafe:
            monotonic = False
            break
        if not verdict.safe:
            seen_unsafe = True
    return {
        "safe_intensity": round(low, 6),
        "unsafe_intensity": round(high, 6),
        "right_censored": False,
        "sampled_monotonic": monotonic,
        "first_violation": high_verdict.first_violation,
        "evaluations": len(cache),
    }


def _probe_score(verdict: Any) -> tuple[int, float]:
    return (int(verdict.safe), verdict.minimum_link_headroom_mb_s)


def sla_stress_search(
    base_workload: list[Request],
    uniform_capacity: dict[str, Any],
    prefill_speeds: list[float],
    decode_speeds: list[float],
    cfg: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    reference = uniform_capacity["unsafe_intensity"]
    if reference is None:
        reference = uniform_capacity["safe_intensity"]
    if reference is None:
        raise ValueError("uniform capacity failed to provide a stress intensity")

    probes: list[tuple[int, Any]] = []
    for shift in (-1, 0, 1):
        probes.append(
            (
                shift,
                _evaluate(
                    base_workload, reference, shift, prefill_speeds, decode_speeds, cfg
                ),
            )
        )
    uniform_verdict = next(verdict for shift, verdict in probes if shift == 0)
    best_shift, best_verdict = max(
        probes,
        key=lambda item: (_probe_score(item[1]), -abs(item[0]), -item[0]),
    )
    if best_shift != 0 and _probe_score(best_verdict) > _probe_score(uniform_verdict):
        extreme = 2 if best_shift > 0 else -2
        extreme_verdict = _evaluate(
            base_workload, reference, extreme, prefill_speeds, decode_speeds, cfg
        )
        probes.append((extreme, extreme_verdict))
        if _probe_score(extreme_verdict) > _probe_score(best_verdict):
            best_shift, best_verdict = extreme, extreme_verdict

    public = [
        {
            "shift": shift,
            "safe": verdict.safe,
            "first_violation": verdict.first_violation,
            "minimum_link_headroom_mb_s": round(verdict.minimum_link_headroom_mb_s, 6),
        }
        for shift, verdict in probes
    ]
    return best_shift, public


def _direction(shifts: list[int]) -> str:
    if shifts and all(shift > 0 for shift in shifts):
        return "l4x2_heavy"
    if shifts and all(shift < 0 for shift in shifts):
        return "t4x4_heavy"
    if shifts == [0]:
        return "uniform"
    return "tie_or_mixed"


def _workload_stats(workload: list[Request]) -> dict[str, Any]:
    inputs = [request.input_tokens for request in workload]
    outputs = [request.output_tokens for request in workload]
    arrivals = [request.arrival_s for request in workload]
    return {
        "request_count": len(workload),
        "first_arrival_s": round(min(arrivals), 6),
        "last_arrival_s": round(max(arrivals), 6),
        "input_mean": round(statistics.mean(inputs), 3),
        "input_median": statistics.median(inputs),
        "input_min": min(inputs),
        "input_max": max(inputs),
        "output_mean": round(statistics.mean(outputs), 3),
        "output_median": statistics.median(outputs),
        "output_min": min(outputs),
        "output_max": max(outputs),
    }


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    checksums = verify_trace_inputs(cfg)
    factors = derive_phase_speed_factors(cfg)
    machines = cfg["phase6"]["stage_machines"]
    prefill_speeds = [factors[machine]["prefill_speed"] for machine in machines]
    decode_speeds = [factors[machine]["decode_speed"] for machine in machines]
    static_shifts = static_baseline_shifts(prefill_speeds, decode_speeds, cfg)

    rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    workload_summaries: list[dict[str, Any]] = []
    log_lines = [f"experiment_id={cfg['experiment_id']}"]

    for seed in cfg["phase11"]["workload_seeds"]:
        base = build_helix_workload(cfg, int(seed))
        workload_summaries.append({"seed": seed, **_workload_stats(base)})
        for regime_name, regime in cfg["phase11"]["regimes"].items():
            regime_cfg = copy.deepcopy(cfg)
            regime_cfg["sla"]["ttft_s"] = regime["ttft_s"]
            regime_cfg["sla"]["tpot_s"] = regime["tpot_s"]

            capacities = {
                shift: adaptive_capacity(
                    base, shift, prefill_speeds, decode_speeds, regime_cfg
                )
                for shift in cfg["phase11"]["candidate_shifts"]
            }
            observed = [
                record["safe_intensity"]
                for record in capacities.values()
                if record["safe_intensity"] is not None
            ]
            oracle_capacity = max(observed) if observed else None
            oracle_shifts = [
                int(shift)
                for shift, record in capacities.items()
                if record["safe_intensity"] == oracle_capacity
            ] if oracle_capacity is not None else []

            proposed_shift, probes = sla_stress_search(
                base,
                capacities[0],
                prefill_speeds,
                decode_speeds,
                regime_cfg,
            )
            methods = dict(static_shifts)
            methods["sla_stress_search"] = proposed_shift
            for method, shift in methods.items():
                record = capacities[int(shift)]
                rows.append(
                    {
                        "seed": seed,
                        "regime": regime_name,
                        "method": method,
                        "shift": int(shift),
                        "partition": "-".join(map(str, shifted_partition(int(shift), regime_cfg))),
                        "safe_intensity": record["safe_intensity"],
                        "unsafe_intensity": record["unsafe_intensity"],
                        "first_violation": record["first_violation"],
                        "sampled_monotonic": record["sampled_monotonic"],
                        "oracle_safe_intensity": oracle_capacity,
                        "oracle_shifts": ",".join(map(str, oracle_shifts)),
                    }
                )

            uniform_capacity = capacities[0]["safe_intensity"]
            proposed_capacity = capacities[proposed_shift]["safe_intensity"]
            gain = None
            if proposed_capacity is not None and uniform_capacity not in (None, 0):
                gain = (proposed_capacity / uniform_capacity - 1.0) * 100.0
            trial = {
                "seed": seed,
                "regime": regime_name,
                "request_count": len(base),
                "oracle_direction": _direction(oracle_shifts),
                "oracle_shifts": oracle_shifts,
                "oracle_safe_intensity": oracle_capacity,
                "proposed_shift": proposed_shift,
                "proposed_safe_intensity": proposed_capacity,
                "uniform_safe_intensity": uniform_capacity,
                "gain_over_uniform_pct": round(gain, 3) if gain is not None else None,
                "matches_oracle": proposed_shift in oracle_shifts,
                "improves_uniform": proposed_capacity is not None and uniform_capacity is not None and proposed_capacity > uniform_capacity,
                "probe_count": len(probes),
                "probe_trace": probes,
                "all_sampled_monotonic": all(record["sampled_monotonic"] for record in capacities.values()),
                "no_right_censoring": not any(record["right_censored"] for record in capacities.values()),
            }
            trials.append(trial)
            log_lines.append(json.dumps(trial, sort_keys=True))

    per_seed = []
    for seed in cfg["phase11"]["workload_seeds"]:
        decode = next(item for item in trials if item["seed"] == seed and item["regime"] == "decode_constrained")
        prefill = next(item for item in trials if item["seed"] == seed and item["regime"] == "prefill_constrained")
        per_seed.append(
            {
                "seed": seed,
                "direction_flip": decode["oracle_direction"] == "t4x4_heavy" and prefill["oracle_direction"] == "l4x2_heavy",
                "decode_direction": decode["oracle_direction"],
                "prefill_direction": prefill["oracle_direction"],
            }
        )

    direction_flips = sum(item["direction_flip"] for item in per_seed)
    oracle_matches = sum(item["matches_oracle"] for item in trials)
    gains = [item["gain_over_uniform_pct"] for item in trials if item["gain_over_uniform_pct"] is not None]
    conclusion = {
        "question": "Do the HELIX-derived SLA-dependent partition direction and low-probe search remain effective when the simplified synthetic workload is replaced by a deterministic 30-second Azure Conversation workload generated from pinned HELIX interval counts and paired lengths?",
        "seeds_tested": len(cfg["phase11"]["workload_seeds"]),
        "trials": len(trials),
        "direction_flip_count": direction_flips,
        "oracle_matches": oracle_matches,
        "uniform_improvements": sum(item["improves_uniform"] for item in trials),
        "median_gain_over_uniform_pct": round(statistics.median(gains), 3) if gains else None,
        "max_probe_count": max(item["probe_count"] for item in trials),
        "all_sampled_monotonic": all(item["all_sampled_monotonic"] for item in trials),
        "no_right_censoring": all(item["no_right_censoring"] for item in trials),
        "same_arrival_schedule_across_seeds": len({(item["request_count"], item["first_arrival_s"], item["last_arrival_s"]) for item in workload_summaries}) == 1,
        "minimum_direction_flips": cfg["phase11"]["minimum_direction_flips"],
        "minimum_oracle_matches": cfg["phase11"]["minimum_oracle_matches"],
    }
    conclusion["answer"] = (
        "yes"
        if direction_flips >= cfg["phase11"]["minimum_direction_flips"]
        and oracle_matches >= cfg["phase11"]["minimum_oracle_matches"]
        and conclusion["all_sampled_monotonic"]
        and conclusion["no_right_censoring"]
        and conclusion["same_arrival_schedule_across_seeds"]
        else "not_yet"
    )
    summary = {
        "experiment_id": cfg["experiment_id"],
        "provenance": {
            "source_repository": "HaochenLv/sla-aware-evaluator",
            "source_repo_commit": cfg["phase11"]["source_repo_commit"],
            "upstream_helix_commit": cfg["phase11"]["upstream_helix_commit"],
            "artifact_sha256": checksums,
            "workload_kind": "generated_from_sequential_HELIX_AzureConversation_interval_counts_and_paired_length_distribution",
        },
        "semantics": "The runtime workload generator reproduces the prior evaluator repository's deterministic finite-workload construction: a sequential 30-second window of HELIX Azure Conversation 3-second interval counts is globally rescaled, residual-rounded, evenly placed within each interval, and paired input/output lengths are sampled with fixed seeds. Capacity then scales only inter-arrival times, preserving request order and lengths. These are Azure-derived generated traces, not raw production timestamps. The event-driven evaluator, network commitment equations, and HELIX-derived Prefill/Decode speed vectors are unchanged.",
        "static_baseline_shifts": static_shifts,
        "workload_summaries": workload_summaries,
        "seed_direction_summaries": per_seed,
        "trial_summaries": trials,
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
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase11_trace.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase11"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

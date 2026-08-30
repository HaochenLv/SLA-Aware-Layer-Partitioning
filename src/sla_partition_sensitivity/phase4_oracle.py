from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterator

from .experiment import _compute_balanced_partition, load_config
from .model import validate_partition
from .phase3 import _full_capacity, boundary_local_search, uniform_partition


def enumerate_partitions(cfg: dict[str, Any]) -> Iterator[list[int]]:
    """Enumerate every feasible contiguous stage-size vector for the reduced oracle case."""
    layers = cfg["model"]["layers"]
    stages = cfg["model"]["stages"]
    lo = cfg["partitions"]["min_layers_per_stage"]
    hi = cfg["partitions"]["max_layers_per_stage"]

    def rec(prefix: list[int], remaining_layers: int, remaining_stages: int) -> Iterator[list[int]]:
        if remaining_stages == 1:
            candidate = prefix + [remaining_layers]
            if lo <= remaining_layers <= hi:
                try:
                    validate_partition(candidate, cfg)
                except ValueError:
                    return
                yield candidate
            return

        minimum_rest = lo * (remaining_stages - 1)
        maximum_rest = hi * (remaining_stages - 1)
        lower = max(lo, remaining_layers - maximum_rest)
        upper = min(hi, remaining_layers - minimum_rest)
        for n in range(lower, upper + 1):
            yield from rec(prefix + [n], remaining_layers - n, remaining_stages - 1)

    yield from rec([], layers, stages)


def _headroom(record: dict[str, Any], reference_lambda: float) -> float:
    return record["verdicts"][reference_lambda].minimum_link_headroom_mb_s


def _capacity_value(record: dict[str, Any]) -> float:
    value = record["safe_lambda"]
    return float("-inf") if value is None else value


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    scenario_summaries: dict[str, Any] = {}
    log = ["experiment_id=phase4-reduced-exhaustive-oracle-v1"]
    all_partitions = list(enumerate_partitions(cfg))
    expected = cfg.get("phase4", {}).get("expected_partition_count")
    if expected is not None and len(all_partitions) != expected:
        raise RuntimeError(f"expected {expected} partitions, enumerated {len(all_partitions)}")

    for scenario, speeds in cfg["scenarios"].items():
        search_partition, history, search_probes, reference_lambda = boundary_local_search(speeds, cfg)
        full_cache: dict[tuple[int, ...], dict[str, Any]] = {}
        exhaustive = [_full_capacity(p, speeds, cfg, full_cache) for p in all_partitions]
        oracle = max(
            exhaustive,
            key=lambda record: (
                _capacity_value(record),
                _headroom(record, reference_lambda),
                -record["weighted_layers"],
                tuple(record["partition"]),
            ),
        )
        search = _full_capacity(search_partition, speeds, cfg, full_cache)
        uniform = _full_capacity(uniform_partition(cfg), speeds, cfg, full_cache)
        balanced = _full_capacity(_compute_balanced_partition(speeds, cfg), speeds, cfg, full_cache)

        step = cfg["sampling"]["lambda_step"]
        oracle_capacity = _capacity_value(oracle)
        search_capacity = _capacity_value(search)
        gap_steps = None
        if oracle_capacity != float("-inf") and search_capacity != float("-inf"):
            gap_steps = round((oracle_capacity - search_capacity) / step)

        summary = {
            "reference_lambda": reference_lambda,
            "feasible_partition_count": len(all_partitions),
            "search_partition_probes": search_probes,
            "search_iterations": len(history) - 1,
            "uniform": {
                "partition": uniform["partition_text"],
                "safe_lambda": uniform["safe_lambda"],
                "headroom_at_reference_mb_s": round(_headroom(uniform, reference_lambda), 6),
            },
            "compute_balanced": {
                "partition": balanced["partition_text"],
                "safe_lambda": balanced["safe_lambda"],
                "headroom_at_reference_mb_s": round(_headroom(balanced, reference_lambda), 6),
            },
            "sla_boundary_search": {
                "partition": search["partition_text"],
                "safe_lambda": search["safe_lambda"],
                "headroom_at_reference_mb_s": round(_headroom(search, reference_lambda), 6),
            },
            "exhaustive_oracle": {
                "partition": oracle["partition_text"],
                "safe_lambda": oracle["safe_lambda"],
                "headroom_at_reference_mb_s": round(_headroom(oracle, reference_lambda), 6),
            },
            "checks": {
                "search_capacity_optimal": gap_steps == 0,
                "search_within_one_lambda_step": gap_steps is not None and gap_steps <= 1,
                "search_not_worse_than_compute_balanced": search_capacity >= _capacity_value(balanced),
                "homogeneous_no_false_gain": (
                    scenario != "homogeneous_control"
                    or search["safe_lambda"] == uniform["safe_lambda"]
                ),
            },
            "capacity_gap_lambda_steps": gap_steps,
        }
        scenario_summaries[scenario] = summary
        log.append(json.dumps({"scenario": scenario, **summary}, sort_keys=True))

        for role, record in [
            ("uniform", uniform),
            ("compute_balanced", balanced),
            ("sla_boundary_search", search),
            ("exhaustive_oracle", oracle),
        ]:
            rows.append(
                {
                    "scenario": scenario,
                    "role": role,
                    "partition": record["partition_text"],
                    "safe_lambda": record["safe_lambda"],
                    "unsafe_lambda": record["unsafe_lambda"],
                    "reference_lambda": reference_lambda,
                    "headroom_at_reference_mb_s": round(_headroom(record, reference_lambda), 6),
                    "weighted_layers": round(record["weighted_layers"], 6),
                    "search_partition_probes": search_probes if role == "sla_boundary_search" else "",
                }
            )

    non_control = [v for k, v in scenario_summaries.items() if k != "homogeneous_control"]
    conclusion = {
        "question": "On a fully enumerable reduced partition space, how close is the stress-probe boundary search to the true sampled-capacity optimum?",
        "reduced_partition_space_size": len(all_partitions),
        "heterogeneous_scenarios_tested": len(non_control),
        "heterogeneous_scenarios_capacity_optimal": sum(x["checks"]["search_capacity_optimal"] for x in non_control),
        "heterogeneous_scenarios_within_one_lambda_step": sum(x["checks"]["search_within_one_lambda_step"] for x in non_control),
        "homogeneous_control_consistent": scenario_summaries["homogeneous_control"]["checks"]["homogeneous_no_false_gain"],
    }
    conclusion["answer"] = (
        "yes"
        if conclusion["heterogeneous_scenarios_within_one_lambda_step"] == len(non_control)
        and conclusion["homogeneous_control_consistent"]
        else "not_yet"
    )
    summary = {
        "experiment_id": "phase4-reduced-exhaustive-oracle-v1",
        "semantics": "synthetic 4-stage reduced abstraction scaled to preserve approximately the 8-stage experiment's total compute and path-level network pressure; exhaustive oracle is for search-quality validation, not hardware realism",
        "scenario_summaries": scenario_summaries,
        "conclusion": conclusion,
    }
    log.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, summary, "\n".join(log) + "\n"


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], log: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "oracle_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase4_oracle.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase4-oracle"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

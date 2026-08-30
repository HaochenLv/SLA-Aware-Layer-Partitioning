from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

from .experiment import _compute_balanced_partition, load_config
from .phase3 import _full_capacity, boundary_local_search, uniform_partition


def _capacity(record: dict[str, Any]) -> float | None:
    return record["safe_lambda"]


def _not_worse(a: float | None, b: float | None) -> bool:
    if a is None:
        return b is None
    if b is None:
        return True
    return a >= b


def _strictly_better(a: float | None, b: float | None) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    return a > b


def run(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    log = ["experiment_id=phase5-six-seed-robustness-v1"]
    phase5 = cfg["phase5"]
    seeds = phase5["workload_seeds"]
    scenarios = phase5["target_scenarios"]

    for seed in seeds:
        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["workload"]["seed"] = seed
        for scenario in scenarios:
            speeds = seed_cfg["scenarios"][scenario]
            search_partition, history, probes, reference_lambda = boundary_local_search(speeds, seed_cfg)
            cache: dict[tuple[int, ...], dict[str, Any]] = {}
            uniform = _full_capacity(uniform_partition(seed_cfg), speeds, seed_cfg, cache)
            balanced = _full_capacity(_compute_balanced_partition(speeds, seed_cfg), speeds, seed_cfg, cache)
            search = _full_capacity(search_partition, speeds, seed_cfg, cache)

            row = {
                "seed": seed,
                "scenario": scenario,
                "reference_lambda": reference_lambda,
                "search_partition": search["partition_text"],
                "search_iterations": len(history) - 1,
                "search_partition_probes": probes,
                "uniform_partition": uniform["partition_text"],
                "uniform_safe_lambda": _capacity(uniform),
                "compute_balanced_partition": balanced["partition_text"],
                "compute_balanced_safe_lambda": _capacity(balanced),
                "search_safe_lambda": _capacity(search),
                "search_improves_uniform": _strictly_better(_capacity(search), _capacity(uniform)),
                "search_matches_or_beats_balanced": _not_worse(_capacity(search), _capacity(balanced)),
                "search_headroom_at_reference_mb_s": round(
                    search["verdicts"][reference_lambda].minimum_link_headroom_mb_s, 6
                ),
                "uniform_headroom_at_reference_mb_s": round(
                    uniform["verdicts"][reference_lambda].minimum_link_headroom_mb_s, 6
                ),
            }
            if _capacity(search) is not None and _capacity(uniform) not in (None, 0):
                row["relative_capacity_gain_pct"] = round(
                    100.0 * (_capacity(search) - _capacity(uniform)) / _capacity(uniform), 6
                )
            else:
                row["relative_capacity_gain_pct"] = None
            rows.append(row)
            log.append(json.dumps(row, sort_keys=True))

    by_scenario: dict[str, Any] = {}
    for scenario in scenarios:
        subset = [row for row in rows if row["scenario"] == scenario]
        gains = [row["relative_capacity_gain_pct"] for row in subset if row["relative_capacity_gain_pct"] is not None]
        by_scenario[scenario] = {
            "cases": len(subset),
            "improves_uniform": sum(row["search_improves_uniform"] for row in subset),
            "matches_or_beats_compute_balanced": sum(row["search_matches_or_beats_balanced"] for row in subset),
            "median_relative_capacity_gain_pct": round(median(gains), 6) if gains else None,
            "min_relative_capacity_gain_pct": round(min(gains), 6) if gains else None,
            "max_relative_capacity_gain_pct": round(max(gains), 6) if gains else None,
            "distinct_search_partitions": len({row["search_partition"] for row in subset}),
        }

    total = len(rows)
    conclusion = {
        "question": "Does the lightweight stress-probe boundary search remain beneficial across the six fixed workload seeds used by the evaluator study?",
        "workload_seeds": seeds,
        "heterogeneous_scenarios": scenarios,
        "cases_tested": total,
        "cases_improving_uniform": sum(row["search_improves_uniform"] for row in rows),
        "cases_matching_or_beating_compute_balanced": sum(row["search_matches_or_beats_balanced"] for row in rows),
    }
    conclusion["answer"] = (
        "yes"
        if conclusion["cases_improving_uniform"] >= int(0.8 * total)
        and conclusion["cases_matching_or_beating_compute_balanced"] >= int(0.8 * total)
        else "mixed"
    )
    summary = {
        "experiment_id": "phase5-six-seed-robustness-v1",
        "semantics": "same 8-stage synthetic evaluator and heterogeneity scenarios as phase 3; workload seed changes only sampled input/output lengths while the finite evenly spaced arrival schedule remains fixed for a given intensity",
        "scenario_summaries": by_scenario,
        "conclusion": conclusion,
    }
    log.append("conclusion=" + json.dumps(conclusion, sort_keys=True))
    return rows, summary, "\n".join(log) + "\n"


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], log: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "multiseed_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "experiment.log").write_text(log, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/phase5.json"))
    parser.add_argument("--output", type=Path, default=Path("results/phase5-multiseed"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    rows, summary, log = run(cfg)
    write_outputs(rows, summary, log, args.output)
    print(json.dumps(summary["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

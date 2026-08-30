import json
import unittest
from pathlib import Path

from sla_partition_sensitivity.diagnostics import evaluate_with_diagnostics
from sla_partition_sensitivity.experiment import (
    build_workload,
    generate_partitions,
    load_config,
)
from sla_partition_sensitivity.model import evaluate
from sla_partition_sensitivity.phase2_common_load import run as run_common_load


ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "phase2.json")


class Phase2Tests(unittest.TestCase):
    def test_diagnostic_replay_matches_phase1_verdict(self):
        probes = [
            ("homogeneous_control", 0.0090),
            ("single_slow_stage", 0.0065),
            ("graded", 0.0085),
        ]
        for scenario, intensity in probes:
            speeds = CFG["scenarios"][scenario]
            partition = generate_partitions(speeds, CFG)[0][1]
            workload = build_workload(CFG, intensity)
            baseline = evaluate(workload, partition, speeds, CFG)
            diagnostic = evaluate_with_diagnostics(
                workload, partition, speeds, CFG
            ).verdict
            self.assertEqual(baseline.safe, diagnostic.safe)
            self.assertEqual(
                baseline.first_violation, diagnostic.first_violation
            )
            self.assertEqual(baseline.event_count, diagnostic.event_count)

    def test_common_load_output_has_three_roles_per_scenario(self):
        rows, summary, _ = run_common_load(CFG)
        expected = {
            (scenario, role)
            for scenario in CFG["phase2"]["scenarios"]
            for role in ("worst", "uniform", "best")
        }
        self.assertEqual(
            {(row["scenario"], row["role"]) for row in rows},
            expected,
        )
        self.assertTrue(
            summary["scenario_summaries"]["homogeneous_control"][
                "mechanism_supported"
            ]
        )
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path

from sla_partition_sensitivity.phase14_revision_experiments import _configured, _mean_std_ci95


class Phase14RevisionExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = json.loads(Path("config/phase14.json").read_text(encoding="utf-8"))

    def test_expanded_family_has_nine_shifts_without_mutating_base(self) -> None:
        configured = _configured(
            self.cfg,
            seeds=list(range(20)),
            radius=4,
            bandwidth_multiplier=1.0,
        )
        self.assertEqual(configured["phase11"]["candidate_shifts"], list(range(-4, 5)))
        self.assertEqual(configured["phase10"]["candidate_shifts"], list(range(-4, 5)))
        self.assertEqual(configured["partitions"]["min_layers_per_stage"], 6)
        self.assertEqual(configured["partitions"]["max_layers_per_stage"], 14)
        self.assertEqual(len(configured["phase11"]["workload_seeds"]), 20)
        self.assertEqual(self.cfg["phase11"]["candidate_shifts"], [-2, -1, 0, 1, 2])
        self.assertEqual(self.cfg["partitions"]["min_layers_per_stage"], 8)
        self.assertEqual(self.cfg["partitions"]["max_layers_per_stage"], 12)

    def test_bandwidth_multiplier_changes_only_copied_config(self) -> None:
        configured = _configured(
            self.cfg,
            seeds=[0],
            radius=4,
            bandwidth_multiplier=0.5,
        )
        self.assertEqual(configured["network"]["link_capacity_mb_s"], 625.0)
        self.assertEqual(self.cfg["network"]["link_capacity_mb_s"], 1250.0)

    def test_ci_summary_is_deterministic(self) -> None:
        result = _mean_std_ci95([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["mean"], 2.5)
        self.assertGreater(result["ci95_high"], result["mean"])
        self.assertLess(result["ci95_low"], result["mean"])


if __name__ == "__main__":
    unittest.main()

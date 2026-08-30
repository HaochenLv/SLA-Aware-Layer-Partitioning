from __future__ import annotations

import unittest
from pathlib import Path

from sla_partition_sensitivity.experiment import load_config
from sla_partition_sensitivity.phase13_evaluator_guided_search import evaluator_guided_local_search


def _record(value: float | None) -> dict[str, object]:
    return {"safe_intensity": value}


class Phase13EvaluatorGuidedSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(Path("config/phase13.json"))

    def test_keeps_full_helix_window_and_fixed_seeds(self) -> None:
        self.assertEqual(self.cfg["phase11"]["duration_s"], 120)
        self.assertEqual(self.cfg["phase11"]["workload_seeds"], [0, 1, 2, 3, 7, 19])
        self.assertEqual(self.cfg["phase11"]["candidate_shifts"], [-2, -1, 0, 1, 2])

    def test_convergence_thresholds_are_predeclared(self) -> None:
        self.assertGreaterEqual(self.cfg["phase13"]["near_oracle_ratio"], 0.98)
        self.assertEqual(self.cfg["phase13"]["minimum_near_oracle_trials"], 10)
        self.assertEqual(self.cfg["phase13"]["minimum_not_worse_than_uniform"], 12)
        self.assertEqual(self.cfg["phase13"]["minimum_not_worse_than_joint_compute"], 11)
        self.assertLessEqual(self.cfg["phase13"]["maximum_candidate_probes"], 4)

    def test_expands_toward_improving_positive_neighbor(self) -> None:
        values = {-1: _record(0.9), 0: _record(1.0), 1: _record(1.1), 2: _record(1.2)}
        shift, probes = evaluator_guided_local_search(lambda item: values[item])
        self.assertEqual(shift, 2)
        self.assertEqual(probes, [-1, 0, 1, 2])

    def test_can_cross_a_flat_neighbor_before_improvement(self) -> None:
        values = {-2: _record(1.2), -1: _record(1.0), 0: _record(1.0), 1: _record(0.9)}
        shift, probes = evaluator_guided_local_search(lambda item: values[item])
        self.assertEqual(shift, -2)
        self.assertEqual(probes, [-1, 0, 1, -2])

    def test_stops_at_uniform_when_both_neighbors_are_worse(self) -> None:
        values = {-1: _record(0.9), 0: _record(1.0), 1: _record(0.95)}
        shift, probes = evaluator_guided_local_search(lambda item: values[item])
        self.assertEqual(shift, 0)
        self.assertEqual(probes, [-1, 0, 1])


if __name__ == "__main__":
    unittest.main()

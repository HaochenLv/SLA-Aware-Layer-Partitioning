from __future__ import annotations

import unittest

from sla_partition_sensitivity.phase13_evaluator_guided_search import evaluator_guided_local_search
from sla_partition_sensitivity.phase15_adaptive_directional_search import (
    bidirectional_adaptive_search,
    single_direction_adaptive_search,
)


class Phase15AdaptiveDirectionalSearchTests(unittest.TestCase):
    @staticmethod
    def capacity(values):
        calls = {}

        def get(shift):
            shift = int(shift)
            calls[shift] = calls.get(shift, 0) + 1
            return {"safe_intensity": values[shift]}

        return get

    def test_single_direction_radius_two_matches_phase13(self) -> None:
        values = {-2: 4.0, -1: 3.0, 0: 2.0, 1: 1.0, 2: 0.5}
        phase13_best, phase13_probes = evaluator_guided_local_search(self.capacity(values))
        phase15_best, phase15_probes = single_direction_adaptive_search(
            self.capacity(values), 2
        )
        self.assertEqual(phase15_best, phase13_best)
        self.assertEqual(phase15_probes, phase13_probes)

    def test_single_direction_continues_until_boundary_when_nondecreasing(self) -> None:
        values = {-4: 6.0, -3: 5.0, -2: 4.0, -1: 3.0, 0: 2.0, 1: 1.0}
        best, probes = single_direction_adaptive_search(self.capacity(values), 4)
        self.assertEqual(best, -4)
        self.assertEqual(probes, [-1, 0, 1, -2, -3, -4])

    def test_single_direction_stops_after_first_decrease(self) -> None:
        values = {-4: 8.0, -3: 2.5, -2: 4.0, -1: 3.0, 0: 2.0, 1: 1.0}
        best, probes = single_direction_adaptive_search(self.capacity(values), 4)
        self.assertEqual(best, -2)
        self.assertEqual(probes, [-1, 0, 1, -2, -3])
        self.assertNotIn(-4, probes)

    def test_bidirectional_expands_both_nondecreasing_sides(self) -> None:
        values = {-3: 4.0, -2: 3.0, -1: 2.0, 0: 1.0, 1: 2.0, 2: 2.5, 3: 2.0}
        best, probes = bidirectional_adaptive_search(self.capacity(values), 3)
        self.assertEqual(best, -3)
        self.assertEqual(probes, [-1, 0, 1, -2, -3, 2, 3])


if __name__ == "__main__":
    unittest.main()

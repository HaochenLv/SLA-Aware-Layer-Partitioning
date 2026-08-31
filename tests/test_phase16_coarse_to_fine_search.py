from __future__ import annotations

import unittest

from sla_partition_sensitivity.phase16_coarse_to_fine_search import coarse_to_fine_search


class Phase16CoarseToFineSearchTests(unittest.TestCase):
    @staticmethod
    def capacity(values):
        def get(shift):
            return {"safe_intensity": values[int(shift)]}
        return get

    def test_top_one_refines_neighbors_of_best_coarse_point(self) -> None:
        values = {-4: 1.0, -3: 2.0, -2: 4.0, -1: 5.0, 0: 3.0, 1: 2.0, 2: 1.0, 3: 0.5, 4: 0.25}
        best, probes = coarse_to_fine_search(self.capacity(values), 4, 1)
        self.assertEqual(best, -1)
        self.assertEqual(probes, [-4, -2, 0, 2, 4, -3, -1])

    def test_top_two_can_refine_two_regions(self) -> None:
        values = {-4: 4.0, -3: 4.5, -2: 3.0, -1: 3.5, 0: 1.0, 1: 2.0, 2: 5.0, 3: 6.0, 4: 2.0}
        best, probes = coarse_to_fine_search(self.capacity(values), 4, 2)
        self.assertEqual(best, 3)
        self.assertIn(-3, probes)
        self.assertIn(3, probes)
        self.assertLessEqual(len(probes), 9)


if __name__ == "__main__":
    unittest.main()

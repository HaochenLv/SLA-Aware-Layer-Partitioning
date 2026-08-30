from __future__ import annotations

import unittest
from pathlib import Path

from sla_partition_sensitivity.experiment import load_config


class Phase12SafeEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(Path("config/phase12.json"))

    def test_uses_full_helix_window_and_fixed_seeds(self) -> None:
        self.assertEqual(self.cfg["phase11"]["duration_s"], 120)
        self.assertEqual(self.cfg["phase11"]["workload_seeds"], [0, 1, 2, 3, 7, 19])

    def test_near_oracle_threshold_is_conservative(self) -> None:
        self.assertGreaterEqual(self.cfg["phase12"]["near_oracle_ratio"], 0.98)
        self.assertEqual(self.cfg["phase11"]["candidate_shifts"], [-2, -1, 0, 1, 2])


if __name__ == "__main__":
    unittest.main()

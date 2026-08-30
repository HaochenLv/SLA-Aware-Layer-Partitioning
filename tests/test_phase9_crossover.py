from __future__ import annotations

import json
import unittest
from pathlib import Path

from sla_partition_sensitivity.phase9_crossover import sla_point


class Phase9CrossoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = json.loads(Path("config/phase9_crossover.json").read_text(encoding="utf-8"))

    def test_inverse_budget_path(self) -> None:
        self.assertEqual(sla_point(0.0, self.cfg), (0.28, 1.0))
        self.assertEqual(sla_point(1.0, self.cfg), (2.0, 0.2))
        self.assertEqual(sla_point(0.5, self.cfg), (0.491228, 0.333333))

    def test_refinement_ratio(self) -> None:
        spec = self.cfg["phase9"]
        ratio = spec["coarse_lambda_step"] / spec["fine_lambda_step"]
        self.assertEqual(round(ratio), 10)


if __name__ == "__main__":
    unittest.main()

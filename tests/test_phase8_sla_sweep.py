from __future__ import annotations

import json
import unittest
from pathlib import Path

from sla_partition_sensitivity.phase6_profiled import shifted_partition
from sla_partition_sensitivity.phase8_sla_sweep import sla_point


class Phase8SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = json.loads(Path("config/phase8_sla_sweep.json").read_text(encoding="utf-8"))

    def test_sla_endpoints_and_midpoint(self) -> None:
        self.assertEqual(sla_point(0.0, self.cfg), (0.28, 1.0))
        self.assertEqual(sla_point(1.0, self.cfg), (2.0, 0.2))
        self.assertEqual(sla_point(0.5, self.cfg), (0.491228, 0.333333))

    def test_shift_direction(self) -> None:
        self.assertEqual(shifted_partition(2, self.cfg), [12, 8, 12, 8, 12, 8, 12, 8])
        self.assertEqual(shifted_partition(-2, self.cfg), [8, 12, 8, 12, 8, 12, 8, 12])
        self.assertEqual(sum(shifted_partition(0, self.cfg)), 80)


if __name__ == "__main__":
    unittest.main()

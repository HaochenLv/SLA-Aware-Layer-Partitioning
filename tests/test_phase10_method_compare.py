from __future__ import annotations

import unittest
from pathlib import Path

from sla_partition_sensitivity.experiment import load_config
from sla_partition_sensitivity.helix_profile import derive_phase_speed_factors
from sla_partition_sensitivity.phase10_method_compare import static_baseline_shifts


class Phase10MethodCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(Path("config/phase10.json"))
        cls.factors = derive_phase_speed_factors(cls.cfg)
        machines = cls.cfg["phase6"]["stage_machines"]
        cls.prefill = [cls.factors[m]["prefill_speed"] for m in machines]
        cls.decode = [cls.factors[m]["decode_speed"] for m in machines]

    def test_six_fixed_seeds(self) -> None:
        self.assertEqual(self.cfg["phase10"]["workload_seeds"], [0, 1, 2, 3, 7, 19])

    def test_phase_specific_static_baselines_disagree(self) -> None:
        shifts = static_baseline_shifts(self.prefill, self.decode, self.cfg)
        self.assertGreater(shifts["prefill_balanced"], 0)
        self.assertLess(shifts["decode_balanced"], 0)
        self.assertEqual(shifts["uniform"], 0)


if __name__ == "__main__":
    unittest.main()

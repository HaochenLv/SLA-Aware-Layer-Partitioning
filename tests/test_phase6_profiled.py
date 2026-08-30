from pathlib import Path
import unittest

from sla_partition_sensitivity.experiment import build_workload, load_config
from sla_partition_sensitivity.helix_profile import derive_phase_speed_factors
from sla_partition_sensitivity.model import evaluate
from sla_partition_sensitivity.phase6_profiled import shifted_partition
from sla_partition_sensitivity.profiled_model import evaluate_profiled


class Phase6ProfiledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(Path("config/phase6.json"))

    def test_helix_profiles_expose_phase_order_cross_over(self) -> None:
        factors = derive_phase_speed_factors(self.cfg)
        self.assertAlmostEqual(factors["a100"]["prefill_speed"], 1.0)
        self.assertAlmostEqual(factors["a100"]["decode_speed"], 1.0)
        self.assertGreater(
            factors["l4x2"]["prefill_speed"], factors["t4x4"]["prefill_speed"]
        )
        self.assertLess(
            factors["l4x2"]["decode_speed"], factors["t4x4"]["decode_speed"]
        )

    def test_shift_family_preserves_all_80_layers(self) -> None:
        for shift in self.cfg["phase6"]["boundary_shifts"]:
            partition = shifted_partition(shift, self.cfg)
            self.assertEqual(sum(partition), 80)
            self.assertTrue(all(8 <= value <= 12 for value in partition))

    def test_profiled_evaluator_reduces_to_original_for_unit_speeds(self) -> None:
        partition = [10] * 8
        workload = build_workload(self.cfg, 0.00025)
        original = evaluate(workload, partition, [1.0] * 8, self.cfg)
        profiled = evaluate_profiled(
            workload, partition, [1.0] * 8, [1.0] * 8, self.cfg
        )
        self.assertEqual(original.safe, profiled.safe)
        self.assertEqual(original.event_count, profiled.event_count)
        self.assertEqual(original.peak_prefill, profiled.peak_prefill)
        self.assertEqual(original.peak_decode, profiled.peak_decode)
        self.assertAlmostEqual(
            original.minimum_link_headroom_mb_s,
            profiled.minimum_link_headroom_mb_s,
            places=9,
        )
        self.assertAlmostEqual(original.peak_memory_gb, profiled.peak_memory_gb, places=9)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest

from sla_partition_sensitivity.experiment import load_config
from sla_partition_sensitivity.phase7_multiseed_profiled import run


class Phase7ProfiledTests(unittest.TestCase):
    def test_phase7_declares_six_fixed_seeds(self) -> None:
        cfg = load_config(Path("config/phase7.json"))
        self.assertEqual(cfg["phase7"]["workload_seeds"], [0, 1, 2, 3, 7, 19])
        self.assertEqual(cfg["phase7"]["minimum_direction_flips_for_robustness"], 5)

    def test_phase7_small_smoke_run(self) -> None:
        cfg = load_config(Path("config/phase7.json"))
        cfg["phase7"]["workload_seeds"] = [7]
        for regime in cfg["phase6"]["regimes"].values():
            regime["lambda_stop"] = regime["lambda_start"] + regime["lambda_step"]
        capacity_rows, seed_rows, summary, _ = run(cfg)
        self.assertEqual(len(seed_rows), 1)
        self.assertEqual(len(capacity_rows), 10)
        self.assertIn("direction_flip_count", summary["conclusion"])


if __name__ == "__main__":
    unittest.main()

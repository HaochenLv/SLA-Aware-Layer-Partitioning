import copy
import unittest
from pathlib import Path

from sla_partition_sensitivity.experiment import build_workload, load_config


ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "phase5.json")


class Phase5MultiSeedTests(unittest.TestCase):
    def test_config_uses_six_fixed_seeds(self):
        self.assertEqual(CFG["phase5"]["workload_seeds"], [0, 1, 2, 3, 7, 19])

    def test_seed_changes_lengths_not_arrival_schedule(self):
        a_cfg = copy.deepcopy(CFG)
        b_cfg = copy.deepcopy(CFG)
        a_cfg["workload"]["seed"] = 0
        b_cfg["workload"]["seed"] = 19
        a = build_workload(a_cfg, 0.01)
        b = build_workload(b_cfg, 0.01)
        self.assertEqual([r.arrival_s for r in a], [r.arrival_s for r in b])
        self.assertNotEqual(
            [(r.input_tokens, r.output_tokens) for r in a],
            [(r.input_tokens, r.output_tokens) for r in b],
        )


if __name__ == "__main__":
    unittest.main()

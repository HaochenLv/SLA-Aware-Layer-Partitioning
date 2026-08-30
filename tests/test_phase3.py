import copy
import unittest
from pathlib import Path

from sla_partition_sensitivity.experiment import load_config
from sla_partition_sensitivity.model import validate_partition
from sla_partition_sensitivity.phase3 import (
    adjacent_boundary_neighbors,
    boundary_local_search,
    uniform_partition,
)


ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "phase3.json")


class Phase3Tests(unittest.TestCase):
    def test_adjacent_neighbors_are_legal_single_boundary_moves(self):
        base = uniform_partition(CFG)
        neighbors = adjacent_boundary_neighbors(base, CFG)
        self.assertEqual(len(neighbors), 14)
        for candidate in neighbors:
            validate_partition(candidate, CFG)
            delta = [b - a for a, b in zip(base, candidate)]
            nonzero = [i for i, value in enumerate(delta) if value]
            self.assertEqual(len(nonzero), 2)
            self.assertEqual(nonzero[1], nonzero[0] + 1)
            self.assertEqual(sorted(delta[i] for i in nonzero), [-1, 1])

    def test_homogeneous_control_does_not_move(self):
        cfg = copy.deepcopy(CFG)
        cfg["sampling"]["lambda_stop"] = 0.010
        cfg["phase3"]["max_iterations"] = 4
        result, history, _, _ = boundary_local_search(
            cfg["scenarios"]["homogeneous_control"], cfg
        )
        self.assertEqual(result["partition"], uniform_partition(cfg))
        self.assertEqual(len(history), 1)

    def test_search_is_deterministic_on_small_grid(self):
        cfg = copy.deepcopy(CFG)
        cfg["sampling"]["lambda_stop"] = 0.010
        cfg["phase3"]["max_iterations"] = 5
        speeds = cfg["scenarios"]["single_slow_stage"]
        a = boundary_local_search(speeds, cfg)
        b = boundary_local_search(speeds, cfg)
        self.assertEqual(a[0]["partition"], b[0]["partition"])
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2:], b[2:])


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from sla_partition_sensitivity.experiment import load_config
from sla_partition_sensitivity.model import validate_partition
from sla_partition_sensitivity.phase4_oracle import enumerate_partitions


ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "phase4_oracle.json")


class Phase4OracleTests(unittest.TestCase):
    def test_reduced_space_has_expected_size(self):
        partitions = list(enumerate_partitions(CFG))
        self.assertEqual(len(partitions), 231)
        self.assertEqual(len({tuple(p) for p in partitions}), 231)

    def test_every_enumerated_partition_is_feasible(self):
        for partition in enumerate_partitions(CFG):
            validate_partition(partition, CFG)
            self.assertEqual(sum(partition), 24)
            self.assertTrue(all(3 <= n <= 9 for n in partition))


if __name__ == "__main__":
    unittest.main()

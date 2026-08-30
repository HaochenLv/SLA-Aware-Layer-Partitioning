import json
import unittest
from pathlib import Path

from sla_partition_sensitivity.experiment import (
    generate_partitions,
    lambda_grid,
    load_config,
    run_experiment,
)
from sla_partition_sensitivity.model import validate_partition


ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config" / "phase1.json")


class ExperimentTests(unittest.TestCase):
    def test_grid_has_paper_phase1_range(self):
        grid = lambda_grid(CFG)
        self.assertEqual(grid[0], 0.006)
        self.assertEqual(grid[-1], 0.022)
        self.assertGreater(len(grid), 20)

    def test_exactly_twenty_legal_partitions_per_scenario(self):
        for speeds in CFG["scenarios"].values():
            partitions = generate_partitions(speeds, CFG)
            self.assertEqual(len(partitions), 20)
            self.assertEqual(len({tuple(p) for _, p in partitions}), 20)
            for _, partition in partitions:
                validate_partition(partition, CFG)

    def test_experiment_is_deterministic_and_monotonic(self):
        rows_a, trials_a, summary_a, log_a = run_experiment(CFG)
        rows_b, trials_b, summary_b, log_b = run_experiment(CFG)
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(trials_a, trials_b)
        self.assertEqual(summary_a, summary_b)
        self.assertEqual(log_a, log_b)
        self.assertEqual(len(rows_a), 100)
        self.assertEqual(len(trials_a), 100 * len(lambda_grid(CFG)))
        self.assertTrue(all(row["sampled_monotonic"] for row in rows_a))

    def test_summary_is_json_serializable(self):
        _, _, summary, _ = run_experiment(CFG)
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()

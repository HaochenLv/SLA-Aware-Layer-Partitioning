from __future__ import annotations

import unittest
from pathlib import Path

from sla_partition_sensitivity.experiment import load_config
from sla_partition_sensitivity.phase11_trace_validation import scale_workload
from sla_partition_sensitivity.model import Request


class Phase11TraceValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(Path("config/phase11_trace.json"))

    def test_declares_first_paper_fixed_seeds(self) -> None:
        self.assertEqual(self.cfg["phase11"]["workload_seeds"], [0, 1, 2, 3, 7, 19])
        self.assertEqual(self.cfg["phase11"]["duration_s"], 30)
        self.assertEqual(self.cfg["phase11"]["interval_offset"], 0)

    def test_intensity_scaling_preserves_lengths_and_order(self) -> None:
        base = [
            Request(0, 1.0, 100, 20),
            Request(1, 3.0, 200, 30),
            Request(2, 7.0, 300, 40),
        ]
        scaled = scale_workload(base, 2.0)
        self.assertEqual([r.input_tokens for r in scaled], [100, 200, 300])
        self.assertEqual([r.output_tokens for r in scaled], [20, 30, 40])
        self.assertEqual([r.request_id for r in scaled], [0, 1, 2])
        self.assertEqual([r.arrival_s for r in scaled], [1.0, 2.0, 4.0])


if __name__ == "__main__":
    unittest.main()

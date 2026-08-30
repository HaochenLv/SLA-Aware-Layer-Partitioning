from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any


def load_profile(path: Path) -> dict[int, float]:
    table: dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            table[int(row[0])] = float(row[1])
    if not table:
        raise ValueError(f"empty HELIX profile: {path}")
    return table


def interpolate(table: dict[int, float], x: float) -> float:
    points = sorted(table)
    if x <= points[0]:
        return table[points[0]]
    if x >= points[-1]:
        return table[points[-1]]
    for left, right in zip(points[:-1], points[1:]):
        if left <= x <= right:
            y0 = table[left]
            y1 = table[right]
            if right == left:
                return y0
            return y0 + (y1 - y0) * (x - left) / (right - left)
    raise RuntimeError("interpolation bracket not found")


def derive_phase_speed_factors(cfg: dict[str, Any]) -> dict[str, Any]:
    spec = cfg["phase6"]
    root = Path(spec["profile_root"])
    reference_machine = spec["reference_machine"]
    machines = spec["machines"]
    prefill_points = spec["prefill_profile_points"]
    decode_points = spec["decode_profile_points"]

    profiles: dict[str, dict[str, dict[int, float]]] = {}
    for machine in machines:
        machine_root = root / machine
        profiles[machine] = {
            "prefill": load_profile(machine_root / "prompt_bs2time.csv"),
            "decode": load_profile(machine_root / "decode_bs2time.csv"),
        }

    reference = profiles[reference_machine]
    result: dict[str, Any] = {}
    for machine in machines:
        prompt_ratios = [
            interpolate(reference["prefill"], point)
            / interpolate(profiles[machine]["prefill"], point)
            for point in prefill_points
        ]
        decode_ratios = [
            interpolate(reference["decode"], point)
            / interpolate(profiles[machine]["decode"], point)
            for point in decode_points
        ]
        result[machine] = {
            "prefill_speed": statistics.median(prompt_ratios),
            "decode_speed": statistics.median(decode_ratios),
            "prefill_ratio_samples": prompt_ratios,
            "decode_ratio_samples": decode_ratios,
        }
    return result

#!/usr/bin/env python3
"""Reproduce, but never silently update, the accepted analytical baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "reference/baselines/radial_analytic_001.json"


def calculate(data: dict) -> list[float]:
    inputs = data["input"]
    return [
        inputs["p0_pa"]
        * math.exp(
            inputs["omega_rad_s"] ** 2
            * (radius**2 - inputs["r0_m"] ** 2)
            / (2 * inputs["gas_constant_j_kg_k"] * inputs["temperature_k"])
        )
        for radius in inputs["r_m"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="explicitly overwrite only the analytical expected values",
    )
    args = parser.parse_args()
    data = json.loads(PATH.read_text(encoding="utf-8"))
    calculated = calculate(data)
    differences = [abs(a - b) for a, b in zip(calculated, data["expected_pressure_pa"])]

    if args.write:
        data["expected_pressure_pa"] = calculated
        PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"UPDATED {PATH.relative_to(ROOT)} by explicit request")
        return 0

    tolerance = data["absolute_tolerance_pa"]
    print(f"max |calculated-reference| = {max(differences):.3e} Pa")
    if max(differences) > tolerance:
        print("FAIL: baseline differs; investigate before any --write")
        return 1
    print("PASS: analytical baseline reproduced; no file changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

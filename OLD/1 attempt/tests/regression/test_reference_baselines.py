import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ReferenceBaselineTests(unittest.TestCase):
    def test_analytical_pressure_samples(self):
        baseline = json.loads((ROOT / "reference/baselines/radial_analytic_001.json").read_text())
        inputs = baseline["input"]
        calculated = [
            inputs["p0_pa"]
            * math.exp(
                inputs["omega_rad_s"] ** 2
                * (radius**2 - inputs["r0_m"] ** 2)
                / (2 * inputs["gas_constant_j_kg_k"] * inputs["temperature_k"])
            )
            for radius in inputs["r_m"]
        ]
        for actual, expected in zip(calculated, baseline["expected_pressure_pa"]):
            self.assertAlmostEqual(actual, expected, delta=baseline["absolute_tolerance_pa"])

    def test_book_control_is_not_mislabelled_as_accepted(self):
        baseline = json.loads((ROOT / "reference/baselines/book_approx_control_001.json").read_text())
        self.assertEqual(baseline["model_status"], "book_approx_diagnostic")
        self.assertFalse(baseline["accepted_physical_solution"])

    def test_baselines_contain_only_finite_numbers(self):
        def check(value):
            if isinstance(value, dict):
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
            elif isinstance(value, float):
                self.assertTrue(math.isfinite(value))

        for path in (ROOT / "reference/baselines").glob("*.json"):
            check(json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()

import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ProjectConstraintTests(unittest.TestCase):
    def test_qn_is_derived_from_stress_not_independent_evidence(self):
        baseline = json.loads((ROOT / "reference/baselines/book_approx_control_001.json").read_text())
        stress = baseline["expected"]["stress_mpa"]
        self.assertAlmostEqual(math.sqrt(stress / 300.0), baseline["expected"]["q_n"], delta=5e-5)

    def test_control_angles_are_inside_book_ranges(self):
        angles = json.loads(
            (ROOT / "reference/baselines/book_approx_control_001.json").read_text()
        )["expected"]["alpha2_deg"]
        self.assertTrue(all(70.0 <= angle <= 90.0 for angle in angles[:2]))
        self.assertTrue(80.0 <= angles[2] <= 90.0)


if __name__ == "__main__":
    unittest.main()

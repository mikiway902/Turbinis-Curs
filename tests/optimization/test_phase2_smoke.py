import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class OptimizerSmokeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is not installed")
    def test_phase2_receives_phase1_feasible_points(self):
        completed = subprocess.run(
            [sys.executable, "test_phase2.py"],
            cwd=ROOT / "Optimizer",
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class LegacyInvariantTests(unittest.TestCase):
    def test_book_approx_invariants(self):
        completed = subprocess.run(
            [sys.executable, "Turbinis-Calc/check_invariants.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()

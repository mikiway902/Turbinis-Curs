"""Focused tests for deterministic, physics-free sampling helpers."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.sampling import (  # noqa: E402
    deterministic_grid,
    iter_grid,
    split_pressure_ratio_log,
    targeted_values,
)


class SamplingTests(unittest.TestCase):
    def test_grid_is_deterministic_and_axis_keys_are_sorted(self):
        axes = {"b": [2, 3], "a": ["x", "y"]}
        first = deterministic_grid(axes)
        second = deterministic_grid({"a": ["x", "y"], "b": [2, 3]})
        self.assertEqual(first, second)
        self.assertEqual(list(first[0]), ["a", "b"])
        self.assertEqual(
            first,
            (
                {"a": "x", "b": 2},
                {"a": "x", "b": 3},
                {"a": "y", "b": 2},
                {"a": "y", "b": 3},
            ),
        )
        self.assertEqual(tuple(iter_grid(axes)), first)

    def test_grid_limit_and_invalid_axes(self):
        self.assertEqual(
            deterministic_grid({"a": [1, 2], "b": [3]}, limit=1),
            ({"a": 1, "b": 3},),
        )
        with self.assertRaises(ValueError):
            deterministic_grid({"a": []})
        with self.assertRaises(ValueError):
            deterministic_grid({})
        with self.assertRaises(ValueError):
            deterministic_grid({"a": [1]}, limit=-1)
        with self.assertRaises(ValueError):
            deterministic_grid({"a": b"12"})  # bytes are not a value sequence
        with self.assertRaises(ValueError):
            deterministic_grid({"a": {1, 2}})  # unordered sets have no stable order

    def test_targeted_values_preserves_order_and_deduplicates(self):
        selected = targeted_values(
            [1.0, 3.05, 3.0, 7.0, 6.95],
            targets=[3.0, 7.0],
            tolerance=0.1,
        )
        self.assertEqual(selected, (3.05, 3.0, 7.0, 6.95))
        self.assertEqual(
            targeted_values([1, 2], targets=[2], tolerance=0),
            (2,),
        )
        with self.assertRaises(ValueError):
            targeted_values([1], targets=[], tolerance=0.1)
        with self.assertRaises(ValueError):
            targeted_values([1], targets=[1], tolerance=-1)

    def test_log_split_preserves_product_and_relative_log_allocation(self):
        parts = split_pressure_ratio_log(12.0, [1.0, 2.0, 3.0])
        self.assertEqual(len(parts), 3)
        self.assertAlmostEqual(math.prod(parts), 12.0, places=12)
        log_parts = [math.log(part) for part in parts]
        self.assertAlmostEqual(
            log_parts[1] / log_parts[0], 2.0, places=12
        )
        self.assertAlmostEqual(
            log_parts[2] / log_parts[0], 3.0, places=12
        )

        with self.assertRaises(ValueError):
            split_pressure_ratio_log(0.0, [1.0])
        with self.assertRaises(ValueError):
            split_pressure_ratio_log(2.0, [])
        with self.assertRaises(ValueError):
            split_pressure_ratio_log(2.0, [1.0, 0.0])


if __name__ == "__main__":
    unittest.main()

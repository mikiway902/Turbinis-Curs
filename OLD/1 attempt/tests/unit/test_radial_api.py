import unittest

import numpy as np

from turbine.radial import entropy_from_state, normalized_residual, pressure_residual


class RadialApiTests(unittest.TestCase):
    def test_entropy_is_zero_at_reference(self):
        entropy = entropy_from_state(
            [900.0, 920.0], [250000.0, 245000.0], cp=1005.0, gas_constant=287.0
        )
        self.assertEqual(entropy[0], 0.0)
        self.assertTrue(np.all(np.isfinite(entropy)))

    def test_rejects_non_positive_state(self):
        with self.assertRaises(ValueError):
            pressure_residual(
                [0.03, 0.04, 0.05],
                [250000.0, 0.0, 250000.0],
                [900.0] * 3,
                [0.0] * 3,
                gas_constant=287.0,
            )

    def test_normalization_requires_physical_scale(self):
        with self.assertRaises(ValueError):
            normalized_residual([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], trim=0)


if __name__ == "__main__":
    unittest.main()

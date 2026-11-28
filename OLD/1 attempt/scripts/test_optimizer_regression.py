"""Runtime regression anchor for one existing optimizer candidate.

PROVENANCE / STATUS: software-drift diagnostic only.

This is NOT a validated physical baseline and NOT an acceptable design. The
candidate is deliberately infeasible (blade length, stress, speed margin and
root reactivity are outside the configured limits), so it must not be used as
evidence that the 3D flow, losses, or radial equilibrium are closed.

The anchor fixes a single hand-evaluated point from the current model:

* scenario ``B87_relaxed`` with ``q_n_limit = 0.995``;
* ``x = [84000, .065, .75, .80, 11, 11, 19, 1.38, 1.12, 1.15]`` (the first
  stage angle is at the book lower bound, ``alpha1 = 11 deg``);
* observed with numpy 2.5.4, scipy 1.18.1, pandas 3.1.1 on CPython 3.12.

After the user-confirmed 2026-11-10 power-plane change, the anchor was
re-evaluated at the same candidate with 110 kW interpreted as gearbox-input
(pre-reducer) power. The changed stage geometry is a deliberate consequence of
that formula change, not an optimizer search result.

The assertions below check that ``eval_candidate`` still returns a
model-evaluated result (``status == "ok"``), that the power target and solved
conservation residuals are unchanged, and that the expected engineering
violations remain present. A change here means software drift or an
intentional model/formula change; investigate and re-approve the anchor
separately. This test never invokes an optimizer search or sweep.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Optimizer"))
sys.path.insert(0, str(ROOT / "Turbinis-Calc"))


class OptimizerCandidateAnchorTests(unittest.TestCase):
    CANDIDATE = (84000.0, 0.065, 0.75, 0.80, 11.0, 11.0, 19.0, 1.38, 1.12, 1.15)
    SCENARIO = "B87_relaxed"
    Q_N_LIMIT = 0.995

    # Observed on the environment listed in the module docstring. Tolerances
    # are tight enough to catch formula/iteration drift, but not so tight that
    # harmless last-bit differences between platforms fail the anchor.
    EXPECTED = {
        "lc1": 0.011295345088530603,
        "a2": (78.43541526121275, 73.67931781281864, 80.81826764406877),
        "sigma": 309.07753384083213,
        "u": 388.0901837762332,
        "lD": 0.26335440171761704,
        "qn": 1.0150164757953966,
        "eta_t": 0.9127057739015165,
    }
    RELATIVE_TOLERANCE = 1e-6

    @classmethod
    def setUpClass(cls) -> None:
        # Importing the model requires the declared runtime dependencies. Skip
        # only when one of those is genuinely unavailable; any other import or
        # model failure must surface as an error, not as a skip.
        try:
            import optimizer_core as oc
            import optimize_extended as oe
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy"}:
                raise unittest.SkipTest(
                    "optimizer regression anchor needs runtime dependency "
                    f"'{dependency}', which is not installed; install the "
                    "project dependencies to run this test"
                ) from exc
            raise
        cls.oc = oc
        cls.oe = oe
        cls.model = oc.load_model()
        cls.ns, _ = oe.build_ns(cls.model, q_n_limit=cls.Q_N_LIMIT)

    def assertFiniteNear(self, actual: float, expected: float) -> None:
        self.assertTrue(math.isfinite(actual), f"expected finite value, got {actual!r}")
        self.assertAlmostEqual(
            actual / expected,
            1.0,
            delta=self.RELATIVE_TOLERANCE,
            msg=f"expected ~{expected!r}, got {actual!r}",
        )

    def test_candidate_remains_a_deterministic_model_evaluation(self) -> None:
        result = self.oe.eval_candidate(
            self.CANDIDATE, self.ns, self.SCENARIO, self.Q_N_LIMIT
        )

        # Calculation health: a solved model evaluation, not a numerical or
        # model exception. Engineering infeasibility is asserted separately.
        self.assertEqual(result["status"], "ok", result.get("err", ""))
        self.assertEqual(result["err"], "")

        # Power target and solver residuals.
        self.assertAlmostEqual(result["N"], 110e3, delta=1e-3)
        self.assertLess(result["cont"], 1e-12, "continuity residual drifted")
        self.assertLess(result["energy"], 1e-12, "energy residual drifted")
        self.assertLess(result["p2z"], 1e-12, "exit-pressure residual drifted")

        # Deterministic design outputs.
        self.assertFiniteNear(result["lc1"], self.EXPECTED["lc1"])
        for index, (actual, expected) in enumerate(
            zip(result["a2"], self.EXPECTED["a2"]), start=1
        ):
            with self.subTest(stage=index, field="alpha2"):
                self.assertFiniteNear(actual, expected)
        for field in ("sigma", "u", "lD", "qn", "eta_t"):
            with self.subTest(field=field):
                self.assertFiniteNear(result[field], self.EXPECTED[field])

        # The point stays infeasible for the expected engineering reasons.
        # These are not calculation failures, so assert them explicitly.
        self.assertFalse(result["feasible"])
        self.assertGreater(result["hard"], 0.0)
        self.assertEqual(result["violation_count"], 5)
        names = [violation["name"] for violation in result["violations"]]
        for prefix in ("l_c1", "σ > 300", "q_n >", "ρ_корня_1", "ρ_корня_2"):
            self.assertTrue(
                any(name.startswith(prefix) for name in names),
                f"expected violation starting with {prefix!r} in {names!r}",
            )


if __name__ == "__main__":
    unittest.main()

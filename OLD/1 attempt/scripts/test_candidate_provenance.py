"""Candidate provenance tests for the existing optimizer adapters.

No optimizer search, sweep, or DE is invoked. The anchor candidates are single
software-drift evaluations, not physical validation or acceptable designs.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Optimizer"))
sys.path.insert(0, str(ROOT / "Turbinis-Calc"))

import optimizer_core as oc  # noqa: E402
import optimize_extended as oe  # noqa: E402


class CandidateProvenanceTests(unittest.TestCase):
    CANDIDATE = (84000.0, 0.065, 0.75, 0.80, 11.0, 11.0, 19.0, 1.38, 1.12, 1.15)
    SCENARIO = "B87_relaxed"
    Q_N_LIMIT = 0.995

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = oc.load_model()
        cls.ns, _ = oe.build_ns(cls.model, q_n_limit=cls.Q_N_LIMIT)

    def _eval(self, soft: bool = False) -> dict:
        return oe.eval_candidate(
            self.CANDIDATE, self.ns, self.SCENARIO, self.Q_N_LIMIT, soft=soft
        )

    def test_extended_success_contains_full_stages_and_requested_actual_fields(self):
        result = self._eval()

        self.assertEqual(result["status"], "ok", result.get("err", ""))
        self.assertEqual(len(result["stages"]), 3)
        self.assertEqual(result["model_variant"], "extended_const_hub")
        self.assertEqual(list(result["a1_requested"]), list(self.CANDIDATE[4:7]))
        self.assertEqual(
            list(result["a1_actual"]),
            [float(stage["α1"]) for stage in result["stages"]],
        )
        self.assertEqual(
            list(result["rho_actual"]),
            [float(stage["ρ"]) for stage in result["stages"]],
        )
        self.assertIsNone(result["rho_requested"])
        adjustment = result["engineering_adjustment"]
        self.assertFalse(adjustment["active"])
        self.assertFalse(adjustment["legacy_cell14_repair"])
        self.assertIn("const_hub_reaction_fixed_point", adjustment["model_construction"])
        self.assertEqual(adjustment["requested"]["a1"], list(result["a1_requested"]))
        self.assertEqual(adjustment["actual"]["rho"], list(result["rho_actual"]))
        self.assertEqual(result["uc0_opt_status"], ["ok"] * 3)

    def test_out_of_graph_diagnostic_survives_without_losing_physics(self):
        baseline = self._eval(soft=False)

        with patch.object(
            oe.cs, "uc0_opt_from_rho",
            side_effect=ValueError("rho (рис. 18) вне диапазона"),
        ):
            degraded = self._eval(soft=False)

        self.assertEqual(degraded["status"], "ok", degraded.get("err", ""))
        self.assertNotEqual(degraded["status"], "numerical")
        self.assertEqual(degraded["uc0_opt"], [None, None, None])
        self.assertEqual(degraded["uc0_opt_status"], ["NOT_CHECKED"] * 3)
        self.assertTrue(
            all("рис. 18" in reason for reason in degraded["uc0_opt_reason"])
        )
        self.assertEqual(degraded["uc0_opt_missing"], [0, 1, 2])
        self.assertEqual(len(degraded["stages"]), 3)

        # A missing optional diagnostic must not change physical outputs,
        # feasibility, phase-1 score, or the non-soft phase-2 score.
        for field in ("lc1", "a2", "sigma", "u", "lD", "qn", "N", "eta_t",
                      "rho", "rho_root", "uc0"):
            with self.subTest(field=field):
                self.assertEqual(degraded[field], baseline[field])
        self.assertEqual(degraded["feasible"], baseline["feasible"])
        self.assertEqual(degraded["score1"], baseline["score1"])
        self.assertEqual(degraded["score2"], baseline["score2"])
        self.assertEqual(
            [v["name"] for v in degraded["violations"]],
            [v["name"] for v in baseline["violations"]],
        )
        self.assertEqual(degraded["scoring_status"], "ok")

    def test_soft_scoring_explicitly_penalizes_missing_graph_diagnostic(self):
        baseline = self._eval(soft=True)

        with patch.object(
            oe.cs, "uc0_opt_from_rho",
            side_effect=ValueError("rho (рис. 18) вне диапазона"),
        ):
            degraded = self._eval(soft=True)
            nonsoft = self._eval(soft=False)

        self.assertEqual(degraded["status"], "ok", degraded.get("err", ""))
        self.assertEqual(degraded["scoring_status"], "unavailable_diagnostic")
        self.assertTrue(degraded["soft_score_applied"])
        self.assertTrue(math.isfinite(degraded["score2"]))
        self.assertGreaterEqual(
            degraded["score2"],
            nonsoft["score2"] + oe.SOFT_MISSING_DIAGNOSTIC_PENALTY * 3 - 1e-9,
        )
        self.assertEqual(degraded["feasible"], baseline["feasible"])
        self.assertEqual(degraded["score1"], baseline["score1"])

    def test_conventional_evaluate_labels_legacy_engineering_repair(self):
        result = oc.evaluate(oc.warm_start({}), self.model, {}, full=False)

        self.assertEqual(result["model_variant"],
                         "legacy_cell16+cell14_engineering_repair")
        self.assertEqual(result["a1_requested"], list(result["p"]["a1"]))
        self.assertEqual(
            result["a1_actual"],
            [float(stage["α1"]) for stage in result["stages"]],
        )
        self.assertEqual(
            result["rho_actual"],
            [float(stage["ρ"]) for stage in result["stages"]],
        )
        adjustment = result["engineering_adjustment"]
        self.assertTrue(adjustment["active"])
        self.assertTrue(adjustment["legacy_repair"])
        mechanisms = {item["name"]: item for item in adjustment["mechanisms"]}
        self.assertIn("rho_root_floor", mechanisms)
        self.assertIn("gamma_inlet_angle_floor", mechanisms)
        self.assertEqual(adjustment["requested"]["a1"], result["a1_requested"])
        self.assertEqual(adjustment["actual"]["a1"], result["a1_actual"])
        self.assertEqual(adjustment["actual"]["rho"], result["rho_actual"])
        self.assertIsNone(adjustment["requested"]["rho"])

    def test_conventional_provenance_helper_keeps_actual_repaired_angles(self):
        ns = {
            "закон_ст": ["α1", "α1", "Γ"],
            "ступень_авто": object(),
            "ступень_с_подбором_ρ": object(),
            "ρ_корня_мин": 0.05,
            "ρ_макс": 0.6,
            "α1_мин_корня": 11.0,
        }
        stages = [
            dict(α1=11.0, ρ=0.30),
            dict(α1=12.0, ρ=0.35),
            dict(α1=14.5, ρ=0.40),
        ]
        provenance = oc._engineering_provenance(
            ns, {"stages": stages}, {"a1": [11.0, 11.0, 12.0]}
        )

        self.assertEqual(provenance["a1_requested"], [11.0, 11.0, 12.0])
        self.assertEqual(provenance["a1_actual"], [11.0, 12.0, 14.5])
        self.assertEqual(
            provenance["engineering_adjustment"]["changed"]["a1"],
            [False, True, True],
        )
        mechanisms = {
            item["name"]: item
            for item in provenance["engineering_adjustment"]["mechanisms"]
        }
        self.assertEqual(mechanisms["rho_root_floor"]["rho_max"], 0.6)
        self.assertEqual(
            mechanisms["gamma_inlet_angle_floor"]["status"], "enabled"
        )

    def test_provenance_helper_keeps_requested_separate_from_actual(self):
        synthetic = [dict(α1=12.5, ρ=0.31)]
        provenance = oe._const_hub_provenance(synthetic, (11.0,))

        self.assertEqual(provenance["a1_requested"], [11.0])
        self.assertEqual(provenance["a1_actual"], [12.5])
        self.assertEqual(provenance["rho_actual"], [0.31])
        self.assertEqual(
            provenance["engineering_adjustment"]["actual"]["a1"],
            provenance["a1_actual"],
        )


if __name__ == "__main__":
    unittest.main()

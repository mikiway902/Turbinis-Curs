"""Diagnostic checks for the read-only Stage 6B-4.5-B entropy budget.

These tests exercise algebra and domains only.  They do not promote the
experimental loss closure, decide existence, or assert a target residual.
"""
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Optimizer"))

import rotor_exit_coupled as rc  # noqa: E402
from turbine.radial import entropy_budget as eb  # noqa: E402
from turbine.radial import rotor_exit as core  # noqa: E402

RESULTS = ROOT / "Optimizer" / "results" / "stage6b45b_results.json"

CP = 1176.1897435517353
R_GAS = 287.4
PSI = 0.96

ST = {
    "T1": 1128.4889811790695,
    "T2": 1077.5660961613316,
    "T2_ад": 1071.397005666692,
    "p1": 670931.142875466,
    "p2": 542505.7849637904,
    "w2": 413.0260690936596,
    "cp": CP,
    "k": 1.3233610672097846,
    "c2a": 122.68425204552908,
    "h_л": 7256.0209668377265,
}


def _synthetic_fields(anchor=1):
    """Positive finite synthetic state whose budget identity is algebraic."""
    g = {
        "T1": np.array([1000.0, 1010.0, 1020.0]),
        "p1": np.array([5.0e5, 5.1e5, 5.2e5]),
        "cp": CP,
        "R": R_GAS,
        "psi": PSI,
        "anchor": anchor,
    }
    f = {
        "T2": np.array([900.0, 910.0, 920.0]),
        "p": np.array([2.0e5, 2.1e5, 2.2e5]),
        "dsl": np.array([5.0, 6.0, 7.0]),
        "w2": np.array([400.0, 410.0, 420.0]),
        "hl": np.array([5000.0, 5500.0, 6000.0]),
        "ca": np.array([100.0, 110.0, 120.0]),
    }
    return g, f


def _copy_state(g, f):
    return ({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in g.items()},
            {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in f.items()})


class EntropyBudgetTests(unittest.TestCase):
    def setUp(self):
        self.g, self.f = _synthetic_fields()
        self.a = self.g["anchor"]

    def test_budget_terms_sum_to_absolute_residual(self):
        budget = eb.entropy_budget(ST, self.g, self.f)
        self.assertAlmostEqual(
            budget["sum"], budget["absolute_entropy_residual"], places=12
        )
        self.assertLessEqual(abs(budget["mismatch"]), 1e-9)
        self.assertEqual(
            set(budget["terms"]),
            {"book_offset", "inlet_state_shift", "exit_temperature_shift",
             "exit_pressure_shift", "loss_shift"},
        )

    def test_each_current_state_perturbation_moves_its_term(self):
        base = eb.entropy_budget(ST, self.g, self.f)
        cases = [
            ("T1", "inlet_state_shift", -CP * math.log(1.01)),
            ("p1", "inlet_state_shift", R_GAS * math.log(1.01)),
        ]
        for key, term, delta in cases:
            with self.subTest(key=key):
                g2, f2 = _copy_state(self.g, self.f)
                g2[key][self.a] *= 1.01
                out = eb.entropy_budget(ST, g2, f2)
                self.assertAlmostEqual(
                    out["terms"][term] - base["terms"][term], delta, places=10
                )
                self.assertLessEqual(abs(out["mismatch"]), 1e-9)

        g2, f2 = _copy_state(self.g, self.f)
        f2["T2"][self.a] *= 1.01
        out = eb.entropy_budget(ST, g2, f2)
        self.assertAlmostEqual(
            out["terms"]["exit_temperature_shift"]
            - base["terms"]["exit_temperature_shift"],
            CP * math.log(1.01), places=10)
        self.assertLessEqual(abs(out["mismatch"]), 1e-9)

        g2, f2 = _copy_state(self.g, self.f)
        f2["p"][self.a] *= 1.01
        out = eb.entropy_budget(ST, g2, f2)
        self.assertAlmostEqual(
            out["terms"]["exit_pressure_shift"] - base["terms"]["exit_pressure_shift"],
            -R_GAS * math.log(1.01), places=10)
        self.assertLessEqual(abs(out["mismatch"]), 1e-9)

        g2, f2 = _copy_state(self.g, self.f)
        f2["dsl"][self.a] += 2.5
        out = eb.entropy_budget(ST, g2, f2)
        self.assertAlmostEqual(
            out["terms"]["loss_shift"] - base["terms"]["loss_shift"], -2.5, places=12
        )
        self.assertLessEqual(abs(out["mismatch"]), 1e-9)

    def test_book_offset_perturbations(self):
        base = eb.entropy_budget(ST, self.g, self.f)
        for key, delta in (
            ("T1", -CP * math.log(1.01)),
            ("p1", R_GAS * math.log(1.01)),
        ):
            with self.subTest(key=key):
                st2 = dict(ST)
                st2[key] *= 1.01
                out = eb.entropy_budget(st2, self.g, self.f)
                self.assertAlmostEqual(
                    out["terms"]["book_offset"] - base["terms"]["book_offset"],
                    delta, places=10)
                self.assertLessEqual(abs(out["mismatch"]), 1e-9)

    def test_zero_loss_psi_one(self):
        g2, f2 = _copy_state(self.g, self.f)
        g2["psi"] = 1.0
        f2["dsl"][:] = 0.0
        budget = eb.entropy_budget(ST, g2, f2)
        self.assertEqual(budget["book"]["hl"], 0.0)
        self.assertEqual(budget["terms"]["loss_shift"], 0.0)
        self.assertLessEqual(abs(budget["mismatch"]), 1e-9)
        self.assertEqual(eb.book_rotor_loss(123.0, 1.0), 0.0)

    def test_small_x_log_matches_quadratic(self):
        hl = 1.0
        T2 = 900.0
        result = eb.log_loss_comparison(hl, T2, CP)
        x = hl / (CP * T2)
        self.assertAlmostEqual(result["x"], x, places=16)
        self.assertAlmostEqual(
            result["difference"], 0.5 * CP * x * x, delta=CP * x ** 3
        )
        self.assertGreater(result["ds_state"], result["ds_l"])

    def test_domain_rejection(self):
        with self.assertRaises(eb.DomainError):
            eb.book_rotor_loss(400.0, 0.0)
        with self.assertRaises(eb.DomainError):
            eb.book_rotor_loss(400.0, 1.2)
        with self.assertRaises(eb.DomainError):
            eb.book_rotor_loss(-1.0, 0.96)
        with self.assertRaises(eb.DomainError):
            eb.log_loss_comparison(CP * 900.0, 900.0, CP)  # x == 1
        with self.assertRaises(eb.DomainError):
            eb.log_loss_comparison(-1.0, 900.0, CP)
        with self.assertRaises(eb.DomainError):
            eb.log_loss_comparison(1.0, 0.0, CP)
        with self.assertRaises(eb.DomainError):
            eb.log_loss_comparison(1.0, 900.0, 0.0)
        bad_st = dict(ST, T2=0.0)
        with self.assertRaises(eb.DomainError):
            eb.book_meanline_diagnostics(bad_st, R=R_GAS, psi=PSI)
        bad_st = dict(ST, p2=-1.0)
        with self.assertRaises(eb.DomainError):
            eb.book_meanline_diagnostics(bad_st, R=R_GAS, psi=PSI)
        bad_g, bad_f = _copy_state(self.g, self.f)
        bad_g["T1"][self.a] = 0.0
        with self.assertRaises(eb.DomainError):
            eb.entropy_budget(ST, bad_g, bad_f)
        bad_g, bad_f = _copy_state(self.g, self.f)
        bad_f["p"][self.a] = -1.0
        with self.assertRaises(eb.DomainError):
            eb.entropy_budget(ST, bad_g, bad_f)

    def test_book_meanline_identities_from_snapshot(self):
        diag = eb.book_meanline_diagnostics(ST, R=R_GAS, psi=PSI)
        self.assertLess(abs(diag["energy_identity_residual_K"]), 1e-9)
        self.assertLess(diag["cp_over_R"]["relative_difference"], 1e-12)
        self.assertTrue(diag["cp_over_R"]["applicable"])
        self.assertLess(abs(diag["pressure_reconstruction"]["relative_residual"]), 1e-12)
        self.assertAlmostEqual(diag["hl"]["key_abs_difference"], 0.0, places=12)
        self.assertEqual(diag["conditional_log"]["status"], "APPLICABLE")
        self.assertAlmostEqual(
            diag["conditional_log"]["ds_state"], diag["ds_state_book"], places=10
        )
        self.assertAlmostEqual(
            diag["conditional_log"]["difference"],
            diag["ds_state_book"] - diag["ds_l"], places=10,
        )
        self.assertLess(abs(diag["conditional_log"]["difference"]), 0.1)

    def test_conditional_identity_is_not_applicable_when_assumption_fails(self):
        st = dict(ST, T2_ад=ST["T2_ад"] + 0.5)
        diag = eb.book_meanline_diagnostics(st, R=R_GAS, psi=PSI)
        self.assertEqual(diag["conditional_log"]["status"], "NOT_APPLICABLE")
        self.assertIsNone(diag["conditional_log"]["ds_state"])
        self.assertIsNone(diag["conditional_log"]["difference"])
        self.assertNotEqual(diag["conditional_log"]["reason"], "")

    @unittest.skipUnless(RESULTS.exists(), "saved Stage 6B-4.5-B artifacts absent")
    def test_all_stages_saved_budget_identity(self):
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        cpnt = rc.control_point()
        for stage in data["stages"]:
            st = cpnt["st"][stage["stage"] - 1]
            finest = stage["meshes"][-1]
            g = {
                k: (np.asarray(v, dtype=float) if isinstance(v, list) else v)
                for k, v in finest["g"].items()
            }
            y = np.asarray(finest["best"]["y"], dtype=float)
            f = core.fields(g, y)
            budget = eb.entropy_budget(st, g, f)
            with self.subTest(stage=stage["stage"]):
                self.assertLessEqual(abs(budget["mismatch"]), 1e-9)
                self.assertLess(budget["absolute"]["span"], 1e-9)
                self.assertTrue(math.isfinite(budget["absolute_entropy_residual"]))
                self.assertAlmostEqual(
                    budget["absolute_entropy_residual"],
                    float(np.asarray(core.absolute_entropy_residual(g, f))[g["anchor"]]),
                    places=12,
                )


if __name__ == "__main__":
    unittest.main()

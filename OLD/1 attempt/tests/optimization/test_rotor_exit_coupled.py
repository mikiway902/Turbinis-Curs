"""Focused checks for the experimental Stage 6B-4.5-B rotor-exit closure."""
import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Optimizer"))

import rotor_exit_coupled as rc  # noqa: E402
from turbine.radial import rotor_exit as core  # noqa: E402
from turbine.radial.equilibrium import pressure_residual  # noqa: E402

# Stage-1 control-point snapshot (diagnostic book_approx values, SI units).
ST = {
    "D_c": 0.07319649716899385,
    "D_л": 0.07478269560502816,
    "l_c": 0.011196497312701255,
    "l_л": 0.01278269622409513,
    "p1": 670931.142875466,
    "p2": 542505.7849637904,
    "p0": 1048333.44,
    "T0": 1250.0,
    "T1_ад": 1120.856606630959,
    "T2": 1077.5660961613316,
    "c1u": 520.7266008954438,
    "c1a": 121.17687589337349,
    "α1": 13.10000000000003,
    "L_u": 194408.14477404466,
    "G_л": 0.6454104119416648,
    "c2a": 122.68425204552908,
    "cp": 1176.1897435517353,
    "k": 1.3233610672097846,
    "ψ": 0.96,
}
N_RPM = 90000.0
R = 287.4
PHI = 0.97

PASS = {
    "epsilon_RRE": True,
    "mass": True,
    "work": True,
    "r_forms": True,
    "positive_finite": True,
    "smooth": True,
    "w2_gt_ca": True,
    "alpha2": True,
    "entropy_anchor": True,
}


def _run(name, kind="cold", criteria=None, success=True, valid=True,
         same=True, delta=0.0, rel=0.0, equation_residual_inf=0.0):
    return {
        "start": name,
        "start_kind": kind,
        "success": success,
        "valid": valid,
        "same_as_best": same,
        "criteria": dict(PASS if criteria is None else criteria),
        "delta_c2a_m_s": delta,
        "rel_from_best": rel,
        "equation_residual_inf": equation_residual_inf,
    }


def _best(y, criteria=None, conditioning=True, success=True,
          equation_residual_inf=0.0):
    return {
        "valid": True,
        "success": success,
        "y": np.asarray(y, dtype=float),
        "criteria": dict(PASS if criteria is None else criteria),
        "conditioning": {"available": conditioning},
        "equation_residual_inf": equation_residual_inf,
    }


def _stage(cold_counts, conditioning=True, second_y=None, criteria=None):
    """Synthetic two-mesh stage result; every mesh gets a warm continuation."""
    y = np.linspace(-0.01, 0.01, 5)
    meshes = []
    for index, (N, count) in enumerate(zip((9, 17), cold_counts)):
        runs = [
            _run(f"cold{i}", criteria=criteria, same=True)
            for i in range(count)
        ]
        runs.append(_run("warm", kind="continuation", criteria=criteria, same=True))
        field = y if index == 0 or second_y is None else second_y
        meshes.append({
            "N": N,
            "runs": runs,
            "best": _best(field, criteria=criteria, conditioning=conditioning),
        })
    return {"meshes": meshes}


def _criterion_stage(key, metric, value):
    """Closed, cold-confirmed, mesh-continuous branch failing one criterion."""
    bad = dict(PASS, **{key: False})
    metrics = {metric: value}
    stage = _stage((2, 2), criteria=bad)
    for sol in stage["meshes"]:
        sol["best"]["metrics"] = dict(metrics)
    stage["bc_hub"] = {
        "best": {"valid": True, "criteria": dict(bad), "metrics": dict(metrics),
                 "equation_residual_inf": 0.0}
    }
    return stage


class RotorExitCoupledTests(unittest.TestCase):
    def setUp(self):
        self.g = rc.build_setup(ST, N_RPM, "Γ", R, PHI, 9, bc="mean", psi=0.96)
        self.y = np.linspace(-0.02, 0.02, 9)

    def test_equation_count_is_n_plus_n_minus_one(self):
        residual = rc.residual_vector(self.y, self.g)
        self.assertEqual(residual.shape, (9,))
        f = rc.fields(self.g, self.y)
        r = self.g["r2"]
        cu = self.g["cu2"]
        expected_fv = (f["p"][1:] - f["p"][:-1]) / np.diff(r) - 0.5 * (
            f["rho"][:-1] * cu[:-1] ** 2 / r[:-1]
            + f["rho"][1:] * cu[1:] ** 2 / r[1:]
        )
        np.testing.assert_allclose(
            residual[:-1], expected_fv / self.g["radial_scale"], rtol=1e-12, atol=1e-15
        )
        expected_mass = (
            np.trapezoid(f["rho"] * f["ca"] * 2 * math.pi * r, r) / self.g["G_l"] - 1.0
        )
        self.assertAlmostEqual(residual[-1], expected_mass, places=13)

    def test_algebraic_invariants(self):
        f = rc.fields(self.g, self.y)
        cp, Rg = self.g["cp"], self.g["R"]
        T2_energy = self.g["T1"] + (
            self.g["w1"] ** 2 - f["w2"] ** 2 + self.g["u2"] ** 2 - self.g["u1"] ** 2
        ) / (2 * cp)
        np.testing.assert_allclose(f["T2"], T2_energy, rtol=1e-14, atol=1e-12)
        work = self.g["cu1"] * self.g["u1"] + self.g["cu2"] * self.g["u2"]
        np.testing.assert_allclose(work, self.g["L_u"], rtol=1e-14)
        np.testing.assert_allclose(f["rho"], f["p"] / (Rg * f["T2"]), rtol=1e-14)
        np.testing.assert_allclose(
            f["s2"], self.g["s1"] + f["hl"] / f["T2"], rtol=1e-14, atol=1e-12
        )
        a = self.g["anchor"]
        entropy_difference = cp * np.log(f["T2"] / f["T2"][a]) - Rg * np.log(
            f["p"] / f["p"][a]
        )
        np.testing.assert_allclose(f["s2"] - f["s2"][a], entropy_difference, rtol=1e-12, atol=1e-10)
        self.assertAlmostEqual(f["p"][a], self.g["p_bc"], places=8)

    def test_book_inlet_pressure_anchor(self):
        p1 = rc._inlet_pressure(ST, "Γ", R, PHI, self.g["r1"])
        self.assertAlmostEqual(p1[self.g["mid"]], ST["p1"], places=6)
        self.assertTrue(np.all(np.isfinite(p1)))
        self.assertTrue(np.all(p1 > 0))

    def test_analytic_jacobian_matches_finite_difference(self):
        jac = rc.jacobian(self.y, self.g).toarray()
        step = 1e-6
        numeric = np.zeros_like(jac)
        base = rc.residual_vector(self.y, self.g)
        for j in range(self.y.size):
            shifted = self.y.copy()
            shifted[j] += step
            numeric[:, j] = (rc.residual_vector(shifted, self.g) - base) / step
        scale = max(float(np.max(np.abs(numeric))), 1e-30)
        self.assertLess(float(np.max(np.abs(jac - numeric))) / scale, 1e-5)

    def test_three_initial_guesses_have_distinct_scales(self):
        starts = rc.initial_guesses(17)
        self.assertEqual(set(starts), {"uniform", "hub-biased", "tip-biased"})
        mid = 8
        self.assertAlmostEqual(math.exp(starts["uniform"][mid]), 1.0, places=12)
        self.assertAlmostEqual(math.exp(starts["hub-biased"][mid]), 0.75, places=12)
        self.assertAlmostEqual(math.exp(starts["tip-biased"][mid]), 1.25, places=12)
        self.assertFalse(np.allclose(starts["hub-biased"], starts["tip-biased"]))

    def test_multiple_start_solve_is_graceful(self):
        starts = rc.initial_guesses(9)
        sol = rc.solve_stage(
            ST, N_RPM, "Γ", R, PHI, 9, stage=1, bc="mean", starts=starts, max_nfev=20
        )
        self.assertEqual(len(sol["runs"]), 3)
        self.assertEqual({run["start"] for run in sol["runs"]}, set(starts))
        self.assertIsNotNone(sol["best"])
        self.assertIn(
            rc.classify({"meshes": [sol]})[0], {"YES", "NO", "INCONCLUSIVE"}
        )

    def test_acceptance_uses_pressure_residual_not_entropy_augmented_form(self):
        f = rc.fields(self.g, self.y)
        r = self.g["r2"]
        dpdr = np.gradient(f["p"], r, edge_order=2) / f["rho"]
        rp = pressure_residual(
            r, f["p"], f["T2"], self.g["cu2"], gas_constant=self.g["R"]
        )
        # Rejected form: (1/rho)dp/dr - cu^2/r + T ds/dr.
        rejected = dpdr - self.g["cu2"] ** 2 / r + f["T2"] * np.gradient(
            f["s2"], r, edge_order=2
        )
        self.assertGreater(float(np.max(np.abs(rejected - rp))), 1e-3)
        m = rc.metrics(self.g, self.y, ST)
        self.assertAlmostEqual(m["R_p_max"], float(np.max(np.abs(rp))), places=10)
        self.assertAlmostEqual(
            m["epsilon_RRE"],
            float(np.max(np.abs(rp)) / np.max(np.abs(dpdr))),
            places=14,
        )
        self.assertNotAlmostEqual(
            m["epsilon_RRE"],
            float(np.max(np.abs(rejected)) / np.max(np.abs(dpdr))),
            places=3,
        )

    def test_continuation_is_not_independent(self):
        stage = _stage((1, 1))
        analysis = rc.branch_analysis(stage)
        self.assertFalse(analysis["confirmed"])
        self.assertEqual(analysis["confirmed_meshes"], [])
        for conf in analysis["confirmations"]:
            self.assertEqual(conf["cold_on_branch"], ["cold0"])
            self.assertNotIn("warm", conf["cold_on_branch"])

    def test_single_cold_branch_is_inconclusive(self):
        label, reasons = rc.classify(_stage((1, 1)))
        self.assertEqual(label, "INCONCLUSIVE")
        self.assertTrue(reasons)

    def test_yes_requires_two_cold_continuity_and_conditioning(self):
        self.assertEqual(rc.classify(_stage((2, 2)))[0], "YES")
        self.assertEqual(rc.classify(_stage((2, 2), conditioning=False))[0], "INCONCLUSIVE")
        self.assertEqual(
            rc.classify(_stage((2, 2), second_y=np.linspace(-0.2, 0.2, 5)))[0],
            "INCONCLUSIVE",
        )
        self.assertEqual(rc.classify(_stage((1, 1)))[0], "INCONCLUSIVE")

    def test_failing_r_forms_does_not_crash_and_is_no(self):
        bad = dict(PASS, r_forms=False)
        metrics = {"delta_T": 1e-2, "delta_GRE": 2e-2}
        stage = _stage((2, 2), criteria=bad)
        for sol in stage["meshes"]:
            sol["best"]["metrics"] = dict(metrics)
        stage["bc_hub"] = {
            "best": {"valid": True, "criteria": dict(bad), "metrics": dict(metrics),
                     "equation_residual_inf": 0.0}
        }
        label, reasons = rc.classify(stage)
        self.assertEqual(label, "NO")
        self.assertIn("r_forms", " ".join(reasons))

    def test_failing_smoothness_does_not_crash_and_is_no(self):
        label, reasons = rc.classify(
            _criterion_stage("smooth", "smoothness", 2.0 * rc.SMOOTH_TOL)
        )
        self.assertEqual(label, "NO")
        self.assertIn("smooth", " ".join(reasons))
        with self.subTest(criterion="smooth", value="1.1 x cap"):
            label, reasons = rc.classify(
                _criterion_stage("smooth", "smoothness", 1.1 * rc.SMOOTH_TOL)
            )
            self.assertEqual(label, "INCONCLUSIVE")
            self.assertIn("marginal", " ".join(reasons))

    def test_failing_work_residual_does_not_crash_and_is_no(self):
        label, reasons = rc.classify(
            _criterion_stage("work", "work_res", 10.0 * rc.WORK_TOL)
        )
        self.assertEqual(label, "NO")
        self.assertIn("work", " ".join(reasons))
        with self.subTest(criterion="work", value="1.1 x cap"):
            label, reasons = rc.classify(
                _criterion_stage("work", "work_res", 1.1 * rc.WORK_TOL)
            )
            self.assertEqual(label, "INCONCLUSIVE")
            self.assertIn("marginal", " ".join(reasons))

    def test_absolute_entropy_exact_anchor_is_zero(self):
        f = rc.fields(self.g, self.y)
        p_exact = self.g["p1"] * np.exp(
            (self.g["cp"] * np.log(f["T2"] / self.g["T1"]) - f["dsl"]) / self.g["R"]
        )
        residual = core.absolute_entropy_residual(self.g, dict(f, p=p_exact))
        self.assertLess(float(np.max(np.abs(residual))), 1e-9)

    def test_absolute_entropy_shifts_exactly_with_pressure_offset(self):
        f = rc.fields(self.g, self.y)
        base = core.absolute_entropy_residual(self.g, f)
        delta = 12.5
        shifted = core.absolute_entropy_residual(
            self.g, dict(f, p=f["p"] * np.exp(delta / self.g["R"]))
        )
        np.testing.assert_allclose(shifted - base, -delta, rtol=0.0, atol=1e-9)

    def test_current_relative_construction_has_constant_offset(self):
        f = rc.fields(self.g, self.y)
        residual = core.absolute_entropy_residual(self.g, f)
        self.assertLess(float(np.max(residual) - np.min(residual)), 1e-9)
        self.assertGreater(float(np.max(np.abs(residual))), 1e-3)

    def test_metrics_and_criteria_reject_entropy_offset(self):
        m = rc.metrics(self.g, self.y, ST)
        crit = rc.criteria_from_metrics(m)
        self.assertIn("entropy_anchor", crit)
        self.assertGreater(m["entropy_max_abs"], rc.ABS_ENTROPY_TOL)
        self.assertFalse(crit["entropy_anchor"])

    def test_unclosed_same_shape_run_is_not_branch_evidence(self):
        y = np.linspace(-0.01, 0.01, 5)
        meshes = []
        for N in (81, 161):
            runs = [_run("cold0"), _run("cold1")]
            if N == 161:
                # Model the Stage-2 N=161 cold endpoint: same shape, unfinished.
                runs.append(_run(
                    "cold_endpoint",
                    criteria=dict(PASS, entropy_anchor=False),
                    delta=500.0, rel=1.0, equation_residual_inf=1e-3,
                ))
            meshes.append({"N": N, "runs": runs, "best": _best(y)})
        stage = {"meshes": meshes}
        analysis = rc.branch_analysis(stage)
        for conf in analysis["confirmations"]:
            self.assertNotIn("cold_endpoint", conf["cold_on_branch"])
        self.assertEqual(
            [c["cold_on_branch"] for c in analysis["confirmations"] if c["N"] == 161],
            [["cold0", "cold1"]],
        )
        self.assertFalse(any(
            item["start"] == "cold_endpoint" for item in analysis["inconsistent"]
        ))
        unfinished = [u for u in analysis["unfinished_near_branch"] if u["N"] == 161]
        self.assertEqual([u["start"] for u in unfinished], ["cold_endpoint"])
        self.assertEqual(unfinished[0]["equation_residual_inf"], 1e-3)
        members = rc._branch_runs(meshes[1]["runs"], meshes[1]["best"])
        self.assertNotIn("cold_endpoint", [r["start"] for r in members])
        self.assertEqual(max(r["delta_c2a_m_s"] for r in members), 0.0)

    def test_unclosed_alternative_bc_cannot_support_no(self):
        bad = dict(PASS, r_forms=False)
        metrics = {"delta_T": 1e-2, "delta_GRE": 2e-2}
        stage = _stage((2, 2), criteria=bad)
        for sol in stage["meshes"]:
            sol["best"]["metrics"] = dict(metrics)
        hub_best = {
            "valid": True, "criteria": dict(bad), "metrics": dict(metrics),
            "equation_residual_inf": 1e-2,
        }
        stage["bc_hub"] = {"best": hub_best}
        label, reasons = rc.classify(stage)
        self.assertEqual(label, "INCONCLUSIVE")
        self.assertIn("not numerically closed", " ".join(reasons))
        hub_best["equation_residual_inf"] = 0.0
        self.assertEqual(rc.classify(stage)[0], "NO")

    def test_branch_spread_units_and_non_branch_exclusion(self):
        old = rc.SAME_SOLUTION_RTOL
        try:
            rc.SAME_SOLUTION_RTOL = 0.0
            sol = rc.solve_stage(
                ST, N_RPM, "Γ", R, PHI, 9, stage=1, bc="mean",
                starts=rc.initial_guesses(9), max_nfev=100,
            )
        finally:
            rc.SAME_SOLUTION_RTOL = old
        best = sol["best"]
        ca_ref = sol["g"]["ca_ref"]
        for run in sol["runs"]:
            if run["valid"] and run.get("y") is not None:
                expected = float(np.max(np.abs(
                    ca_ref * (np.exp(run["y"]) - np.exp(best["y"]))
                )))
                np.testing.assert_allclose(
                    run["delta_c2a_m_s"], expected, rtol=1e-12, atol=1e-9
                )
        branch = rc._branch_runs(sol["runs"], best)
        self.assertEqual(best["branch_run_count"], len(branch))
        spread = max(r["delta_c2a_m_s"] for r in branch)
        self.assertAlmostEqual(best["branch_spread_max_m_s"], spread, places=12)
        sol["runs"].append(
            _run("different-branch", same=False, delta=123.0, rel=1.0)
        )
        branch_after = rc._branch_runs(sol["runs"], best)
        self.assertEqual(best["branch_run_count"], len(branch_after))
        self.assertNotIn("different-branch", [r["start"] for r in branch_after])
        self.assertAlmostEqual(best["branch_spread_max_m_s"], spread, places=12)

    def test_sparse_conditioning_matches_dense_small(self):
        g = rc.build_setup(ST, N_RPM, "Γ", R, PHI, 17, bc="mean", psi=0.96)
        cond = rc.conditioning_estimate(np.linspace(-0.02, 0.02, 17), g)
        self.assertTrue(cond["available"])
        self.assertIn("dense_cond_1", cond)
        dense = cond["dense_cond_1"]
        estimate = cond["condition_1norm_estimate"]
        self.assertLess(abs(estimate - dense) / dense, 0.05)

    def test_optimizer_wrappers_match_shared_core(self):
        g1 = rc.build_setup(ST, N_RPM, "Γ", R, PHI, 9, bc="mean", psi=0.96)
        g2 = core.build_setup(ST, N_RPM, "Γ", R, PHI, 9, bc="mean", psi=0.96)
        self.assertEqual(set(g1), set(g2))
        for key in g1:
            left, right = g1[key], g2[key]
            if isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            else:
                self.assertEqual(left, right)
        for law in ("Γ", "α1"):
            for left, right in zip(
                rc._stator(ST, N_RPM, law, R, PHI, g1["r1"]),
                core.stator_profile(ST, law, R, PHI, g1["r1"]),
            ):
                np.testing.assert_array_equal(left, right)
            for left, right in zip(
                rc._inlet_pressure(ST, law, R, PHI, g1["r1"]),
                core.inlet_pressure(ST, law, R, PHI, g1["r1"]),
            ):
                np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(
            rc._trapezoid_weights(g1["r2"]), core.trapezoid_weights(g1["r2"])
        )
        f1, f2 = rc.fields(g1, self.y), core.fields(g2, self.y)
        self.assertEqual(set(f1), set(f2))
        for key in f1:
            np.testing.assert_array_equal(f1[key], f2[key])
        for left, right in zip(
            rc._derivatives(g1, f1), core.derivatives(g2, f2)
        ):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(
            core.absolute_entropy_residual(g1, f1),
            core.absolute_entropy_residual(g2, f2),
        )
        np.testing.assert_array_equal(
            rc.residual_vector(self.y, g1), core.residual(self.y, g2)
        )
        np.testing.assert_array_equal(
            rc.jacobian(self.y, g1).toarray(), core.jacobian(self.y, g2).toarray()
        )

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy is not installed")
    def test_control_point_resolves_alpha1_stage1(self):
        cp = rc.control_point()
        self.assertEqual(len(cp["st"]), 3)
        self.assertAlmostEqual(cp["alpha1_stage1"], 13.10, delta=0.01)
        self.assertAlmostEqual(cp["n"], 90000.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()

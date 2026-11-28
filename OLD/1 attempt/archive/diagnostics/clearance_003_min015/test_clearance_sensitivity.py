"""Focused checks for the explicit EXPERIMENTAL clearance variant.

The variant (delta_rel = 0.003, delta_min = 0.15 mm) is a diagnostic entry
point, not a project default. All physical work is done by the existing
notebook stage calculator, reached through the existing optimizer namespaces.
"""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Optimizer"))

import clearance_sensitivity as cs  # noqa: E402
import optimizer_core as oc  # noqa: E402
import optimize_extended as oe  # noqa: E402
import rotor_exit_coupled as rc  # noqa: E402

STRICT_ALPHA2 = ((70.0, 90.0), (70.0, 90.0), (80.0, 90.0))


class TestClearanceVariant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = cs.run()
        cls.model = oc.load_model()
        cls.ns_new, _ = oe.build_ns(cls.model, cs.Q_N, cs.Q_N**2 * oe.SIGMA_LIM)
        cls.ns_new["δ_отн"] = cs.CASE["delta_rel"]
        cls.ns_new["δ_мин"] = cs.CASE["delta_min_m"]
        cls.ns_old, _ = oe.build_ns(cls.model, cs.Q_N, cs.Q_N**2 * oe.SIGMA_LIM)
        cls.ns_old["δ_отн"] = cs.LEGACY["delta_rel"]
        cls.ns_old["δ_мин"] = cs.LEGACY["delta_min_m"]
        cls.ns_zero, _ = oe.build_ns(cls.model, cs.Q_N, cs.Q_N**2 * oe.SIGMA_LIM)
        cls.ns_zero["δ_отн"] = 0.0
        cls.ns_zero["δ_мин"] = 0.0

    def _scaled_fixture(self, height_m):
        row = self.res["frozen"]["stages"][0]
        inputs = dict(row["inputs"])
        inputs["G_л"] = inputs["G_л"] * height_m / row["old"]["l_rotor_m"]
        return inputs

    @staticmethod
    def _three(inputs):
        return [dict(inputs) for _ in range(3)]

    def test_old_defaults_unchanged(self):
        ns, _ = oe.build_ns(self.model, cs.Q_N, cs.Q_N**2 * oe.SIGMA_LIM)
        self.assertEqual(ns["δ_отн"], 0.005)
        self.assertEqual(ns["δ_мин"], 0.2e-3)
        self.assertEqual((cs.CASE["delta_rel"], cs.CASE["delta_min_m"]), (0.003, 0.15e-3))

    def test_new_case_gap_is_150_micrometres_at_current_heights(self):
        for row in self.res["frozen"]["stages"]:
            self.assertGreater(row["new"]["l_rotor_m"], 0.0)
            self.assertLess(row["new"]["l_rotor_m"], 0.050)
            self.assertAlmostEqual(row["new"]["delta_m"], 0.15e-3, delta=1e-15)
            self.assertAlmostEqual(row["old"]["delta_m"], 0.2e-3, delta=1e-15)

    def test_frozen_hz_ratio_is_0_75(self):
        expected = cs.CASE["delta_min_m"] / cs.LEGACY["delta_min_m"]
        self.assertAlmostEqual(expected, 0.75, delta=1e-15)
        for row in self.res["frozen"]["stages"]:
            self.assertLessEqual(abs(row["ratio"] - 0.75), 1e-12)
        self.assertLessEqual(self.res["frozen"]["max_ratio_error"], 1e-12)

    def test_zero_clearance_gives_zero_loss(self):
        inputs = dict(self.res["frozen"]["stages"][0]["inputs"])
        row = cs.local_losses(self.ns_zero, [inputs])[0]
        self.assertEqual(row["hz_J_kg"], 0.0)

    def test_50mm_boundary_matches_floor_and_relative_branch(self):
        row = cs.local_losses(self.ns_new, [self._scaled_fixture(0.050)])[0]
        self.assertAlmostEqual(row["l_rotor_m"], 0.050, delta=1e-12)
        self.assertAlmostEqual(row["delta_m"], 0.15e-3, delta=1e-15)
        # At 50 mm the relative gap (0.003*0.050) equals the floor.
        self.assertAlmostEqual(row["delta_m"], 0.003 * row["l_rotor_m"], delta=1e-15)

    def test_100mm_boundary_uses_relative_branch(self):
        row = cs.local_losses(self.ns_new, [self._scaled_fixture(0.100)])[0]
        self.assertAlmostEqual(row["l_rotor_m"], 0.100, delta=1e-12)
        self.assertAlmostEqual(row["delta_m"], 0.3e-3, delta=1e-15)
        self.assertGreater(row["delta_m"], 0.15e-3)
        self.assertAlmostEqual(row["delta_m"], 0.003 * row["l_rotor_m"], delta=1e-15)

    def test_whole_meanline_power_closure_and_finite_fields(self):
        list_fields = (
            "T0_K", "p0_Pa", "T1_K", "p1_Pa", "T2_K", "p2_Pa",
            "heights_m", "delta_m", "hz_J_kg", "L_u_J_kg", "L_t_J_kg",
            "G_stator_kg_s", "G_rotor_kg_s",
        )
        for case in ("old", "new"):
            m = self.res["meanline"][case]
            self.assertLessEqual(abs(m["N_rel"]), cs.TOL_POWER)
            self.assertAlmostEqual(m["N_W"], 110e3, delta=1e-10 * 110e3)
            for key in list_fields:
                for value in m[key]:
                    self.assertTrue(math.isfinite(value) and value > 0, (case, key, value))
            for key in ("alpha1_deg", "alpha2_deg"):
                for value in m[key]:
                    self.assertTrue(math.isfinite(value), (case, key, value))
            for key in ("G_air_kg_s", "eta_t", "sigma_max_MPa", "q_n",
                        "u_max_m_s", "u_mean_m_s"):
                self.assertTrue(math.isfinite(m[key]) and m[key] > 0, (case, key))
            ok = all(lo <= a <= hi for (lo, hi), a in zip(STRICT_ALPHA2, m["alpha2_deg"]))
            self.assertEqual(m["strict_alpha2"], ok)
            if not ok:
                self.assertFalse(self.res["feasible"])

    def test_current_16mm_root_height_is_not_feasible(self):
        self.assertFalse(self.res["feasible"])
        height_checks = [c for c in self.res["constraints"] if "l_rk1" in c["name"]]
        self.assertTrue(height_checks)
        self.assertTrue(all(not c["pass"] for c in height_checks))
        for case in ("old", "new"):
            self.assertLess(self.res["meanline"][case]["heights_m"][0], 0.016)

    def test_same_control_point_x_for_both_cases(self):
        self.assertEqual(self.res["meanline"]["old"]["x"],
                         self.res["meanline"]["new"]["x"])
        self.assertEqual(self.res["control_point"]["x"],
                         self.res["meanline"]["new"]["x"])

    def test_old_regression_and_new_case_is_not_a_tuned_target(self):
        baseline = json.loads(
            (ROOT / "reference/baselines/book_approx_control_001.json").read_text(encoding="utf-8")
        )
        old = self.res["meanline"]["old"]["eta_t"]
        self.assertAlmostEqual(old, baseline["expected"]["eta_t_total"], delta=1e-4)
        comp = self.res["comparison"]
        self.assertEqual(comp["eta_old"], old)
        self.assertAlmostEqual(comp["delta_abs"], comp["eta_new"] - comp["eta_old"], places=15)
        self.assertAlmostEqual(comp["delta_points"], 100.0 * comp["delta_abs"], places=12)
        self.assertAlmostEqual(comp["delta_rel_pct"],
                               100.0 * (comp["eta_new"] / comp["eta_old"] - 1.0), places=12)
        self.assertNotEqual(comp["delta_points"], comp["delta_rel_pct"])
        self.assertIs(self.res["case"]["global_default_migrated"], False)

    def test_invalid_summary_fails_closed(self):
        bad = copy.deepcopy(self.res)
        bad["comparison"]["eta_new"] = float("nan")
        summary = cs.classify(bad)
        self.assertFalse(summary["software_pass"])
        self.assertTrue(summary["problems"])
        self.assertTrue(cs.classify(self.res)["software_pass"])

    def test_static_heated_continuity_uses_static_state(self):
        # Re-evaluate the existing stage calculator from the recorded inputs.
        inp = self.res["meanline"]["new"]["stage_inputs"][0]
        s = self.ns_new["ступень"](inp["T0"], inp["p0"], inp["H0"], inp["ρ"],
                                   inp["D_c"], inp["D_л"], inp["n_rpm"],
                                   inp["G_c"], inp["G_л"], inp["α1"], inp["κ"])
        gas_r = float(self.ns_new["R"])
        area = math.pi * s["D_л"] * s["l_л"]
        rho_static = s["p2"] / (gas_r * s["T_ст"])
        signed = rho_static * area * s["c2a"] / s["G_л"] - 1.0
        # Existing continuity identity A = G*v2/(c2a*pi*D) makes this T2/Tст-1.
        self.assertAlmostEqual(signed, s["T2"] / s["T_ст"] - 1.0, places=13)
        m = self.res["meanline"]["new"]
        self.assertAlmostEqual(m["mass_static_heated_signed_rel_error"][0], signed, delta=1e-12)
        self.assertAlmostEqual(m["rho_static_heated_kg_m3"][0], rho_static, delta=1e-15)
        self.assertAlmostEqual(m["p_static_heated_Pa"][0], s["p2"], delta=1e-6)
        self.assertAlmostEqual(m["T_static_heated_K"][0], s["T_ст"], delta=1e-12)
        # Total state is reported separately and is not used as a density.
        self.assertAlmostEqual(m["T_total_heated_K"][0], s["T_ст_п"], delta=1e-12)
        self.assertAlmostEqual(m["p_total_heated_Pa"][0], s["p_ст_п"], delta=1e-6)
        old_total = s["p_ст_п"] / (gas_r * s["T_ст_п"]) * area * s["c2a"] / s["G_л"] - 1.0
        self.assertGreater(abs(old_total - signed), 1e-3)
        for key in ("p_static_heated_Pa", "T_static_heated_K", "rho_static_heated_kg_m3"):
            for value in m[key]:
                self.assertTrue(math.isfinite(value) and value > 0, (key, value))

    def test_classify_requires_every_expected_check_once_and_true(self):
        names = [c["name"] for c in self.res["checks"]]
        self.assertEqual(len(names), len(set(names)))
        missing = copy.deepcopy(self.res)
        missing["checks"] = missing["checks"][:-1]
        duplicate = copy.deepcopy(self.res)
        duplicate["checks"].append(copy.deepcopy(duplicate["checks"][0]))
        false = copy.deepcopy(self.res)
        false["checks"][0]["pass"] = False
        for label, bad in (("missing", missing), ("duplicate", duplicate), ("false", false)):
            summary = cs.classify(bad)
            self.assertFalse(summary["software_pass"], label)
            self.assertTrue(summary["problems"], label)

    def test_classify_recomputes_checks_instead_of_trusting_booleans(self):
        bad = copy.deepcopy(self.res)
        bad["meanline"]["new"]["N_rel"] = 1e-3
        summary = cs.classify(bad)
        self.assertFalse(summary["software_pass"])
        self.assertTrue(any("stale" in p or "recomputed" in p for p in summary["problems"]),
                        summary["problems"])

    def test_classify_rejects_nan_or_missing_stage_arrays(self):
        mutations = (
            lambda r: r["meanline"]["new"].__setitem__("heights_m", []),
            lambda r: r["meanline"]["new"].__setitem__("T2_K", [1.0, float("nan"), 1.0]),
            lambda r: r["meanline"]["new"].__setitem__("N_rel", float("nan")),
            lambda r: r["meanline"]["new"].pop("rho_static_heated_kg_m3"),
            lambda r: r["comparison"].__setitem__("eta_new", float("nan")),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.res)
            mutate(bad)
            self.assertFalse(cs.classify(bad)["software_pass"])

    def test_classify_rejects_missing_or_empty_protected_hashes(self):
        mutations = (
            lambda r: r.__setitem__("protected", {}),
            lambda r: r["protected"].pop("before"),
            lambda r: r["protected"].__setitem__("after", {}),
            lambda r: r["protected"]["before"].__setitem__("x", "not-a-hash"),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.res)
            mutate(bad)
            self.assertFalse(cs.classify(bad)["software_pass"])

    def test_classification_never_claims_engineering_pass(self):
        summary = cs.classify(self.res)
        self.assertIs(summary["engineering_pass"], False)
        self.assertIn("EXPERIMENTAL", summary["engineering_status"])
        inflated = copy.deepcopy(self.res)
        inflated["feasible"] = True
        summary = cs.classify(inflated)
        self.assertIs(summary["engineering_pass"], False)
        self.assertIn("EXPERIMENTAL", summary["engineering_status"])

    def test_root_reaction_is_a_geometry_estimate_not_the_baseline(self):
        info = self.res["rho_root_method"]
        self.assertEqual(info["method"], "approximate_geometry_exponent_estimate")
        self.assertIs(info["not_a_radial_profile"], True)
        self.assertIs(info["same_as_baseline"], False)
        self.assertIs(info["baseline_changed"], False)
        self.assertIn("optimize_extended.py", info["source"])
        values = self.res["meanline"]["old"]["rho_root_stages_1_2"]
        self.assertAlmostEqual(values[0], 0.047859, delta=1e-5)
        baseline = json.loads(
            (ROOT / "reference/baselines/book_approx_control_001.json").read_text(encoding="utf-8")
        )
        stored = baseline["expected"]["root_reaction_book_approx"]
        self.assertGreater(abs(values[0] - stored[0]), 1e-3)
        self.assertAlmostEqual(values[1], stored[1], delta=5e-5)

    def test_main_exit_code_tracks_software_acceptance_only(self):
        ok = copy.deepcopy(self.res)
        bad = copy.deepcopy(self.res)
        bad["checks"][0]["pass"] = False
        bad["classification"] = cs.classify(bad)
        write = mock.patch.object(cs, "write_evidence",
                                  return_value=(Path("x.md"), Path("x.json")))
        with mock.patch.object(cs, "run", side_effect=[ok, bad, bad]), write, \
                mock.patch("builtins.print"):
            self.assertEqual(cs.main(), 0)   # engineering FAIL only -> exit 0
            self.assertEqual(cs.main(), 1)   # software FAIL -> exit 1
            with self.assertRaises(SystemExit) as caught:
                raise SystemExit(cs.main())
            self.assertEqual(caught.exception.code, 1)

    def test_write_evidence_rejects_nan(self):
        bad = copy.deepcopy(self.res)
        bad["comparison"]["eta_new"] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                cs.write_evidence(bad, Path(tmp) / "r.md", Path(tmp) / "r.json")

    def test_report_separates_status_and_lists_unchecked_items(self):
        md = cs._markdown(self.res)
        for needle in ("## Software checks", "## Engineering acceptance: FAIL",
                       "## Static heated exit continuity", "## Unresolved / NOT_CHECKED",
                       "stage-3 root reaction", "stator", "mesh", "multiple starts",
                       "material", "life"):
            self.assertIn(needle, md)

    def test_protected_sources_unchanged(self):
        protected = self.res["protected"]
        self.assertTrue(protected["before"])
        self.assertEqual(protected["before"], protected["after"])
        self.assertIn("Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb", protected["before"])
        self.assertIn("Optimizer/rotor_exit_coupled.py", protected["before"])

    def test_compare_frozen_requires_exactly_three_rows(self):
        inputs = self.res["frozen"]["stages"][0]["inputs"]
        for bad in (self._three(inputs)[:2], self._three(inputs) + [dict(inputs)]):
            with self.assertRaises(ValueError):
                cs.compare_frozen(bad, self.ns_old, self.ns_new)

    def test_compare_frozen_rejects_nonpositive_hz(self):
        inputs = self.res["frozen"]["stages"][0]["inputs"]
        with self.assertRaises(ValueError):
            cs.compare_frozen(self._three(inputs), self.ns_zero, self.ns_new)
        with self.assertRaises(ValueError):
            cs.compare_frozen(self._three(inputs), self.ns_old, self.ns_zero)

    def test_compare_frozen_rejects_nonfinite_rows(self):
        inputs = self.res["frozen"]["stages"][0]["inputs"]
        good_old = cs.local_losses(self.ns_old, self._three(inputs))
        good_new = cs.local_losses(self.ns_new, self._three(inputs))
        for key in ("hz_J_kg", "delta_m", "l_rotor_m"):
            broken = copy.deepcopy(good_new)
            broken[0][key] = float("nan")
            with mock.patch.object(cs, "local_losses", side_effect=[good_old, broken]):
                with self.assertRaises(ValueError):
                    cs.compare_frozen(self._three(inputs), self.ns_old, self.ns_new)

    def test_compare_frozen_ratio_follows_delta_on_both_branches(self):
        # The empirical hz formula is linear in delta/l; l and all other factors
        # are identical, so the expected ratio is delta_new/delta_old.
        floor_inputs = self.res["frozen"]["stages"][0]["inputs"]
        floor = cs.compare_frozen(self._three(floor_inputs), self.ns_old, self.ns_new)
        self.assertEqual(len(floor["stages"]), 3)
        for row in floor["stages"]:
            expected = row["new"]["delta_m"] / row["old"]["delta_m"]
            self.assertAlmostEqual(expected, 0.75, delta=1e-12)
            self.assertTrue(math.isclose(row["ratio"], expected, rel_tol=1e-12, abs_tol=1e-15))
            self.assertTrue(row["local_fields_unchanged"])
        self.assertLessEqual(floor["max_ratio_error"], 1e-12)

        relative = cs.compare_frozen(self._three(self._scaled_fixture(0.100)),
                                     self.ns_old, self.ns_new)
        for row in relative["stages"]:
            expected = row["new"]["delta_m"] / row["old"]["delta_m"]
            self.assertAlmostEqual(expected, 0.6, delta=1e-12)
            self.assertGreater(row["new"]["delta_m"], cs.CASE["delta_min_m"])
            self.assertTrue(math.isclose(row["ratio"], expected, rel_tol=1e-12, abs_tol=1e-15))
        self.assertLessEqual(relative["max_ratio_error"], 1e-12)

        for result in (floor, relative):
            recomputed = max(abs(row["ratio"] - row["new"]["delta_m"] / row["old"]["delta_m"])
                             for row in result["stages"])
            self.assertTrue(math.isclose(result["max_ratio_error"], recomputed,
                                         rel_tol=1e-12, abs_tol=1e-15))

    def test_classify_rejects_malformed_frozen_evidence(self):
        mutations = (
            lambda r: r["frozen"]["stages"].pop(),
            lambda r: r["frozen"]["stages"][0]["old"].__setitem__("hz_J_kg", 0.0),
            lambda r: r["frozen"]["stages"][0]["new"].__setitem__("hz_J_kg", float("nan")),
            lambda r: r["frozen"]["stages"][0].__setitem__("ratio", 0.5),
            lambda r: r["frozen"].__setitem__("max_ratio_error", 0.25),
            lambda r: r["frozen"].__setitem__("max_ratio_error", float("nan")),
            lambda r: r["frozen"]["stages"][0]["old"].__setitem__("delta_m", 0.0),
            lambda r: r["frozen"]["stages"][0]["old"].__setitem__("delta_m", 0.2e-3 * 1.01),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.res)
            mutate(bad)
            summary = cs.classify(bad)
            self.assertFalse(summary["software_pass"], summary["problems"])

    def test_classify_rejects_inconsistent_constraint_evidence(self):
        mutations = (
            lambda r: r["constraints"].pop(),
            lambda r: r["constraints"].append(copy.deepcopy(r["constraints"][0])),
            lambda r: r["constraints"].reverse(),
            lambda r: r["constraints"][0].__setitem__(
                "pass", not r["constraints"][0]["pass"]),
            lambda r: r["constraints"][6].__setitem__(
                "pass", not r["constraints"][6]["pass"]),
            lambda r: r["constraints"][0].__setitem__("pass", 1),
            lambda r: r["constraints"][0].__setitem__("detail", "corrupted"),
            lambda r: r["constraints"][0].__setitem__("limit", "corrupted"),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.res)
            mutate(bad)
            summary = cs.classify(bad)
            self.assertFalse(summary["software_pass"], summary["problems"])

    def test_classify_rejects_feasible_mismatch_or_non_bool(self):
        self.assertIs(self.res["feasible"], False)
        for value in (True, 0, 1, "False", None):
            bad = copy.deepcopy(self.res)
            bad["feasible"] = value
            summary = cs.classify(bad)
            self.assertFalse(summary["software_pass"], (value, summary["problems"]))

    def test_classify_recomputes_comparison_from_meanline(self):
        mutations = (
            lambda r: r["comparison"].__setitem__(
                "delta_points", r["comparison"]["delta_points"] + 1e-6),
            lambda r: r["comparison"].__setitem__(
                "eta_old", r["comparison"]["eta_old"] * (1.0 + 1e-9)),
            lambda r: r["comparison"].__setitem__(
                "power_old_kW", r["comparison"]["power_old_kW"] * (1.0 + 1e-9)),
            lambda r: r["comparison"].pop("delta_rel_pct"),
            lambda r: r.__setitem__("comparison", "not-a-dict"),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.res)
            mutate(bad)
            summary = cs.classify(bad)
            self.assertFalse(summary["software_pass"], summary["problems"])

    @staticmethod
    def _resync_all(res):
        """Refresh every stored summary from the raw meanline, as run() does."""
        res["checks"] = cs._checks(res)
        res["constraints"] = cs._constraints(res)
        res["comparison"] = cs._comparison(res["meanline"]["old"], res["meanline"]["new"])
        res["feasible"] = all(c["pass"] for c in res["constraints"])

    def test_classify_rejects_power_that_does_not_match_n_w(self):
        honest = copy.deepcopy(self.res)
        self._resync_all(honest)
        self.assertTrue(cs.classify(honest)["software_pass"])
        self.assertTrue(cs.classify(json.loads(json.dumps(honest)))["software_pass"])

        # (a) N_W moved alone: the stored N_rel and N_W no longer agree.
        stray = copy.deepcopy(self.res)
        stray["meanline"]["new"]["N_W"] *= 1.001
        self.assertFalse(cs.classify(stray)["software_pass"])

        # (b) N_W moved and every stored summary re-derived from it, so only the
        # unchanged N_rel contradicts N_W/N_TARGET - 1. This is the reviewer case.
        relabelled = copy.deepcopy(self.res)
        relabelled["meanline"]["new"]["N_W"] *= 1.001
        self._resync_all(relabelled)
        self.assertAlmostEqual(relabelled["comparison"]["power_new_kW"],
                               relabelled["meanline"]["new"]["N_W"] / 1e3, delta=1e-12)
        summary = cs.classify(relabelled)
        self.assertFalse(summary["software_pass"], summary["problems"])
        self.assertIn("recomputed_fail:power_new", summary["problems"])

    def test_classify_honest_result_is_software_only(self):
        summary = cs.classify(copy.deepcopy(self.res))
        self.assertTrue(summary["software_pass"], summary["problems"])
        self.assertEqual(summary["problems"], [])
        self.assertIs(summary["engineering_pass"], False)
        self.assertIs(summary["verified_claim"], False)
        self.assertIn("EXPERIMENTAL", summary["engineering_status"])

    @staticmethod
    def _resync_constraints(res):
        res["constraints"] = cs._constraints(res)
        res["feasible"] = all(c["pass"] for c in res["constraints"])

    def test_classify_rejects_frozen_inputs_altered_or_removed(self):
        mutations = (
            lambda r: r["frozen"]["stages"][0]["inputs"].__setitem__(
                "G_л", r["frozen"]["stages"][0]["inputs"]["G_л"] * 8.0),
            lambda r: r["frozen"]["stages"][1]["inputs"].pop("n_rpm"),
            lambda r: r["frozen"]["stages"][2].pop("inputs"),
            lambda r: r["frozen"]["stages"][0].__setitem__("inputs", {"T0": 1250.0}),
            lambda r: r["meanline"]["old"]["stage_inputs"][0].pop("T0"),
            lambda r: r["meanline"]["old"]["stage_inputs"][0].__setitem__("κ", 2.0),
        )
        for mutate in mutations:
            bad = copy.deepcopy(self.res)
            mutate(bad)
            summary = cs.classify(bad)
            self.assertFalse(summary["software_pass"], summary["problems"])

    def test_classify_rejects_wrong_floor_branch_flag(self):
        for value in (False, 1, "true", None):
            bad = copy.deepcopy(self.res)
            bad["frozen"]["stages"][0]["floor_branch"] = value
            summary = cs.classify(bad)
            self.assertFalse(summary["software_pass"], (value, summary["problems"]))

    def test_classify_rejects_consistent_relative_branch_frozen_rows(self):
        relative = cs.compare_frozen(
            self._three(self._scaled_fixture(0.100)), self.ns_old, self.ns_new)
        for row in relative["stages"]:
            self.assertAlmostEqual(row["ratio"], 0.6, delta=1e-12)
            self.assertFalse(row["floor_branch"])
        bad = copy.deepcopy(self.res)
        bad["frozen"] = relative
        summary = cs.classify(bad)
        self.assertFalse(summary["software_pass"], summary["problems"])

    def test_classify_rejects_missing_or_trimmed_root_reaction_array(self):
        for values in ([0.05], [0.05, 0.06, 0.07], [0.05, float("nan")]):
            bad = copy.deepcopy(self.res)
            bad["meanline"]["new"]["rho_root_stages_1_2"] = values
            self._resync_constraints(bad)
            summary = cs.classify(bad)
            self.assertFalse(summary["software_pass"], (values, summary["problems"]))
        bad = copy.deepcopy(self.res)
        bad["meanline"]["new"].pop("rho_root_stages_1_2")
        self.assertFalse(cs.classify(bad)["software_pass"])

    def test_classify_rejects_stale_strict_alpha2_boolean(self):
        bad = copy.deepcopy(self.res)
        bad["meanline"]["new"]["strict_alpha2"] = False
        self._resync_constraints(bad)
        summary = cs.classify(bad)
        self.assertFalse(summary["software_pass"], summary["problems"])

        bad = copy.deepcopy(self.res)
        bad["meanline"]["new"]["alpha2_deg"][0] = 10.0
        bad["meanline"]["new"]["strict_alpha2"] = True
        self._resync_constraints(bad)
        summary = cs.classify(bad)
        self.assertFalse(summary["software_pass"], summary["problems"])

    def test_classify_rejects_non_bool_eval_feasible(self):
        for value in (0, 1, "False", None):
            bad = copy.deepcopy(self.res)
            bad["meanline"]["new"]["eval_feasible"] = value
            self._resync_constraints(bad)
            summary = cs.classify(bad)
            self.assertFalse(summary["software_pass"], (value, summary["problems"]))


if __name__ == "__main__":
    unittest.main()

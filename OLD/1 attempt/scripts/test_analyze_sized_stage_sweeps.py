"""Focused cheap tests for the cross-run sized-stage sweep analysis."""

from __future__ import annotations

import contextlib, hashlib, importlib.util, io, json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import analyze_sized_stage_sweeps as analysis  # noqa: E402
import audit_sized_stage_roots as auditor  # noqa: E402

PI_ANCHOR = ROOT / "reference" / "architecture_stage_smoke" / "2026-11-11-pi-anchor-split"


def _metrics(sigma=200.0):
    return {"sigma_MPa": sigma, "M_c1": 0.7, "M_w2": 0.5, "l_l_m": 0.02, "D_l_m": 0.1, "pi": 2.0, "rho": 0.5, "alpha2_deg": 85.0}


def _root(cid="cand_same", architecture="SINGLE_SHAFT_DUAL_CC", case="A_PI8_W0.5", sigma=200.0, numerical=True):
    constraints = {key: {"status": "PASS", "min_signed_margin": 10.0} for key in analysis.CONSTRAINTS}
    return {"root_candidate_id": cid, "case": case, "architecture": architecture, "constraints": constraints,
        "source_domain": {"status": "WITHIN_SOURCE_DOMAIN", "min_source_domain_margin": 0.05},
        "numerical_closed": numerical, "numerical_power_closed": numerical,
        "mass_fuel_air": {"air_mass_flow_kg_s": 0.7, "fuel_mass_flow_kg_s": 0.013, "gas_mass_flow_kg_s": 0.713,
            "fuel_air_ratio": 0.01857, "mass_balance_residual_kg_s": 0.0, "status": "PASS"},
        "groups": {"main": {"stages": [{"metrics": _metrics(sigma)}],
            "closure": {"controls": {"n_rpm": 78000.0, "D_hub_m": 0.07}},
            "power": {"net_mechanical_power_W": 110000.0, "target_W": 110000.0, "signed_power_residual_W": 0.0,
                      "relative_power_residual": 0.0, "engine_power_closed": True, "status": "PASS"}}},
        "reported_residuals": {"main_power_signed_residual_W": 0.0, "continuity_pressure_delta_Pa": None}}


def _two_role_root(cid="cand_c", gg_margin=-2.0, fpt_margin=5.0, gg_sigma=200.0, fpt_sigma=150.0):
    root = _root(cid=cid, architecture="FREE_POWER_TURBINE", case="C_PI8_W0.5", sigma=gg_sigma)
    gas = root["groups"]["main"]
    gas["stages"][0]["checks"] = {"alpha2_source_page_38": {"margin": gg_margin}}
    gas["closure"]["controls"] = {"n_rpm": 72000.0, "D_hub_m": 0.065}
    free = {"stages": [{"metrics": _metrics(fpt_sigma), "checks": {"alpha2_source_page_38": {"margin": fpt_margin}}}],
        "closure": {"controls": {"n_rpm": 61000.0, "D_hub_m": 0.08}}, "power": gas["power"]}
    root["groups"] = {"gas_generator": gas, "free_power": free}
    root["constraints"]["alpha2_source_page_38"] = {"status": "FAIL" if min(gg_margin, fpt_margin) < 0 else "PASS",
        "min_signed_margin": min(gg_margin, fpt_margin)}
    return root


def _record(cid="cand_same", architecture="SINGLE_SHAFT_DUAL_CC", ratios=(2.8284271247461903, 2.8284271247461903)):
    return SimpleNamespace(candidate_id=cid, architecture=architecture,
        payload={"architecture_cycle_source": {"compressor_pressure_ratios": list(ratios)}, "stage_counts": {"main": 3}})


def _failed(case="A_PI3_W0.5"):
    return {"case": case, "status": "NUMERICAL_FAILURE", "reason": "outer_bracket_not_bracketed",
        "architecture": "SINGLE_SHAFT_DUAL_CC", "source_point_index": 2,
        "evaluated_states": [{"candidate_id": "cand_f1", "air_mass_flow_kg_s": 0.6, "calculation_status": "SUCCESS",
            "net_mechanical_power_W": 90000.0, "signed_power_residual_W": -20000.0, "relative_power_residual": 0.181818},
            {"candidate_id": "cand_f2", "air_mass_flow_kg_s": 0.9, "calculation_status": "SUCCESS",
             "net_mechanical_power_W": 93000.0, "signed_power_residual_W": -17000.0, "relative_power_residual": 0.154545}]}


def _audit(run_id, journal, roots, failed, outer=None, inner=None, records=0):
    return {"audit_version": auditor.AUDIT_VERSION, "audit_digest": "a" * 64,
        "run": {"run_id": run_id, "journal_sha256": journal, "record_count": records, "complete": True},
        "roots": roots, "failed_cases": failed, "case_count": len(roots) + len(failed),
        "outer_case_status_counts": outer if outer is not None else {"SUCCESS": len(roots), "NUMERICAL_FAILURE": len(failed)},
        "inner_trial_status_counts": inner if inner is not None else {"SUCCESS": records}}


class AnalyzeSizedStageSweepsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sized-analysis-")
        self.base = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _build(self, specs):
        audits, records, paths = {}, {}, []
        for name, audit, record_list in specs:
            path = self.base / name
            path.mkdir()
            audits[str(path.resolve())] = audit
            records[str(path.resolve())] = tuple(record_list)
            paths.append(path)
        with mock.patch.object(analysis.auditor, "audit_dataset", side_effect=lambda path: audits[str(Path(path).resolve())]), \
             mock.patch.object(analysis, "read_records", side_effect=lambda path: records[str(Path(path).resolve())]):
            return analysis.build_analysis(paths)

    def test_audit_and_raw_read_once_per_dataset(self):
        path = self.base / "once"
        path.mkdir()
        audit = _audit("run", "6" * 64, [_root()], [], records=1)
        with mock.patch.object(analysis.auditor, "audit_dataset", return_value=audit) as audit_call, \
             mock.patch.object(analysis, "read_records", return_value=(_record(),)) as read_call:
            report = analysis.build_analysis([path])
        self.assertEqual(report["root_count"], 1)
        self.assertEqual(audit_call.call_count, 1)
        self.assertEqual(read_call.call_count, 1)

    def _assert_conflict(self, first, second, cid="cand_same", architecture="SINGLE_SHAFT_DUAL_CC"):
        with self.assertRaises(analysis.AnalysisError):
            self._build([("one", _audit("run1", "a" * 64, [first], [], records=1), [_record(cid=cid, architecture=architecture)]),
                         ("two", _audit("run2", "b" * 64, [second], [], records=1), [_record(cid=cid, architecture=architecture)])])

    def test_duplicate_input_path_is_rejected(self):
        path = self.base / "dup"
        path.mkdir()
        with self.assertRaises(analysis.AnalysisError):
            analysis.analyze([path, path], self.base / "out")
        self.assertFalse((self.base / "out").exists())

    def test_root_dedup_preserves_source_memberships(self):
        first = _audit("run1", "a" * 64, [_root(case="A_PI8_W0.4")], [], outer={"SUCCESS": 1}, inner={"SUCCESS": 3}, records=3)
        second = _audit("run2", "b" * 64, [_root(case="A_PI8_W0.5")], [], outer={"SUCCESS": 1}, inner={"SUCCESS": 3}, records=3)
        report = self._build([("d1", first, [_record()]), ("d2", second, [_record()])])
        self.assertEqual(report["root_count"], 1)
        row = report["roots"][0]
        self.assertEqual(row["source_run_ids"], ["run1", "run2"])
        self.assertEqual(len(row["source_cases"]), 2)
        self.assertEqual([source["root_candidate_ids"] for source in report["sources"]], [["cand_same"], ["cand_same"]])
        self.assertEqual(report["source_counts"]["datasets"], 2)
        self.assertEqual(report["source_counts"]["unique_journals"], 2)
        self.assertEqual(report["source_counts"]["root_source_memberships"], 2)
        self.assertEqual(report["outer_case_status_counts"]["SUCCESS"], 2)

    def test_root_dedup_conflicting_metrics_is_rejected(self):
        first = _audit("run1", "a" * 64, [_root(sigma=200.0)], [], records=1)
        second = _audit("run2", "b" * 64, [_root(sigma=250.0)], [], records=1)
        with self.assertRaises(analysis.AnalysisError):
            self._build([("d1", first, [_record()]), ("d2", second, [_record()])])

    def test_conflict_on_closure_controls_is_rejected(self):
        first, second = _root(), _root()
        second["groups"]["main"]["closure"]["controls"]["n_rpm"] = 80000.0
        self._assert_conflict(first, second)

    def test_conflict_on_reported_residual_is_rejected(self):
        first, second = _root(), _root()
        second["reported_residuals"]["main_power_signed_residual_W"] = 1.0
        self._assert_conflict(first, second)

    def test_conflict_on_per_role_alpha_margin_is_rejected(self):
        first = _two_role_root(gg_margin=-2.0, fpt_margin=5.0)
        second = _two_role_root(gg_margin=5.0, fpt_margin=-2.0)
        self._assert_conflict(first, second, cid="cand_c", architecture="FREE_POWER_TURBINE")

    def test_conflict_on_role_extrema_is_rejected(self):
        first = _two_role_root(gg_sigma=200.0, fpt_sigma=150.0)
        second = _two_role_root(gg_sigma=150.0, fpt_sigma=200.0)
        self._assert_conflict(first, second, cid="cand_c", architecture="FREE_POWER_TURBINE")

    def test_conflict_on_group_closure_flag_is_rejected(self):
        first, second = _root(), _root()
        second["groups"]["main"]["power"]["engine_power_closed"] = False
        self._assert_conflict(first, second)

    def test_conflict_on_power_only_closure_is_rejected(self):
        first, second = _root(numerical=True), _root(numerical=True)
        second["numerical_power_closed"] = False
        self._assert_conflict(first, second)

    def test_per_role_alpha_margin_is_role_specific_and_plotted(self):
        report = self._build([("c", _audit("run", "c" * 64, [_two_role_root()], [], records=1),
            [_record(cid="cand_c", architecture="FREE_POWER_TURBINE", ratios=(8.0,))])])
        row = report["roots"][0]
        self.assertEqual(row["per_role_alpha2_margin"]["gas_generator"], {"min_margin_deg": -2.0, "status": "OBSERVED"})
        self.assertEqual(row["per_role_alpha2_margin"]["free_power"], {"min_margin_deg": 5.0, "status": "OBSERVED"})
        self.assertEqual(row["profile_margins"]["alpha2_source_page_38"]["min_signed_margin"], -2.0)
        coordinates = {point["role"]: point["alpha2_margin_deg"] for point in analysis._control_points([row])}
        self.assertEqual(coordinates, {"gas_generator": -2.0, "free_power": 5.0})
        missing = _two_role_root(cid="cand_c_missing")
        missing["groups"]["free_power"]["stages"][0]["checks"]["alpha2_source_page_38"]["margin"] = None
        missing["groups"]["free_power"]["stages"].append({"metrics": _metrics(150.0), "checks": {"alpha2_source_page_38": {"margin": 1.0}}})
        missing_report = self._build([("missing-role", _audit("run2", "d" * 64, [missing], [], records=1),
            [_record(cid="cand_c_missing", architecture="FREE_POWER_TURBINE", ratios=(8.0,))])])
        role = missing_report["roots"][0]["per_role_alpha2_margin"]["free_power"]
        self.assertIsNone(role["min_margin_deg"])
        self.assertEqual(role["status"], "NOT_CHECKED")

    def test_outer_failure_counts_do_not_turn_inner_success_into_feasibility(self):
        audit = _audit("run", "c" * 64, [], [_failed()], outer={"NUMERICAL_FAILURE": 1}, inner={"SUCCESS": 159}, records=159)
        report = self._build([("failed", audit, [])])
        self.assertEqual(report["outer_case_status_counts"], {"SUCCESS": 0, "NUMERICAL_FAILURE": 1, "NON_PHYSICAL_STATE": 0, "MODEL_ERROR": 0})
        self.assertEqual(report["inner_trial_status_counts"], {"SUCCESS": 159, "NUMERICAL_FAILURE": 0, "NON_PHYSICAL_STATE": 0, "MODEL_ERROR": 0})
        self.assertEqual((report["root_count"], report["failed_case_count"]), (0, 1))
        failed = report["failed_cases"][0]
        self.assertEqual((failed["status"], failed["reason"]), ("NUMERICAL_FAILURE", "outer_bracket_not_bracketed"))
        self.assertEqual(failed["evaluated_state_ids"], ["cand_f1", "cand_f2"])
        self.assertEqual(failed["last_state"]["air_mass_flow_kg_s"], 0.9)
        self.assertFalse(failed["four_profile_screening_pass"])
        self.assertIsNone(failed["overall_engineering_feasibility"])
        self.assertEqual(failed["overall_engineering_feasibility_status"], "NOT_CHECKED")
        self.assertEqual(report["failed_case_inner_power_residuals"]["status"], "OBSERVED")

    def test_missing_metric_is_null_not_zero(self):
        root = _root(sigma=None)
        root["constraints"]["sigma_project_290MPa"] = {"status": "NOT_CHECKED", "min_signed_margin": None}
        report = self._build([("missing", _audit("run", "d" * 64, [root], [], records=1), [_record()])])
        row = report["roots"][0]
        self.assertIsNone(row["stage_extremes"]["peak_sigma_MPa"])
        self.assertFalse(row["available_metrics"]["peak_sigma_MPa"])
        self.assertIn("peak_sigma_MPa", row["not_checked_metrics"])
        self.assertFalse(row["four_profile_screening_pass"])
        json.dumps(report, allow_nan=False)

    def test_four_profile_screen_is_not_full_feasibility(self):
        report = self._build([("pass", _audit("run", "e" * 64, [_root(numerical=True)], [], records=1), [_record()])])
        self.assertTrue(report["roots"][0]["four_profile_screening_pass"])
        self.assertEqual(report["roots"][0]["screening_scope"], analysis.SCOPE)
        self.assertEqual(report["scope"], "FOUR_PROFILE_SCREENING_NOT_OVERALL_FEASIBILITY")
        self.assertIsNone(report["overall_engineering_feasibility"])
        self.assertEqual(report["overall_engineering_feasibility_status"], "NOT_CHECKED")
        self.assertIsNone(report["selection"])
        self.assertIsNone(report["pareto_front"])
        self.assertIsNone(report["production_readiness"])
        report = self._build([("unknown", _audit("run2", "f" * 64, [_root(cid="cand_other", numerical=None)], [], records=1), [_record(cid="cand_other")])])
        self.assertFalse(report["roots"][0]["four_profile_screening_pass"])
        self.assertIsNone(report["roots"][0]["numerical_closed"])

    def test_no_overwrite_and_raw_bytes_unchanged(self):
        dataset = self.base / "raw"
        dataset.mkdir()
        names = ("dataset.jsonl", "manifest.json", "checkpoint.json", "summary.json")
        for name in names:
            (dataset / name).write_bytes(b"raw-" + name.encode())
        before = {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest() for name in names}
        audit = _audit("run", "9" * 64, [_root()], [], records=1)
        destination = self.base / "out"
        patches = (mock.patch.object(analysis.auditor, "audit_dataset", return_value=audit),
                   mock.patch.object(analysis, "read_records", return_value=(_record(),)),
                   mock.patch.object(analysis, "_make_plots", return_value={"status": "SKIPPED_FOR_TEST", "files": [], "root_count": 1}))
        with patches[0], patches[1], patches[2]:
            self.assertEqual(analysis.main(["--dataset", str(dataset), "--destination", str(destination)]), 0)
            first = (destination / "analysis.json").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(analysis.main(["--dataset", str(dataset), "--destination", str(destination)]), 2)
        self.assertEqual((destination / "analysis.json").read_bytes(), first)
        after = {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest() for name in names}
        self.assertEqual(before, after)

    def test_matplotlib_absence_still_writes_json(self):
        dataset = self.base / "nodep"
        dataset.mkdir()
        destination = self.base / "nodep-out"
        audit = _audit("run", "8" * 64, [_root()], [], records=1)
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with mock.patch.object(analysis.auditor, "audit_dataset", return_value=audit), \
             mock.patch.object(analysis, "read_records", return_value=(_record(),)), \
             mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(analysis.main(["--dataset", str(dataset), "--destination", str(destination)]), 0)
        report = json.loads((destination / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(report["plot_status"]["status"], "NOT_AVAILABLE")
        self.assertIn("matplotlib", report["plot_status"]["reason"])
        self.assertEqual(report["root_count"], 1)

    def test_scatter_generation_when_matplotlib_available(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not available")
        report = self._build([("plot", _audit("run", "7" * 64, [_root()], [], records=1), [_record()])])
        destination = self.base / "plots"
        destination.mkdir()
        status = analysis._make_plots(report["roots"], destination)
        self.assertEqual(status["status"], "GENERATED")
        self.assertEqual(len(status["files"]), 2)
        for name in status["files"]:
            self.assertGreater((destination / name).stat().st_size, 0)

    def test_reference_pi_anchor_split_is_failure_aware(self):
        if not PI_ANCHOR.is_dir():
            self.skipTest("saved pi-anchor-split reference dataset is absent")
        report = analysis.build_analysis([PI_ANCHOR])
        self.assertEqual(report["scope"], analysis.SCOPE)
        self.assertEqual(report["source_counts"]["datasets"], 1)
        self.assertEqual(report["source_counts"]["unique_journals"], 1)
        self.assertEqual(report["source_counts"]["inner_trials"], 159)
        self.assertEqual(report["root_count"], 8)
        self.assertEqual(report["failed_case_count"], 3)
        self.assertEqual(report["outer_case_status_counts"]["SUCCESS"], 8)
        self.assertEqual(report["outer_case_status_counts"]["NUMERICAL_FAILURE"], 3)
        self.assertEqual(report["inner_trial_status_counts"]["SUCCESS"], 159)
        self.assertTrue(all(abs(root["pi_product"] - 8.0) < 1e-12 or abs(root["pi_product"] - 13.0) < 1e-12 for root in report["roots"]))
        self.assertEqual(len({root["source_run_ids"][0] for root in report["roots"]}), 1)
        self.assertTrue(all(root["logsplit_weight_A_status"] in ("OBSERVED", "NOT_APPLICABLE") for root in report["roots"]))
        failed = {item["case"]: item for item in report["failed_cases"]}
        for name in ("A_PI3_W0.5", "B_PI3_W0.5", "C_PI3_W0.5"):
            self.assertEqual(failed[name]["status"], "NUMERICAL_FAILURE")
            self.assertFalse(failed[name]["four_profile_screening_pass"])
            self.assertIsNone(failed[name]["overall_engineering_feasibility"])
        self.assertIsNone(report["overall_engineering_feasibility"])
        self.assertEqual(report["overall_engineering_feasibility_status"], "NOT_CHECKED")


if __name__ == "__main__":
    unittest.main()

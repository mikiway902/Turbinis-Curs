"""Focused tests for generic immutable RAW candidate records.

These tests are synthetic and contain no turbine physics.  They check status
semantics, input retention, deterministic identifiers, strict JSON rejection,
hash integrity, serialization-vs-physics failure separation, and deep-copy
immutability.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.raw import (  # noqa: E402
    RAW_SCHEMA_VERSION,
    CalculationStatus,
    ConstraintStatus,
    EvaluationFailure,
    EvaluationOutcome,
    EvaluationSerializationError,
    ModelError,
    NonPhysicalStateError,
    NumericalFailure,
    PayloadValidationError,
    RawCandidate,
    RecordIntegrityError,
    canonical_hash,
    evaluate_candidate,
    make_candidate_id,
    make_run_id,
)


RUN_ID = make_run_id(
    {"model_variant": "synthetic-v1", "evaluator_version": "eval-test"}
)
MODEL_VARIANT = "synthetic-v1"


def successful_evaluator(payload):
    return {
        "rho": payload["x"] / payload["denominator"],
        "work": payload["x"] * 2,
    }


def rehash(record):
    body = {key: value for key, value in record.items() if key != "record_hash"}
    body["record_hash"] = canonical_hash(body)
    return body


def evaluate(payload, evaluator=successful_evaluator, **kwargs):
    parameters = {
        "architecture": "SYNTHETIC",
        "model_variant": MODEL_VARIANT,
        "run_id": RUN_ID,
        "sampling_method": "unit_test",
    }
    parameters.update(kwargs)
    return evaluate_candidate(evaluator, payload, **parameters)


class RawCandidateTests(unittest.TestCase):
    def test_success_record_has_strict_statuses_and_roundtrips(self):
        payload = {"x": 2.5, "denominator": 5.0}
        record = evaluate(payload, sampling_metadata={"axis": "x", "index": 0})

        self.assertEqual(record.schema_version, RAW_SCHEMA_VERSION)
        self.assertIs(record.calculation_status, CalculationStatus.SUCCESS)
        self.assertIs(record.constraint_status, ConstraintStatus.NOT_EVALUATED)
        self.assertIsNone(record.failure_stage)
        self.assertIsNone(record.failure_reason)
        self.assertIsNone(record.last_valid_state)
        self.assertEqual(record.payload, payload)
        self.assertEqual(record.outputs, {"rho": 0.5, "work": 5.0})
        self.assertEqual(record.sampling, {"axis": "x", "index": 0})

        restored = RawCandidate.from_json(record.to_json())
        self.assertEqual(restored.to_dict(), record.to_dict())
        self.assertEqual(restored.record_hash, record.record_hash)

    def test_identifiers_are_deterministic_from_config_not_random(self):
        payload_a = {"x": 1, "denominator": 2}
        payload_b = {"denominator": 2, "x": 1}
        first = evaluate(payload_a)
        second = evaluate(payload_b)
        self.assertEqual(first.candidate_id, second.candidate_id)
        self.assertEqual(first.record_hash, second.record_hash)

        other_architecture = evaluate(payload_a, architecture="OTHER")
        self.assertNotEqual(first.candidate_id, other_architecture.candidate_id)

        other_variant = evaluate(payload_a, model_variant="synthetic-v2")
        self.assertNotEqual(first.candidate_id, other_variant.candidate_id)

        run_a = make_run_id(
            {"model_variant": "synthetic-v1", "preset": "quick"},
            evaluator_version="eval-1",
        )
        run_b = make_run_id(
            {"preset": "quick", "model_variant": "synthetic-v1"},
            evaluator_version="eval-1",
        )
        run_c = make_run_id(
            {"model_variant": "synthetic-v1", "preset": "quick"},
            evaluator_version="eval-2",
        )
        self.assertEqual(run_a, run_b)
        self.assertNotEqual(run_a, run_c)

    def test_payload_cannot_alias_record_state(self):
        payload = {"x": 3, "denominator": 4, "nested": {"values": [1, 2]}}
        record = evaluate(payload)
        payload["x"] = 999
        payload["nested"]["values"].append(3)
        self.assertEqual(record.payload["x"], 3)
        self.assertEqual(record.payload["nested"]["values"], [1, 2])

        exposed = record.payload
        exposed["x"] = -1
        exposed["nested"]["values"].append(4)
        self.assertEqual(record.payload["x"], 3)
        self.assertEqual(record.payload["nested"]["values"], [1, 2])

        dictionary = record.to_dict()
        dictionary["outputs"]["rho"] = 100.0
        self.assertEqual(record.outputs["rho"], 0.75)

    def test_invalid_payload_is_rejected_before_evaluator(self):
        calls = []

        def evaluator(payload):
            calls.append(payload)
            return {"ok": True}

        invalid_payloads = [
            {"value": math.nan},
            {"value": math.inf},
            {"value": (1, 2)},
            {"value": b"bytes"},
            {"value": {1: "non-string-key"}},
            {"value": object()},
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=repr(payload)):
                with self.assertRaises(PayloadValidationError):
                    evaluate(payload, evaluator=evaluator)
        self.assertEqual(calls, [])

    def test_record_hash_and_identifiers_are_verified(self):
        record = evaluate({"x": 1, "denominator": 2})
        data = record.to_dict()
        data["outputs"]["rho"] = 10.0
        with self.assertRaises(RecordIntegrityError):
            RawCandidate.from_dict(data)

        changed_payload = record.to_dict()
        changed_payload["payload"]["x"] = 50
        changed_payload = rehash(changed_payload)
        with self.assertRaises(RecordIntegrityError):
            RawCandidate.from_dict(changed_payload)

        wrong_input_hash = record.to_dict()
        wrong_input_hash["input_hash"] = "0" * 64
        wrong_input_hash = rehash(wrong_input_hash)
        with self.assertRaises(RecordIntegrityError):
            RawCandidate.from_dict(wrong_input_hash)

        mutated_constraint = record.to_dict()
        mutated_constraint["constraint_status"] = ConstraintStatus.PASS.value
        mutated_constraint = rehash(mutated_constraint)
        with self.assertRaises(RecordIntegrityError):
            RawCandidate.from_dict(mutated_constraint)

    def test_candidate_id_helper_matches_record(self):
        payload = {"x": 1, "denominator": 2}
        record = evaluate(payload)
        self.assertEqual(
            record.candidate_id,
            make_candidate_id("SYNTHETIC", MODEL_VARIANT, payload),
        )

    def test_numerical_failure_retains_inputs_and_diagnostic(self):
        def evaluator(payload):
            raise NumericalFailure(
                "solver",
                "root did not converge",
                last_valid_state={"temperature": 300.0, "pressure": 100000.0},
                diagnostics={"iterations": 50},
            )

        payload = {"x": 1, "denominator": 0.5}
        record = evaluate(payload, evaluator=evaluator)
        self.assertIs(record.calculation_status, CalculationStatus.NUMERICAL_FAILURE)
        self.assertEqual(record.failure_stage, "solver")
        self.assertEqual(record.failure_reason, "root did not converge")
        self.assertEqual(record.payload, payload)
        self.assertEqual(record.last_valid_state["pressure"], 100000.0)
        self.assertEqual(record.diagnostics, {"iterations": 50})

    def test_non_physical_and_model_failures_are_retained(self):
        cases = [
            (
                NonPhysicalStateError("state", "temperature must be positive"),
                CalculationStatus.NON_PHYSICAL_STATE,
            ),
            (
                ModelError("model", "missing model coefficient"),
                CalculationStatus.MODEL_ERROR,
            ),
        ]
        for failure, expected_status in cases:
            with self.subTest(status=expected_status.value):
                def evaluator(payload, failure=failure):
                    raise failure

                payload = {"x": 7, "denominator": 2}
                record = evaluate(payload, evaluator=evaluator)
                self.assertIs(record.calculation_status, expected_status)
                self.assertEqual(record.payload, payload)
                self.assertEqual(record.failure_stage, failure.stage)
                self.assertEqual(record.failure_reason, failure.reason)

    def test_unexpected_evaluator_exception_is_model_error(self):
        def evaluator(payload):
            raise RuntimeError("unexpected synthetic bug")

        record = evaluate({"x": 1, "denominator": 2}, evaluator=evaluator)
        self.assertIs(record.calculation_status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(record.failure_stage, "evaluator_exception")
        self.assertIn("RuntimeError", record.failure_reason)
        self.assertIn("unexpected synthetic bug", record.failure_reason)
        self.assertEqual(record.payload, {"x": 1, "denominator": 2})

    def test_serialization_failure_is_not_physics_failure(self):
        def evaluator(payload):
            return {"rho": math.nan}

        record = evaluate({"x": 1, "denominator": 2}, evaluator=evaluator)
        self.assertIs(record.calculation_status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(record.failure_stage, "result_serialization")
        self.assertIsNone(record.outputs)
        self.assertTrue(record.diagnostics["serialization_failure"])
        self.assertEqual(record.payload, {"x": 1, "denominator": 2})

        def unsupported(payload):
            return object()

        record = evaluate({"x": 1, "denominator": 2}, evaluator=unsupported)
        self.assertIs(record.calculation_status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(record.failure_stage, "result_serialization")

        def explicit(payload):
            raise EvaluationSerializationError("cannot serialize externally supplied data")

        record = evaluate({"x": 1, "denominator": 2}, evaluator=explicit)
        self.assertIs(record.calculation_status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(record.failure_stage, "result_serialization")
        self.assertEqual(record.failure_reason, "cannot serialize externally supplied data")

    def test_interrupts_are_never_converted_to_records(self):
        def interrupt(payload):
            raise KeyboardInterrupt("stop")

        def exit_process(payload):
            raise SystemExit(3)

        for evaluator in (interrupt, exit_process):
            with self.subTest(evaluator=evaluator.__name__):
                with self.assertRaises((KeyboardInterrupt, SystemExit)):
                    evaluate({"x": 1, "denominator": 2}, evaluator=evaluator)

    def test_outcome_exposes_intermediates_and_causal_diagnostics(self):
        def evaluator(payload):
            return EvaluationOutcome.success(
                outputs={"rho": 0.5},
                intermediates={"stage_pressure_ratio": 2.0, "area": 1.5},
                diagnostics={"converged": True},
            )

        record = evaluate({"x": 1, "denominator": 2}, evaluator=evaluator)
        self.assertEqual(record.intermediates["stage_pressure_ratio"], 2.0)
        self.assertEqual(record.diagnostics, {"converged": True})

    def test_evaluator_mutation_does_not_change_stored_payload(self):
        def evaluator(payload):
            payload["x"] = 999
            payload["injected"] = True
            return {"seen": 999}

        record = evaluate({"x": 1, "denominator": 2}, evaluator=evaluator)
        self.assertEqual(record.payload, {"x": 1, "denominator": 2})
        self.assertEqual(record.outputs, {"seen": 999})

    def test_failure_exception_requires_stage_and_reason(self):
        with self.assertRaises(ValueError):
            NumericalFailure("", "reason")
        with self.assertRaises(ValueError):
            ModelError("stage", "")
        with self.assertRaises(ValueError):
            EvaluationFailure(CalculationStatus.SUCCESS, "stage", "reason")


if __name__ == "__main__":
    unittest.main()

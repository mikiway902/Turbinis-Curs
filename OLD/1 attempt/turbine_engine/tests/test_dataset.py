"""Focused tests for deterministic RAW dataset persistence and recovery.

The tests use synthetic evaluators only.  They verify append-only records,
manifest/checkpoint integrity, crash reconciliation, duplicate semantics, tamper
rejection, reproduce comparison, single-writer locking, and optional Parquet
export.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.dataset import (  # noqa: E402
    CHECKPOINT_NAME,
    JOURNAL_NAME,
    MANIFEST_NAME,
    DatasetIntegrityError,
    DatasetLockError,
    DatasetWriter,
    DuplicateCandidateError,
    OptionalDependencyError,
    RunConfigMismatchError,
    export_parquet,
    load_manifest,
    read_records,
    reproduce_candidate,
)
from turbine_engine.raw import (  # noqa: E402
    EvaluationOutcome,
    NumericalFailure,
    canonical_json,
    evaluate_candidate,
    make_run_id,
)


CONFIG = {
    "model_variant": "synthetic-v1",
    "evaluator_version": "eval-test",
    "preset": "unit-test",
}
RUN_ID = make_run_id(CONFIG)


def synthetic_evaluator(payload):
    return {
        "rho": payload["x"] / payload["denominator"],
        "work": payload["x"] * 2.0,
    }


def make_record(payload, evaluator=synthetic_evaluator):
    return evaluate_candidate(
        evaluator,
        payload,
        architecture="SYNTHETIC",
        model_variant="synthetic-v1",
        run_id=RUN_ID,
        sampling_method="unit_test",
    )


def make_payloads(count):
    return [
        {"x": float(index + 1), "denominator": float(index + 2)}
        for index in range(count)
    ]


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def _new_dataset(self, name="dataset"):
        return DatasetWriter.create(self.root / name, run_config=CONFIG)

    def test_append_success_and_failure_records_with_manifest(self):
        records = [
            make_record({"x": 2.0, "denominator": 4.0}),
        ]

        def numerical_failure(payload):
            raise NumericalFailure("solver", "synthetic non-convergence")

        def nonphysical(payload):
            raise NumericalFailure("state", "synthetic failure")

        records.append(make_record({"x": 3.0, "denominator": 4.0}, numerical_failure))
        records.append(make_record({"x": 4.0, "denominator": 4.0}, nonphysical))

        writer = self._new_dataset()
        try:
            self.assertEqual(writer.append_many(records), 3)
            self.assertFalse(writer.complete)
        finally:
            writer.close()

        manifest = load_manifest(writer.directory)
        self.assertEqual(manifest["record_count"], 3)
        self.assertTrue(manifest["complete"])
        journal = (writer.directory / JOURNAL_NAME).read_bytes()
        self.assertEqual(manifest["journal_size_bytes"], len(journal))
        self.assertEqual(
            manifest["journal_sha256"],
            hashlib.sha256(journal).hexdigest(),
        )
        loaded = read_records(writer.directory)
        self.assertEqual([record.candidate_id for record in loaded],
                         [record.candidate_id for record in records])
        self.assertEqual(loaded[0].outputs["rho"], 0.5)
        self.assertEqual(loaded[1].failure_stage, "solver")
        self.assertEqual(loaded[2].payload["x"], 4.0)

    def test_exact_duplicate_is_idempotent_and_conflict_is_rejected(self):
        record = make_record({"x": 1.0, "denominator": 2.0})
        writer = self._new_dataset()
        try:
            writer.append(record)
            before = (writer.directory / JOURNAL_NAME).read_bytes()
            returned = writer.append(record)
            after = (writer.directory / JOURNAL_NAME).read_bytes()
            self.assertEqual(returned.record_hash, record.record_hash)
            self.assertEqual(before, after)
            self.assertEqual(writer.record_count, 1)

            def different_diagnostics(payload):
                return EvaluationOutcome.success(
                    outputs={"rho": 0.5, "work": 2.0},
                    diagnostics={"variant": "different"},
                )

            conflicting = make_record(
                {"x": 1.0, "denominator": 2.0}, different_diagnostics
            )
            self.assertEqual(conflicting.candidate_id, record.candidate_id)
            self.assertNotEqual(conflicting.record_hash, record.record_hash)
            with self.assertRaises(DuplicateCandidateError):
                writer.append(conflicting)
            self.assertEqual(
                (writer.directory / JOURNAL_NAME).read_bytes(), after
            )
        finally:
            writer.close()

    def test_wrong_run_record_is_rejected(self):
        other = evaluate_candidate(
            synthetic_evaluator,
            {"x": 1.0, "denominator": 2.0},
            architecture="SYNTHETIC",
            model_variant="synthetic-v1",
            run_id=make_run_id(
                {"model_variant": "synthetic-v1", "evaluator_version": "other"}
            ),
            sampling_method="unit_test",
        )
        writer = self._new_dataset()
        try:
            with self.assertRaises(RunConfigMismatchError):
                writer.append(other)
        finally:
            writer.close()

    def test_resume_after_record_before_manifest_is_identical(self):
        payloads = make_payloads(4)
        candidates = [make_record(payload) for payload in payloads]

        uninterrupted = self.root / "uninterrupted"
        writer = DatasetWriter.create(uninterrupted, run_config=CONFIG)
        writer.append_many(candidates)
        writer.close()

        interrupted = self.root / "interrupted"
        writer = DatasetWriter.create(interrupted, run_config=CONFIG)
        writer.append_many(candidates[:2])
        saved_manifest = (interrupted / MANIFEST_NAME).read_bytes()
        saved_checkpoint = (interrupted / CHECKPOINT_NAME).read_bytes()
        writer.append_many(candidates[2:3])
        # Simulate a crash after the journal record but before cursor commits.
        (interrupted / MANIFEST_NAME).write_bytes(saved_manifest)
        (interrupted / CHECKPOINT_NAME).write_bytes(saved_checkpoint)
        writer.close(finalize=False)

        resumed = DatasetWriter.open(interrupted, expected_run_config=CONFIG)
        self.assertEqual(resumed.record_count, 3)
        resumed.append_many(candidates[3:])
        resumed.close()

        self.assertEqual(
            (interrupted / JOURNAL_NAME).read_bytes(),
            (uninterrupted / JOURNAL_NAME).read_bytes(),
        )
        self.assertEqual(
            (interrupted / MANIFEST_NAME).read_bytes(),
            (uninterrupted / MANIFEST_NAME).read_bytes(),
        )
        self.assertEqual(
            [record.to_dict() for record in read_records(interrupted)],
            [record.to_dict() for record in read_records(uninterrupted)],
        )
        self.assertEqual(
            load_manifest(interrupted)["journal_sha256"],
            load_manifest(uninterrupted)["journal_sha256"],
        )

    def test_open_refuses_config_and_evaluator_mismatch(self):
        writer = self._new_dataset()
        writer.close()
        with self.assertRaises(RunConfigMismatchError):
            DatasetWriter.open(
                writer.directory,
                expected_run_config={
                    "model_variant": "different-variant",
                    "evaluator_version": "eval-test",
                },
            )
        with self.assertRaises(RunConfigMismatchError):
            DatasetWriter.open(
                writer.directory,
                expected_run_config=CONFIG,
                expected_evaluator_version="eval-other",
            )
        reopened = DatasetWriter.open(writer.directory, expected_run_config=CONFIG)
        reopened.close()

    def test_tampered_journal_manifest_and_truncation_are_refused(self):
        writer = self._new_dataset()
        writer.append_many([make_record(payload) for payload in make_payloads(3)])
        writer.close()
        source = writer.directory

        tampered = self.root / "tampered"
        shutil.copytree(source, tampered)
        data = (tampered / JOURNAL_NAME).read_bytes()
        first_line, remainder = data.split(b"\n", 1)
        record = json.loads(first_line.decode("utf-8"))
        record["outputs"]["rho"] = 999.0
        (tampered / JOURNAL_NAME).write_bytes(
            canonical_json(record).encode("utf-8") + b"\n" + remainder
        )
        with self.assertRaises(DatasetIntegrityError):
            read_records(tampered)
        with self.assertRaises(DatasetIntegrityError):
            DatasetWriter.open(tampered)

        truncated = self.root / "truncated"
        shutil.copytree(source, truncated)
        data = (truncated / JOURNAL_NAME).read_bytes()
        (truncated / JOURNAL_NAME).write_bytes(data.rsplit(b"\n", 2)[0] + b"\n")
        with self.assertRaises(DatasetIntegrityError):
            read_records(truncated)
        with self.assertRaises(DatasetIntegrityError):
            DatasetWriter.open(truncated)

        bad_manifest = self.root / "bad-manifest"
        shutil.copytree(source, bad_manifest)
        manifest = json.loads((bad_manifest / MANIFEST_NAME).read_text("utf-8"))
        manifest["record_count"] += 1
        (bad_manifest / MANIFEST_NAME).write_text(
            canonical_json(manifest) + "\n", encoding="utf-8"
        )
        with self.assertRaises(DatasetIntegrityError):
            read_records(bad_manifest)

    def test_incomplete_final_line_is_repaired_and_noted(self):
        writer = self._new_dataset()
        writer.append_many([make_record(payload) for payload in make_payloads(2)])
        writer.close()
        original = (writer.directory / JOURNAL_NAME).read_bytes()
        with open(writer.directory / JOURNAL_NAME, "ab") as handle:
            handle.write(b'{"schema_version":"raw-candidate-v1"')

        resumed = DatasetWriter.open(writer.directory, expected_run_config=CONFIG)
        notes = resumed.repair_notes
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["kind"], "truncated_journal_tail")
        self.assertEqual(resumed.record_count, 2)
        resumed.close()
        self.assertEqual(
            (writer.directory / JOURNAL_NAME).read_bytes(), original
        )
        self.assertEqual(len(read_records(writer.directory)), 2)

    def test_single_writer_lock(self):
        writer = self._new_dataset()
        try:
            with self.assertRaises(DatasetLockError):
                DatasetWriter.open(writer.directory)
        finally:
            writer.close()
        reopened = DatasetWriter.open(writer.directory)
        reopened.close()

    def test_reproduce_requires_explicit_tolerance_and_ignores_diagnostics(self):
        payload = {"x": 2.0, "denominator": 4.0}
        writer = self._new_dataset()
        record = writer.evaluate(
            synthetic_evaluator,
            payload,
            architecture="SYNTHETIC",
            sampling_method="unit_test",
        )
        writer.append(record)
        writer.close()

        result = reproduce_candidate(
            writer.directory, record.candidate_id, synthetic_evaluator, rtol=0, atol=0
        )
        self.assertTrue(result.matches)
        self.assertEqual(result.differences, ())

        def different_diagnostics(payload):
            return EvaluationOutcome.success(
                outputs={"rho": 0.5, "work": 4.0},
                diagnostics={"nondeterministic": "ignored"},
            )

        result = reproduce_candidate(
            writer.directory,
            record.candidate_id,
            different_diagnostics,
            rtol=0,
            atol=0,
        )
        self.assertTrue(result.matches)

        def slightly_changed(payload):
            return {"rho": 0.5, "work": 4.0 + 1e-6}

        strict = reproduce_candidate(
            writer.directory, record.candidate_id, slightly_changed, rtol=0, atol=1e-9
        )
        self.assertFalse(strict.matches)
        self.assertTrue(strict.differences)
        relaxed = reproduce_candidate(
            writer.directory, record.candidate_id, slightly_changed, rtol=0, atol=1e-5
        )
        self.assertTrue(relaxed.matches)
        with self.assertRaises(TypeError):
            reproduce_candidate(
                writer.directory, record.candidate_id, synthetic_evaluator
            )

    def test_parquet_export_is_explicit_and_consistent_when_installed(self):
        writer = self._new_dataset()
        writer.append_many([make_record(payload) for payload in make_payloads(2)])
        writer.close()
        output = self.root / "export.parquet"

        if importlib.util.find_spec("pyarrow") is None:
            with self.assertRaises(OptionalDependencyError):
                export_parquet(writer.directory, output)
            return

        import pyarrow.parquet as pq

        exported = export_parquet(writer.directory, output)
        self.assertEqual(exported, output)
        table = pq.read_table(output)
        rows = table.to_pylist()
        records = read_records(writer.directory)
        self.assertEqual(len(rows), len(records))
        for row, record in zip(rows, records):
            self.assertEqual(row["candidate_id"], record.candidate_id)
            self.assertEqual(row["record_hash"], record.record_hash)
            self.assertEqual(json.loads(row["payload"]), record.payload)
            self.assertEqual(json.loads(row["outputs"]), record.outputs)
            self.assertEqual(row["calculation_status"], record.calculation_status.value)
        with self.assertRaises(FileExistsError):
            export_parquet(writer.directory, output)
        export_parquet(writer.directory, output, overwrite=True)

    def test_parquet_export_reports_missing_dependency_explicitly(self):
        writer = self._new_dataset()
        writer.append(make_record({"x": 1.0, "denominator": 2.0}))
        writer.close()
        output = self.root / "missing-pyarrow.parquet"
        with mock.patch.dict(
            sys.modules, {"pyarrow": None, "pyarrow.parquet": None}
        ):
            with self.assertRaises(OptionalDependencyError) as context:
                export_parquet(writer.directory, output)
        self.assertIn("pyarrow", str(context.exception))
        self.assertIn("JSONL", str(context.exception))
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

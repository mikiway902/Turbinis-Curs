"""Deterministic, crash-recoverable RAW dataset persistence.

The authoritative on-disk format is a canonical JSONL journal plus a hash
manifest and checkpoint.  Parquet export is an optional derived product; it is
never the recovery source of truth.

Layout
======
``<dataset>/dataset.jsonl``
    Append-only canonical JSON records, one :class:`~turbine_engine.raw.RawCandidate`
    per line.
``<dataset>/manifest.json``
    Run metadata, record count, journal byte size, prefix SHA-256, completion
    flag, and repair notes.
``<dataset>/checkpoint.json``
    The same committed cursor, updated before the manifest.  If a crash happens
    between the journal and either cursor write, :meth:`DatasetWriter.open`
    reconciles against the journal.
``<dataset>/.dataset.lock``
    OS advisory single-writer lock.  The lock is released automatically if a
    process dies; the file itself is harmless.

Recovery rules
==============
* A complete record is authoritative once its line and newline are written.
* If the journal has complete lines beyond the last committed cursor, ``open``
  verifies the committed prefix hash, accepts the extra records, and repairs the
  cursors.
* If the journal is shorter than the committed cursor, or the committed prefix
  hash does not match, the dataset is refused rather than silently dropping
  attempts.
* A final line without a newline is treated as an interrupted write.  It is
  truncated only after all committed data verifies, and the event is recorded
  in ``repair_notes``.
* Duplicate candidate ids are exact-idempotent only when the complete record
  hash matches; otherwise the append is rejected.

No wall-clock timestamps are written, so the same evaluation sequence produces
the same journal bytes, record ids, count, and hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from turbine_engine.raw import (
    JOURNAL_FORMAT_JSONL,
    RAW_SCHEMA_VERSION,
    CalculationStatus,
    RawCandidate,
    RecordIntegrityError,
    canonical_hash,
    canonical_json,
    copy_json_value,
    evaluate_candidate,
    evaluator_version_hash,
    make_run_id,
    normalize_run_config,
)


DATASET_VERSION = "raw-dataset-v1"
JOURNAL_NAME = "dataset.jsonl"
MANIFEST_NAME = "manifest.json"
CHECKPOINT_NAME = "checkpoint.json"
LOCK_NAME = ".dataset.lock"

_SNAPSHOT_KEYS = frozenset(
    {
        "dataset_version",
        "schema_version",
        "format",
        "run_id",
        "run_config",
        "run_config_hash",
        "evaluator_version",
        "evaluator_version_hash",
        "record_count",
        "journal_size_bytes",
        "journal_sha256",
        "last_candidate_id",
        "complete",
        "repair_notes",
    }
)


class DatasetError(Exception):
    """Base class for dataset persistence errors."""


class DatasetExistsError(DatasetError):
    """The destination already contains dataset or unrelated files."""


class DatasetLockError(DatasetError):
    """Another writer currently holds the dataset lock."""


class DatasetIntegrityError(DatasetError, ValueError):
    """Stored bytes or metadata do not match the manifest/checkpoint."""


class RunConfigMismatchError(DatasetError):
    """The requested run config/evaluator version differs from the dataset."""


class DuplicateCandidateError(DatasetError):
    """A candidate id already exists with a different record hash."""


class CandidateNotFoundError(DatasetError):
    """No RAW record with the requested candidate id exists."""


class OptionalDependencyError(DatasetError):
    """An optional runtime dependency is required for this operation."""


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {token}")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DatasetIntegrityError(f"missing dataset file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text, parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError, ValueError, TypeError) as exc:
        raise DatasetIntegrityError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetIntegrityError(f"{path.name} must contain a JSON object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    text = canonical_json(value) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


class _DatasetLock:
    """Best-effort OS advisory lock held for the writer lifetime."""

    def __init__(self, path: Path) -> None:
        self._path = path
        try:
            self._file = open(path, "r+b")
        except FileNotFoundError:
            self._file = open(path, "w+b")
        try:
            self._file.seek(0, os.SEEK_END)
            if self._file.tell() == 0:
                self._file.write(b"\0")
                self._file.flush()
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._file.close()
            raise DatasetLockError(
                f"dataset is already locked by another writer: {path}"
            ) from exc

    def release(self) -> None:
        if self._file.closed:
            return
        try:
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self._file.close()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _split_complete_journal(raw: bytes) -> tuple[bytes, bytes]:
    if not raw:
        return b"", b""
    if raw.endswith(b"\n"):
        return raw, b""
    cut = raw.rfind(b"\n") + 1
    return raw[:cut], raw[cut:]


def _iter_record_lines(data: bytes, source: str) -> Iterator[RawCandidate]:
    if not data:
        return
    if not data.endswith(b"\n"):
        raise DatasetIntegrityError(f"{source}: journal does not end with a newline")
    for index, raw_line in enumerate(data[:-1].split(b"\n"), start=1):
        if not raw_line:
            raise DatasetIntegrityError(f"{source}: empty journal line {index}")
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DatasetIntegrityError(
                f"{source}: journal line {index} is not valid UTF-8"
            ) from exc
        try:
            yield RawCandidate.from_json(line)
        except RecordIntegrityError as exc:
            raise DatasetIntegrityError(
                f"{source}: journal line {index} failed record verification: {exc}"
            ) from exc


def _ensure_unique(records: Sequence[RawCandidate], source: str) -> None:
    seen: set[str] = set()
    for record in records:
        if record.candidate_id in seen:
            raise DatasetIntegrityError(
                f"{source}: duplicate candidate_id {record.candidate_id!r}"
            )
        seen.add(record.candidate_id)


def _validate_snapshot(
    snapshot: Mapping[str, Any], complete_bytes: bytes, source: str
) -> list[RawCandidate]:
    if not isinstance(snapshot, Mapping):
        raise DatasetIntegrityError(f"{source} must be a JSON object")
    keys = set(snapshot)
    if keys != _SNAPSHOT_KEYS:
        missing = sorted(_SNAPSHOT_KEYS - keys)
        extra = sorted(keys - _SNAPSHOT_KEYS)
        raise DatasetIntegrityError(
            f"{source} keys mismatch; missing={missing!r}, extra={extra!r}"
        )
    if snapshot["dataset_version"] != DATASET_VERSION:
        raise DatasetIntegrityError(
            f"{source} has unsupported dataset version {snapshot['dataset_version']!r}"
        )
    if snapshot["schema_version"] != RAW_SCHEMA_VERSION:
        raise DatasetIntegrityError(
            f"{source} has unsupported RAW schema version {snapshot['schema_version']!r}"
        )
    if snapshot["format"] != JOURNAL_FORMAT_JSONL:
        raise DatasetIntegrityError(f"{source} has unsupported journal format")

    try:
        config, version = normalize_run_config(
            snapshot["run_config"], evaluator_version=snapshot["evaluator_version"]
        )
    except (TypeError, ValueError) as exc:
        raise DatasetIntegrityError(f"{source} run_config is invalid: {exc}") from exc
    if config != snapshot["run_config"]:
        raise DatasetIntegrityError(
            f"{source} run_config is not in canonical normalized form"
        )
    if canonical_hash(config) != snapshot["run_config_hash"]:
        raise DatasetIntegrityError(f"{source} run_config_hash mismatch")
    if make_run_id(config) != snapshot["run_id"]:
        raise DatasetIntegrityError(f"{source} run_id does not match run_config")
    if evaluator_version_hash(version) != snapshot["evaluator_version_hash"]:
        raise DatasetIntegrityError(f"{source} evaluator_version_hash mismatch")

    count = snapshot["record_count"]
    size = snapshot["journal_size_bytes"]
    if type(count) is not int or count < 0:
        raise DatasetIntegrityError(f"{source} record_count must be a non-negative int")
    if type(size) is not int or size < 0:
        raise DatasetIntegrityError(
            f"{source} journal_size_bytes must be a non-negative int"
        )
    if type(snapshot["complete"]) is not bool:
        raise DatasetIntegrityError(f"{source} complete must be a bool")
    if not isinstance(snapshot["repair_notes"], list):
        raise DatasetIntegrityError(f"{source} repair_notes must be a list")
    try:
        for note in snapshot["repair_notes"]:
            copy_json_value(note, path=f"$.{source}.repair_notes")
    except (TypeError, ValueError) as exc:
        raise DatasetIntegrityError(f"{source} repair_notes are invalid: {exc}") from exc
    digest = snapshot["journal_sha256"]
    if type(digest) is not str or len(digest) != 64:
        raise DatasetIntegrityError(f"{source} journal_sha256 is invalid")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise DatasetIntegrityError(f"{source} journal_sha256 is not hex") from exc

    if size > len(complete_bytes):
        raise DatasetIntegrityError(
            f"{source} cursor {size} exceeds journal length {len(complete_bytes)}; "
            "committed records may be missing"
        )
    prefix = complete_bytes[:size]
    if _sha256_bytes(prefix) != digest:
        raise DatasetIntegrityError(
            f"{source} committed journal prefix hash mismatch; dataset was tampered "
            "with or the journal was truncated"
        )
    if size and not prefix.endswith(b"\n"):
        raise DatasetIntegrityError(f"{source} cursor is not on a line boundary")
    records = list(_iter_record_lines(prefix, source))
    if len(records) != count:
        raise DatasetIntegrityError(
            f"{source} record_count {count} does not match prefix {len(records)}"
        )
    last_candidate_id = snapshot["last_candidate_id"]
    if count:
        if last_candidate_id != records[-1].candidate_id:
            raise DatasetIntegrityError(
                f"{source} last_candidate_id does not match the committed prefix"
            )
    elif last_candidate_id is not None:
        raise DatasetIntegrityError(
            f"{source} empty dataset must have last_candidate_id=null"
        )
    return records


def _compare_snapshot_identity(
    first: Mapping[str, Any], second: Mapping[str, Any], *, first_name: str, second_name: str
) -> None:
    for key in (
        "dataset_version",
        "schema_version",
        "format",
        "run_id",
        "run_config_hash",
        "evaluator_version",
        "evaluator_version_hash",
    ):
        if first[key] != second[key]:
            raise DatasetIntegrityError(
                f"{first_name} and {second_name} disagree on {key}"
            )


def _verify_dataset(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise DatasetIntegrityError(f"dataset directory does not exist: {directory}")
    manifest = _read_json_object(directory / MANIFEST_NAME)
    checkpoint = _read_json_object(directory / CHECKPOINT_NAME)
    journal_path = directory / JOURNAL_NAME
    if not journal_path.is_file():
        raise DatasetIntegrityError(f"missing dataset file: {journal_path}")
    try:
        raw = journal_path.read_bytes()
    except OSError as exc:
        raise DatasetIntegrityError(f"cannot read journal: {exc}") from exc
    if raw and not raw.endswith(b"\n"):
        raise DatasetIntegrityError(
            "journal has an incomplete final line; open the dataset as a writer "
            "to repair it or investigate the interrupted write"
        )

    manifest_records = _validate_snapshot(manifest, raw, MANIFEST_NAME)
    checkpoint_records = _validate_snapshot(checkpoint, raw, CHECKPOINT_NAME)
    _compare_snapshot_identity(
        manifest, checkpoint, first_name=MANIFEST_NAME, second_name=CHECKPOINT_NAME
    )
    if (
        manifest["record_count"] != checkpoint["record_count"]
        or manifest["journal_size_bytes"] != checkpoint["journal_size_bytes"]
        or manifest["journal_sha256"] != checkpoint["journal_sha256"]
    ):
        raise DatasetIntegrityError(
            "manifest and checkpoint cursors differ; resume the dataset as a writer"
        )
    if manifest["journal_size_bytes"] != len(raw):
        raise DatasetIntegrityError(
            "journal contains uncommitted records; resume the dataset as a writer"
        )
    if manifest["journal_sha256"] != _sha256_bytes(raw):
        raise DatasetIntegrityError("manifest journal hash mismatch")
    if manifest["record_count"] != len(manifest_records):
        raise DatasetIntegrityError("manifest record count mismatch")
    if manifest["record_count"] != len(checkpoint_records):
        raise DatasetIntegrityError("checkpoint record count mismatch")

    records = list(_iter_record_lines(raw, "journal"))
    _ensure_unique(records, "journal")
    if len(records) != manifest["record_count"]:
        raise DatasetIntegrityError("journal record count mismatch")
    for record in records:
        if record.run_id != manifest["run_id"]:
            raise DatasetIntegrityError(
                f"record {record.candidate_id} belongs to a different run_id"
            )
    return manifest


def iter_records(directory: str | os.PathLike[str]) -> Iterator[RawCandidate]:
    """Verify the dataset and yield RAW records one at a time.

    The dataset must be in a clean committed state.  An interrupted dataset
    must first be repaired by opening it as a :class:`DatasetWriter`.
    """

    path = Path(directory)
    _verify_dataset(path)
    with open(path / JOURNAL_NAME, "rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                raise DatasetIntegrityError(
                    f"journal line {line_number} is incomplete"
                )
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise DatasetIntegrityError(
                    f"journal line {line_number} is not valid UTF-8"
                ) from exc
            try:
                yield RawCandidate.from_json(line)
            except RecordIntegrityError as exc:
                raise DatasetIntegrityError(
                    f"journal line {line_number} failed verification: {exc}"
                ) from exc


def read_records(directory: str | os.PathLike[str]) -> tuple[RawCandidate, ...]:
    """Return all RAW records after full integrity verification."""

    return tuple(iter_records(directory))


def load_manifest(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a verified, fresh manifest dictionary."""

    manifest = _verify_dataset(Path(directory))
    return copy_json_value(manifest)


class DatasetWriter:
    """Append-only single-writer RAW dataset.

    Use :meth:`create` for a new dataset and :meth:`open` to resume.  A writer
    commits count/hash cursors after each append (or once per
    :meth:`append_many` batch).  The journal remains the recovery authority.
    """

    def __init__(
        self,
        directory: Path,
        run_config: Mapping[str, Any],
        evaluator_version: str,
        journal,
        lock: _DatasetLock,
        records: Mapping[str, RawCandidate],
        size: int,
        hasher: Any,
        repair_notes: Sequence[Mapping[str, Any]],
        *,
        complete: bool,
    ) -> None:
        self._directory = directory
        self._run_config = copy_json_value(run_config, path="$.run_config")
        self._evaluator_version = evaluator_version
        self._journal = journal
        self._lock = lock
        self._records_by_id: dict[str, RawCandidate] = dict(records)
        self._size = size
        self._hasher = hasher
        self._repair_notes = [copy_json_value(note) for note in repair_notes]
        self._complete = complete
        self._closed = False
        self.run_id = make_run_id(self._run_config)
        self.run_config_hash = canonical_hash(self._run_config)
        self._last_candidate_id = (
            next(reversed(self._records_by_id)) if self._records_by_id else None
        )
        # The dict order is insertion/record order only while loaded from the
        # journal.  Recompute from journal order on open to avoid any ambiguity.
        if self._records_by_id:
            self._last_candidate_id = list(self._records_by_id)[-1]

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def model_variant(self) -> str:
        return self._run_config["model_variant"]

    @property
    def evaluator_version(self) -> str:
        return self._evaluator_version

    @property
    def run_config(self) -> dict[str, Any]:
        return copy_json_value(self._run_config)

    @property
    def record_count(self) -> int:
        return len(self._records_by_id)

    @property
    def repair_notes(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy_json_value(note) for note in self._repair_notes)

    @property
    def complete(self) -> bool:
        return self._complete

    def _check_open(self) -> None:
        if self._closed:
            raise DatasetError("dataset writer is closed")

    def _snapshot(self, *, complete: bool) -> dict[str, Any]:
        return {
            "dataset_version": DATASET_VERSION,
            "schema_version": RAW_SCHEMA_VERSION,
            "format": JOURNAL_FORMAT_JSONL,
            "run_id": self.run_id,
            "run_config": copy_json_value(self._run_config),
            "run_config_hash": self.run_config_hash,
            "evaluator_version": self._evaluator_version,
            "evaluator_version_hash": evaluator_version_hash(
                self._evaluator_version
            ),
            "record_count": len(self._records_by_id),
            "journal_size_bytes": self._size,
            "journal_sha256": self._hasher.hexdigest(),
            "last_candidate_id": self._last_candidate_id,
            "complete": bool(complete),
            "repair_notes": [copy_json_value(note) for note in self._repair_notes],
        }

    def _commit(self, *, complete: bool) -> None:
        snapshot = self._snapshot(complete=complete)
        # Checkpoint first, then manifest.  A crash between them is recoverable
        # because both are prefix cursors and the journal is authoritative.
        _atomic_write_json(self._directory / CHECKPOINT_NAME, snapshot)
        _atomic_write_json(self._directory / MANIFEST_NAME, snapshot)
        self._complete = complete

    @property
    def manifest(self) -> dict[str, Any]:
        """Fresh snapshot of the current in-memory committed state."""

        return self._snapshot(complete=self._complete)

    def _validate_record_for_run(self, record: RawCandidate) -> None:
        if not isinstance(record, RawCandidate):
            raise TypeError("only RawCandidate records can be appended")
        if record.run_id != self.run_id:
            raise RunConfigMismatchError(
                f"record run_id {record.run_id!r} does not match dataset run_id "
                f"{self.run_id!r}"
            )
        if record.model_variant != self.model_variant:
            raise RunConfigMismatchError(
                f"record model_variant {record.model_variant!r} does not match "
                f"dataset model_variant {self.model_variant!r}"
            )

    def _write_bytes(self, data: bytes) -> None:
        self._journal.write(data)
        self._journal.flush()
        os.fsync(self._journal.fileno())
        self._hasher.update(data)
        self._size += len(data)

    @staticmethod
    def _journal_bytes(record: RawCandidate) -> bytes:
        return record.to_json().encode("utf-8") + b"\n"

    def append(self, record: RawCandidate) -> RawCandidate:
        """Append one record or exactly no-op an identical duplicate."""

        self._check_open()
        self._validate_record_for_run(record)
        existing = self._records_by_id.get(record.candidate_id)
        if existing is not None:
            if existing.record_hash == record.record_hash:
                return existing
            raise DuplicateCandidateError(
                f"candidate_id {record.candidate_id!r} already exists with a "
                "different record hash"
            )
        data = self._journal_bytes(record)
        self._write_bytes(data)
        self._records_by_id[record.candidate_id] = record
        self._last_candidate_id = record.candidate_id
        self._commit(complete=False)
        return record

    def append_many(self, records: Sequence[RawCandidate]) -> int:
        """Append a batch, validating duplicates before writing anything.

        Returns the number of newly written records.  Exact duplicate records
        are idempotent no-ops; conflicting duplicate ids raise before any bytes
        are written.  One cursor commit follows the batch.
        """

        self._check_open()
        prepared: list[RawCandidate] = []
        batch_seen: dict[str, str] = {}
        for record in records:
            self._validate_record_for_run(record)
            existing = self._records_by_id.get(record.candidate_id)
            if existing is not None:
                if existing.record_hash == record.record_hash:
                    continue
                raise DuplicateCandidateError(
                    f"candidate_id {record.candidate_id!r} already exists with a "
                    "different record hash"
                )
            previous_hash = batch_seen.get(record.candidate_id)
            if previous_hash is not None:
                if previous_hash == record.record_hash:
                    continue
                raise DuplicateCandidateError(
                    f"candidate_id {record.candidate_id!r} appears twice in the "
                    "batch with different record hashes"
                )
            batch_seen[record.candidate_id] = record.record_hash
            prepared.append(record)

        if not prepared:
            return 0
        data = b"".join(self._journal_bytes(record) for record in prepared)
        self._write_bytes(data)
        for record in prepared:
            self._records_by_id[record.candidate_id] = record
        self._last_candidate_id = prepared[-1].candidate_id
        self._commit(complete=False)
        return len(prepared)

    def evaluate(
        self,
        evaluator,
        input_payload: Mapping[str, Any],
        *,
        architecture: str,
        sampling_method: str = "manual",
        sampling_metadata: Mapping[str, Any] | None = None,
    ) -> RawCandidate:
        """Evaluate one payload with this dataset's run id/model variant."""

        self._check_open()
        return evaluate_candidate(
            evaluator,
            input_payload,
            architecture=architecture,
            model_variant=self.model_variant,
            run_id=self.run_id,
            sampling_method=sampling_method,
            sampling_metadata=sampling_metadata,
        )

    def evaluate_and_append(
        self,
        evaluator,
        input_payload: Mapping[str, Any],
        *,
        architecture: str,
        sampling_method: str = "manual",
        sampling_metadata: Mapping[str, Any] | None = None,
    ) -> RawCandidate:
        """Evaluate one payload and append the resulting RAW record."""

        record = self.evaluate(
            evaluator,
            input_payload,
            architecture=architecture,
            sampling_method=sampling_method,
            sampling_metadata=sampling_metadata,
        )
        return self.append(record)

    def finish(self) -> None:
        """Mark the current cursor complete without closing the lock/file."""

        self._check_open()
        self._commit(complete=True)

    def close(self, *, finalize: bool = True) -> None:
        """Close the writer.

        ``finalize=True`` writes a complete cursor.  ``finalize=False`` releases
        the lock without committing, which is useful for interruption tests and
        is safe because the journal is already written.
        """

        if self._closed:
            return
        try:
            if finalize:
                self._commit(complete=True)
        finally:
            self._closed = True
            try:
                self._journal.close()
            finally:
                self._lock.release()

    def __enter__(self) -> "DatasetWriter":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close(finalize=True)
        return False

    @classmethod
    def create(
        cls,
        directory: str | os.PathLike[str],
        *,
        run_config: Mapping[str, Any],
        evaluator_version: str | None = None,
    ) -> "DatasetWriter":
        """Create a new dataset in an empty/new directory."""

        config, version = normalize_run_config(
            run_config, evaluator_version=evaluator_version
        )
        path = Path(directory)
        if path.exists():
            if not path.is_dir():
                raise DatasetExistsError(f"not a directory: {path}")
            if any(path.iterdir()):
                raise DatasetExistsError(
                    f"dataset directory is not empty: {path}; refusing to overwrite"
                )
        else:
            path.mkdir(parents=True)

        lock = _DatasetLock(path / LOCK_NAME)
        journal = None
        try:
            journal_path = path / JOURNAL_NAME
            if journal_path.exists():
                raise DatasetExistsError(f"dataset journal already exists: {journal_path}")
            journal_path.touch(exist_ok=False)
            journal = open(journal_path, "ab", buffering=0)
            writer = cls(
                path,
                config,
                version,
                journal,
                lock,
                {},
                0,
                hashlib.sha256(),
                (),
                complete=False,
            )
            writer._commit(complete=False)
            return writer
        except Exception:
            if journal is not None:
                journal.close()
            lock.release()
            raise

    @classmethod
    def open(
        cls,
        directory: str | os.PathLike[str],
        *,
        expected_run_config: Mapping[str, Any] | None = None,
        expected_evaluator_version: str | None = None,
    ) -> "DatasetWriter":
        """Open and repair an existing dataset, refusing mismatched config/data."""

        path = Path(directory)
        if not path.is_dir():
            raise DatasetIntegrityError(f"dataset directory does not exist: {path}")
        lock = _DatasetLock(path / LOCK_NAME)
        journal = None
        try:
            manifest = _read_json_object(path / MANIFEST_NAME)
            checkpoint = _read_json_object(path / CHECKPOINT_NAME)
            journal_path = path / JOURNAL_NAME
            if not journal_path.is_file():
                raise DatasetIntegrityError(f"missing dataset file: {journal_path}")
            try:
                raw = journal_path.read_bytes()
            except OSError as exc:
                raise DatasetIntegrityError(f"cannot read journal: {exc}") from exc
            complete_bytes, tail = _split_complete_journal(raw)

            manifest_records = _validate_snapshot(
                manifest, complete_bytes, MANIFEST_NAME
            )
            checkpoint_records = _validate_snapshot(
                checkpoint, complete_bytes, CHECKPOINT_NAME
            )
            _compare_snapshot_identity(
                manifest,
                checkpoint,
                first_name=MANIFEST_NAME,
                second_name=CHECKPOINT_NAME,
            )

            if expected_run_config is not None:
                if (
                    expected_evaluator_version is not None
                    and expected_evaluator_version != manifest["evaluator_version"]
                ):
                    raise RunConfigMismatchError(
                        "expected evaluator_version does not match the dataset"
                    )
                try:
                    expected_config, expected_version = normalize_run_config(
                        expected_run_config,
                        evaluator_version=expected_evaluator_version,
                    )
                except (TypeError, ValueError) as exc:
                    raise RunConfigMismatchError(
                        f"expected run_config is invalid or mismatched: {exc}"
                    ) from exc
                if canonical_hash(expected_config) != manifest["run_config_hash"]:
                    raise RunConfigMismatchError(
                        "expected run_config does not match the dataset"
                    )
                if expected_version != manifest["evaluator_version"]:
                    raise RunConfigMismatchError(
                        "expected evaluator_version does not match the dataset"
                    )
            elif expected_evaluator_version is not None:
                if expected_evaluator_version != manifest["evaluator_version"]:
                    raise RunConfigMismatchError(
                        "expected evaluator_version does not match the dataset"
                    )

            all_records = list(_iter_record_lines(complete_bytes, "journal"))
            _ensure_unique(all_records, "journal")
            if len(all_records) < max(
                manifest["record_count"], checkpoint["record_count"]
            ):
                raise DatasetIntegrityError(
                    "journal contains fewer records than a committed cursor"
                )

            repair_notes = [
                copy_json_value(note) for note in manifest.get("repair_notes", [])
            ]
            repaired_tail = False
            if tail:
                repair_notes.append(
                    {
                        "kind": "truncated_journal_tail",
                        "bytes": len(tail),
                        "sha256": _sha256_bytes(tail),
                    }
                )
                with open(journal_path, "r+b") as handle:
                    handle.truncate(len(complete_bytes))
                repaired_tail = True

            journal = open(journal_path, "ab", buffering=0)
            records_by_id = {record.candidate_id: record for record in all_records}
            writer = cls(
                path,
                manifest["run_config"],
                manifest["evaluator_version"],
                journal,
                lock,
                records_by_id,
                len(complete_bytes),
                hashlib.sha256(complete_bytes),
                repair_notes,
                complete=bool(manifest["complete"]),
            )
            if writer._last_candidate_id != (
                all_records[-1].candidate_id if all_records else None
            ):
                raise DatasetIntegrityError(
                    "journal order does not match the loaded record index"
                )

            current = (
                not repaired_tail
                and manifest["record_count"] == len(all_records)
                and checkpoint["record_count"] == len(all_records)
                and manifest["journal_size_bytes"] == len(complete_bytes)
                and checkpoint["journal_size_bytes"] == len(complete_bytes)
                and manifest["journal_sha256"] == _sha256_bytes(complete_bytes)
                and checkpoint["journal_sha256"] == _sha256_bytes(complete_bytes)
                and manifest["last_candidate_id"]
                == (all_records[-1].candidate_id if all_records else None)
                and checkpoint["last_candidate_id"]
                == (all_records[-1].candidate_id if all_records else None)
            )
            if not current:
                writer._commit(complete=False)
            return writer
        except Exception:
            if journal is not None:
                journal.close()
            lock.release()
            raise


@dataclass(frozen=True)
class ReproduceResult:
    """Outcome of reproducing one stored candidate."""

    candidate_id: str
    matches: bool
    differences: tuple[str, ...]
    expected: RawCandidate
    actual: RawCandidate

    def __bool__(self) -> bool:
        return self.matches


def _validate_tolerance(rtol: Any, atol: Any) -> tuple[float, float]:
    values: list[float] = []
    for name, value in (("rtol", rtol), ("atol", atol)):
        if type(value) is bool or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite non-negative number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number")
        values.append(numeric)
    return values[0], values[1]


def _compare_json(
    expected: Any,
    actual: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
    differences: list[str],
) -> None:
    if type(expected) is bool or type(actual) is bool:
        if type(expected) is not type(actual) or expected != actual:
            differences.append(f"{path}: {expected!r} != {actual!r}")
        return

    if isinstance(expected, (int, float)) or isinstance(actual, (int, float)):
        if not isinstance(expected, (int, float)) or not isinstance(
            actual, (int, float)
        ):
            if expected != actual:
                differences.append(f"{path}: {expected!r} != {actual!r}")
            return
        if not math.isclose(
            float(expected), float(actual), rel_tol=rtol, abs_tol=atol
        ):
            differences.append(f"{path}: {expected!r} != {actual!r}")
        return

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            if expected_keys - actual_keys:
                differences.append(
                    f"{path}: missing keys {sorted(expected_keys - actual_keys)!r}"
                )
            if actual_keys - expected_keys:
                differences.append(
                    f"{path}: extra keys {sorted(actual_keys - expected_keys)!r}"
                )
        for key in sorted(expected_keys & actual_keys):
            _compare_json(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
                differences=differences,
            )
        return

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append(
                f"{path}: list length {len(expected)} != {len(actual)}"
            )
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare_json(
                left,
                right,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
                differences=differences,
            )
        return

    if type(expected) is not type(actual) or expected != actual:
        differences.append(f"{path}: {expected!r} != {actual!r}")


def compare_candidates(
    expected: RawCandidate,
    actual: RawCandidate,
    *,
    rtol: Any,
    atol: Any,
) -> tuple[str, ...]:
    """Compare causal reproduction fields under explicit numeric tolerance.

    Identifiers, timestamps, hashes, and free-form ``diagnostics`` are ignored.
    Compared fields are calculation status, failure stage, outputs, causal
    intermediates, and last valid state.  Failure reason text is diagnostic, not
    causal.  Tolerances are required and never widened automatically.
    """

    if not isinstance(expected, RawCandidate) or not isinstance(actual, RawCandidate):
        raise TypeError("expected and actual must be RawCandidate instances")
    rtol_value, atol_value = _validate_tolerance(rtol, atol)
    differences: list[str] = []
    if expected.calculation_status is not actual.calculation_status:
        differences.append(
            "$.calculation_status: "
            f"{expected.calculation_status.value!r} != "
            f"{actual.calculation_status.value!r}"
        )
    if expected.failure_stage != actual.failure_stage:
        differences.append(
            f"$.failure_stage: {expected.failure_stage!r} != "
            f"{actual.failure_stage!r}"
        )
    for field_name in ("outputs", "intermediates", "last_valid_state"):
        _compare_json(
            getattr(expected, field_name),
            getattr(actual, field_name),
            path=f"$.{field_name}",
            rtol=rtol_value,
            atol=atol_value,
            differences=differences,
        )
    return tuple(differences)


def reproduce_candidate(
    directory: str | os.PathLike[str],
    candidate_id: str,
    evaluator,
    *,
    rtol: Any,
    atol: Any,
) -> ReproduceResult:
    """Re-evaluate one stored candidate and compare causal fields.

    The evaluator receives the stored payload.  Callers must pass explicit
    ``rtol`` and ``atol``; no automatic tolerance broadening is performed.
    """

    if type(candidate_id) is not str or not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    path = Path(directory)
    _verify_dataset(path)
    expected: RawCandidate | None = None
    for record in iter_records(path):
        if record.candidate_id == candidate_id:
            expected = record
            break
    if expected is None:
        raise CandidateNotFoundError(
            f"candidate {candidate_id!r} not found in {path}"
        )

    actual = evaluate_candidate(
        evaluator,
        expected.payload,
        architecture=expected.architecture,
        model_variant=expected.model_variant,
        run_id=expected.run_id,
        sampling_method=expected.sampling_method,
        sampling_metadata=expected.sampling,
    )
    differences = compare_candidates(expected, actual, rtol=rtol, atol=atol)
    return ReproduceResult(
        candidate_id=candidate_id,
        matches=not differences,
        differences=differences,
        expected=expected,
        actual=actual,
    )


_PARQUET_SCHEMA_FIELDS = (
    "schema_version",
    "run_id",
    "candidate_id",
    "architecture",
    "model_variant",
    "sampling_method",
    "input_hash",
    "calculation_status",
    "constraint_status",
    "failure_stage",
    "failure_reason",
    "record_hash",
)
_PARQUET_JSON_FIELDS = (
    "payload",
    "sampling",
    "intermediates",
    "outputs",
    "diagnostics",
    "last_valid_state",
)


def _parquet_row(record: RawCandidate) -> dict[str, Any]:
    data = record.to_dict()
    row: dict[str, Any] = {name: data[name] for name in _PARQUET_SCHEMA_FIELDS}
    for name in _PARQUET_JSON_FIELDS:
        value = data[name]
        row[name] = None if value is None else canonical_json(value)
    return row


def export_parquet(
    directory: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    batch_size: int = 1000,
    overwrite: bool = False,
) -> Path:
    """Export a verified RAW dataset to Parquet in bounded batches.

    Parquet is a derived export, not the recovery authority.  Arbitrary nested
    payload/output structures are stored as canonical JSON strings in nullable
    string columns so the table schema is stable and streaming-friendly.
    ``pyarrow`` is imported only here; if it is absent this raises
    :class:`OptionalDependencyError` instead of claiming Parquet support.
    """

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise OptionalDependencyError(
            "Parquet export requires the optional 'pyarrow' dependency; "
            "install pyarrow in the active environment or use the JSONL journal"
        ) from exc

    source = Path(directory)
    destination = Path(output_path)
    authoritative = {
        (source / JOURNAL_NAME).resolve(),
        (source / MANIFEST_NAME).resolve(),
        (source / CHECKPOINT_NAME).resolve(),
    }
    if destination.resolve() in authoritative:
        raise ValueError("refusing to overwrite an authoritative dataset file")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {destination}")
    _verify_dataset(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            pa.field(
                name,
                pa.string(),
                nullable=name in {"failure_stage", "failure_reason"},
            )
            for name in _PARQUET_SCHEMA_FIELDS
        ]
        + [
            pa.field(name, pa.string(), nullable=True)
            for name in _PARQUET_JSON_FIELDS
        ]
    )

    handle, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    writer = None
    try:
        writer = pq.ParquetWriter(temporary, schema)
        batch: list[dict[str, Any]] = []
        for record in _iter_journal_records(source):
            batch.append(_parquet_row(record))
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        writer.close()
        writer = None
        os.replace(temporary, destination)
        return destination
    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _iter_journal_records(directory: Path) -> Iterator[RawCandidate]:
    with open(directory / JOURNAL_NAME, "rb") as handle:
        for raw_line in handle:
            if not raw_line.endswith(b"\n"):
                raise DatasetIntegrityError("journal line is incomplete")
            yield RawCandidate.from_json(raw_line.decode("utf-8"))


__all__ = [
    "DATASET_VERSION",
    "JOURNAL_NAME",
    "MANIFEST_NAME",
    "CHECKPOINT_NAME",
    "LOCK_NAME",
    "DatasetError",
    "DatasetExistsError",
    "DatasetLockError",
    "DatasetIntegrityError",
    "RunConfigMismatchError",
    "DuplicateCandidateError",
    "CandidateNotFoundError",
    "OptionalDependencyError",
    "DatasetWriter",
    "ReproduceResult",
    "iter_records",
    "read_records",
    "load_manifest",
    "compare_candidates",
    "reproduce_candidate",
    "export_parquet",
]

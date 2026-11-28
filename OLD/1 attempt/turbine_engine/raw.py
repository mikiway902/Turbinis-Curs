"""Generic, versioned RAW candidate records for design-space calculation.

M3/M5 infrastructure only
=========================
This module does not contain or import any turbine physics.  It accepts an
arbitrary user-supplied evaluator callable and a JSON-compatible input payload,
then builds one immutable :class:`RawCandidate` record.  It deliberately makes
no assumption about the payload schema, architecture names, sampling boxes, or
engineering acceptance limits.

Calculation semantics
=====================
One record stores inputs, causal intermediates, outputs, diagnostics, and the
calculation status.  ``calculation_status`` is distinct from
``constraint_status``: this module never marks a calculable point as rejected.
The RAW record always carries ``constraint_status = NOT_EVALUATED``.  Constraint
annotations are produced separately in :mod:`turbine_engine.constraints` and
never mutate a RAW record.

Statuses are strict and versioned:

* ``SUCCESS`` -- evaluator returned a complete, serializable result.
* ``NUMERICAL_FAILURE`` -- explicit solver/root/division failure.
* ``NON_PHYSICAL_STATE`` -- explicit physically meaningless state.
* ``MODEL_ERROR`` -- explicit model error, unexpected evaluator exception, or
  an evaluator result that cannot be stored as strict JSON.

Records deliberately do not swallow ``KeyboardInterrupt`` or ``SystemExit``.

JSON and immutability
=====================
All stored sub-structures are canonical JSON (sorted keys, compact separators,
finite numbers only).  Invalid payloads are rejected before a record exists.
Each record has ``input_hash`` (the canonical payload hash) and ``record_hash``
(hash of the complete record except ``record_hash``).  The immutable dataclass
stores canonical JSON strings internally, so accessors return fresh copies and
caller mutations cannot alias record state.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping


RAW_SCHEMA_VERSION = "raw-candidate-v1"
JOURNAL_FORMAT_JSONL = "jsonl"

_JSON_FIELDS = (
    "payload_json",
    "sampling_json",
    "intermediates_json",
    "outputs_json",
    "diagnostics_json",
    "last_valid_state_json",
)

_REQUIRED_RECORD_KEYS = frozenset(
    {
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
        "last_valid_state",
        "intermediates",
        "outputs",
        "diagnostics",
        "payload",
        "sampling",
        "record_hash",
    }
)


class PayloadValidationError(ValueError):
    """The supplied value is not strict JSON-compatible data."""


class RecordIntegrityError(ValueError):
    """A RAW record failed schema, identifier, or hash verification."""


class CalculationStatus(str, Enum):
    """Strict calculation outcomes for a RAW candidate."""

    SUCCESS = "SUCCESS"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    NON_PHYSICAL_STATE = "NON_PHYSICAL_STATE"
    MODEL_ERROR = "MODEL_ERROR"


class ConstraintStatus(str, Enum):
    """Strict post-calculation acceptance outcomes.

    ``NOT_EVALUATED`` belongs to an untouched RAW record.  ``NOT_CHECKED`` is
    used when an acceptance rule cannot be evaluated because a value is
    missing, null, or non-finite; it is never treated as a pass.
    """

    NOT_EVALUATED = "NOT_EVALUATED"
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_CHECKED = "NOT_CHECKED"


_FAILURE_STATUSES = frozenset(
    {
        CalculationStatus.NUMERICAL_FAILURE,
        CalculationStatus.NON_PHYSICAL_STATE,
        CalculationStatus.MODEL_ERROR,
    }
)


def _reject_constant(token: str) -> None:
    raise PayloadValidationError(f"non-finite JSON number is not allowed: {token}")


def _check_text(name: str, value: Any) -> str:
    if type(value) is not str or not value:
        raise PayloadValidationError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PayloadValidationError(f"{name} must be valid UTF-8 text") from exc
    return value


def _copy_json_value(value: Any, *, path: str = "$") -> Any:
    """Validate and deep-copy a strict JSON value.

    Accepted leaf types are exactly ``None``, ``bool``, ``int``, finite
    ``float``, and ``str``.  Mappings are copied to plain ``dict`` objects with
    string keys; lists are copied to plain lists.  Tuples, sets, bytes, NumPy
    scalars, ``Decimal``, NaN, and infinities are rejected rather than silently
    coerced.
    """

    if value is None:
        return None
    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PayloadValidationError(
                f"{path} must be valid UTF-8 text"
            ) from exc
        return value
    if value_type is int:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise PayloadValidationError(f"{path} must be a finite number")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise PayloadValidationError(
                    f"{path} has a non-string key {key!r}; JSON object keys must be strings"
                )
            try:
                key.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise PayloadValidationError(
                    f"{path} has a key that is not valid UTF-8 text"
                ) from exc
            copied[key] = _copy_json_value(item, path=f"{path}.{key}")
        return copied
    if isinstance(value, list):
        return [
            _copy_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise PayloadValidationError(
        f"{path} has unsupported JSON type {type(value).__name__}; "
        "use only null, bool, int, finite float, str, list, and dict"
    )


def copy_json_value(value: Any, *, path: str = "$") -> Any:
    """Return a validated, deep-copied strict JSON value."""

    return _copy_json_value(value, path=path)


def _dumps_canonical(value: Any) -> str:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PayloadValidationError(f"value is not canonical JSON: {exc}") from exc
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PayloadValidationError("value contains invalid UTF-8 text") from exc
    return text


def canonical_json(value: Any) -> str:
    """Serialize a strict JSON value in canonical form.

    Canonical form is sorted object keys, compact separators, UTF-8 text, and
    no NaN/Infinity tokens.  A deep copy is validated before serialization.
    """

    return _dumps_canonical(_copy_json_value(value))


def canonical_hash(value: Any) -> str:
    """Return the SHA-256 hex digest of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_candidate_id(
    architecture: str, model_variant: str, payload: Mapping[str, Any]
) -> str:
    """Deterministically derive a candidate id from architecture/model/payload.

    No randomness, clock, process id, or host name participates.  The schema
    version is included so a deliberate schema change changes ids.
    """

    architecture = _check_text("architecture", architecture)
    model_variant = _check_text("model_variant", model_variant)
    payload_value = _copy_json_value(payload, path="$.payload")
    digest = canonical_hash(
        {
            "schema_version": RAW_SCHEMA_VERSION,
            "architecture": architecture,
            "model_variant": model_variant,
            "payload": payload_value,
        }
    )
    return f"cand_{digest}"


def evaluator_version_hash(evaluator_version: str) -> str:
    """Hash an explicit evaluator version string."""

    evaluator_version = _check_text("evaluator_version", evaluator_version)
    return canonical_hash(evaluator_version)


def normalize_run_config(
    run_config: Mapping[str, Any], *, evaluator_version: str | None = None
) -> tuple[dict[str, Any], str]:
    """Validate run metadata and return ``(canonical_config, evaluator_version)``.

    ``model_variant`` is required.  The evaluator version is explicit and is
    stored inside the normalized config so it participates in the run id.
    Passing a different evaluator version alongside a config that already
    records one is an error.
    """

    config = _copy_json_value(run_config, path="$.run_config")
    if not isinstance(config, dict):
        raise PayloadValidationError("run_config must be a JSON object")
    if "model_variant" not in config:
        raise PayloadValidationError("run_config must contain model_variant")
    _check_text("run_config.model_variant", config["model_variant"])

    recorded = config.get("evaluator_version")
    if recorded is not None:
        _check_text("run_config.evaluator_version", recorded)
    if evaluator_version is None:
        if recorded is None:
            raise PayloadValidationError(
                "evaluator_version is required, either as an argument or in run_config"
            )
        evaluator_version = recorded
    else:
        _check_text("evaluator_version", evaluator_version)
        if recorded is not None and recorded != evaluator_version:
            raise PayloadValidationError(
                "run_config.evaluator_version does not match the explicit evaluator_version"
            )

    config["evaluator_version"] = evaluator_version
    return config, evaluator_version


def run_config_hash(
    run_config: Mapping[str, Any], *, evaluator_version: str | None = None
) -> str:
    """Return the canonical hash of a normalized run configuration."""

    config, _ = normalize_run_config(run_config, evaluator_version=evaluator_version)
    return canonical_hash(config)


def make_run_id(
    run_config: Mapping[str, Any], *, evaluator_version: str | None = None
) -> str:
    """Deterministically derive a run id from config metadata only."""

    config, _ = normalize_run_config(run_config, evaluator_version=evaluator_version)
    digest = canonical_hash(
        {"schema_version": RAW_SCHEMA_VERSION, "run_config": config}
    )
    return f"run_{digest}"


class EvaluationFailure(Exception):
    """Base class for an evaluator-declared, non-success outcome.

    Raise a subclass from an evaluator to retain the candidate inputs together
    with a structured diagnostic record.  ``KeyboardInterrupt`` and
    ``SystemExit`` are never converted into records.
    """

    status: CalculationStatus

    def __init__(
        self,
        status: CalculationStatus,
        stage: str,
        reason: str,
        *,
        last_valid_state: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        intermediates: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            resolved_status = CalculationStatus(status)
        except ValueError as exc:
            raise ValueError(f"unknown calculation status {status!r}") from exc
        if resolved_status not in _FAILURE_STATUSES:
            raise ValueError("EvaluationFailure status must be a failure status")
        stage = _check_text("failure_stage", stage)
        reason = _check_text("failure_reason", reason)
        super().__init__(f"{resolved_status.value} at {stage}: {reason}")
        self.status = resolved_status
        self.stage = stage
        self.reason = reason
        self.last_valid_state = last_valid_state
        self.diagnostics = diagnostics
        self.outputs = outputs
        self.intermediates = intermediates

    def to_outcome(self) -> "EvaluationOutcome":
        """Convert this exception to an :class:`EvaluationOutcome`."""

        return EvaluationOutcome(
            status=self.status,
            outputs=self.outputs,
            intermediates=self.intermediates,
            diagnostics=self.diagnostics,
            failure_stage=self.stage,
            failure_reason=self.reason,
            last_valid_state=self.last_valid_state,
        )


class NumericalFailure(EvaluationFailure):
    """Explicit numerical failure (non-convergence, root/division failure)."""

    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        last_valid_state: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        intermediates: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            CalculationStatus.NUMERICAL_FAILURE,
            stage,
            reason,
            last_valid_state=last_valid_state,
            diagnostics=diagnostics,
            outputs=outputs,
            intermediates=intermediates,
        )


class NonPhysicalStateError(EvaluationFailure):
    """Explicit physically meaningless state (for example non-positive T/p)."""

    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        last_valid_state: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        intermediates: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            CalculationStatus.NON_PHYSICAL_STATE,
            stage,
            reason,
            last_valid_state=last_valid_state,
            diagnostics=diagnostics,
            outputs=outputs,
            intermediates=intermediates,
        )


class ModelError(EvaluationFailure):
    """Explicit evaluator/model error that is not a numerical/physical failure."""

    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        last_valid_state: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        intermediates: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            CalculationStatus.MODEL_ERROR,
            stage,
            reason,
            last_valid_state=last_valid_state,
            diagnostics=diagnostics,
            outputs=outputs,
            intermediates=intermediates,
        )


class EvaluationSerializationError(EvaluationFailure):
    """Explicit evaluator result serialization failure, not a physics failure."""

    def __init__(
        self,
        reason: str,
        *,
        stage: str = "result_serialization",
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            CalculationStatus.MODEL_ERROR,
            stage,
            reason,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class EvaluationOutcome:
    """Evaluator result before it is committed to a RAW record."""

    status: CalculationStatus = CalculationStatus.SUCCESS
    outputs: Mapping[str, Any] | None = None
    intermediates: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None
    last_valid_state: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        try:
            status = CalculationStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"unknown calculation status {self.status!r}") from exc
        object.__setattr__(self, "status", status)

        if status is CalculationStatus.SUCCESS:
            if self.failure_stage is not None or self.failure_reason is not None:
                raise ValueError("a SUCCESS outcome must not carry failure fields")
        else:
            _check_text("failure_stage", self.failure_stage)
            _check_text("failure_reason", self.failure_reason)

        for name in ("failure_stage", "failure_reason"):
            value = getattr(self, name)
            if value is not None:
                _check_text(name, value)

    @classmethod
    def success(
        cls,
        outputs: Mapping[str, Any] | None = None,
        *,
        intermediates: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> "EvaluationOutcome":
        """Build a successful outcome."""

        return cls(
            status=CalculationStatus.SUCCESS,
            outputs=outputs,
            intermediates=intermediates,
            diagnostics=diagnostics,
        )

    @classmethod
    def failure(
        cls,
        status: CalculationStatus,
        stage: str,
        reason: str,
        *,
        last_valid_state: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        outputs: Mapping[str, Any] | None = None,
        intermediates: Mapping[str, Any] | None = None,
    ) -> "EvaluationOutcome":
        """Build a failure outcome with a mandatory stage and reason."""

        return cls(
            status=status,
            outputs=outputs,
            intermediates=intermediates,
            diagnostics=diagnostics,
            failure_stage=stage,
            failure_reason=reason,
            last_valid_state=last_valid_state,
        )


def _json_text(value: Any, *, name: str) -> str:
    try:
        return canonical_json(value)
    except PayloadValidationError as exc:
        raise PayloadValidationError(f"{name} is not serializable: {exc}") from exc


@dataclass(frozen=True)
class RawCandidate:
    """One immutable, versioned RAW candidate record.

    The JSON sub-structures are stored as canonical text.  Use the properties
    (for example :attr:`payload`) to obtain fresh Python copies; mutating a
    returned object cannot mutate this record.
    """

    schema_version: str
    run_id: str
    candidate_id: str
    architecture: str
    model_variant: str
    sampling_method: str
    input_hash: str
    calculation_status: CalculationStatus
    constraint_status: ConstraintStatus
    failure_stage: str | None
    failure_reason: str | None
    record_hash: str
    payload_json: str = field(repr=False)
    sampling_json: str = field(repr=False)
    intermediates_json: str = field(repr=False)
    outputs_json: str = field(repr=False)
    diagnostics_json: str = field(repr=False)
    last_valid_state_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.schema_version != RAW_SCHEMA_VERSION:
            raise RecordIntegrityError(
                f"unsupported RAW schema version {self.schema_version!r}"
            )
        for name in (
            "run_id",
            "candidate_id",
            "architecture",
            "model_variant",
            "sampling_method",
        ):
            try:
                _check_text(name, getattr(self, name))
            except PayloadValidationError as exc:
                raise RecordIntegrityError(str(exc)) from exc
        try:
            status = CalculationStatus(self.calculation_status)
            constraint_status = ConstraintStatus(self.constraint_status)
        except ValueError as exc:
            raise RecordIntegrityError(f"invalid status: {exc}") from exc
        object.__setattr__(self, "calculation_status", status)
        object.__setattr__(self, "constraint_status", constraint_status)

        if constraint_status is not ConstraintStatus.NOT_EVALUATED:
            raise RecordIntegrityError(
                "RAW records must keep constraint_status=NOT_EVALUATED; "
                "use turbine_engine.constraints for post-calculation annotation"
            )

        if status is CalculationStatus.SUCCESS:
            if self.failure_stage is not None or self.failure_reason is not None:
                raise RecordIntegrityError(
                    "a SUCCESS record must not carry failure_stage/failure_reason"
                )
        else:
            try:
                _check_text("failure_stage", self.failure_stage)
                _check_text("failure_reason", self.failure_reason)
            except PayloadValidationError as exc:
                raise RecordIntegrityError(str(exc)) from exc

        for name in _JSON_FIELDS:
            text = getattr(self, name)
            if type(text) is not str:
                raise RecordIntegrityError(f"{name} must be a canonical JSON string")
            try:
                decoded = json.loads(text)
                if _dumps_canonical(decoded) != text:
                    raise RecordIntegrityError(f"{name} is not canonical JSON")
            except (TypeError, ValueError, PayloadValidationError) as exc:
                raise RecordIntegrityError(f"{name} is not valid canonical JSON") from exc

        payload = json.loads(self.payload_json)
        expected_input_hash = canonical_hash(payload)
        if self.input_hash != expected_input_hash:
            raise RecordIntegrityError(
                "input_hash does not match the canonical payload"
            )
        expected_candidate_id = make_candidate_id(
            self.architecture, self.model_variant, payload
        )
        if self.candidate_id != expected_candidate_id:
            raise RecordIntegrityError(
                "candidate_id does not match architecture/model_variant/payload"
            )
        if self.record_hash != canonical_hash(self._body()):
            raise RecordIntegrityError("record_hash does not match the record body")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "architecture": self.architecture,
            "model_variant": self.model_variant,
            "sampling_method": self.sampling_method,
            "input_hash": self.input_hash,
            "calculation_status": self.calculation_status.value,
            "constraint_status": self.constraint_status.value,
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "last_valid_state": json.loads(self.last_valid_state_json),
            "intermediates": json.loads(self.intermediates_json),
            "outputs": json.loads(self.outputs_json),
            "diagnostics": json.loads(self.diagnostics_json),
            "payload": json.loads(self.payload_json),
            "sampling": json.loads(self.sampling_json),
        }

    @property
    def payload(self) -> Any:
        """Fresh copy of the input payload."""

        return json.loads(self.payload_json)

    @property
    def sampling(self) -> Any:
        """Fresh copy of the sampling metadata."""

        return json.loads(self.sampling_json)

    @property
    def intermediates(self) -> Any:
        """Fresh copy of causal intermediates."""

        return json.loads(self.intermediates_json)

    @property
    def outputs(self) -> Any:
        """Fresh copy of evaluator outputs."""

        return json.loads(self.outputs_json)

    @property
    def diagnostics(self) -> Any:
        """Fresh copy of evaluator diagnostics."""

        return json.loads(self.diagnostics_json)

    @property
    def last_valid_state(self) -> Any:
        """Fresh copy of the last valid state, if recorded."""

        return json.loads(self.last_valid_state_json)

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh plain-dict representation including ``record_hash``."""

        body = self._body()
        body["record_hash"] = self.record_hash
        return body

    def to_json(self) -> str:
        """Return canonical single-line JSON for the journal."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RawCandidate":
        """Validate and reconstruct a record, including both hashes."""

        if not isinstance(record, Mapping):
            raise RecordIntegrityError("record must be a JSON object")
        keys = set(record)
        if keys != _REQUIRED_RECORD_KEYS:
            missing = sorted(_REQUIRED_RECORD_KEYS - keys)
            extra = sorted(keys - _REQUIRED_RECORD_KEYS)
            raise RecordIntegrityError(
                f"record keys mismatch; missing={missing!r}, extra={extra!r}"
            )

        body = {key: value for key, value in record.items() if key != "record_hash"}
        try:
            expected_hash = canonical_hash(body)
        except PayloadValidationError as exc:
            raise RecordIntegrityError(f"record body is not JSON-safe: {exc}") from exc
        if record["record_hash"] != expected_hash:
            raise RecordIntegrityError("record_hash does not match the record body")

        try:
            payload_text = _json_text(record["payload"], name="payload")
            sampling_text = _json_text(record["sampling"], name="sampling")
            intermediates_text = _json_text(
                record["intermediates"], name="intermediates"
            )
            outputs_text = _json_text(record["outputs"], name="outputs")
            diagnostics_text = _json_text(record["diagnostics"], name="diagnostics")
            last_valid_text = _json_text(
                record["last_valid_state"], name="last_valid_state"
            )
        except PayloadValidationError as exc:
            raise RecordIntegrityError(str(exc)) from exc

        try:
            return cls(
                schema_version=record["schema_version"],
                run_id=record["run_id"],
                candidate_id=record["candidate_id"],
                architecture=record["architecture"],
                model_variant=record["model_variant"],
                sampling_method=record["sampling_method"],
                input_hash=record["input_hash"],
                calculation_status=record["calculation_status"],
                constraint_status=record["constraint_status"],
                failure_stage=record["failure_stage"],
                failure_reason=record["failure_reason"],
                record_hash=record["record_hash"],
                payload_json=payload_text,
                sampling_json=sampling_text,
                intermediates_json=intermediates_text,
                outputs_json=outputs_text,
                diagnostics_json=diagnostics_text,
                last_valid_state_json=last_valid_text,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, RecordIntegrityError):
                raise
            raise RecordIntegrityError(str(exc)) from exc

    @classmethod
    def from_json(cls, text: str) -> "RawCandidate":
        """Parse one canonical journal line."""

        if type(text) is not str:
            raise RecordIntegrityError("record JSON must be text")
        try:
            value = json.loads(text, parse_constant=_reject_constant)
        except (TypeError, ValueError, PayloadValidationError) as exc:
            raise RecordIntegrityError(f"invalid record JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise RecordIntegrityError("record JSON must be an object")
        return cls.from_dict(value)


def _safe_exception_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = type(exc).__name__
    return text


def _serialization_failure(
    reason: str, *, component: str, attempted_status: Any, attempted_stage: Any
) -> EvaluationOutcome:
    diagnostics: dict[str, Any] = {
        "serialization_failure": True,
        "component": component,
    }
    if isinstance(attempted_status, CalculationStatus):
        diagnostics["attempted_status"] = attempted_status.value
    elif type(attempted_status) is str:
        diagnostics["attempted_status"] = attempted_status
    if type(attempted_stage) is str:
        diagnostics["attempted_failure_stage"] = attempted_stage
    return EvaluationOutcome.failure(
        CalculationStatus.MODEL_ERROR,
        "result_serialization",
        reason,
        diagnostics=diagnostics,
    )


def _coerce_outcome(result: Any) -> EvaluationOutcome:
    if isinstance(result, EvaluationOutcome):
        return result
    if isinstance(result, Mapping):
        return EvaluationOutcome.success(outputs=result)
    return _serialization_failure(
        f"evaluator returned unsupported result type {type(result).__name__}",
        component="result",
        attempted_status=None,
        attempted_stage=None,
    )


def _record_from_outcome(
    *,
    run_id: str,
    architecture: str,
    model_variant: str,
    sampling_method: str,
    payload_text: str,
    sampling_text: str,
    outcome: EvaluationOutcome,
) -> RawCandidate:
    payload = json.loads(payload_text)
    try:
        outputs_text = _json_text(outcome.outputs, name="outputs")
        intermediates_text = _json_text(outcome.intermediates, name="intermediates")
        diagnostics_text = _json_text(outcome.diagnostics, name="diagnostics")
        last_valid_text = _json_text(
            outcome.last_valid_state, name="last_valid_state"
        )
    except PayloadValidationError as exc:
        failed_outcome = _serialization_failure(
            str(exc),
            component="evaluator_result",
            attempted_status=outcome.status,
            attempted_stage=outcome.failure_stage,
        )
        return _record_from_outcome(
            run_id=run_id,
            architecture=architecture,
            model_variant=model_variant,
            sampling_method=sampling_method,
            payload_text=payload_text,
            sampling_text=sampling_text,
            outcome=failed_outcome,
        )

    body: dict[str, Any] = {
        "schema_version": RAW_SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_id": make_candidate_id(architecture, model_variant, payload),
        "architecture": architecture,
        "model_variant": model_variant,
        "sampling_method": sampling_method,
        "input_hash": canonical_hash(payload),
        "calculation_status": outcome.status.value,
        "constraint_status": ConstraintStatus.NOT_EVALUATED.value,
        "failure_stage": outcome.failure_stage,
        "failure_reason": outcome.failure_reason,
        "last_valid_state": json.loads(last_valid_text),
        "intermediates": json.loads(intermediates_text),
        "outputs": json.loads(outputs_text),
        "diagnostics": json.loads(diagnostics_text),
        "payload": payload,
        "sampling": json.loads(sampling_text),
    }
    body["record_hash"] = canonical_hash(body)
    return RawCandidate.from_dict(body)


def evaluate_candidate(
    evaluator: Callable[[Any], Any],
    input_payload: Mapping[str, Any],
    *,
    architecture: str,
    model_variant: str,
    run_id: str,
    sampling_method: str = "manual",
    sampling_metadata: Mapping[str, Any] | None = None,
) -> RawCandidate:
    """Evaluate one input payload and return an immutable RAW record.

    The evaluator receives a fresh deep copy of the validated payload and may
    return either an :class:`EvaluationOutcome`, any JSON object (treated as
    successful ``outputs``), or raise :class:`EvaluationFailure`.

    Explicit failure exceptions are stored as their declared status.  An
    unexpected ``Exception`` becomes ``MODEL_ERROR`` with an ``evaluator_exception``
    stage.  A non-serializable result becomes ``MODEL_ERROR`` with a
    ``result_serialization`` stage, which is distinct from numerical/physical
    failure.  ``KeyboardInterrupt`` and ``SystemExit`` always propagate.
    """

    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    architecture = _check_text("architecture", architecture)
    model_variant = _check_text("model_variant", model_variant)
    run_id = _check_text("run_id", run_id)
    sampling_method = _check_text("sampling_method", sampling_method)
    try:
        payload_text = canonical_json(input_payload)
    except PayloadValidationError:
        raise
    sampling_text = canonical_json({} if sampling_metadata is None else sampling_metadata)

    try:
        evaluator_input = json.loads(payload_text)
        result = evaluator(evaluator_input)
    except (KeyboardInterrupt, SystemExit):
        raise
    except EvaluationFailure as exc:
        try:
            outcome = exc.to_outcome()
        except Exception as conversion_error:
            outcome = EvaluationOutcome.failure(
                CalculationStatus.MODEL_ERROR,
                "evaluator_failure_conversion",
                _safe_exception_text(conversion_error),
            )
    except Exception as exc:
        outcome = EvaluationOutcome.failure(
            CalculationStatus.MODEL_ERROR,
            "evaluator_exception",
            _safe_exception_text(exc),
            diagnostics={"exception_type": type(exc).__name__},
        )
    else:
        outcome = _coerce_outcome(result)

    return _record_from_outcome(
        run_id=run_id,
        architecture=architecture,
        model_variant=model_variant,
        sampling_method=sampling_method,
        payload_text=payload_text,
        sampling_text=sampling_text,
        outcome=outcome,
    )


__all__ = [
    "RAW_SCHEMA_VERSION",
    "JOURNAL_FORMAT_JSONL",
    "CalculationStatus",
    "ConstraintStatus",
    "PayloadValidationError",
    "RecordIntegrityError",
    "EvaluationFailure",
    "NumericalFailure",
    "NonPhysicalStateError",
    "ModelError",
    "EvaluationSerializationError",
    "EvaluationOutcome",
    "RawCandidate",
    "canonical_json",
    "canonical_hash",
    "copy_json_value",
    "make_candidate_id",
    "make_run_id",
    "run_config_hash",
    "normalize_run_config",
    "evaluator_version_hash",
    "evaluate_candidate",
]

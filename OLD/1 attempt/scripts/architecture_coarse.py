#!/usr/bin/env python3
"""Bounded coarse ``pi_k`` experiment on the reviewed legacy RAW path.

SCOPE
=====
This is an **illustrative methodological experiment**, not a production sweep,
not a compressor selection, and not a physical validation.  It evaluates a
fixed 55-point plan through the existing reviewed opt-in 0D architecture cycle
and its versioned RAW adapter:

* 11 points for ``SINGLE_SHAFT_SINGLE_CC`` with integer ``pi`` from 3 to 13;
* 11 points for ``FREE_POWER_TURBINE`` with the same integer ``pi`` values;
* 33 points for ``SINGLE_SHAFT_DUAL_CC``: three logarithmic pressure-ratio
  splits (``w = 0.4, 0.5, 0.6``) at each integer ``pi``.  The unit ratios are
  ``pi**w`` and ``pi**(1-w)``; the second ratio is formed as ``pi / first`` so
  the floating product remains as close as possible to the requested ``pi``.

All fixture values come from :mod:`architecture_smoke`: 288 K / 101300 Pa
inlet, 1250 K legacy combustor exit with 0.96 pressure recovery, 110 kW
pre-reducer target, 0.98 gearbox, 0.85 compressor polytropic efficiency,
0.90 turbine polytropic efficiency, 0.99 mechanical efficiency, and 150 kPa
turbine outlet total pressure.  The gas model is the same explicit fixture
gas model.

There is **no compressor map, viability gate, stage count, reaction ratio, or
stage strength model**.  Every point is labelled illustrative and unvalidated;
``rho`` and strength are ``NOT_CHECKED`` and are never counted as feasible.
Some points may fail; every attempt is stored.  A post-hoc ``rho`` annotation
reads the existing record and does not recalculate physics.

DURABILITY
==========
The output is a normal :class:`~turbine_engine.dataset.DatasetWriter` dataset.
The ordered plan count and hash are part of the deterministic run config, so a
resume with a changed plan, model variant, evaluator version, assumptions, or
backend is rejected.  Timestamps, git state, and host/runtime details are
recorded only in ``summary.json`` -- never in the manifest run config -- so
they cannot make an otherwise identical resume fail.  The summary is written
atomically only after the journal is closed.  An interrupted run leaves the
manifest incomplete and writes no summary; resuming repairs and continues.

The dataset is deliberately small: 55 points, capped at 128.  No large grid,
optimizer, GPU, or production-search path is provided.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from architecture_smoke import GAS, GEARBOX, P_T_OUT, TARGET, request_for  # noqa: E402
from turbine_engine.architecture import Architecture  # noqa: E402
from turbine_engine.architecture_cycle import (  # noqa: E402
    ARCHITECTURE_CYCLE_LIMITATIONS,
)
from turbine_engine.architecture_raw import (  # noqa: E402
    ARCHITECTURE_RAW_MODEL_VARIANT,
    evaluate_architecture_payload,
    input_to_payload,
)
from turbine_engine.constraints import (  # noqa: E402
    RHO_LE_0_65_PROFILE_NAME,
    annotate_candidate,
    rho_065_profile,
)
from turbine_engine.dataset import (  # noqa: E402
    DatasetError,
    DatasetWriter,
    compare_candidates,
    load_manifest,
    read_records,
)
from turbine_engine.raw import (  # noqa: E402
    CalculationStatus,
    ConstraintStatus,
    PayloadValidationError,
    canonical_hash,
    copy_json_value,
    evaluate_candidate,
    make_candidate_id,
)


EXPERIMENT_VERSION = "architecture-coarse-v1"
CONFIG_VERSION = "architecture-coarse-config-v1"
SUMMARY_VERSION = "architecture-coarse-summary-v1"
EVALUATOR_VERSION = "architecture-cycle-raw-coarse-v1"
PLAN_NAME = "coarse-pi3-13-legacy"
SUMMARY_NAME = "summary.json"
BACKEND = "reference_python_float64"

PI_VALUES: tuple[int, ...] = tuple(range(3, 14))
SPLIT_WEIGHTS: tuple[float, ...] = (0.4, 0.5, 0.6)
PLAN_POINT_COUNT = len(PI_VALUES) * (2 + len(SPLIT_WEIGHTS))
MAX_PLAN_POINTS = 128

ARCHITECTURES_IN_PLAN: tuple[Architecture, ...] = (
    Architecture.SINGLE_SHAFT_SINGLE_CC,
    Architecture.FREE_POWER_TURBINE,
    Architecture.SINGLE_SHAFT_DUAL_CC,
)

PLAN_ORDER = (
    "ascending pi for SINGLE_SHAFT_SINGLE_CC (11 points), then ascending pi "
    "for FREE_POWER_TURBINE (11 points), then ascending pi with split weights "
    "0.4, 0.5, 0.6 for SINGLE_SHAFT_DUAL_CC (33 points)"
)

MODEL_VARIANTS: dict[str, Any] = {
    "architecture_raw": ARCHITECTURE_RAW_MODEL_VARIANT,
    "architecture_cycle_0d": "uncooled_no_bleed_0d_ideal_mixing",
    "gas_properties": GAS.model_variant,
    "combustor_closure": "legacy",
    "backend": BACKEND,
}

ASSUMPTIONS: dict[str, Any] = {
    "inlet_total_temperature_K": 288.0,
    "inlet_total_pressure_Pa": 101_300.0,
    "combustor": {
        "kind": "legacy",
        "outlet_total_temperature_K": 1250.0,
        "total_pressure_recovery": 0.96,
    },
    "target_gearbox_input_power_W": TARGET,
    "gearbox_efficiency": GEARBOX,
    "compressor_polytropic_efficiency": 0.85,
    "turbine_polytropic_efficiency": {
        "single_shaft": 0.90,
        "free_power_turbine": [0.90, 0.90],
    },
    "turbine_mechanical_efficiency": 0.99,
    "turbine_outlet_total_pressure_Pa": P_T_OUT,
    "pressure_ratio": {
        "pi_values": list(PI_VALUES),
        "dual_split_weights": list(SPLIT_WEIGHTS),
        "dual_split_definition": (
            "unit ratios are pi**w and pi**(1-w); the second is formed as "
            "pi / first so the floating product is preserved"
        ),
    },
    "gas_model": {
        "R": GAS.R,
        "l0": GAS.l0,
        "q_lhv_kj": GAS.q_lhv_kj,
        "eta_combustion": GAS.eta_combustion,
        "T_ref": GAS.T_ref,
        "alpha_basis": GAS.alpha_basis.value,
        "model_variant": GAS.model_variant,
    },
    "stage_reaction_ratio": {
        "status": ConstraintStatus.NOT_CHECKED.value,
        "reason": "MISSING_STAGE_MODEL",
    },
    "stage_strength": {
        "status": ConstraintStatus.NOT_CHECKED.value,
        "reason": "MISSING_STAGE_STRENGTH_MODEL",
    },
    "compressor_viability_gate": "NOT_APPLIED",
    "physically_validated": False,
    "illustrative_only": True,
}

_RHO_PROFILE = rho_065_profile()


class ArchitectureCoarseError(RuntimeError):
    """The bounded coarse experiment cannot proceed safely."""


@dataclass(frozen=True)
class PlanPoint:
    """One deterministic point in the fixed 55-point plan."""

    index: int
    architecture: Architecture
    pi: int
    split_weight: float | None
    pressure_ratios: tuple[float, ...]


def pressure_ratios_for(pi: int, split_weight: float | None) -> tuple[float, ...]:
    """Return the requested unit pressure ratios for one plan point."""

    if isinstance(pi, bool) or not isinstance(pi, int):
        raise ArchitectureCoarseError(f"pi must be an integer, got {pi!r}")
    if split_weight is None:
        return (float(pi),)
    weight = float(split_weight)
    if not math.isfinite(weight) or not 0.0 < weight < 1.0:
        raise ArchitectureCoarseError(
            f"split_weight must be in (0, 1), got {split_weight!r}"
        )
    first = float(pi) ** weight
    # Deriving the second unit from the product preserves the requested
    # product numerically while remaining pi**(1-w) in exact arithmetic.
    second = float(pi) / first
    return (first, second)


def plan_points() -> tuple[PlanPoint, ...]:
    """Return the ordered fixed plan (deterministic; no sweep)."""

    points: list[PlanPoint] = []
    index = 0
    for architecture in (
        Architecture.SINGLE_SHAFT_SINGLE_CC,
        Architecture.FREE_POWER_TURBINE,
    ):
        for pi in PI_VALUES:
            points.append(
                PlanPoint(
                    index=index,
                    architecture=architecture,
                    pi=pi,
                    split_weight=None,
                    pressure_ratios=pressure_ratios_for(pi, None),
                )
            )
            index += 1
    for pi in PI_VALUES:
        for weight in SPLIT_WEIGHTS:
            points.append(
                PlanPoint(
                    index=index,
                    architecture=Architecture.SINGLE_SHAFT_DUAL_CC,
                    pi=pi,
                    split_weight=weight,
                    pressure_ratios=pressure_ratios_for(pi, weight),
                )
            )
            index += 1
    if len(points) != PLAN_POINT_COUNT:  # pragma: no cover - constant check
        raise ArchitectureCoarseError(
            f"internal plan has {len(points)} points, expected {PLAN_POINT_COUNT}"
        )
    return tuple(points)


def payload_for(point: PlanPoint) -> dict[str, Any]:
    """Build the strict RAW payload for one point from the fixture helper."""

    request = request_for(point.architecture, ratios=point.pressure_ratios)
    return input_to_payload(request)


def plan_payloads() -> list[dict[str, Any]]:
    """Return the ordered strict JSON payloads of the fixed plan."""

    return [payload_for(point) for point in plan_points()]


def validate_plan_size(payloads: Sequence[Mapping[str, Any]]) -> None:
    """Enforce the deliberately small plan bounds."""

    try:
        count = len(payloads)
    except TypeError as exc:  # pragma: no cover - caller error
        raise ArchitectureCoarseError("plan payloads must be a sequence") from exc
    if not 1 <= count <= MAX_PLAN_POINTS:
        raise ArchitectureCoarseError(
            f"plan must contain 1..{MAX_PLAN_POINTS} points, got {count}"
        )


def make_run_config(
    payloads: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the deterministic manifest run config.

    No timestamp, git state, host name, or path participates.  Those belong to
    the summary only, so an identical resume never fails because of them.
    """

    if payloads is None:
        payloads = plan_payloads()
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "config_version": CONFIG_VERSION,
        "plan_name": PLAN_NAME,
        "plan_order": PLAN_ORDER,
        "payload_plan_count": len(payloads),
        "payload_plan_sha256": canonical_hash(payloads),
        "model_variant": ARCHITECTURE_RAW_MODEL_VARIANT,
        "evaluator_version": EVALUATOR_VERSION,
        "backend": BACKEND,
        "model_variants": copy_json_value(MODEL_VARIANTS),
        "assumptions": copy_json_value(ASSUMPTIONS),
        "limitations": list(ARCHITECTURE_CYCLE_LIMITATIONS),
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "compressor_viability_gate": "NOT_APPLIED",
    }


def _check_reference_backend() -> None:
    if struct.calcsize("d") != 8 or sys.float_info.mant_dig != 53:
        raise ArchitectureCoarseError(
            "this experiment requires a float64 reference Python backend"
        )


def _validate_plan_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> None:
    validate_plan_size(payloads)
    identifiers: list[str] = []
    for index, payload in enumerate(payloads):
        strict = copy_json_value(payload, path=f"$.payloads[{index}]")
        architecture = strict.get("architecture")
        if not isinstance(architecture, str):
            raise ArchitectureCoarseError(
                f"payload {index} has no string architecture"
            )
        identifiers.append(
            make_candidate_id(
                architecture,
                ARCHITECTURE_RAW_MODEL_VARIANT,
                strict,
            )
        )
    if len(set(identifiers)) != len(identifiers):
        raise ArchitectureCoarseError(
            "plan contains duplicate candidate inputs; deterministic ids must "
            "be unique"
        )


def _open_writer(
    destination: Path,
    *,
    resume: bool,
    config: Mapping[str, Any],
) -> DatasetWriter:
    summary_path = destination / SUMMARY_NAME
    if summary_path.exists() and not resume:
        raise ArchitectureCoarseError(
            f"{summary_path} already exists; refusing to overwrite an existing "
            "report (pass --resume to continue an existing dataset)"
        )
    if destination.exists():
        if not destination.is_dir():
            raise ArchitectureCoarseError(
                f"destination is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            if not resume:
                raise ArchitectureCoarseError(
                    f"destination is not empty: {destination}; refusing to "
                    "overwrite an existing dataset (pass --resume to continue "
                    "it)"
                )
            return DatasetWriter.open(
                destination,
                expected_run_config=config,
                expected_evaluator_version=EVALUATOR_VERSION,
            )
        if resume:
            raise ArchitectureCoarseError(
                "cannot resume an empty destination; create the dataset "
                "without --resume first"
            )
        return DatasetWriter.create(destination, run_config=config)

    if resume:
        raise ArchitectureCoarseError(
            f"cannot resume missing dataset: {destination}"
        )
    return DatasetWriter.create(destination, run_config=config)


def _ordered_plan_records(
    points: Sequence[PlanPoint],
    payloads: Sequence[Mapping[str, Any]],
    records: Sequence[Any],
) -> list[Any]:
    if len(records) != len(points):
        raise ArchitectureCoarseError(
            f"dataset has {len(records)} records for a {len(points)}-point "
            "plan; authoritative journal differs from the fixed plan"
        )
    expected = [
        make_candidate_id(
            point.architecture.value,
            ARCHITECTURE_RAW_MODEL_VARIANT,
            payload,
        )
        for point, payload in zip(points, payloads)
    ]
    by_id = {record.candidate_id: record for record in records}
    missing = [identifier for identifier in expected if identifier not in by_id]
    extra = sorted(set(by_id) - set(expected))
    if missing or extra:
        raise ArchitectureCoarseError(
            "dataset records do not match the fixed plan; "
            f"missing={missing!r} extra={extra!r}"
        )
    return [by_id[identifier] for identifier in expected]


def _rho_entry(record: Any) -> dict[str, Any]:
    annotation = annotate_candidate(record, _RHO_PROFILE)
    if annotation.constraint_status is not ConstraintStatus.NOT_CHECKED:
        raise ArchitectureCoarseError(
            "the 0D cycle has no stage model, so rho must be NOT_CHECKED, got "
            f"{annotation.constraint_status.value}"
        )
    return {
        "profile": RHO_LE_0_65_PROFILE_NAME,
        "path": "outputs.rho",
        "status": annotation.constraint_status.value,
        "reason": "MISSING_STAGE_MODEL",
        "feasible": annotation.feasible,
    }


def _success_values(record: Any) -> dict[str, Any]:
    outputs = record.outputs
    shaft = outputs["shaft"]
    return {
        "compressor_pressure_ratios": list(
            outputs["compressor"]["pressure_ratios"]
        ),
        "pressure_ratio_product": outputs["pressure_ratio_product"],
        "air_mass_flow": outputs["air_mass_flow"],
        "fuel_mass_flow": outputs["fuel_mass_flow"],
        "gas_mass_flow": outputs["gas_mass_flow"],
        "fuel_air_ratio": outputs["combustor"]["fuel_air_ratio"],
        "net_specific_power_per_air": outputs[
            "net_specific_power_per_air"
        ],
        "compressor_power": outputs["compressor"]["power"],
        "turbine_pressure_ratios": [
            turbine["pressure_ratio"] for turbine in outputs["turbines"]
        ],
        "turbine_powers": [
            turbine["power"] for turbine in outputs["turbines"]
        ],
        "turbine_mechanical_powers": list(
            shaft["turbine_mechanical_powers"]
        ),
        "actual_gearbox_input_power": outputs[
            "actual_gearbox_input_power"
        ],
        "gearbox_output_power": outputs["gearbox_output_power"],
        "main_net_power": shaft["main_net_power"],
        "gas_generator_net_power": shaft["gas_generator_net_power"],
        "free_power_turbine_mechanical_power": shaft[
            "free_power_turbine_mechanical_power"
        ],
        "target_power_status": outputs["target_power_status"],
        "compressor_pressure_residual": outputs[
            "compressor_pressure_residual"
        ],
        "turbine_pressure_residuals": list(
            outputs["turbine_pressure_residuals"]
        ),
        "mass_balance_residual": outputs["mass_balance_residual"],
        "target_power_residual": outputs["target_power_residual"],
        "shaft_residuals": dict(outputs["shaft_residuals"]),
    }


def _point_entry(point: PlanPoint, record: Any) -> dict[str, Any]:
    successful = record.calculation_status is CalculationStatus.SUCCESS
    return {
        "index": point.index,
        "candidate_id": record.candidate_id,
        "architecture": point.architecture.value,
        "pi": point.pi,
        "split_weight": point.split_weight,
        "requested_pressure_ratios": list(point.pressure_ratios),
        "requested_pressure_ratio_product": math.prod(
            point.pressure_ratios
        ),
        "illustrative_only": True,
        "physically_validated": False,
        "compressor_viability_gate": "NOT_APPLIED",
        "calculation_status": record.calculation_status.value,
        "failure_stage": record.failure_stage,
        "failure_reason": record.failure_reason,
        "rho": _rho_entry(record),
        "stage_strength": {
            "status": ConstraintStatus.NOT_CHECKED.value,
            "reason": "MISSING_STAGE_STRENGTH_MODEL",
        },
        "values": _success_values(record) if successful else None,
    }


def _stored_flow_values(record: Any) -> tuple[float, float] | None:
    if record.calculation_status is not CalculationStatus.SUCCESS:
        return None
    outputs = record.outputs
    return (
        float(outputs["gas_mass_flow"]),
        float(outputs["fuel_mass_flow"]),
    )


def _adjacent_pi_slopes(
    points: Sequence[PlanPoint],
    records: Sequence[Any],
) -> list[dict[str, Any]]:
    by_key = {
        (point.architecture.value, point.pi, point.split_weight): record
        for point, record in zip(points, records)
    }
    groups: list[tuple[Architecture, float | None]] = [
        (Architecture.SINGLE_SHAFT_SINGLE_CC, None),
        (Architecture.FREE_POWER_TURBINE, None),
        *[
            (Architecture.SINGLE_SHAFT_DUAL_CC, weight)
            for weight in SPLIT_WEIGHTS
        ],
    ]
    series: list[dict[str, Any]] = []
    for architecture, split_weight in groups:
        pairs: list[dict[str, Any]] = []
        for pi_from in PI_VALUES[:-1]:
            pi_to = pi_from + 1
            record_from = by_key[
                (architecture.value, pi_from, split_weight)
            ]
            record_to = by_key[(architecture.value, pi_to, split_weight)]
            values_from = _stored_flow_values(record_from)
            values_to = _stored_flow_values(record_to)
            pair: dict[str, Any] = {
                "pi_from": pi_from,
                "pi_to": pi_to,
            }
            if values_from is None or values_to is None:
                pair["status"] = "break"
                pair["reason"] = (
                    "adjacent point is not SUCCESS; no interpolation or "
                    "skipping across the failed point"
                )
                pair["pi_from_status"] = (
                    record_from.calculation_status.value
                )
                pair["pi_to_status"] = record_to.calculation_status.value
                pair["d_gas_mass_flow_d_pi"] = None
                pair["d_fuel_mass_flow_d_pi"] = None
            else:
                gas_from, fuel_from = values_from
                gas_to, fuel_to = values_to
                delta_pi = float(pi_to - pi_from)
                pair["status"] = "computed"
                pair["gas_mass_flow_from"] = gas_from
                pair["gas_mass_flow_to"] = gas_to
                pair["fuel_mass_flow_from"] = fuel_from
                pair["fuel_mass_flow_to"] = fuel_to
                pair["d_gas_mass_flow_d_pi"] = (
                    gas_to - gas_from
                ) / delta_pi
                pair["d_fuel_mass_flow_d_pi"] = (
                    fuel_to - fuel_from
                ) / delta_pi
            pairs.append(pair)
        series.append(
            {
                "architecture": architecture.value,
                "split_weight": split_weight,
                "pairs": pairs,
            }
        )
    return series


def _dual_split_ranges(
    points: Sequence[PlanPoint],
    records: Sequence[Any],
) -> list[dict[str, Any]]:
    by_key = {
        (point.pi, point.split_weight): record
        for point, record in zip(points, records)
        if point.architecture is Architecture.SINGLE_SHAFT_DUAL_CC
    }
    result: list[dict[str, Any]] = []
    for pi in PI_VALUES:
        splits: list[dict[str, Any]] = []
        successful: list[tuple[float, float, float]] = []
        for weight in SPLIT_WEIGHTS:
            record = by_key[(pi, weight)]
            values = _stored_flow_values(record)
            if values is None:
                splits.append(
                    {
                        "split_weight": weight,
                        "calculation_status": (
                            record.calculation_status.value
                        ),
                        "gas_mass_flow": None,
                        "fuel_mass_flow": None,
                    }
                )
                continue
            gas, fuel = values
            successful.append((weight, gas, fuel))
            splits.append(
                {
                    "split_weight": weight,
                    "calculation_status": (
                        record.calculation_status.value
                    ),
                    "gas_mass_flow": gas,
                    "fuel_mass_flow": fuel,
                }
            )

        def spread(index: int) -> dict[str, Any] | None:
            if not successful:
                return None
            values = [item[index] for item in successful]
            minimum = min(values)
            maximum = max(values)
            return {
                "count": len(values),
                "min": minimum,
                "max": maximum,
                "range": maximum - minimum,
            }

        result.append(
            {
                "pi": pi,
                "splits": splits,
                "gas_mass_flow_spread": spread(1),
                "fuel_mass_flow_spread": spread(2),
            }
        )
    return result


def _status_by_architecture(
    points: Sequence[PlanPoint],
    records: Sequence[Any],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for architecture in ARCHITECTURES_IN_PLAN:
        counts = {
            status.value: 0 for status in CalculationStatus
        }
        for point, record in zip(points, records):
            if point.architecture is architecture:
                counts[record.calculation_status.value] += 1
        counts["total"] = sum(
            1 for point in points if point.architecture is architecture
        )
        result[architecture.value] = counts
    return result


def _git_provenance() -> dict[str, Any]:
    def run(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )

    try:
        commit_result = run(["rev-parse", "HEAD"])
        status_result = run(["status", "--porcelain"])
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": None,
            "dirty": None,
            "source": "unavailable",
        }
    commit = (
        commit_result.stdout.strip()
        if commit_result.returncode == 0
        else None
    )
    if status_result.returncode != 0:
        dirty: bool | None = None
        source = "unavailable"
    else:
        dirty = bool(status_result.stdout.strip())
        source = "git"
    return {
        "commit": commit or None,
        "dirty": dirty,
        "source": source,
    }


def _runtime_provenance() -> dict[str, Any]:
    return {
        "backend": BACKEND,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "float_mant_dig": sys.float_info.mant_dig,
        "pointer_bytes": struct.calcsize("P"),
    }


def _reproduce_all(records: Sequence[Any]) -> dict[str, Any]:
    """Re-evaluate every stored record with explicit strict tolerances.

    ``read_records`` has already verified the journal and manifest, so this
    mirrors ``reproduce_candidate`` without re-verifying the whole dataset for
    each of the 55 records.
    """

    differences: list[dict[str, Any]] = []
    for expected in records:
        actual = evaluate_candidate(
            evaluate_architecture_payload,
            expected.payload,
            architecture=expected.architecture,
            model_variant=expected.model_variant,
            run_id=expected.run_id,
            sampling_method=expected.sampling_method,
            sampling_metadata=expected.sampling,
        )
        candidate_differences = compare_candidates(
            expected,
            actual,
            rtol=1e-12,
            atol=1e-9,
        )
        if candidate_differences:
            differences.append(
                {
                    "candidate_id": expected.candidate_id,
                    "differences": list(candidate_differences),
                }
            )
    return {
        "checked": len(records),
        "matches": not differences,
        "differences": differences,
        "rtol": 1e-12,
        "atol": 1e-9,
    }


def _build_summary(
    *,
    destination: Path,
    points: Sequence[PlanPoint],
    payloads: Sequence[Mapping[str, Any]],
    records: Sequence[Any],
    manifest: Mapping[str, Any],
    appended: int,
    elapsed: float,
    reproduce: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(
        record.calculation_status.value for record in records
    )
    return {
        "report_version": SUMMARY_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "provenance_scope": "summary_only_not_manifest_run_config",
        "provenance": {
            "git": _git_provenance(),
            "runtime": _runtime_provenance(),
        },
        "physically_validated": False,
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "illustrative_only": True,
        "experiment": {
            "experiment_version": EXPERIMENT_VERSION,
            "config_version": CONFIG_VERSION,
            "plan_name": PLAN_NAME,
            "plan_order": PLAN_ORDER,
            "plan_point_count": len(points),
            "payload_plan_count": len(payloads),
            "payload_plan_sha256": canonical_hash(payloads),
            "model_variants": copy_json_value(MODEL_VARIANTS),
            "assumptions": copy_json_value(ASSUMPTIONS),
            "backend": BACKEND,
            "limitations": list(ARCHITECTURE_CYCLE_LIMITATIONS),
            "compressor_viability_gate": "NOT_APPLIED",
        },
        "run": {
            "destination": str(destination.resolve()),
            "run_id": manifest["run_id"],
            "run_config_hash": manifest["run_config_hash"],
            "evaluator_version": manifest["evaluator_version"],
            "model_variant": manifest["run_config"]["model_variant"],
            "record_count": manifest["record_count"],
            "complete": manifest["complete"],
            "journal_size_bytes": manifest["journal_size_bytes"],
            "journal_sha256": manifest["journal_sha256"],
            "points_attempted": len(points),
            "records_appended_this_invocation": appended,
            "elapsed_seconds": round(elapsed, 6),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "status_by_architecture": _status_by_architecture(points, records),
        "posthoc_stage_rho": {
            "status": ConstraintStatus.NOT_CHECKED.value,
            "reason": "MISSING_STAGE_MODEL",
            "profile": RHO_LE_0_65_PROFILE_NAME,
            "note": "read-only annotation; no physics recalculation",
        },
        "posthoc_stage_strength": {
            "status": ConstraintStatus.NOT_CHECKED.value,
            "reason": "MISSING_STAGE_STRENGTH_MODEL",
        },
        "points": [
            _point_entry(point, record)
            for point, record in zip(points, records)
        ],
        "controlled_slices": {
            "method": (
                "descriptive adjacent-pi finite differences of stored gas and "
                "fuel mass flows; no fit, correlation, ranking, interpolation, "
                "or physics recalculation"
            ),
            "symbols": {
                "G": "gas_mass_flow",
                "fuel": "fuel_mass_flow",
                "pi": "requested integer compressor pressure-ratio product",
            },
            "adjacent_pi_slopes": _adjacent_pi_slopes(points, records),
            "dual_split_ranges": _dual_split_ranges(points, records),
        },
        "reproduce": copy_json_value(reproduce),
    }


def _write_summary_atomic(
    path: Path,
    summary: Mapping[str, Any],
) -> None:
    try:
        strict = copy_json_value(summary, path="$.summary")
    except PayloadValidationError as exc:
        raise ArchitectureCoarseError(
            f"summary is not strict finite JSON: {exc}"
        ) from exc
    text = (
        json.dumps(
            strict,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".summary-",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def run_coarse(
    destination: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate the fixed bounded plan, store all attempts, and summarise."""

    _check_reference_backend()
    points = plan_points()
    payloads = [payload_for(point) for point in points]
    _validate_plan_payloads(payloads)
    config = make_run_config(payloads)

    path = Path(destination)
    writer = _open_writer(path, resume=resume, config=config)
    appended = 0
    started = time.perf_counter()
    completed = False
    try:
        for index, (point, payload) in enumerate(zip(points, payloads)):
            before = writer.record_count
            writer.evaluate_and_append(
                evaluate_architecture_payload,
                payload,
                architecture=point.architecture.value,
                sampling_method="coarse_fixture",
                sampling_metadata={
                    "plan": PLAN_NAME,
                    "index": index,
                    "pi": point.pi,
                    "split_weight": point.split_weight,
                    "architecture": point.architecture.value,
                },
            )
            if writer.record_count > before:
                appended += 1
        completed = True
    finally:
        # An interrupted run leaves the manifest incomplete and writes no
        # summary.  A later --resume repairs and continues from the journal.
        writer.close(finalize=completed)
    elapsed = time.perf_counter() - started

    records = _ordered_plan_records(
        points,
        payloads,
        read_records(path),
    )
    manifest = load_manifest(path)
    reproduce = _reproduce_all(records)
    summary = _build_summary(
        destination=path,
        points=points,
        payloads=payloads,
        records=records,
        manifest=manifest,
        appended=appended,
        elapsed=elapsed,
        reproduce=reproduce,
    )
    _write_summary_atomic(path / SUMMARY_NAME, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded 55-point coarse pi_k experiment on the reviewed "
            "architecture RAW path. Illustrative, not physically validated."
        )
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="required output directory for the RAW dataset and summary.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing compatible dataset; never overwrite",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the JSON summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_coarse(args.destination, resume=args.resume)
    except (ArchitectureCoarseError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    if not summary["reproduce"]["matches"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

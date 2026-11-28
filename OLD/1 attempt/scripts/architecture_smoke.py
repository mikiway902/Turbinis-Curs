#!/usr/bin/env python3
"""Bounded architecture-cycle RAW smoke CLI.

This is a **manual/fixture smoke run**, not a production sweep and not a
physical validation.  It evaluates a small deterministic set (13 points) of the
explicit 0D architecture cycles for all three topologies, using illustrative
assumptions copied from the existing test fixtures:

* 110000 W at the gearbox input (pre-reducer);
* 0.98 gearbox efficiency, so the illustrative gearbox output is 107800 W;
* R = 287.4 J/(kg K), l0 = 14.61, q_lhv_kj = 43150 kJ/kg,
  eta_combustion = 0.98, T_ref = 288 K;
* compressor polytropic efficiency 0.85, turbine polytropic efficiency 0.90,
  turbine mechanical efficiency 0.99;
* total states only, no bleed/cooling, and no stage/rho model.

None of these numbers is a validated design value.  Every JSON summary states
``physically_validated: false`` and marks the missing stage reaction ratio as
``NOT_CHECKED``.

``--gas-property-basis textbook_true_cp`` promotes the same ordered legacy JSON
fixtures to explicit ``architecture-cycle-input-v2`` objects.  Incompatible
external-state and empirical-alpha fixtures are stored as ``MODEL_ERROR`` rather
than dropped or silently changed.  The dataset config records the textbook RAW
model variant, the v2 plan hash/count, and the basis, so a cross-basis resume is
rejected.  The default remains the original v1 legacy run exactly.

Durability
==========
The output is a normal :class:`~turbine_engine.dataset.DatasetWriter` dataset.
All attempts are stored, including nonphysical/configured failures.  Existing
non-empty destinations are refused unless ``--resume`` is passed; resume uses
the dataset's own run-config/evaluator-version checks and rejects incompatible
configuration.  The run config also records the ordered fixture plan's count
and hash, so a changed fixture set is rejected even when the preset and
evaluator version are unchanged.  No interrupt behaviour is simulated.  The CLI
then reproduces every stored record with explicit tolerances and reports timings
and journal size.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turbine_engine.architecture import Architecture  # noqa: E402
from turbine_engine.architecture_cycle import (  # noqa: E402
    ARCHITECTURE_CYCLE_LIMITATIONS,
    ARCHITECTURE_CYCLE_TEXTBOOK_LIMITATIONS,
    ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
    ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
    AlphaBasis,
    ArchitectureCycleInput,
    ExternalCombustorClosure,
    ExternalCombustorState,
    LegacyCombustorClosure,
    PolynomialGasModel,
)
from turbine_engine.architecture_raw import (  # noqa: E402
    ARCHITECTURE_RAW_MODEL_VARIANT,
    ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT,
    INPUT_SCHEMA_VERSION_V2,
    architecture_from_payload,
    evaluate_architecture_payload,
    input_to_payload,
    model_variant_for_payload,
)
from turbine_engine.constraints import (  # noqa: E402
    RHO_LE_0_65_PROFILE_NAME,
    annotate_candidate,
    rho_065_profile,
)
from turbine_engine.cycle import (  # noqa: E402
    calc_cp_kerosene,
    calc_k_kerosene,
)
from turbine_engine.dataset import (  # noqa: E402
    DatasetError,
    DatasetWriter,
    load_manifest,
    read_records,
    reproduce_candidate,
)
from turbine_engine.raw import (  # noqa: E402
    ConstraintStatus,
    canonical_hash,
)


GAS = PolynomialGasModel(
    R=287.4,
    l0=14.61,
    q_lhv_kj=43150.0,
    eta_combustion=0.98,
    T_ref=288.0,
    alpha_basis=AlphaBasis.FUEL_BASED,
)

TARGET = 110_000.0
GEARBOX = 0.98
P_T_OUT = 150_000.0
DEFAULT_EVALUATOR_VERSION = "architecture-cycle-raw-v1"
DEFAULT_PRESET = "manual"
MAX_SMOKE_POINTS = 24


class ArchitectureSmokeError(RuntimeError):
    """The requested bounded smoke run cannot proceed safely."""


def _ratios(architecture: Architecture) -> tuple[float, ...]:
    if architecture is Architecture.SINGLE_SHAFT_SINGLE_CC:
        return (11.0,)
    if architecture is Architecture.SINGLE_SHAFT_DUAL_CC:
        return (math.sqrt(11.0), math.sqrt(11.0))
    return (8.0,)


def _turbine_etas(architecture: Architecture) -> tuple[float, ...]:
    if architecture is Architecture.FREE_POWER_TURBINE:
        return (0.90, 0.90)
    return (0.90,)


def request_for(
    architecture: Architecture,
    *,
    ratios: tuple[float, ...] | None = None,
    combustor: LegacyCombustorClosure
    | ExternalCombustorClosure
    | None = None,
    gas_model: PolynomialGasModel = GAS,
    p_t_out: float = P_T_OUT,
    target: float = TARGET,
) -> ArchitectureCycleInput:
    """Build one illustrative fixture request (same values as the tests)."""

    if ratios is None:
        ratios = _ratios(architecture)
    if combustor is None:
        combustor = LegacyCombustorClosure(1250.0, 0.96)
    etas = _turbine_etas(architecture)
    return ArchitectureCycleInput(
        architecture=architecture,
        inlet_total_temperature=288.0,
        inlet_total_pressure=101_300.0,
        compressor_pressure_ratios=ratios,
        compressor_polytropic_efficiency=0.85,
        combustor=combustor,
        turbine_outlet_total_pressure=p_t_out,
        turbine_polytropic_efficiencies=etas,
        turbine_mechanical_efficiencies=(0.99,) * len(etas),
        target_gearbox_input_power=target,
        gearbox_efficiency=GEARBOX,
        gas_model=gas_model,
    )


def external_closure() -> ExternalCombustorClosure:
    """The external-combustor fixture used by the architecture-cycle tests."""

    temperature = 1200.0
    return ExternalCombustorClosure(
        ExternalCombustorState(
            T_total=temperature,
            p_total=900_000.0,
            fuel_air_ratio=0.017,
            cp=calc_cp_kerosene(temperature, 4.0),
            k=calc_k_kerosene(temperature, 4.0, GAS.R),
        )
    )


def nonphysical_high_ratio_request() -> ArchitectureCycleInput:
    """A deliberately nonphysical point: no positive net shaft power."""

    request = request_for(
        Architecture.SINGLE_SHAFT_SINGLE_CC,
        ratios=(30.0,),
        p_t_out=P_T_OUT,
    )
    p_gas = request.inlet_total_pressure * 30.0 * 0.96
    return replace(
        request,
        turbine_outlet_total_pressure=p_gas * 0.999,
    )


def _payload(request: ArchitectureCycleInput) -> dict[str, Any]:
    return input_to_payload(request)


def manual_payloads() -> list[dict[str, Any]]:
    """Return the fixed 13-point manual fixture set."""

    empirical = replace(GAS, alpha_basis=AlphaBasis.EMPIRICAL_POLYNOMIAL)
    requests = [
        request_for(Architecture.SINGLE_SHAFT_SINGLE_CC),
        request_for(
            Architecture.SINGLE_SHAFT_SINGLE_CC, p_t_out=140_000.0
        ),
        request_for(
            Architecture.SINGLE_SHAFT_SINGLE_CC,
            combustor=external_closure(),
        ),
        request_for(Architecture.SINGLE_SHAFT_DUAL_CC),
        request_for(
            Architecture.SINGLE_SHAFT_DUAL_CC,
            ratios=(math.sqrt(11.0), math.sqrt(11.0) * 1.05),
        ),
        request_for(
            Architecture.SINGLE_SHAFT_DUAL_CC,
            combustor=external_closure(),
        ),
        request_for(Architecture.FREE_POWER_TURBINE),
        request_for(
            Architecture.FREE_POWER_TURBINE, p_t_out=135_000.0
        ),
        request_for(
            Architecture.FREE_POWER_TURBINE,
            combustor=external_closure(),
        ),
        request_for(
            Architecture.SINGLE_SHAFT_SINGLE_CC, gas_model=empirical
        ),
        nonphysical_high_ratio_request(),
        request_for(
            Architecture.SINGLE_SHAFT_SINGLE_CC,
            combustor=LegacyCombustorClosure(600.0, 0.96),
        ),
        request_for(Architecture.FREE_POWER_TURBINE, p_t_out=160_000.0),
    ]
    return [_payload(request) for request in requests]


def grid_payloads() -> list[dict[str, Any]]:
    """Return a deterministic 13-point grid fixture set (no sweep)."""

    variants = (
        (P_T_OUT, "legacy"),
        (140_000.0, "legacy"),
        (160_000.0, "legacy"),
        (P_T_OUT, "external"),
    )
    payloads: list[dict[str, Any]] = []
    for architecture in Architecture:
        for p_t_out, closure in variants:
            combustor = (
                external_closure()
                if closure == "external"
                else LegacyCombustorClosure(1250.0, 0.96)
            )
            payloads.append(
                _payload(
                    request_for(
                        architecture,
                        combustor=combustor,
                        p_t_out=p_t_out,
                    )
                )
            )
    payloads.append(_payload(nonphysical_high_ratio_request()))
    return payloads


def payloads_for_preset(preset: str) -> list[dict[str, Any]]:
    if preset == "manual":
        return manual_payloads()
    if preset == "grid":
        return grid_payloads()
    raise ArchitectureSmokeError(
        f"unknown preset {preset!r}; choose 'manual' or 'grid'"
    )


def _as_source_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Promote a legacy fixture object to an explicit v2 textbook payload.

    The legacy fixture is generated first so an incompatible external/empirical
    fixture is never turned into an invalid typed request.  The v2 method field
    is explicit, and evaluation later records it as ``MODEL_ERROR`` when the
    source option cannot use that fixture.
    """

    source = copy.deepcopy(payload)
    source["schema_version"] = INPUT_SCHEMA_VERSION_V2
    source["gas_property_basis"] = (
        ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK
    )
    return source


def source_payloads_for_preset(preset: str) -> list[dict[str, Any]]:
    """Return the same ordered fixture set with the explicit textbook basis."""

    return [
        _as_source_payload(payload)
        for payload in payloads_for_preset(preset)
    ]


def make_run_config(
    preset: str,
    evaluator_version: str = DEFAULT_EVALUATOR_VERSION,
    payloads: list[dict[str, Any]] | None = None,
    gas_property_basis: str = ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
) -> dict[str, Any]:
    if gas_property_basis not in (
        ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
        ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
    ):
        raise ArchitectureSmokeError(
            f"unknown gas_property_basis {gas_property_basis!r}"
        )
    source = (
        gas_property_basis == ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK
    )
    if payloads is None:
        payloads = (
            source_payloads_for_preset(preset)
            if source
            else payloads_for_preset(preset)
        )
    config = {
        "model_variant": (
            ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT
            if source
            else ARCHITECTURE_RAW_MODEL_VARIANT
        ),
        "evaluator_version": evaluator_version,
        "preset": preset,
        "payload_plan_count": len(payloads),
        "payload_plan_sha256": canonical_hash(payloads),
    }
    if source:
        config["gas_property_basis"] = gas_property_basis
    return config


def _open_writer(
    destination: Path,
    *,
    preset: str,
    resume: bool,
    evaluator_version: str,
    payloads: list[dict[str, Any]],
    gas_property_basis: str = ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
) -> DatasetWriter:
    config = make_run_config(
        preset,
        evaluator_version,
        payloads,
        gas_property_basis=gas_property_basis,
    )
    if destination.exists():
        if not destination.is_dir():
            raise ArchitectureSmokeError(
                f"destination is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            if not resume:
                raise ArchitectureSmokeError(
                    f"destination is not empty: {destination}; refusing to "
                    "overwrite an existing dataset (pass --resume to continue "
                    "it)"
                )
            return DatasetWriter.open(
                destination,
                expected_run_config=config,
                expected_evaluator_version=evaluator_version,
            )
        if resume:
            raise ArchitectureSmokeError(
                "cannot resume an empty destination; create the dataset "
                "without --resume first"
            )
        return DatasetWriter.create(destination, run_config=config)

    if resume:
        raise ArchitectureSmokeError(
            f"cannot resume missing dataset: {destination}"
        )
    return DatasetWriter.create(destination, run_config=config)


def _annotation_summary(records) -> dict[str, Any]:
    profile = rho_065_profile()
    annotations = [annotate_candidate(record, profile) for record in records]
    not_checked = sum(
        1
        for annotation in annotations
        if annotation.constraint_status is ConstraintStatus.NOT_CHECKED
    )
    return {
        "profile": RHO_LE_0_65_PROFILE_NAME,
        "path": "outputs.rho",
        "status": ConstraintStatus.NOT_CHECKED.value,
        "reason": (
            "stage reaction ratio rho is not produced by the 0D "
            "architecture cycle"
        ),
        "records_not_checked": not_checked,
    }


def run_smoke(
    destination: str | Path,
    *,
    preset: str = DEFAULT_PRESET,
    resume: bool = False,
    evaluator_version: str = DEFAULT_EVALUATOR_VERSION,
    gas_property_basis: str = ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
) -> dict[str, Any]:
    """Run the bounded fixture set, save all attempts, and reproduce them."""

    if gas_property_basis not in (
        ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
        ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
    ):
        raise ArchitectureSmokeError(
            f"unknown gas_property_basis {gas_property_basis!r}; choose "
            f"{ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY!r} or "
            f"{ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK!r}"
        )
    source = (
        gas_property_basis == ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK
    )
    payloads = (
        source_payloads_for_preset(preset)
        if source
        else payloads_for_preset(preset)
    )
    model_variant = (
        ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT
        if source
        else ARCHITECTURE_RAW_MODEL_VARIANT
    )
    validation_scope = (
        list(ARCHITECTURE_CYCLE_TEXTBOOK_LIMITATIONS)
        if source
        else list(ARCHITECTURE_CYCLE_LIMITATIONS)
    )
    if not 1 <= len(payloads) <= MAX_SMOKE_POINTS:
        raise ArchitectureSmokeError(
            f"smoke preset {preset!r} must contain 1..{MAX_SMOKE_POINTS} "
            f"points, got {len(payloads)}"
        )

    path = Path(destination)
    writer = _open_writer(
        path,
        preset=preset,
        resume=resume,
        evaluator_version=evaluator_version,
        payloads=payloads,
        gas_property_basis=gas_property_basis,
    )
    appended = 0
    started = time.perf_counter()
    completed = False
    try:
        for index, payload in enumerate(payloads):
            architecture = architecture_from_payload(payload)
            if model_variant_for_payload(payload) != model_variant:
                raise ArchitectureSmokeError(
                    "fixture payload variant does not match the run config"
                )
            before = writer.record_count
            writer.evaluate_and_append(
                evaluate_architecture_payload,
                payload,
                architecture=architecture.value,
                sampling_method="fixture",
                sampling_metadata={"preset": preset, "index": index},
            )
            if writer.record_count > before:
                appended += 1
        completed = True
    finally:
        # An interrupted/erroring run must leave the manifest incomplete so a
        # later resume repairs and continues instead of pretending success.
        writer.close(finalize=completed)
    elapsed = time.perf_counter() - started

    records = read_records(path)
    manifest = load_manifest(path)

    reproduce_differences: list[dict[str, Any]] = []
    for record in records:
        result = reproduce_candidate(
            path,
            record.candidate_id,
            evaluate_architecture_payload,
            rtol=1e-12,
            atol=1e-9,
        )
        if not result.matches:
            reproduce_differences.append(
                {
                    "candidate_id": record.candidate_id,
                    "differences": list(result.differences),
                }
            )

    status_counts = Counter(
        record.calculation_status.value for record in records
    )
    rho_annotation = _annotation_summary(records)

    summary = {
        "physically_validated": False,
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "validation_scope": validation_scope,
        "stage_rho": {
            "status": ConstraintStatus.NOT_CHECKED.value,
            "reason": "MISSING_STAGE_MODEL",
        },
        "rho_annotation": rho_annotation,
        "destination": str(path.resolve()),
        "preset": preset,
        "evaluator_version": evaluator_version,
        "model_variant": model_variant,
        "payload_plan_count": len(payloads),
        "payload_plan_sha256": canonical_hash(payloads),
        "points_attempted": len(payloads),
        "records_appended": appended,
        "record_count": manifest["record_count"],
        "status_counts": dict(sorted(status_counts.items())),
        "journal_size_bytes": manifest["journal_size_bytes"],
        "journal_sha256": manifest["journal_sha256"],
        "run_id": manifest["run_id"],
        "complete": manifest["complete"],
        "elapsed_seconds": round(elapsed, 6),
        "reproduce": {
            "checked": len(records),
            "matches": not reproduce_differences,
            "differences": reproduce_differences,
            "rtol": 1e-12,
            "atol": 1e-9,
        },
        "records": [
            {
                "candidate_id": record.candidate_id,
                "architecture": record.architecture,
                "calculation_status": record.calculation_status.value,
                "failure_stage": record.failure_stage,
            }
            for record in records
        ],
    }
    if source:
        summary["gas_property_basis"] = gas_property_basis
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded, deterministic architecture-cycle RAW smoke run. "
            "This is not a production sweep and not a physical validation."
        )
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="required output directory for the RAW dataset",
    )
    parser.add_argument(
        "--preset",
        choices=("manual", "grid"),
        default=DEFAULT_PRESET,
        help="fixed fixture set to evaluate (default: manual)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing compatible dataset; never overwrite",
    )
    parser.add_argument(
        "--evaluator-version",
        default=DEFAULT_EVALUATOR_VERSION,
        help=(
            "explicit evaluator version; a resume with a different version is "
            "rejected"
        ),
    )
    parser.add_argument(
        "--gas-property-basis",
        choices=(
            ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
            ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
        ),
        default=ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
        help=(
            "explicit gas-property method; the default preserves the original "
            "saved fixtures exactly"
        ),
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
        summary = run_smoke(
            args.destination,
            preset=args.preset,
            resume=args.resume,
            evaluator_version=args.evaluator_version,
            gas_property_basis=args.gas_property_basis,
        )
    except (ArchitectureSmokeError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["reproduce"]["matches"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicit, versioned JSON adapter from architecture cycles to generic RAW.

SCOPE
=====
This module is an **adapter only**.  It does not change the physics in
:mod:`turbine_engine.architecture_cycle`, the generic records in
:mod:`turbine_engine.raw`, the dataset writer, or the legacy cycle.  It adds an
explicit input schema, typed JSON conversion, and one evaluator that returns a
:class:`~turbine_engine.raw.EvaluationOutcome`.

The architecture cycles remain opt-in 0D reference closures.  They are **not**
a physical validation of any topology, compressor map, blading, cooling
system, gearbox, propeller, or stage inventory.  Every successful outcome
carries the complete result ``validation_scope`` tuple (the legacy or textbook
limitation set selected by the explicit basis).  The adapter does
not add a stage model, so a reaction ratio ``rho`` is absent; a post-hoc
``rho`` constraint annotation must therefore be ``NOT_CHECKED``, never a pass.

INPUT SCHEMA
============
``architecture-cycle-input-v1`` is a strict JSON object.  Unknown, missing, or
wrongly typed keys are rejected.  Numbers must be genuine finite JSON numbers
(``int``/``float``); booleans, numeric strings, and integers that cannot be
represented exactly as a finite float are rejected, and no physics field has a
silent default.  Tuples are written as JSON arrays and enums as their exact
string values.  Nested dataclass validation failures (for example ``R <= 0`` or
an efficiency outside ``(0, 1]``) are schema errors, not generic evaluator
exceptions.  Both combustor closures are represented explicitly:

``legacy``
    ``{"kind": "legacy", "outlet_total_temperature": ...,
    "total_pressure_recovery": ...}``.  Fuel flow is obtained from the retained
    ``calc_g_m`` relation.

``external``
    ``{"kind": "external", "state": {"T_total": ..., "p_total": ...,
    "fuel_air_ratio": ..., "cp": ..., "k": ...}}``.  No fuel flow, composition,
    ``alpha``, or gas property is inferred.

Gas assumptions are always explicit in ``gas_model`` (``R``, ``l0``,
``q_lhv_kj``, ``eta_combustion``, ``T_ref``, ``alpha_basis``, and
``model_variant``).

``architecture-cycle-input-v2`` is the same strict object plus exactly one
required top-level key, ``gas_property_basis`` (``"legacy_mean_cp"`` or
``"textbook_true_cp"``).  Version 1 has no such key and is always parsed as the
legacy basis; adding the key to a version-1 object is rejected so a hidden
method cannot be selected.  ``input_to_payload`` writes version 1 for the
legacy basis and version 2 for the textbook basis; a source request is never
silently downgraded.

``gas_model.model_variant`` remains caller provenance and **not** a method
selector.  The explicit basis, the result's ``model_variant`` label, and the
``gas_property_basis`` field are the authoritative method record.

OUTPUT SCHEMAS
==============
The legacy basis keeps ``architecture-cycle-output-v1`` with exactly the
previous keys.  The textbook basis emits ``architecture-cycle-output-v2`` and
adds the actual ``gas_property_basis`` plus the complete process records
(process-mean temperature, ``cp``/``k``, composition, residuals, tolerances,
requested/achieved work, and gas constant).  Endpoint and mass states are
already present in the component states.  Compressor process records use
``alpha = +inf`` for dry air; JSON has no infinity token, so that exact value is
written as the documented string ``"dry_air"`` (no fuel, ``alpha`` unbounded).
Turbine process records have a finite fuel-based ``alpha``.

STATUS MAPPING
==============
* :class:`~turbine_engine.architecture_cycle.ArchitectureCyclePhysicalError`
  becomes ``NON_PHYSICAL_STATE``;
* :class:`~turbine_engine.cycle.CycleSolverError` becomes
  ``NUMERICAL_FAILURE`` with the available input state;
* :class:`~turbine_engine.diagnostics.SolverDiagnosticsError` preserves its
  declared ``status`` (``NUMERICAL_FAILURE`` or ``NON_PHYSICAL_STATE``) and
  retains whatever structured state is available;
* an invalid input schema becomes ``MODEL_ERROR``;
* other :class:`~turbine_engine.architecture_cycle.ArchitectureCycleError`
  configuration failures become ``MODEL_ERROR``;
* unexpected exceptions are deliberately **not** converted here.  The generic
  RAW evaluator turns them into ``MODEL_ERROR`` with an ``evaluator_exception``
  stage, so they remain visible instead of being mislabelled as physical.

ARCHITECTURE LABELS
===================
``evaluate_architecture_payload`` derives the architecture from the payload.
``evaluate_architecture_record`` additionally checks that an existing RAW
record's ``architecture`` label equals that derived value; a mismatch is a
``MODEL_ERROR``.  A textbook-basis payload must also carry
``ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT``; this prevents a source result from
passing under the legacy label.  The generic RAW evaluator cannot enforce this:
it has no payload-derived label, so callers writing datasets must use
``model_variant_for_payload`` (the smoke CLI does).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from turbine_engine.architecture import Architecture, StationState
from turbine_engine.architecture_cycle import (
    ARCHITECTURE_GAS_PROPERTY_BASES,
    ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
    ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
    AlphaBasis,
    ArchitectureCycleError,
    ArchitectureCycleInput,
    ArchitectureCyclePhysicalError,
    ArchitectureCycleResult,
    CombustorClosureResult,
    CompressorResult,
    ExternalCombustorClosure,
    ExternalCombustorState,
    LegacyCombustorClosure,
    PolynomialGasModel,
    TurbineResult,
    calculate_architecture_cycle,
)
from turbine_engine.cycle import CycleSolverError
from turbine_engine.diagnostics import (
    STATUSES as SOLVER_STATUSES,
    SolverDiagnosticsError,
)
from turbine_engine.raw import (
    CalculationStatus,
    EvaluationOutcome,
    RawCandidate,
    copy_json_value,
)


INPUT_SCHEMA_VERSION = "architecture-cycle-input-v1"
INPUT_SCHEMA_VERSION_V2 = "architecture-cycle-input-v2"
OUTPUT_SCHEMA_VERSION = "architecture-cycle-output-v1"
OUTPUT_SCHEMA_VERSION_V2 = "architecture-cycle-output-v2"

#: Dataset-level model variants.  The cycle result keeps its own
#: ``uncooled_no_bleed_0d_ideal_mixing`` / textbook label.
ARCHITECTURE_RAW_MODEL_VARIANT = "architecture_cycle_0d_raw_v1"
ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT = "architecture_cycle_0d_textbook_raw_v2"

#: Serialized value for a compressor process ``alpha == +inf``: dry air, i.e.
#: no fuel and an unbounded excess-air ratio.  JSON has no infinity token.
COMPRESSOR_DRY_AIR_ALPHA = "dry_air"


class ArchitectureRawSchemaError(ValueError):
    """The supplied payload is not a supported architecture-cycle input."""


_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "architecture",
        "inlet_total_temperature",
        "inlet_total_pressure",
        "compressor_pressure_ratios",
        "compressor_polytropic_efficiency",
        "combustor",
        "turbine_outlet_total_pressure",
        "turbine_polytropic_efficiencies",
        "turbine_mechanical_efficiencies",
        "target_gearbox_input_power",
        "gearbox_efficiency",
        "gas_model",
    }
)
_INPUT_KEYS_V2 = _INPUT_KEYS | {"gas_property_basis"}
_LEGACY_COMBUSTOR_KEYS = frozenset(
    {"kind", "outlet_total_temperature", "total_pressure_recovery"}
)
_EXTERNAL_COMBUSTOR_KEYS = frozenset({"kind", "state"})
_EXTERNAL_STATE_KEYS = frozenset(
    {"T_total", "p_total", "fuel_air_ratio", "cp", "k"}
)
_GAS_MODEL_KEYS = frozenset(
    {
        "R",
        "l0",
        "q_lhv_kj",
        "eta_combustion",
        "T_ref",
        "alpha_basis",
        "model_variant",
    }
)


def _reason(exc: BaseException) -> str:
    text = str(exc).strip()
    return text if text else type(exc).__name__


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArchitectureRawSchemaError(
            f"{path} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _require_exact_keys(
    value: Mapping[str, Any], required: frozenset[str], path: str
) -> None:
    keys = set(value)
    if keys != required:
        missing = sorted(required - keys)
        extra = sorted(keys - required, key=repr)
        raise ArchitectureRawSchemaError(
            f"{path} keys mismatch; missing={missing!r}, extra={extra!r}"
        )


def _number(value: Any, path: str) -> float:
    if type(value) is int:
        try:
            result = float(value)
        except OverflowError as exc:
            raise ArchitectureRawSchemaError(
                f"{path} integer is too large to represent as a finite float"
            ) from exc
        if not math.isfinite(result):
            raise ArchitectureRawSchemaError(f"{path} must be finite")
        if result != value:
            raise ArchitectureRawSchemaError(
                f"{path} integer is not exactly representable as a float"
            )
        return result
    if type(value) is not float:
        raise ArchitectureRawSchemaError(
            f"{path} must be a JSON number, got {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise ArchitectureRawSchemaError(f"{path} must be finite")
    return value


def _string(value: Any, path: str) -> str:
    if type(value) is not str or not value:
        raise ArchitectureRawSchemaError(f"{path} must be a non-empty string")
    return value


def _enum(enum_type: Any, value: Any, path: str) -> Any:
    text = _string(value, path)
    try:
        return enum_type(text)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise ArchitectureRawSchemaError(
            f"{path} must be one of {{{allowed}}}, got {text!r}"
        ) from exc


def _list(value: Any, path: str) -> list[Any]:
    if type(value) is not list:
        raise ArchitectureRawSchemaError(
            f"{path} must be a JSON array, got {type(value).__name__}"
        )
    return value


def _number_tuple(value: Any, path: str) -> tuple[float, ...]:
    values = _list(value, path)
    return tuple(
        _number(item, f"{path}[{index}]")
        for index, item in enumerate(values)
    )


def _state_to_json(state: StationState) -> dict[str, Any]:
    return {
        "T_total": state.T_total,
        "p_total": state.p_total,
        "mass_flow": state.mass_flow,
        "cp": state.cp,
        "k": state.k,
        "identifier": state.identifier,
    }


def _gas_model_to_json(model: PolynomialGasModel) -> dict[str, Any]:
    return {
        "R": model.R,
        "l0": model.l0,
        "q_lhv_kj": model.q_lhv_kj,
        "eta_combustion": model.eta_combustion,
        "T_ref": model.T_ref,
        "alpha_basis": model.alpha_basis.value,
        "model_variant": model.model_variant,
    }


def _compressor_process_to_json(record: Any) -> dict[str, Any]:
    return {
        "T_in": record.T_in,
        "T_out": record.T_out,
        "T_mean": record.T_mean,
        "cp": record.cp,
        "k": record.k,
        "pressure_ratio": record.pressure_ratio,
        "eta_polytropic": record.eta_polytropic,
        "gas_constant_R": record.gas_constant_R,
        "alpha": (
            COMPRESSOR_DRY_AIR_ALPHA
            if math.isinf(record.alpha)
            else record.alpha
        ),
        "method": record.method,
        "fixed_point_residual_K": record.fixed_point_residual_K,
        "tolerance_K": record.tolerance_K,
    }


def _turbine_process_to_json(record: Any) -> dict[str, Any]:
    return {
        "T_in": record.T_in,
        "T_out": record.T_out,
        "T_mean": record.T_mean,
        "cp": record.cp,
        "k": record.k,
        "alpha": record.alpha,
        "gas_constant_R": record.gas_constant_R,
        "method": record.method,
        "tolerance_K": record.tolerance_K,
        "temperature_residual_K": record.temperature_residual_K,
        "pressure_formula_temperature_residual_K": (
            record.pressure_formula_temperature_residual_K
        ),
        "requested_specific_work": record.requested_specific_work,
        "achieved_specific_work": record.achieved_specific_work,
        "work_relative_residual": record.work_relative_residual,
    }


def _compressor_to_json(
    result: CompressorResult, *, include_process: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "inlet": _state_to_json(result.inlet),
        "outlet": _state_to_json(result.outlet),
        "station_states": [
            _state_to_json(state) for state in result.station_states
        ],
        "pressure_ratios": list(result.pressure_ratios),
        "pressure_ratio_product": result.pressure_ratio_product,
        "per_unit_specific_work": list(result.per_unit_specific_work),
        "specific_work": result.specific_work,
        "power": result.power,
        "residual": result.residual,
    }
    if include_process:
        payload["gas_property_basis"] = result.gas_property_basis
        payload["process_properties"] = [
            _compressor_process_to_json(record)
            for record in result.process_properties
        ]
    return payload


def _combustor_to_json(result: CombustorClosureResult) -> dict[str, Any]:
    return {
        "inlet": _state_to_json(result.inlet),
        "outlet": _state_to_json(result.outlet),
        "fuel_air_ratio": result.fuel_air_ratio,
        "alpha": result.alpha,
        "alpha_basis": result.alpha_basis,
        "closure": result.closure,
        "fuel_mass_flow": result.fuel_mass_flow,
        "gas_mass_flow": result.gas_mass_flow,
        "heat_input": result.heat_input,
        "mass_balance_residual": result.mass_balance_residual,
    }


def _turbine_to_json(
    result: TurbineResult, *, include_process: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "inlet": _state_to_json(result.inlet),
        "outlet": _state_to_json(result.outlet),
        "pressure_ratio": result.pressure_ratio,
        "eta_turbine_poly": result.eta_turbine_poly,
        "specific_work": result.specific_work,
        "power": result.power,
        "efficiency": result.efficiency,
        "residual": result.residual,
    }
    if include_process:
        payload["gas_property_basis"] = result.gas_property_basis
        process = result.process_properties
        payload["process_properties"] = (
            None if process is None else _turbine_process_to_json(process)
        )
    return payload


def _shaft_to_json(result: ArchitectureCycleResult) -> dict[str, Any]:
    shaft = result.shaft
    return {
        "architecture": shaft.architecture.value,
        "compressor_gas_powers": list(shaft.compressor_gas_powers),
        "turbine_gas_powers": list(shaft.turbine_gas_powers),
        "mechanical_efficiencies": list(shaft.mechanical_efficiencies),
        "required_gearbox_input_power": shaft.required_gearbox_input_power,
        "gearbox_efficiency": shaft.gearbox_efficiency,
        # Derived properties.  dataclasses.asdict() would omit all of these.
        "turbine_mechanical_powers": list(shaft.turbine_mechanical_powers),
        "compressor_gas_power_total": shaft.compressor_gas_power_total,
        "turbine_mechanical_power_total": shaft.turbine_mechanical_power_total,
        "main_net_power": shaft.main_net_power,
        "gas_generator_net_power": shaft.gas_generator_net_power,
        "free_power_turbine_mechanical_power": (
            shaft.free_power_turbine_mechanical_power
        ),
        "actual_gearbox_input_power": shaft.actual_gearbox_input_power,
        "gearbox_output_power": shaft.gearbox_output_power,
        "main_residual": shaft.main_residual,
        "gas_generator_residual": shaft.gas_generator_residual,
        "free_power_turbine_residual": shaft.free_power_turbine_residual,
    }


def _turbine_role_indices(result: ArchitectureCycleResult) -> dict[str, Any]:
    if result.architecture is Architecture.FREE_POWER_TURBINE:
        return {"gas_generator": 0, "free_power": 1, "main": None}
    return {"gas_generator": None, "free_power": None, "main": 0}


def _is_textbook(result: ArchitectureCycleResult) -> bool:
    return (
        result.gas_property_basis
        == ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK
    )


def _intermediates(result: ArchitectureCycleResult) -> dict[str, Any]:
    source = _is_textbook(result)
    payload: dict[str, Any] = {
        "schema_version": (
            OUTPUT_SCHEMA_VERSION_V2 if source else OUTPUT_SCHEMA_VERSION
        ),
        "architecture": result.architecture.value,
        "model_variant": result.model_variant,
        "combustor_closure": result.combustor_closure,
        "validation_scope": list(result.validation_scope),
        "gas_model": _gas_model_to_json(result.gas_model),
        "inlet": _state_to_json(result.inlet),
        "compressor": _compressor_to_json(
            result.compressor, include_process=source
        ),
        "combustor": _combustor_to_json(result.combustor),
        "turbines": [
            _turbine_to_json(turbine, include_process=source)
            for turbine in result.turbines
        ],
        "turbine_roles": _turbine_role_indices(result),
        "shaft": _shaft_to_json(result),
    }
    if source:
        payload["gas_property_basis"] = result.gas_property_basis
    return payload


def _outputs(result: ArchitectureCycleResult) -> dict[str, Any]:
    source = _is_textbook(result)
    payload: dict[str, Any] = {
        "schema_version": (
            OUTPUT_SCHEMA_VERSION_V2 if source else OUTPUT_SCHEMA_VERSION
        ),
        "architecture": result.architecture.value,
        "model_variant": result.model_variant,
        "combustor_closure": result.combustor_closure,
        "inlet_total_temperature": result.inlet.T_total,
        "inlet_total_pressure": result.inlet.p_total,
        "air_mass_flow": result.air_mass_flow,
        "fuel_mass_flow": result.fuel_mass_flow,
        "gas_mass_flow": result.gas_mass_flow,
        "target_gearbox_input_power": result.target_gearbox_input_power,
        "actual_gearbox_input_power": result.actual_gearbox_input_power,
        "gearbox_output_power": result.gearbox_output_power,
        "net_specific_power_per_air": result.net_specific_power_per_air,
        "pressure_ratio_product": result.pressure_ratio_product,
        "compressor_pressure_residual": result.compressor_pressure_residual,
        "turbine_pressure_residuals": list(result.turbine_pressure_residuals),
        "mass_balance_residual": result.mass_balance_residual,
        "target_power_residual": result.target_power_residual,
        "target_power_status": result.target_power_status,
        "residuals": result.residual_report(),
        "shaft_residuals": {
            "main": result.shaft.main_residual,
            "gas_generator": result.shaft.gas_generator_residual,
            "free_power_turbine": result.shaft.free_power_turbine_residual,
        },
        # Complete components/states are repeated here so the output record is
        # self-contained; the same causal state is also in ``intermediates``.
        "gas_model": _gas_model_to_json(result.gas_model),
        "inlet": _state_to_json(result.inlet),
        "compressor": _compressor_to_json(
            result.compressor, include_process=source
        ),
        "combustor": _combustor_to_json(result.combustor),
        "turbines": [
            _turbine_to_json(turbine, include_process=source)
            for turbine in result.turbines
        ],
        "turbine_roles": _turbine_role_indices(result),
        "shaft": _shaft_to_json(result),
        "validation_scope": list(result.validation_scope),
        "limitations": list(result.validation_scope),
    }
    if source:
        payload["gas_property_basis"] = result.gas_property_basis
    return payload


def input_to_payload(request: ArchitectureCycleInput) -> dict[str, Any]:
    """Convert a validated :class:`ArchitectureCycleInput` to strict JSON.

    Enums become their string values, tuples become lists, and both combustor
    closures are written with an explicit ``kind``.  The result is a fresh
    deep copy; mutating it cannot affect ``request``.
    """

    if not isinstance(request, ArchitectureCycleInput):
        raise TypeError(
            "request must be an ArchitectureCycleInput, "
            f"got {type(request).__name__}"
        )

    combustor = request.combustor
    if isinstance(combustor, LegacyCombustorClosure):
        combustor_payload: dict[str, Any] = {
            "kind": "legacy",
            "outlet_total_temperature": combustor.outlet_total_temperature,
            "total_pressure_recovery": combustor.total_pressure_recovery,
        }
    elif isinstance(combustor, ExternalCombustorClosure):
        state = combustor.state
        combustor_payload = {
            "kind": "external",
            "state": {
                "T_total": state.T_total,
                "p_total": state.p_total,
                "fuel_air_ratio": state.fuel_air_ratio,
                "cp": state.cp,
                "k": state.k,
            },
        }
    else:  # pragma: no cover - validated by ArchitectureCycleInput
        raise ArchitectureRawSchemaError(
            "combustor must be LegacyCombustorClosure or "
            "ExternalCombustorClosure"
        )

    source = (
        request.gas_property_basis
        == ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK
    )
    payload = {
        "schema_version": (
            INPUT_SCHEMA_VERSION_V2 if source else INPUT_SCHEMA_VERSION
        ),
        "architecture": request.architecture.value,
        "inlet_total_temperature": request.inlet_total_temperature,
        "inlet_total_pressure": request.inlet_total_pressure,
        "compressor_pressure_ratios": list(
            request.compressor_pressure_ratios
        ),
        "compressor_polytropic_efficiency": (
            request.compressor_polytropic_efficiency
        ),
        "combustor": combustor_payload,
        "turbine_outlet_total_pressure": (
            request.turbine_outlet_total_pressure
        ),
        "turbine_polytropic_efficiencies": list(
            request.turbine_polytropic_efficiencies
        ),
        "turbine_mechanical_efficiencies": list(
            request.turbine_mechanical_efficiencies
        ),
        "target_gearbox_input_power": request.target_gearbox_input_power,
        "gearbox_efficiency": request.gearbox_efficiency,
        "gas_model": _gas_model_to_json(request.gas_model),
    }
    if source:
        payload["gas_property_basis"] = request.gas_property_basis
    return copy_json_value(payload, path="$.payload")


def _parse_combustor(
    value: Any,
) -> LegacyCombustorClosure | ExternalCombustorClosure:
    combustor = _require_mapping(value, "$.combustor")
    kind = _string(combustor.get("kind"), "$.combustor.kind")
    if kind == "legacy":
        _require_exact_keys(
            combustor, _LEGACY_COMBUSTOR_KEYS, "$.combustor"
        )
        return LegacyCombustorClosure(
            outlet_total_temperature=_number(
                combustor["outlet_total_temperature"],
                "$.combustor.outlet_total_temperature",
            ),
            total_pressure_recovery=_number(
                combustor["total_pressure_recovery"],
                "$.combustor.total_pressure_recovery",
            ),
        )
    if kind == "external":
        _require_exact_keys(
            combustor, _EXTERNAL_COMBUSTOR_KEYS, "$.combustor"
        )
        state = _require_mapping(combustor["state"], "$.combustor.state")
        _require_exact_keys(
            state, _EXTERNAL_STATE_KEYS, "$.combustor.state"
        )
        return ExternalCombustorClosure(
            ExternalCombustorState(
                T_total=_number(
                    state["T_total"], "$.combustor.state.T_total"
                ),
                p_total=_number(
                    state["p_total"], "$.combustor.state.p_total"
                ),
                fuel_air_ratio=_number(
                    state["fuel_air_ratio"],
                    "$.combustor.state.fuel_air_ratio",
                ),
                cp=_number(state["cp"], "$.combustor.state.cp"),
                k=_number(state["k"], "$.combustor.state.k"),
            )
        )
    raise ArchitectureRawSchemaError(
        "$.combustor.kind must be 'legacy' or 'external', "
        f"got {kind!r}"
    )


def _parse_gas_model(value: Any) -> PolynomialGasModel:
    gas = _require_mapping(value, "$.gas_model")
    _require_exact_keys(gas, _GAS_MODEL_KEYS, "$.gas_model")
    return PolynomialGasModel(
        R=_number(gas["R"], "$.gas_model.R"),
        l0=_number(gas["l0"], "$.gas_model.l0"),
        q_lhv_kj=_number(gas["q_lhv_kj"], "$.gas_model.q_lhv_kj"),
        eta_combustion=_number(
            gas["eta_combustion"], "$.gas_model.eta_combustion"
        ),
        T_ref=_number(gas["T_ref"], "$.gas_model.T_ref"),
        alpha_basis=_enum(
            AlphaBasis, gas["alpha_basis"], "$.gas_model.alpha_basis"
        ),
        model_variant=_string(
            gas["model_variant"], "$.gas_model.model_variant"
        ),
    )


def payload_to_input(payload: Mapping[str, Any]) -> ArchitectureCycleInput:
    """Parse a strict architecture-cycle input JSON object (v1 or v2).

    Unknown, missing, or wrongly typed keys raise
    :class:`ArchitectureRawSchemaError`; physics fields have no defaults.
    Version 1 has no ``gas_property_basis`` key and always selects the legacy
    basis.  Version 2 requires exactly one such key.
    """

    payload = _require_mapping(payload, "$")
    schema_version = _string(
        payload.get("schema_version"), "$.schema_version"
    )
    if schema_version == INPUT_SCHEMA_VERSION:
        _require_exact_keys(payload, _INPUT_KEYS, "$")
        gas_property_basis = ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY
    elif schema_version == INPUT_SCHEMA_VERSION_V2:
        _require_exact_keys(payload, _INPUT_KEYS_V2, "$")
        gas_property_basis = _string(
            payload["gas_property_basis"], "$.gas_property_basis"
        )
        if gas_property_basis not in ARCHITECTURE_GAS_PROPERTY_BASES:
            allowed = ", ".join(ARCHITECTURE_GAS_PROPERTY_BASES)
            raise ArchitectureRawSchemaError(
                "$.gas_property_basis must be one of "
                f"{{{allowed}}}, got {gas_property_basis!r}"
            )
    else:
        raise ArchitectureRawSchemaError(
            "$.schema_version must be "
            f"{INPUT_SCHEMA_VERSION!r} or {INPUT_SCHEMA_VERSION_V2!r}, "
            f"got {schema_version!r}"
        )

    architecture = _enum(
        Architecture, payload["architecture"], "$.architecture"
    )

    try:
        combustor = _parse_combustor(payload["combustor"])
        gas_model = _parse_gas_model(payload["gas_model"])
        return ArchitectureCycleInput(
            architecture=architecture,
            inlet_total_temperature=_number(
                payload["inlet_total_temperature"],
                "$.inlet_total_temperature",
            ),
            inlet_total_pressure=_number(
                payload["inlet_total_pressure"],
                "$.inlet_total_pressure",
            ),
            compressor_pressure_ratios=_number_tuple(
                payload["compressor_pressure_ratios"],
                "$.compressor_pressure_ratios",
            ),
            compressor_polytropic_efficiency=_number(
                payload["compressor_polytropic_efficiency"],
                "$.compressor_polytropic_efficiency",
            ),
            combustor=combustor,
            turbine_outlet_total_pressure=_number(
                payload["turbine_outlet_total_pressure"],
                "$.turbine_outlet_total_pressure",
            ),
            turbine_polytropic_efficiencies=_number_tuple(
                payload["turbine_polytropic_efficiencies"],
                "$.turbine_polytropic_efficiencies",
            ),
            turbine_mechanical_efficiencies=_number_tuple(
                payload["turbine_mechanical_efficiencies"],
                "$.turbine_mechanical_efficiencies",
            ),
            target_gearbox_input_power=_number(
                payload["target_gearbox_input_power"],
                "$.target_gearbox_input_power",
            ),
            gearbox_efficiency=_number(
                payload["gearbox_efficiency"],
                "$.gearbox_efficiency",
            ),
            gas_model=gas_model,
            gas_property_basis=gas_property_basis,
        )
    except ArchitectureCycleError as exc:
        raise ArchitectureRawSchemaError(
            f"invalid architecture cycle input: {exc}"
        ) from exc


def architecture_from_payload(payload: Mapping[str, Any]) -> Architecture:
    """Return the architecture declared by a payload.

    This is intentionally a narrow read: the full payload is still validated by
    :func:`payload_to_input` before any calculation.
    """

    payload = _require_mapping(payload, "$")
    if "architecture" not in payload:
        raise ArchitectureRawSchemaError(
            "$ is missing required key 'architecture'"
        )
    return _enum(Architecture, payload["architecture"], "$.architecture")


def model_variant_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the dataset-level variant selected by a payload.

    Version 1 selects :data:`ARCHITECTURE_RAW_MODEL_VARIANT`; version 2 selects
    the textbook variant only when its explicit ``gas_property_basis`` says so.
    This is a structural derivation, not a physics evaluation: incompatible
    source configurations still derive the textbook label and then fail as a
    ``MODEL_ERROR`` during evaluation.  Generic RAW writers must call this (or
    an equivalent derivation) themselves; the evaluator cannot enforce a record
    label it does not receive.
    """

    payload = _require_mapping(payload, "$")
    schema_version = _string(
        payload.get("schema_version"), "$.schema_version"
    )
    if schema_version == INPUT_SCHEMA_VERSION:
        _require_exact_keys(payload, _INPUT_KEYS, "$")
        return ARCHITECTURE_RAW_MODEL_VARIANT
    if schema_version == INPUT_SCHEMA_VERSION_V2:
        _require_exact_keys(payload, _INPUT_KEYS_V2, "$")
        basis = _string(
            payload["gas_property_basis"], "$.gas_property_basis"
        )
        if basis == ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK:
            return ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT
        if basis == ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY:
            return ARCHITECTURE_RAW_MODEL_VARIANT
        raise ArchitectureRawSchemaError(
            "$.gas_property_basis must be one of "
            f"{{{', '.join(ARCHITECTURE_GAS_PROPERTY_BASES)}}}, "
            f"got {basis!r}"
        )
    raise ArchitectureRawSchemaError(
        "$.schema_version must be "
        f"{INPUT_SCHEMA_VERSION!r} or {INPUT_SCHEMA_VERSION_V2!r}, "
        f"got {schema_version!r}"
    )


def _declared_input_schema(payload: Any) -> str:
    if isinstance(payload, Mapping):
        value = payload.get("schema_version")
        if type(value) is str:
            return value
    return INPUT_SCHEMA_VERSION


def _failure_input_state(
    payload: Any,
) -> dict[str, Any] | None:
    try:
        return {"input": copy_json_value(payload, path="$.input")}
    except (TypeError, ValueError):
        return None


def evaluate_architecture_payload(
    payload: Mapping[str, Any],
) -> EvaluationOutcome:
    """Evaluate one strict architecture-cycle payload as a RAW outcome.

    The architecture is derived from the payload itself.  The original payload
    is never mutated; the generic RAW evaluator stores it separately.
    """

    input_schema_version = _declared_input_schema(payload)
    try:
        request = payload_to_input(payload)
    except ArchitectureRawSchemaError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.MODEL_ERROR,
            "schema",
            _reason(exc),
            last_valid_state=_failure_input_state(payload),
            diagnostics={
                "error_type": type(exc).__name__,
                "input_schema_version": input_schema_version,
            },
        )

    input_payload = input_to_payload(request)
    try:
        result = calculate_architecture_cycle(request)
    except ArchitectureCyclePhysicalError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.NON_PHYSICAL_STATE,
            "architecture_cycle_physical",
            _reason(exc),
            last_valid_state={"input": input_payload},
            diagnostics={
                "error_type": type(exc).__name__,
                "input_schema_version": input_schema_version,
                "architecture": request.architecture.value,
            },
        )
    except SolverDiagnosticsError as exc:
        last_valid_state: dict[str, Any] = {"input": input_payload}
        if exc.last_valid_state:
            last_valid_state["solver_last_valid_state"] = (
                exc.last_valid_state
            )
        if exc.status in SOLVER_STATUSES:
            solver_status = CalculationStatus(exc.status)
        else:  # pragma: no cover - the exception validates its own status
            solver_status = CalculationStatus.MODEL_ERROR
        return EvaluationOutcome.failure(
            solver_status,
            exc.stage or "solver_diagnostics",
            exc.reason or _reason(exc),
            last_valid_state=last_valid_state,
            diagnostics={
                "error_type": type(exc).__name__,
                "input_schema_version": input_schema_version,
                "architecture": request.architecture.value,
                "solver": exc.as_record(),
            },
        )
    except CycleSolverError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.NUMERICAL_FAILURE,
            "cycle_solver",
            _reason(exc),
            last_valid_state={"input": input_payload},
            diagnostics={
                "error_type": type(exc).__name__,
                "input_schema_version": input_schema_version,
                "architecture": request.architecture.value,
            },
        )
    except ArchitectureCycleError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.MODEL_ERROR,
            "architecture_cycle_config",
            _reason(exc),
            last_valid_state={"input": input_payload},
            diagnostics={
                "error_type": type(exc).__name__,
                "input_schema_version": input_schema_version,
                "architecture": request.architecture.value,
            },
        )

    source = _is_textbook(result)
    return EvaluationOutcome.success(
        outputs=_outputs(result),
        intermediates=_intermediates(result),
        diagnostics={
            "schema_version": (
                OUTPUT_SCHEMA_VERSION_V2
                if source
                else OUTPUT_SCHEMA_VERSION
            ),
            "input_schema_version": input_schema_version,
            "architecture": result.architecture.value,
            "model_variant": result.model_variant,
            "combustor_closure": result.combustor_closure,
            "validation_scope": list(result.validation_scope),
            "limitations": list(result.validation_scope),
        },
    )


def evaluate_architecture_record(record: RawCandidate) -> EvaluationOutcome:
    """Re-evaluate a RAW record after checking its payload-derived labels.

    The architecture is derived from ``record.payload``.  If the record's
    ``architecture`` field names a different topology, the result is a
    ``MODEL_ERROR``.  A textbook payload must also use the textbook RAW model
    variant.  Callers using the generic RAW evaluator directly must derive that
    label themselves; this check needs the record that the payload evaluator
    does not receive.
    """

    if not isinstance(record, RawCandidate):
        raise TypeError(
            "record must be a RawCandidate, got "
            f"{type(record).__name__}"
        )
    try:
        architecture = architecture_from_payload(record.payload)
    except ArchitectureRawSchemaError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.MODEL_ERROR,
            "schema",
            _reason(exc),
            diagnostics={
                "error_type": type(exc).__name__,
                "record_architecture": record.architecture,
            },
        )
    if record.architecture != architecture.value:
        return EvaluationOutcome.failure(
            CalculationStatus.MODEL_ERROR,
            "architecture_label_mismatch",
            "record architecture "
            f"{record.architecture!r} does not match payload architecture "
            f"{architecture.value!r}",
            diagnostics={
                "record_architecture": record.architecture,
                "payload_architecture": architecture.value,
            },
        )
    try:
        expected_model_variant = model_variant_for_payload(record.payload)
    except ArchitectureRawSchemaError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.MODEL_ERROR,
            "schema",
            _reason(exc),
            diagnostics={
                "error_type": type(exc).__name__,
                "record_model_variant": record.model_variant,
            },
        )
    if (
        expected_model_variant == ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT
        and record.model_variant != expected_model_variant
    ):
        return EvaluationOutcome.failure(
            CalculationStatus.MODEL_ERROR,
            "model_variant_label_mismatch",
            "record model_variant "
            f"{record.model_variant!r} does not match textbook payload "
            f"variant {expected_model_variant!r}",
            diagnostics={
                "record_model_variant": record.model_variant,
                "payload_model_variant": expected_model_variant,
            },
        )
    return evaluate_architecture_payload(record.payload)


__all__ = [
    "ARCHITECTURE_RAW_MODEL_VARIANT",
    "ARCHITECTURE_RAW_TEXTBOOK_MODEL_VARIANT",
    "COMPRESSOR_DRY_AIR_ALPHA",
    "INPUT_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION_V2",
    "OUTPUT_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_VERSION_V2",
    "ArchitectureRawSchemaError",
    "architecture_from_payload",
    "evaluate_architecture_payload",
    "evaluate_architecture_record",
    "input_to_payload",
    "model_variant_for_payload",
    "payload_to_input",
]

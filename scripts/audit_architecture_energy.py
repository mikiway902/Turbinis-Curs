#!/usr/bin/env python3
"""Independent read-only energy/property audit of a saved architecture RAW set.

SCOPE
=====
This script reads a saved ``raw-candidate-v1`` dataset through
:func:`turbine_engine.dataset.read_records` (integrity verification only) and
re-derives selected algebraic diagnostics **independently** in this file.  It
does not import production cycle, gas, compressor, turbine or architecture
physics as an oracle, it does not evaluate a new cycle, and it never rewrites the
dataset.  All equations below are transcribed from the cited sources and are
labelled as diagnostics, not as corrected or validated physics.

The central unresolved question is the mean-``cp`` temperature basis: the
legacy polynomial returns a mean value, while the stage gas helper converts it
to a local value with a 273.15 K offset.  A 0 K offset is mathematically
possible and gives different local values.  This audit therefore reports both
conventions side by side; neither is treated as the accepted answer.

The second unresolved question is the free-power-turbine (FPT) gas-property
state: the saved FPT inlet inherits the gas-generator inlet ``cp``/``k``.  This
audit reports that inherited value, its apparent evaluation temperature, the
independent mean/local value at the actual inlet, and the discrepancy.  It does
not refresh the saved model.

USAGE
=====
    python scripts/audit_architecture_energy.py \
        --dataset reference/architecture_smoke/2026-11-11-grid \
        --output audit.json

``--dataset`` and ``--output`` are required.  An existing output file is refused
(no silent overwrite).  The process exits non-zero only for dataset/integrity,
data-shape or JSON-serialization failures; finite diagnostic residuals never
change the exit code.

JSON is strict: every non-finite diagnostic would be replaced by ``null`` and
listed under ``non_finite_replacements``; missing values are ``null`` with an
adjacent reason.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turbine_engine.dataset import (  # noqa: E402
    DatasetError,
    load_manifest,
    read_records,
)


AUDIT_SCHEMA_VERSION = "architecture-energy-audit-v1"
CP_BRANCH_TEMPERATURE_K = 700.0
CP_ZERO_CONVENTIONS: tuple[tuple[str, float], ...] = (
    ("zero_0K", 0.0),
    ("zero_273_15K", 273.15),
)
AUDIT_LIMITATIONS: tuple[str, ...] = (
    "NOT_PHYSICALLY_VALIDATED",
    "DIAGNOSTIC_ONLY: no physical acceptance threshold is applied; finite "
    "non-zero residuals are reported, not failed",
    "SOURCE_BASIS_UNRESOLVED: the mean-cp zero offset (0 K vs 273.15 K) is not "
    "resolved; both conventions are sensitivity diagnostics, not corrected "
    "physics",
    "FPT_INHERITED_GAS_PROPERTIES: saved free-power-turbine states inherit "
    "upstream cp/k; the audit reports the discrepancy and does not refresh them",
    "EXTERNAL_CLOSURE_UNKNOWN: external-combustor heat input and alpha are not "
    "supplied by the dataset and are not inferred",
    "ONE_SIDED_700K: branch values and local cp are evaluated one-sided; no "
    "finite difference crosses the 700 K branch",
    "The complete architecture-cycle limitation list remains stored in each "
    "SUCCESS record; this audit does not replace it",
)

UNITS: dict[str, str] = {
    "temperature": "K (absolute total temperature)",
    "pressure": "Pa",
    "mass_flow": "kg/s",
    "fuel_air_ratio": "kg fuel per kg air",
    "mean_cp": "J/(kg K)",
    "local_cp": "J/(kg K)",
    "cp_derivative": "J/(kg K^2)",
    "isentropic_exponent_k": "dimensionless",
    "specific_work": "J/kg of the component gas basis",
    "power": "W",
    "absolute_enthalpy_H_z": "J/kg of the selected gas basis",
    "sensible_enthalpy": "J/kg of the selected gas basis",
    "fuel_relation_residual": "J/kg air",
    "combustor_energy_residual": "J/kg air",
    "heat_input": "W",
    "gross_chemical_power": "W",
    "relative_difference": "dimensionless",
    "alpha": "dimensionless (infinity means dry air)",
}

SCAN_NOTE = (
    "Source references document the intended provenance only. This audit tool "
    "does not scan or verify the cited files, functions or line numbers; it "
    "reads only the saved dataset. The independent equations in this file are "
    "transcriptions, not imports of production physics."
)

SOURCE_REFERENCES: tuple[dict[str, str], ...] = (
    {
        "path": "turbine_engine/cycle.py",
        "function": "calc_cp_kerosene",
        "lines": "262-284",
        "usage": "piecewise mean-cp polynomial transcribed independently",
    },
    {
        "path": "turbine_engine/cycle.py",
        "function": "calc_k_kerosene",
        "lines": "287-296",
        "usage": "k = cp / (cp - R) transcribed independently",
    },
    {
        "path": "turbine_engine/cycle.py",
        "function": "calc_alpha",
        "lines": "299-318",
        "usage": "legacy excess-air polynomial transcribed independently",
    },
    {
        "path": "turbine_engine/cycle.py",
        "function": "calc_g_m",
        "lines": "405-427",
        "usage": "legacy N/D fuel relation transcribed independently",
    },
    {
        "path": "turbine_engine/gas.py",
        "function": "calc_gas_cp_k",
        "lines": "75-114",
        "usage": "local cp = cpbar + (T - 273.15) * dcpbar/dT; finite "
        "difference is the production method, not used here",
    },
    {
        "path": "turbine_engine/architecture_cycle.py",
        "function": "_resolve_combustor",
        "lines": "1082-1175",
        "usage": "legacy heat_per_air = f * Q * eta; external closure has no "
        "inferred heat or alpha",
    },
    {
        "path": "turbine_engine/architecture_cycle.py",
        "function": "calculate_architecture_cycle",
        "lines": "1221-1517",
        "usage": "FPT consumes the actual GG outlet state without refreshing "
        "cp/k; shaft and state-transfer context",
    },
    {
        "path": "turbine_engine/architecture_raw.py",
        "function": "_outputs",
        "lines": "353-393",
        "usage": "stored outputs/intermediates field layout",
    },
    {
        "path": "turbine_engine/compressor.py",
        "function": "CompressorResult",
        "lines": "94-262",
        "usage": "per-unit work equals exit cp times the stage temperature rise",
    },
    {
        "path": "turbine_engine/turbine.py",
        "function": "TurbineResult",
        "lines": "86-230",
        "usage": "specific work equals inlet cp times the inlet-outlet "
        "temperature difference",
    },
    {
        "path": "turbine_engine/dataset.py",
        "function": "read_records",
        "lines": "451-454",
        "usage": "integrity-verified dataset read (permitted non-physics input)",
    },
    {
        "path": "docs/ARCHITECTURE_RAW_SMOKE.md",
        "function": "SOURCE_DISCREPANCY - FPT gas-property evaluation "
        "temperature",
        "lines": "73-90",
        "usage": "description of the saved inherited-property discrepancy",
    },
)


class AuditError(RuntimeError):
    """Base class for controlled audit failures."""


class AuditDataError(AuditError):
    """The saved dataset does not have the expected shape for this audit."""


class AuditSerializationError(AuditError):
    """The report cannot be represented as strict finite JSON."""


# ---------------------------------------------------------------------------
# Strict value helpers
# ---------------------------------------------------------------------------
def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuditDataError(
            f"{path} must be a JSON object, got {type(value).__name__}"
        )
    return value


def _require_key(value: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in value:
        raise AuditDataError(f"{path} is missing required key {key!r}")
    return value[key]


def _as_finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditDataError(
            f"{name} must be a JSON number, got {type(value).__name__}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise AuditDataError(f"{name} must be finite, got {value!r}")
    return result


def _positive(name: str, value: Any) -> float:
    result = _as_finite(name, value)
    if result <= 0.0:
        raise AuditDataError(f"{name} must be > 0, got {value!r}")
    return result


def _alpha_json(alpha: float | None) -> float | str | None:
    """Represent infinity explicitly so strict JSON never carries ``Infinity``."""

    if alpha is None:
        return None
    if math.isinf(alpha):
        return "infinity"
    return float(alpha)


def _relative(value: float, reference: float, name: str) -> tuple[float | None, str | None]:
    if reference == 0.0:
        return None, f"{name} is undefined: reference is zero"
    result = value / reference
    if not math.isfinite(result):
        return None, f"{name} is not finite"
    return result, None


# ---------------------------------------------------------------------------
# Independent gas-property transcriptions
# ---------------------------------------------------------------------------
def _inverse_alpha(alpha: float) -> float:
    if alpha is None or math.isnan(alpha) or alpha <= 0.0:
        raise AuditDataError(f"alpha must be > 0 or infinity, got {alpha!r}")
    if math.isinf(alpha):
        return 0.0
    return 1.0 / alpha


def _cpbar_branch(T: float, alpha: float, branch: str) -> float:
    """Piecewise mean cp on an explicitly selected branch."""

    inverse = _inverse_alpha(alpha)
    if branch == "low":
        return (
            (1.2 + 2.25 * inverse) / 1e5 * (T - 70.0) + 0.236
        ) * 4187.0
    if branch == "high":
        return (
            (2.2 + 1.25 * inverse) / 1e5 * (T + 450.0) + 0.218
        ) * 4187.0
    raise AuditDataError(f"branch must be 'low' or 'high', got {branch!r}")


def mean_cp(T: float, alpha: float) -> float:
    """Independent piecewise mean cp in J/(kg K).

    The source branch is ``T < 700`` low, otherwise high.
    """

    T_value = _positive("T", T)
    return _cpbar_branch(T_value, alpha, "low" if T_value < CP_BRANCH_TEMPERATURE_K else "high")


def mean_cp_one_sided(T: float, alpha: float, side: str) -> float:
    """Mean cp on one side of the 700 K branch (no crossing)."""

    T_value = _positive("T", T)
    if side == "left":
        return _cpbar_branch(T_value, alpha, "low")
    if side == "right":
        return _cpbar_branch(T_value, alpha, "high")
    raise AuditDataError(f"side must be 'left' or 'right', got {side!r}")


def _dcpbar_branch(alpha: float, branch: str) -> float:
    inverse = _inverse_alpha(alpha)
    if branch == "low":
        return (1.2 + 2.25 * inverse) / 1e5 * 4187.0
    if branch == "high":
        return (2.2 + 1.25 * inverse) / 1e5 * 4187.0
    raise AuditDataError(f"branch must be 'low' or 'high', got {branch!r}")


def mean_cp_derivative(T: float, alpha: float) -> float:
    """Analytical dcpbar/dT on the branch selected by the temperature."""

    T_value = _positive("T", T)
    branch = "low" if T_value < CP_BRANCH_TEMPERATURE_K else "high"
    return _dcpbar_branch(alpha, branch)


def local_cp(T: float, alpha: float, zero: float) -> float:
    """Analytical local cp = cpbar + (T - zero) * dcpbar/dT."""

    T_value = _positive("T", T)
    return mean_cp(T_value, alpha) + (T_value - zero) * mean_cp_derivative(
        T_value, alpha
    )


def enthalpy(T: float, alpha: float, zero: float) -> float:
    """H_z(T, alpha) = cpbar(T, alpha) * (T - zero), J/kg."""

    T_value = _positive("T", T)
    return mean_cp(T_value, alpha) * (T_value - zero)


def sensible_enthalpy(
    T: float, alpha: float, zero: float, T_ref: float
) -> float:
    """h_sensible = H_z(T) - H_z(T_ref), J/kg."""

    return enthalpy(T, alpha, zero) - enthalpy(T_ref, alpha, zero)


def _k_from_cp(cp: float, R: float) -> tuple[float | None, str | None]:
    denominator = cp - R
    if not math.isfinite(cp) or not math.isfinite(denominator) or denominator <= 0.0:
        return None, "k is unavailable because cp - R is not finite and positive"
    value = cp / denominator
    if not math.isfinite(value):
        return None, "k is not finite"
    return value, None


def _empirical_alpha(T_k: float, T_g: float) -> tuple[float | None, str | None]:
    """Transcribed legacy ``calc_alpha`` polynomial; no production import."""

    x = _positive("T_k", T_k) / 1000.0
    y = _positive("T_g", T_g) / 1000.0
    denominator = (
        -0.004149
        + 0.367730 * y
        - 0.011584 * y**2
        + 0.027054 * y**3
        - 0.398434 * x
        + 0.031502 * x * y
        - 0.022030 * x * y**2
        + 0.062033 * x**2
        - 0.003988 * x**2 * y
        - 0.049657 * x**3
    )
    if denominator == 0.0:
        return None, "empirical alpha polynomial denominator is zero"
    value = 1.0 / denominator
    if not math.isfinite(value):
        return None, "empirical alpha polynomial returned a non-finite value"
    return value, None


def gas_state_diagnostics(T: float, alpha: float, R: float) -> dict[str, Any]:
    """Mean and analytical-local gas properties at one temperature."""

    cpbar = mean_cp(T, alpha)
    derivative = mean_cp_derivative(T, alpha)
    mean_k, mean_k_reason = _k_from_cp(cpbar, R)
    local: dict[str, Any] = {}
    for label, zero in CP_ZERO_CONVENTIONS:
        cp_local = cpbar + (T - zero) * derivative
        k_local, k_local_reason = _k_from_cp(cp_local, R)
        local[label] = {
            "zero_K": zero,
            "local_cp_J_per_kg_K": cp_local,
            "local_k": k_local,
            "local_k_unavailable_reason": k_local_reason,
        }
    return {
        "T_K": float(T),
        "alpha": _alpha_json(alpha),
        "mean_cp_J_per_kg_K": cpbar,
        "dcpbar_dT_J_per_kg_K2": derivative,
        "mean_k": mean_k,
        "mean_k_unavailable_reason": mean_k_reason,
        "local": local,
    }


def branch_diagnostics(alpha: float) -> dict[str, Any]:
    """One-sided 700 K branch report for one alpha basis."""

    left_cp = mean_cp_one_sided(CP_BRANCH_TEMPERATURE_K, alpha, "left")
    right_cp = mean_cp_one_sided(CP_BRANCH_TEMPERATURE_K, alpha, "right")
    left_derivative = _dcpbar_branch(alpha, "low")
    right_derivative = _dcpbar_branch(alpha, "high")
    cpbar_jump = right_cp - left_cp
    local: dict[str, Any] = {}
    enthalpy_block: dict[str, Any] = {}
    for label, zero in CP_ZERO_CONVENTIONS:
        left_local = left_cp + (CP_BRANCH_TEMPERATURE_K - zero) * left_derivative
        right_local = right_cp + (CP_BRANCH_TEMPERATURE_K - zero) * right_derivative
        left_h = left_cp * (CP_BRANCH_TEMPERATURE_K - zero)
        right_h = right_cp * (CP_BRANCH_TEMPERATURE_K - zero)
        local[label] = {
            "zero_K": zero,
            "left_local_cp_J_per_kg_K": left_local,
            "right_local_cp_J_per_kg_K": right_local,
            "local_cp_jump_J_per_kg_K": right_local - left_local,
            "left_dcpbar_dT_J_per_kg_K2": left_derivative,
            "right_dcpbar_dT_J_per_kg_K2": right_derivative,
        }
        enthalpy_block[label] = {
            "zero_K": zero,
            "left_H_J_per_kg": left_h,
            "right_H_J_per_kg": right_h,
            "enthalpy_jump_J_per_kg": right_h - left_h,
        }
    return {
        "alpha": _alpha_json(alpha),
        "branch_T_K": CP_BRANCH_TEMPERATURE_K,
        "mean_cp_left_J_per_kg_K": left_cp,
        "mean_cp_right_J_per_kg_K": right_cp,
        "mean_cp_jump_J_per_kg_K": cpbar_jump,
        "local_cp": local,
        "enthalpy_H_z": enthalpy_block,
        "method": "one-sided branch evaluation; no finite difference crosses 700 K",
    }


# ---------------------------------------------------------------------------
# Dataset field helpers
# ---------------------------------------------------------------------------
def _closure_kind(payload: Any, outputs: Any = None) -> str:
    if isinstance(payload, Mapping):
        combustor = payload.get("combustor")
        if isinstance(combustor, Mapping):
            kind = combustor.get("kind")
            if kind in ("legacy", "external"):
                return kind
    if isinstance(outputs, Mapping):
        combustor = outputs.get("combustor")
        if isinstance(combustor, Mapping):
            closure = combustor.get("closure")
            if closure == "external_state":
                return "external"
            if isinstance(closure, str) and "legacy" in closure:
                return "legacy"
    return "unknown"


def _gas_model(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    gas = _require_mapping(
        _require_key(payload, "gas_model", "$.payload"), "$.payload.gas_model"
    )
    for key in ("R", "l0", "q_lhv_kj", "eta_combustion", "T_ref"):
        _require_key(gas, key, "$.payload.gas_model")
    return gas


def _success_outputs(outputs: Any, candidate: str) -> Mapping[str, Any]:
    if outputs is None:
        raise AuditDataError(
            f"SUCCESS record {candidate} has no outputs; data shape is invalid"
        )
    return _require_mapping(outputs, "$.outputs")


def _combustor(outputs: Mapping[str, Any]) -> Mapping[str, Any]:
    return _require_mapping(
        _require_key(outputs, "combustor", "$.outputs"),
        "$.outputs.combustor",
    )


# ---------------------------------------------------------------------------
# Legacy fuel relation
# ---------------------------------------------------------------------------
def audit_legacy_fuel(
    payload: Mapping[str, Any], outputs: Mapping[str, Any]
) -> dict[str, Any]:
    """Independent legacy N/D fuel relation for both zero conventions."""

    closure = _closure_kind(payload, outputs)
    if closure != "legacy":
        return {
            "available": False,
            "reason": "external closure: fuel relation/alpha not supplied and "
            "not inferred",
            "z_conventions": {},
        }

    gas = _gas_model(payload)
    R = _positive("R", gas["R"])
    l0 = _positive("l0", gas["l0"])
    q_lhv = _positive("q_lhv_kj", gas["q_lhv_kj"]) * 1000.0
    eta = _positive("eta_combustion", gas["eta_combustion"])
    T_ref = _positive("T_ref", gas["T_ref"])

    combustor = _combustor(outputs)
    inlet = _require_mapping(
        _require_key(combustor, "inlet", "$.outputs.combustor"),
        "$.outputs.combustor.inlet",
    )
    outlet = _require_mapping(
        _require_key(combustor, "outlet", "$.outputs.combustor"),
        "$.outputs.combustor.outlet",
    )
    Tc = _positive("combustor.inlet.T_total", inlet["T_total"])
    Tg = _positive("combustor.outlet.T_total", outlet["T_total"])
    f = _positive("combustor.fuel_air_ratio", combustor["fuel_air_ratio"])

    conventions: dict[str, Any] = {}
    for label, zero in CP_ZERO_CONVENTIONS:
        D = (
            eta * q_lhv
            - (1.0 + l0)
            * (
                enthalpy(Tg, 1.0, zero)
                - enthalpy(T_ref, 1.0, zero)
            )
            + l0
            * (
                enthalpy(Tg, math.inf, zero)
                - enthalpy(T_ref, math.inf, zero)
            )
        )
        N = enthalpy(Tg, math.inf, zero) - enthalpy(Tc, math.inf, zero)
        implied_f, implied_f_reason = _relative(N, D, "implied fuel/air ratio")
        entry: dict[str, Any] = {
            "zero_K": zero,
            "D_J_per_kg_air": D,
            "N_J_per_kg_air": N,
            "f_stored": f,
            "implied_f": implied_f,
            "implied_f_unavailable_reason": implied_f_reason,
        }
        if zero == 0.0:
            entry["retained_residual_J_per_kg_air"] = f * D - N
            entry["retained_formula"] = "stored_f * D - N"
        else:
            alt_f = implied_f
            entry["alternative_f"] = alt_f
            if alt_f is None:
                entry["relative_change_vs_stored"] = None
                entry["relative_change_unavailable_reason"] = implied_f_reason
            else:
                relative, reason = _relative(alt_f - f, f, "relative f change")
                entry["relative_change_vs_stored"] = relative
                entry["relative_change_unavailable_reason"] = reason
        conventions[label] = entry

    return {
        "available": True,
        "closure": closure,
        "R_J_per_kg_K": R,
        "l0": l0,
        "q_lhv_J_per_kg": q_lhv,
        "eta_combustion": eta,
        "T_ref_K": T_ref,
        "Tc_K": Tc,
        "Tg_K": Tg,
        "z_conventions": conventions,
        "properties": {
            "Tg": {
                "products_alpha_1": gas_state_diagnostics(Tg, 1.0, R),
                "air_alpha_inf": gas_state_diagnostics(Tg, math.inf, R),
            },
            "Tc": {"air_alpha_inf": gas_state_diagnostics(Tc, math.inf, R)},
            "T_ref": {
                "products_alpha_1": gas_state_diagnostics(T_ref, 1.0, R),
                "air_alpha_inf": gas_state_diagnostics(T_ref, math.inf, R),
            },
        },
    }


# ---------------------------------------------------------------------------
# Alpha and heat diagnostics
# ---------------------------------------------------------------------------
def audit_alpha(
    payload: Mapping[str, Any], outputs: Mapping[str, Any]
) -> dict[str, Any]:
    closure = _closure_kind(payload, outputs)
    combustor = _combustor(outputs)
    stored_alpha = combustor.get("alpha")
    stored_basis = combustor.get("alpha_basis")
    if closure != "legacy" or stored_alpha is None:
        return {
            "available": False,
            "reason": "external closure: alpha is not supplied and is not "
            "inferred",
            "stored_alpha": None,
            "stored_alpha_basis": stored_basis,
        }

    gas = _gas_model(payload)
    R = _positive("R", gas["R"])
    l0 = _positive("l0", gas["l0"])
    inlet = _require_mapping(
        _require_key(combustor, "inlet", "$.outputs.combustor"),
        "$.outputs.combustor.inlet",
    )
    outlet = _require_mapping(
        _require_key(combustor, "outlet", "$.outputs.combustor"),
        "$.outputs.combustor.outlet",
    )
    Tc = _positive("combustor.inlet.T_total", inlet["T_total"])
    Tg = _positive("combustor.outlet.T_total", outlet["T_total"])
    f = _positive("combustor.fuel_air_ratio", combustor["fuel_air_ratio"])
    alpha_stored = _positive("combustor.alpha", stored_alpha)

    alpha_fuel = 1.0 / (f * l0)
    alpha_empirical, alpha_empirical_reason = _empirical_alpha(Tc, Tg)
    relative_alpha, relative_alpha_reason = (
        (None, alpha_empirical_reason)
        if alpha_empirical is None
        else _relative(alpha_empirical - alpha_fuel, alpha_fuel, "alpha difference")
    )

    gas_at_Tg: dict[str, Any] = {
        "stored": {
            "cp_J_per_kg_K": _positive("combustor.outlet.cp", outlet["cp"]),
            "k": _positive("combustor.outlet.k", outlet["k"]),
            "alpha_basis": stored_basis,
        },
        "fuel_based": gas_state_diagnostics(Tg, alpha_fuel, R),
    }
    if alpha_empirical is not None:
        gas_at_Tg["empirical_polynomial"] = gas_state_diagnostics(
            Tg, alpha_empirical, R
        )

    cp_fuel = mean_cp(Tg, alpha_fuel)
    cp_stored = _positive("combustor.outlet.cp", outlet["cp"])
    relative_cp_stored_vs_fuel, relative_cp_reason = _relative(
        cp_stored - cp_fuel, cp_fuel, "stored cp versus independent cpbar"
    )
    sensitivity: dict[str, Any] = {
        "stored_cp_vs_independent_fuel_based_cpbar_relative": relative_cp_stored_vs_fuel,
        "stored_cp_vs_independent_fuel_based_cpbar_relative_reason": relative_cp_reason,
    }
    if alpha_empirical is not None:
        cp_empirical = mean_cp(Tg, alpha_empirical)
        relative_empirical, relative_empirical_reason = _relative(
            cp_empirical - cp_fuel,
            cp_fuel,
            "empirical versus fuel-based cpbar",
        )
        sensitivity.update(
            {
                "empirical_cpbar_vs_fuel_based_cpbar_relative": relative_empirical,
                "empirical_cpbar_vs_fuel_based_cpbar_relative_reason": relative_empirical_reason,
                "empirical_cpbar_J_per_kg_K": cp_empirical,
            }
        )

    return {
        "available": True,
        "stored_alpha": alpha_stored,
        "stored_alpha_basis": stored_basis,
        "alpha_fuel_based_1_over_f_l0": alpha_fuel,
        "alpha_empirical_polynomial": alpha_empirical,
        "alpha_empirical_unavailable_reason": alpha_empirical_reason,
        "relative_difference_empirical_minus_fuel": relative_alpha,
        "relative_difference_empirical_minus_fuel_reason": relative_alpha_reason,
        "gas_at_Tg": gas_at_Tg,
        "cp_k_sensitivity": sensitivity,
    }


def audit_heat(
    payload: Mapping[str, Any], outputs: Mapping[str, Any]
) -> dict[str, Any]:
    closure = _closure_kind(payload, outputs)
    combustor = _combustor(outputs)
    stored_heat = combustor.get("heat_input")
    if closure != "legacy" or stored_heat is None:
        return {
            "available": False,
            "reason": "external closure: heat_input is not supplied and is not "
            "inferred; gross chemical power is also not reported",
            "stored_heat_input_W": None,
            "expected_fuel_heat_release_W": None,
            "gross_chemical_power_W": None,
            "residual_W": None,
            "relative_residual": None,
        }

    gas = _gas_model(payload)
    q_lhv = _positive("q_lhv_kj", gas["q_lhv_kj"]) * 1000.0
    eta = _positive("eta_combustion", gas["eta_combustion"])
    fuel_flow = _positive("combustor.fuel_mass_flow", combustor["fuel_mass_flow"])
    stored = _as_finite("combustor.heat_input", stored_heat)

    expected = fuel_flow * q_lhv * eta
    gross = fuel_flow * q_lhv
    residual = stored - expected
    relative, relative_reason = _relative(residual, expected, "heat residual")
    return {
        "available": True,
        "formula": "stored heat_input compared with m_fuel * Q * eta",
        "stored_heat_input_W": stored,
        "expected_fuel_heat_release_W": expected,
        "gross_chemical_power_W": gross,
        "residual_W": residual,
        "relative_residual": relative,
        "relative_residual_unavailable_reason": relative_reason,
    }


def audit_combustor_energy_residual(
    payload: Mapping[str, Any], outputs: Mapping[str, Any]
) -> dict[str, Any]:
    """Illustrative sensible-enthalpy residual, explicitly assumption-labelled."""

    closure = _closure_kind(payload, outputs)
    if closure != "legacy":
        return {
            "available": False,
            "reason": "external closure: alpha and heat input are not supplied "
            "and not inferred",
            "cases": [],
        }

    gas = _gas_model(payload)
    R = _positive("R", gas["R"])
    l0 = _positive("l0", gas["l0"])
    q_lhv = _positive("q_lhv_kj", gas["q_lhv_kj"]) * 1000.0
    eta = _positive("eta_combustion", gas["eta_combustion"])
    T_ref = _positive("T_ref", gas["T_ref"])
    combustor = _combustor(outputs)
    inlet = _require_mapping(
        _require_key(combustor, "inlet", "$.outputs.combustor"),
        "$.outputs.combustor.inlet",
    )
    outlet = _require_mapping(
        _require_key(combustor, "outlet", "$.outputs.combustor"),
        "$.outputs.combustor.outlet",
    )
    Tc = _positive("combustor.inlet.T_total", inlet["T_total"])
    Tg = _positive("combustor.outlet.T_total", outlet["T_total"])
    f = _positive("combustor.fuel_air_ratio", combustor["fuel_air_ratio"])
    alpha_fuel = 1.0 / (f * l0)
    alpha_empirical, alpha_empirical_reason = _empirical_alpha(Tc, Tg)

    alpha_bases: list[tuple[str, float | None, str | None]] = [
        ("fuel_based_1_over_f_l0", alpha_fuel, None),
        ("empirical_polynomial", alpha_empirical, alpha_empirical_reason),
    ]
    cases: list[dict[str, Any]] = []
    lhs = eta * f * q_lhv
    for zero_label, zero in CP_ZERO_CONVENTIONS:
        h_air = sensible_enthalpy(Tc, math.inf, zero, T_ref)
        for alpha_label, alpha, alpha_reason in alpha_bases:
            case: dict[str, Any] = {
                "zero_convention": zero_label,
                "zero_K": zero,
                "alpha_basis": alpha_label,
                "alpha": _alpha_json(alpha),
                "lhs_eta_f_Q_J_per_kg_air": lhs,
                "h_air_sensible_J_per_kg": h_air,
            }
            if alpha is None:
                case["available"] = False
                case["reason"] = alpha_reason or "alpha is unavailable"
                case["h_gas_sensible_J_per_kg"] = None
                case["residual_J_per_kg_air"] = None
            else:
                h_gas = sensible_enthalpy(Tg, alpha, zero, T_ref)
                rhs = (1.0 + f) * h_gas - h_air
                case["available"] = True
                case["h_gas_sensible_J_per_kg"] = h_gas
                case["rhs_sensible_energy_J_per_kg_air"] = rhs
                case["residual_J_per_kg_air"] = lhs - rhs
            cases.append(case)

    return {
        "available": True,
        "closure": closure,
        "assumption": (
            "Illustrative energy residual only: h_sensible(T,a,z) = "
            "H_z(T,a) - H_z(T_ref,a). Fuel sensible enthalpy is assumed zero "
            "at T_ref. This is not equivalent to stoichiometric-product "
            "bookkeeping and is not a physics pass/fail."
        ),
        "formula": (
            "residual = eta*f*Q - ((1+f)*h_gas(Tg,alpha,z) - "
            "h_air(Tc,alpha=inf,z))"
        ),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Component work diagnostics
# ---------------------------------------------------------------------------
def audit_compressor_components(
    payload: Mapping[str, Any], outputs: Mapping[str, Any]
) -> dict[str, Any]:
    gas = _gas_model(payload)
    R = _positive("R", gas["R"])
    compressor = _require_mapping(
        _require_key(outputs, "compressor", "$.outputs"),
        "$.outputs.compressor",
    )
    inlet = _require_mapping(
        _require_key(compressor, "inlet", "$.outputs.compressor"),
        "$.outputs.compressor.inlet",
    )
    states = compressor.get("station_states")
    works = compressor.get("per_unit_specific_work")
    if not isinstance(states, list) or not isinstance(works, list):
        raise AuditDataError(
            "$.outputs.compressor station_states and per_unit_specific_work "
            "must be arrays"
        )
    if len(states) != len(works) or not states:
        raise AuditDataError(
            "compressor station_states and per_unit_specific_work must have "
            "the same non-zero length"
        )

    previous_temperature = _positive(
        "compressor.inlet.T_total", inlet["T_total"]
    )
    units: list[dict[str, Any]] = []
    total_stored_cp_identity = 0.0
    total_independent_cp_identity = 0.0
    endpoint_totals = {label: 0.0 for label, _zero in CP_ZERO_CONVENTIONS}
    for index, (state, work) in enumerate(zip(states, works)):
        state_mapping = _require_mapping(
            state, f"$.outputs.compressor.station_states[{index}]"
        )
        T_out = _positive(
            f"compressor.station_states[{index}].T_total",
            state_mapping["T_total"],
        )
        stored_cp = _positive(
            f"compressor.station_states[{index}].cp", state_mapping["cp"]
        )
        stored_work = _as_finite(
            f"compressor.per_unit_specific_work[{index}]", work
        )
        delta_t = T_out - previous_temperature
        mean_temperature = 0.5 * (previous_temperature + T_out)
        independent_cp = mean_cp(mean_temperature, math.inf)
        stored_cp_identity = stored_cp * delta_t
        independent_cp_identity = independent_cp * delta_t
        total_stored_cp_identity += stored_cp_identity
        total_independent_cp_identity += independent_cp_identity
        endpoint: dict[str, Any] = {}
        for label, zero in CP_ZERO_CONVENTIONS:
            difference = enthalpy(T_out, math.inf, zero) - enthalpy(
                previous_temperature, math.inf, zero
            )
            endpoint_totals[label] += difference
            endpoint[label] = {
                "zero_K": zero,
                "endpoint_sensible_enthalpy_difference_J_per_kg": difference,
                "stored_work_minus_endpoint_difference_J_per_kg": (
                    stored_work - difference
                ),
            }
        units.append(
            {
                "index": index,
                "T_in_K": previous_temperature,
                "T_out_K": T_out,
                "delta_T_K": delta_t,
                "stored_mean_cp_J_per_kg_K": stored_cp,
                "independent_cpbar_at_mean_T_J_per_kg_K": independent_cp,
                "stored_work_J_per_kg": stored_work,
                "cp_identity_work_from_stored_mean_cp_J_per_kg": stored_cp_identity,
                "cp_identity_work_from_independent_cpbar_J_per_kg": independent_cp_identity,
                "stored_work_minus_stored_cp_identity_J_per_kg": (
                    stored_work - stored_cp_identity
                ),
                "stored_work_minus_independent_cpbar_identity_J_per_kg": (
                    stored_work - independent_cp_identity
                ),
                "endpoint_sensible_enthalpy_difference": endpoint,
            }
        )
        previous_temperature = T_out

    stored_specific = _as_finite(
        "compressor.specific_work", compressor["specific_work"]
    )
    total_endpoint = {
        label: {
            "zero_K": zero,
            "endpoint_sensible_enthalpy_difference_J_per_kg": endpoint_totals[
                label
            ],
            "stored_specific_work_minus_endpoint_difference_J_per_kg": (
                stored_specific - endpoint_totals[label]
            ),
        }
        for label, zero in CP_ZERO_CONVENTIONS
    }
    return {
        "available": True,
        "R_J_per_kg_K": R,
        "inlet_T_K": _positive("compressor.inlet.T_total", inlet["T_total"]),
        "outlet_T_K": previous_temperature,
        "specific_work_stored_J_per_kg": stored_specific,
        "cp_identity_work_from_stored_mean_cp_J_per_kg": total_stored_cp_identity,
        "cp_identity_work_from_independent_cpbar_J_per_kg": total_independent_cp_identity,
        "stored_specific_work_minus_stored_cp_identity_J_per_kg": (
            stored_specific - total_stored_cp_identity
        ),
        "stored_specific_work_minus_independent_cpbar_identity_J_per_kg": (
            stored_specific - total_independent_cp_identity
        ),
        "endpoint_sensible_enthalpy_difference": total_endpoint,
        "comparison_direction": "stored positive compression work compared "
        "with H_out - H_in (positive sensible rise)",
        "per_unit": units,
    }


def audit_turbine_components(
    payload: Mapping[str, Any],
    outputs: Mapping[str, Any],
    alpha_stored: float | None,
    alpha_basis: Any,
) -> list[dict[str, Any]]:
    """Return turbine work diagnostics.

    A turbine produces positive extraction work for expansion, so the stored
    work is compared with the positive sensible enthalpy drop
    ``H_z(T_in) - H_z(T_out)``.  The reverse difference would have the wrong
    sign and is not reported.
    """

    gas = _gas_model(payload)
    R = _positive("R", gas["R"])
    turbines = outputs.get("turbines")
    if not isinstance(turbines, list) or not turbines:
        raise AuditDataError("$.outputs.turbines must be a non-empty array")

    results: list[dict[str, Any]] = []
    for index, turbine in enumerate(turbines):
        turbine_mapping = _require_mapping(
            turbine, f"$.outputs.turbines[{index}]"
        )
        inlet = _require_mapping(
            _require_key(
                turbine_mapping, "inlet", f"$.outputs.turbines[{index}]"
            ),
            f"$.outputs.turbines[{index}].inlet",
        )
        outlet = _require_mapping(
            _require_key(
                turbine_mapping, "outlet", f"$.outputs.turbines[{index}]"
            ),
            f"$.outputs.turbines[{index}].outlet",
        )
        T_in = _positive(
            f"turbines[{index}].inlet.T_total", inlet["T_total"]
        )
        T_out = _positive(
            f"turbines[{index}].outlet.T_total", outlet["T_total"]
        )
        cp_in = _positive(f"turbines[{index}].inlet.cp", inlet["cp"])
        k_in = _positive(f"turbines[{index}].inlet.k", inlet["k"])
        stored_work = _as_finite(
            f"turbines[{index}].specific_work",
            turbine_mapping["specific_work"],
        )
        cp_identity = cp_in * (T_in - T_out)

        endpoint: dict[str, Any] = {}
        for label, zero in CP_ZERO_CONVENTIONS:
            if alpha_stored is None:
                endpoint[label] = {
                    "zero_K": zero,
                    "endpoint_sensible_enthalpy_drop_J_per_kg": None,
                    "stored_work_minus_endpoint_drop_J_per_kg": None,
                    "unavailable_reason": "external closure: alpha is not "
                    "supplied and not inferred",
                }
            else:
                # Positive extraction work corresponds to a positive drop.
                drop = enthalpy(T_in, alpha_stored, zero) - enthalpy(
                    T_out, alpha_stored, zero
                )
                endpoint[label] = {
                    "zero_K": zero,
                    "endpoint_sensible_enthalpy_drop_J_per_kg": drop,
                    "stored_work_minus_endpoint_drop_J_per_kg": (
                        stored_work - drop
                    ),
                }

        results.append(
            {
                "index": index,
                "T_in_K": T_in,
                "T_out_K": T_out,
                "inlet_cp_J_per_kg_K": cp_in,
                "inlet_k": k_in,
                "alpha_used": _alpha_json(alpha_stored),
                "alpha_basis": alpha_basis,
                "specific_work_stored_J_per_kg": stored_work,
                "constant_cp_identity_J_per_kg": cp_identity,
                "stored_work_minus_constant_cp_identity_J_per_kg": (
                    stored_work - cp_identity
                ),
                "endpoint_sensible_enthalpy_drop": endpoint,
                "comparison_direction": "stored positive extraction work "
                "compared with H_in - H_out (positive sensible drop)",
                "cp_k_inherited_from_inlet": outlet["cp"] == inlet["cp"]
                and outlet["k"] == inlet["k"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Free-power-turbine inherited-property diagnosis
# ---------------------------------------------------------------------------
def audit_fpt_inherited_properties(
    payload: Mapping[str, Any],
    outputs: Mapping[str, Any],
    architecture: str,
) -> dict[str, Any] | None:
    if architecture != "FREE_POWER_TURBINE":
        return {
            "available": False,
            "reason": "record is not a FREE_POWER_TURBINE architecture",
        }
    turbines = outputs.get("turbines")
    if not isinstance(turbines, list) or len(turbines) < 2:
        raise AuditDataError(
            "FREE_POWER_TURBINE outputs must contain two turbine results"
        )
    gg_inlet = _require_mapping(
        _require_key(turbines[0], "inlet", "$.outputs.turbines[0]"),
        "$.outputs.turbines[0].inlet",
    )
    fpt_inlet = _require_mapping(
        _require_key(turbines[1], "inlet", "$.outputs.turbines[1]"),
        "$.outputs.turbines[1].inlet",
    )
    combustor = _combustor(outputs)
    combustor_outlet = _require_mapping(
        _require_key(combustor, "outlet", "$.outputs.combustor"),
        "$.outputs.combustor.outlet",
    )
    combustor_alpha = combustor.get("alpha")
    alpha_basis = combustor.get("alpha_basis")
    gas = _gas_model(payload)
    R = _positive("R", gas["R"])

    inherited_cp = _positive("fpt.inlet.cp", fpt_inlet["cp"])
    inherited_k = _positive("fpt.inlet.k", fpt_inlet["k"])
    actual_temperature = _positive(
        "fpt.inlet.T_total", fpt_inlet["T_total"]
    )

    candidates: list[tuple[str, Mapping[str, Any]]] = [
        ("combustor_exit", combustor_outlet),
        ("gg_turbine_inlet", gg_inlet),
        (
            "gg_turbine_outlet",
            _require_mapping(
                _require_key(
                    turbines[0], "outlet", "$.outputs.turbines[0]"
                ),
                "$.outputs.turbines[0].outlet",
            ),
        ),
    ]
    origin: dict[str, Any] | None = None
    for role, state in candidates:
        state_cp = _positive(f"{role}.cp", state["cp"])
        state_k = _positive(f"{role}.k", state["k"])
        if state_cp == inherited_cp and state_k == inherited_k:
            origin = {
                "role": role,
                "T_K": _positive(f"{role}.T_total", state["T_total"]),
                "cp_J_per_kg_K": state_cp,
                "k": state_k,
                "matched_inherited_cp_and_k_exactly": True,
            }
            break

    base: dict[str, Any] = {
        "available": True,
        "fpt_inlet_T_K": actual_temperature,
        "inherited_cp_J_per_kg_K": inherited_cp,
        "inherited_k": inherited_k,
        "properties_originated_at": origin,
        "properties_originated_at_unavailable_reason": (
            None
            if origin is not None
            else "no earlier stored state carries exactly the inherited cp/k"
        ),
        "alpha_used": _alpha_json(combustor_alpha),
        "alpha_basis": alpha_basis,
        "independent_cpbar_at_actual_inlet_J_per_kg_K": None,
        "independent_k_at_actual_inlet": None,
        "local_cp_at_actual_inlet": {},
        "relative_cp_discrepancy_inherited_minus_independent": None,
        "relative_cp_discrepancy_unavailable_reason": None,
        "relative_k_discrepancy_inherited_minus_independent": None,
        "note": "Inherited properties are reported as stored; the audit does "
        "not refresh or replace them in the model.",
    }
    if combustor_alpha is None:
        base["relative_cp_discrepancy_unavailable_reason"] = (
            "external closure: alpha is not supplied and not inferred"
        )
        return base

    alpha = _positive("combustor.alpha", combustor_alpha)
    independent_cp = mean_cp(actual_temperature, alpha)
    independent_k, independent_k_reason = _k_from_cp(independent_cp, R)
    local: dict[str, Any] = {}
    for label, zero in CP_ZERO_CONVENTIONS:
        cp_local = local_cp(actual_temperature, alpha, zero)
        k_local, k_local_reason = _k_from_cp(cp_local, R)
        local[label] = {
            "zero_K": zero,
            "local_cp_J_per_kg_K": cp_local,
            "local_k": k_local,
            "local_k_unavailable_reason": k_local_reason,
        }
    relative_cp, relative_cp_reason = _relative(
        inherited_cp - independent_cp,
        independent_cp,
        "FPT inherited cp discrepancy",
    )
    relative_k: float | None = None
    relative_k_reason: str | None = None
    if independent_k is not None:
        relative_k, relative_k_reason = _relative(
            inherited_k - independent_k,
            independent_k,
            "FPT inherited k discrepancy",
        )
    else:
        relative_k_reason = independent_k_reason

    base.update(
        {
            "independent_cpbar_at_actual_inlet_J_per_kg_K": independent_cp,
            "independent_k_at_actual_inlet": independent_k,
            "independent_k_unavailable_reason": independent_k_reason,
            "local_cp_at_actual_inlet": local,
            "relative_cp_discrepancy_inherited_minus_independent": relative_cp,
            "relative_cp_discrepancy_unavailable_reason": relative_cp_reason,
            "relative_k_discrepancy_inherited_minus_independent": relative_k,
            "relative_k_discrepancy_unavailable_reason": relative_k_reason,
        }
    )
    return base


# ---------------------------------------------------------------------------
# Per-record audit
# ---------------------------------------------------------------------------
def audit_record(record: Any, index: int) -> dict[str, Any]:
    payload = record.payload
    if not isinstance(payload, Mapping):
        raise AuditDataError(
            f"record {record.candidate_id!r} payload is not a JSON object"
        )
    status = record.calculation_status.value
    closure = _closure_kind(payload)
    gas_model = payload.get("gas_model")
    entry: dict[str, Any] = {
        "index": index,
        "candidate_id": record.candidate_id,
        "architecture": record.architecture,
        "model_variant": record.model_variant,
        "payload_schema_version": payload.get("schema_version"),
        "sampling_method": record.sampling_method,
        "input_hash": record.input_hash,
        "record_hash": record.record_hash,
        "calculation_status": status,
        "constraint_status": record.constraint_status.value,
        "combustor_closure": closure,
        "gas_model": gas_model,
    }
    if status != "SUCCESS":
        entry["failure"] = {
            "stage": record.failure_stage,
            "reason": record.failure_reason,
        }
        entry["skipped"] = {
            "reason": "record is not SUCCESS; numerical/property diagnostics "
            "are skipped while the failure is retained",
            "failure_stage": record.failure_stage,
            "failure_reason": record.failure_reason,
        }
        entry["input_payload"] = payload
        entry["diagnostics"] = record.diagnostics
        return entry

    outputs = _success_outputs(record.outputs, record.candidate_id)
    combustor = _combustor(outputs)
    alpha_stored = combustor.get("alpha")
    alpha_basis = combustor.get("alpha_basis")
    entry.update(
        {
            "fuel_relation": audit_legacy_fuel(payload, outputs),
            "heat_identity": audit_heat(payload, outputs),
            "alpha_analysis": audit_alpha(payload, outputs),
            "combustor_energy_residual": audit_combustor_energy_residual(
                payload, outputs
            ),
            "components": {
                "compressor": audit_compressor_components(payload, outputs),
                "turbines": audit_turbine_components(
                    payload,
                    outputs,
                    _positive("combustor.alpha", alpha_stored)
                    if alpha_stored is not None
                    else None,
                    alpha_basis,
                ),
            },
            "fpt_inherited_properties": audit_fpt_inherited_properties(
                payload, outputs, record.architecture
            ),
        }
    )
    return entry


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def _finite_values(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            result.append(float(value))
    return result


def _residual_stats(values: Iterable[Any], unit: str) -> dict[str, Any]:
    finite = _finite_values(values)
    if not finite:
        return {"count": 0, "max_abs": None, "unit": unit}
    return {
        "count": len(finite),
        "max_abs": max(abs(value) for value in finite),
        "min": min(finite),
        "max": max(finite),
        "unit": unit,
    }


def _range_stats(values: Iterable[Any], unit: str) -> dict[str, Any]:
    finite = _finite_values(values)
    if not finite:
        return {"count": 0, "min": None, "max": None, "max_abs": None, "unit": unit}
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "max_abs": max(abs(value) for value in finite),
        "unit": unit,
    }


def _collect_summary(
    records: list[dict[str, Any]],
    status_counts: Counter[str],
    closure_counts: Counter[str],
    closure_status_counts: Counter[tuple[str, str]],
) -> dict[str, Any]:
    fuel_residuals: list[float] = []
    heat_residuals: list[float] = []
    turbine_cp_identities: list[float] = []
    compressor_cp_identities: list[float] = []
    compressor_independent_identities: list[float] = []

    fuel_relative_changes: list[float] = []
    alpha_relative_differences: list[float] = []
    heat_relative_residuals: list[float] = []
    fpt_cp_relative_discrepancies: list[float] = []
    combustor_residuals: list[float] = []

    for entry in records:
        if entry.get("skipped") is not None:
            continue
        fuel = entry.get("fuel_relation") or {}
        if fuel.get("available"):
            z0 = (fuel.get("z_conventions") or {}).get("zero_0K") or {}
            residual = z0.get("retained_residual_J_per_kg_air")
            if residual is not None:
                fuel_residuals.append(residual)
            z273 = (fuel.get("z_conventions") or {}).get("zero_273_15K") or {}
            relative = z273.get("relative_change_vs_stored")
            if relative is not None:
                fuel_relative_changes.append(relative)

        alpha = entry.get("alpha_analysis") or {}
        if alpha.get("available"):
            relative = alpha.get("relative_difference_empirical_minus_fuel")
            if relative is not None:
                alpha_relative_differences.append(relative)

        heat = entry.get("heat_identity") or {}
        if heat.get("available"):
            residual = heat.get("residual_W")
            if residual is not None:
                heat_residuals.append(residual)
            relative = heat.get("relative_residual")
            if relative is not None:
                heat_relative_residuals.append(relative)

        combustor = entry.get("combustor_energy_residual") or {}
        for case in combustor.get("cases", []):
            residual = case.get("residual_J_per_kg_air")
            if residual is not None:
                combustor_residuals.append(residual)

        components = entry.get("components") or {}
        compressor = components.get("compressor") or {}
        if compressor.get("available"):
            residual = compressor.get(
                "stored_specific_work_minus_stored_cp_identity_J_per_kg"
            )
            if residual is not None:
                compressor_cp_identities.append(residual)
            residual = compressor.get(
                "stored_specific_work_minus_independent_cpbar_identity_J_per_kg"
            )
            if residual is not None:
                compressor_independent_identities.append(residual)

        for turbine in components.get("turbines", []):
            residual = turbine.get(
                "stored_work_minus_constant_cp_identity_J_per_kg"
            )
            if residual is not None:
                turbine_cp_identities.append(residual)

        fpt = entry.get("fpt_inherited_properties") or {}
        if fpt.get("available"):
            relative = fpt.get(
                "relative_cp_discrepancy_inherited_minus_independent"
            )
            if relative is not None:
                fpt_cp_relative_discrepancies.append(relative)

    retained = {
        "fuel_relation_zero_0K": _residual_stats(
            fuel_residuals, "J/kg air"
        ),
        "heat_input_identity": _residual_stats(heat_residuals, "W"),
        "turbine_constant_cp_identity": _residual_stats(
            turbine_cp_identities, "J/kg gas"
        ),
        "compressor_constant_cp_identity_stored_cp": _residual_stats(
            compressor_cp_identities, "J/kg air"
        ),
    }

    return {
        "record_count": len(records),
        "success_count": status_counts.get("SUCCESS", 0),
        "failure_count": len(records) - status_counts.get("SUCCESS", 0),
        "counts": {
            "by_status": dict(sorted(status_counts.items())),
            "by_closure": dict(sorted(closure_counts.items())),
            "by_closure_and_status": {
                closure: dict(sorted(status_counts_for_closure.items()))
                for closure, status_counts_for_closure in sorted(
                    _group_closure_status(closure_status_counts).items()
                )
            },
        },
        "retained_algebraic_residuals": retained,
        "discrepancy_ranges": {
            "fuel_zero_273_15_relative_change": _range_stats(
                fuel_relative_changes, "dimensionless"
            ),
            "empirical_minus_fuel_alpha_relative": _range_stats(
                alpha_relative_differences, "dimensionless"
            ),
            "heat_input_relative_residual": _range_stats(
                heat_relative_residuals, "dimensionless"
            ),
            "fpt_inherited_cp_relative_discrepancy": _range_stats(
                fpt_cp_relative_discrepancies, "dimensionless"
            ),
            "combustor_illustrative_energy_residual": _range_stats(
                combustor_residuals, "J/kg air"
            ),
            "compressor_stored_minus_independent_cpbar_identity": _range_stats(
                compressor_independent_identities, "J/kg air"
            ),
        },
    }


def _group_closure_status(
    closure_status_counts: Counter[tuple[str, str]]
) -> dict[str, Counter[str]]:
    grouped: dict[str, Counter[str]] = {}
    for (closure, status), count in closure_status_counts.items():
        grouped.setdefault(closure, Counter())[status] += count
    return grouped


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _collect_branch_alphas(records: list[Any]) -> list[float]:
    alphas: set[float] = {1.0, math.inf}
    for record in records:
        if record.calculation_status.value != "SUCCESS":
            continue
        outputs = record.outputs
        if not isinstance(outputs, Mapping):
            continue
        combustor = outputs.get("combustor")
        if not isinstance(combustor, Mapping):
            continue
        alpha = combustor.get("alpha")
        if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
            value = float(alpha)
            if value > 0.0 and math.isfinite(value):
                alphas.add(value)
    return sorted(alphas, key=lambda value: (math.isinf(value), value))


def _sanitize(value: Any, path: str, notes: list[dict[str, str]]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            notes.append(
                {
                    "path": path,
                    "reason": "non-finite value replaced with null for strict "
                    "finite JSON",
                }
            )
            return None
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(child, f"{path}.{key}", notes)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitize(child, f"{path}[{index}]", notes)
            for index, child in enumerate(value)
        ]
    raise AuditSerializationError(
        f"unsupported JSON value at {path}: {type(value).__name__}"
    )


def build_report(dataset: str | Path) -> dict[str, Any]:
    """Read the saved dataset and build the independent audit report."""

    dataset_path = Path(dataset)
    records = read_records(dataset_path)
    manifest = load_manifest(dataset_path)

    status_counts: Counter[str] = Counter()
    closure_counts: Counter[str] = Counter()
    closure_status_counts: Counter[tuple[str, str]] = Counter()
    audited: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        entry = audit_record(record, index)
        audited.append(entry)
        status = record.calculation_status.value
        closure = entry["combustor_closure"]
        status_counts[status] += 1
        closure_counts[closure] += 1
        closure_status_counts[(closure, status)] += 1

    run_config = manifest.get("run_config") or {}
    dataset_block = {
        "directory": str(dataset_path.resolve()),
        "journal_sha256": manifest.get("journal_sha256"),
        "journal_size_bytes": manifest.get("journal_size_bytes"),
        "raw_schema_version": manifest.get("schema_version"),
        "dataset_version": manifest.get("dataset_version"),
        "evaluator_version": manifest.get("evaluator_version"),
        "model_variant": run_config.get("model_variant"),
        "record_count": manifest.get("record_count"),
        "complete": manifest.get("complete"),
        "run_id": manifest.get("run_id"),
        "input_schema_version": (
            audited[0].get("payload_schema_version") if audited else None
        ),
    }

    report: dict[str, Any] = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "physically_validated": False,
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "limitations": list(AUDIT_LIMITATIONS),
        "units": dict(UNITS),
        "scan_note": SCAN_NOTE,
        "source_references": [dict(item) for item in SOURCE_REFERENCES],
        "dataset": dataset_block,
        "gas_property_diagnostics": {
            "zero_conventions": [
                {"label": label, "zero_K": zero}
                for label, zero in CP_ZERO_CONVENTIONS
            ],
            "branch_temperature_K": CP_BRANCH_TEMPERATURE_K,
            "method": "analytical derivative at fixed alpha; branch values are "
            "one-sided; no finite difference crosses 700 K",
            "branch_700K": [
                branch_diagnostics(alpha)
                for alpha in _collect_branch_alphas(records)
            ],
        },
        "records": audited,
        "summary": _collect_summary(
            audited,
            status_counts,
            closure_counts,
            closure_status_counts,
        ),
    }

    notes: list[dict[str, str]] = []
    report = _sanitize(report, "$", notes)
    report["non_finite_replacements"] = notes
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent read-only numerical energy/property audit of a saved "
            "architecture RAW dataset. This is a diagnostic, not a physical "
            "validation, and it never rewrites the dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="required saved RAW dataset directory (integrity-verified read)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="required JSON output file; an existing file is refused",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the report to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        print(
            f"error: output already exists and will not be overwritten: {output}",
            file=sys.stderr,
        )
        return 2

    try:
        report = build_report(args.dataset)
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        if output.parent != Path("."):
            output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.write("\n")
    except FileExistsError:
        print(
            f"error: output already exists and will not be overwritten: {output}",
            file=sys.stderr,
        )
        return 2
    except (AuditError, DatasetError, OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

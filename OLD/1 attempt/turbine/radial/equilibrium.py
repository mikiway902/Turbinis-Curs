"""Accepted radial-equilibrium residuals.

Inputs are one-dimensional SI arrays: radius [m], pressure [Pa], temperature
[K], velocity [m/s], entropy [J/(kg K)], and enthalpy [J/kg].
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _fields(radius: ArrayLike, **values: ArrayLike) -> tuple[FloatArray, dict[str, FloatArray]]:
    r = np.asarray(radius, dtype=float)
    fields = {name: np.asarray(value, dtype=float) for name, value in values.items()}
    if r.ndim != 1 or r.size < 3 or np.any(~np.isfinite(r)) or np.any(np.diff(r) <= 0):
        raise ValueError("radius must be a finite, strictly increasing 1-D array with at least 3 points")
    if any(value.shape != r.shape or np.any(~np.isfinite(value)) for value in fields.values()):
        raise ValueError("all fields must be finite 1-D arrays matching radius")
    return r, fields


def entropy_from_state(
    temperature: ArrayLike,
    pressure: ArrayLike,
    *,
    cp: float,
    gas_constant: float,
    reference_index: int = 0,
) -> FloatArray:
    """Return ideal-gas entropy relative to one array element."""
    t = np.asarray(temperature, dtype=float)
    p = np.asarray(pressure, dtype=float)
    if t.shape != p.shape or t.ndim != 1 or np.any(t <= 0) or np.any(p <= 0):
        raise ValueError("temperature and pressure must be positive matching 1-D arrays")
    if not (cp > 0 and gas_constant > 0):
        raise ValueError("cp and gas_constant must be positive")
    if not -t.size <= reference_index < t.size:
        raise IndexError("reference_index is outside the fields")
    return cp * np.log(t / t[reference_index]) - gas_constant * np.log(p / p[reference_index])


def pressure_residual(
    radius: ArrayLike,
    pressure: ArrayLike,
    temperature: ArrayLike,
    tangential_velocity: ArrayLike,
    *,
    gas_constant: float,
) -> FloatArray:
    """Return ``(1/rho) dp/dr - c_u**2/r`` in m/s²."""
    r, f = _fields(radius, p=pressure, t=temperature, cu=tangential_velocity)
    if gas_constant <= 0 or np.any(f["p"] <= 0) or np.any(f["t"] <= 0) or np.any(r <= 0):
        raise ValueError("gas_constant, pressure, temperature, and radius must be positive")
    density = f["p"] / (gas_constant * f["t"])
    return np.gradient(f["p"], r, edge_order=2) / density - f["cu"] ** 2 / r


def thermodynamic_residual(
    radius: ArrayLike,
    temperature: ArrayLike,
    entropy: ArrayLike,
    tangential_velocity: ArrayLike,
    *,
    cp: float,
) -> FloatArray:
    """Return ``cp dT/dr - T ds/dr - c_u**2/r`` in m/s²."""
    r, f = _fields(radius, t=temperature, s=entropy, cu=tangential_velocity)
    if cp <= 0 or np.any(f["t"] <= 0) or np.any(r <= 0):
        raise ValueError("cp, temperature, and radius must be positive")
    return (
        cp * np.gradient(f["t"], r, edge_order=2)
        - f["t"] * np.gradient(f["s"], r, edge_order=2)
        - f["cu"] ** 2 / r
    )


def general_residual(
    radius: ArrayLike,
    temperature: ArrayLike,
    entropy: ArrayLike,
    axial_velocity: ArrayLike,
    tangential_velocity: ArrayLike,
    *,
    cp: float,
) -> FloatArray:
    """Return the general radial-equilibrium residual in m/s²."""
    r, f = _fields(
        radius,
        t=temperature,
        s=entropy,
        ca=axial_velocity,
        cu=tangential_velocity,
    )
    if cp <= 0 or np.any(f["t"] <= 0) or np.any(r <= 0):
        raise ValueError("cp, temperature, and radius must be positive")
    h0 = cp * f["t"] + (f["ca"] ** 2 + f["cu"] ** 2) / 2
    return (
        np.gradient(h0, r, edge_order=2)
        - f["t"] * np.gradient(f["s"], r, edge_order=2)
        - f["ca"] * np.gradient(f["ca"], r, edge_order=2)
        - f["cu"] / r * np.gradient(r * f["cu"], r, edge_order=2)
    )


def normalized_residual(residual: ArrayLike, reference: ArrayLike, *, trim: int = 1) -> float:
    """Return max absolute residual divided by a non-zero reference scale."""
    residual_array = np.asarray(residual, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    if residual_array.shape != reference_array.shape or residual_array.ndim != 1:
        raise ValueError("residual and reference must be matching 1-D arrays")
    if trim < 0 or 2 * trim >= residual_array.size:
        raise ValueError("trim removes the complete field")
    section = slice(trim, -trim or None)
    scale = float(np.max(np.abs(reference_array[section])))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("reference scale must be finite and non-zero")
    return float(np.max(np.abs(residual_array[section])) / scale)

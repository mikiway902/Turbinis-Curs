"""Meridional flowpath geometry from notebook cell 22.

PROVENANCE
==========
This module is a source-preserving extraction of the numerical meridional
geometry block in code cell 22 of
``Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb``.  It executes no notebook,
imports no NumPy/SciPy/pandas, and keeps no mutable global state.

The notebook remains the independent legacy comparison path.  The helpers
``γ_участка``, ``ширина`` and ``r_нв``, the contour/geometry construction, the
strict ``γ_цель`` comparisons, the exact 50-iteration bisection, the
``γ_макс`` transition constraint and the final contour-angle aggregation are
copied with the same operation order.  Calculated points are returned even
when the resulting geometry is engineering-poor; no guard, filter or clamp is
added.  Like the source, the function works on the stage mappings it is given
and does not mutate them.

The source's pandas display objects (``контур_df``, ``участки``, ``табл_мер``,
``ширины_df``) are presentation only and are not part of the wired path.  The
optimizer consumes only ``контур``, ``геом``, ``γ_max``, ``dx_т`` and
``γ_турб``; :meth:`MeridionalGeometry.legacy_values` returns those exact
identifiers.

Tests compare this module with the observed legacy cell-22 execution at one
fixed candidate only as a software-drift diagnostic, never as physical
validation.

UNITS
=====
SI throughout.  ``x``, ``dx``, ``dx_t``, blade lengths and chord widths
``b_c``/``b_l`` are m; radii are m.  ``gamma_max``, ``gamma_turbine``,
``gamma_target`` and ``gamma_limit`` are degrees.  ``delta_a_ratio`` is the
dimensionless ``δ_a / b_a`` source ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

# Source-fixed numerical control (do not tune without changing the notebook).
WIDTH_ITER_MAX = 50


@dataclass(frozen=True)
class MeridionalGeometryConfig:
    """Cell-22 constants and angle limits.

    ======================  ==================
    field                   notebook identifier
    ======================  ==================
    ``b_a_min``             ``b_a_мин``
    ``lb_c``                ``lb_c``
    ``lb_l``                ``lb_л``
    ``lb_min_c``            ``lb_мин_c``
    ``lb_min_l``            ``lb_мин_л``
    ``delta_a_ratio``       ``δа_к_ba``
    ``gamma_target``        ``γ_цель``
    ``gamma_limit``         ``γ_макс``
    ======================  ==================

    ``gamma_limit`` is the maximum wall inclination of the transition channel;
    the result field ``gamma_max`` is the largest per-section opening angle.
    The two notebook identifiers differ only in the underscore position
    (``γ_макс`` / ``γ_max``).
    """

    b_a_min: float
    lb_c: float
    lb_l: float
    lb_min_c: float
    lb_min_l: float
    delta_a_ratio: float
    gamma_target: float
    gamma_limit: float


@dataclass(frozen=True)
class ContourPoint:
    """One meridional contour section: ``(x, r_outer, r_inner, label)``."""

    x: float
    r_outer: float
    r_inner: float
    section: str

    def legacy_tuple(self) -> tuple[float, float, float, str]:
        """Return the source tuple shape used by ``контур``."""

        return (self.x, self.r_outer, self.r_inner, self.section)


@dataclass(frozen=True)
class BladeWidths:
    """One ``геом`` entry: stator and rotor axial chord widths, m."""

    b_c: float
    b_l: float

    def legacy_dict(self) -> dict[str, float]:
        """Return the source dict shape (``b_c``, ``b_л``) in insertion order."""

        return {"b_c": self.b_c, "b_л": self.b_l}


@dataclass(frozen=True)
class MeridionalGeometry:
    """Numerical result of the cell-22 meridional geometry block.

    ===================  ==================
    result field         notebook identifier
    ===================  ==================
    ``contour``          ``контур``
    ``geom``             ``геом``
    ``gamma_max``        ``γ_max``
    ``dx_t``             ``dx_т``
    ``gamma_turbine``    ``γ_турб``
    ===================  ==================
    """

    contour: tuple[ContourPoint, ...]
    geom: tuple[BladeWidths, ...]
    gamma_max: float
    dx_t: float
    gamma_turbine: float

    def legacy_values(self) -> dict[str, Any]:
        """Return the values under the exact notebook identifiers."""

        return {
            "контур": [point.legacy_tuple() for point in self.contour],
            "геом": [widths.legacy_dict() for widths in self.geom],
            "γ_max": self.gamma_max,
            "dx_т": self.dx_t,
            "γ_турб": self.gamma_turbine,
        }


def section_opening_angle(
    r_n1: float,
    r_v1: float,
    r_n2: float,
    r_v2: float,
    dx: float,
) -> float:
    """Return ``γ = γ_н + γ_в`` over a section of axial length ``dx``."""

    return math.degrees(
        math.atan((r_n2 - r_n1) / dx) + math.atan((r_v1 - r_v2) / dx)
    )


def blade_width(
    l: float,
    lb: float,
    lb_min: float,
    delta_do: float,
    r1: tuple[float, float],
    r2: tuple[float, float],
    *,
    b_a_min: float,
    gamma_target: float,
) -> float:
    """Return the axial chord width ``b_a`` with the source bisection.

    The width starts at ``max(l / lb, b_a_min)`` and is expanded up to
    ``l / lb_min`` until the section opening angle no longer exceeds
    ``gamma_target``.  The 50-iteration limit and both strict comparisons are
    the source's.
    """

    b0 = max(l / lb, b_a_min)
    b_max = max(l / lb_min, b0)
    gamma_at = lambda b: section_opening_angle(*r1, *r2, delta_do + b)
    if gamma_at(b0) <= gamma_target:
        return b0
    if gamma_at(b_max) > gamma_target:
        return b_max
    lo, hi = b0, b_max
    for _ in range(WIDTH_ITER_MAX):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gamma_at(mid) > gamma_target else (lo, mid)
    return hi


def outer_inner_radii(D: float, l: float) -> tuple[float, float]:
    """Return ``(r_outer, r_inner)`` for a mean diameter and blade length."""

    return (D / 2 + l / 2, D / 2 - l / 2)


def calculate_meridional_geometry(
    stages: Sequence[Mapping[str, Any]],
    config: MeridionalGeometryConfig,
) -> MeridionalGeometry:
    """Evaluate the cell-22 meridional geometry without notebook execution.

    Parameters
    ----------
    stages:
        Stage mappings in flow order.  Each stage supplies ``D_c``, ``l_c``,
        ``D_л`` and ``l_л`` in m.  No stage-count or field-name guard is added.
    config:
        The cell-22 constants from ``b_a_мин`` through ``γ_макс``.

    Returns
    -------
    MeridionalGeometry
        Every numerical contour point, blade width and angle aggregate.
    """

    contour: list[ContourPoint] = []
    geom: list[BladeWidths] = []
    x = 0.0
    b_c = max(stages[0]["l_c"] / config.lb_c, config.b_a_min)
    r_prev = outer_inner_radii(stages[0]["D_c"], stages[0]["l_c"])
    contour.append(ContourPoint(x, r_prev[0], r_prev[1], "СА1 выход"))
    for i, s in enumerate(stages):
        delta_a = config.delta_a_ratio * b_c
        r_l = outer_inner_radii(s["D_л"], s["l_л"])
        b_l = blade_width(
            s["l_л"],
            config.lb_l,
            config.lb_min_l,
            delta_a,
            r_prev,
            r_l,
            b_a_min=config.b_a_min,
            gamma_target=config.gamma_target,
        )
        x += delta_a + b_l
        contour.append(ContourPoint(x, r_l[0], r_l[1], f"РК{i + 1} выход"))
        geom.append(BladeWidths(b_c=b_c, b_l=b_l))
        r_prev = r_l
        if i + 1 < len(stages):
            next_stage = stages[i + 1]
            r_c = outer_inner_radii(next_stage["D_c"], next_stage["l_c"])
            delta_a_l = config.delta_a_ratio * b_l
            b_c = blade_width(
                next_stage["l_c"],
                config.lb_c,
                config.lb_min_c,
                delta_a_l,
                r_prev,
                r_c,
                b_a_min=config.b_a_min,
                gamma_target=config.gamma_target,
            )
            # Transition channel: the wall slope is no steeper than γ_макс.
            x += max(
                delta_a_l + b_c,
                max(abs(r_c[0] - r_prev[0]), abs(r_c[1] - r_prev[1]))
                / math.tan(math.radians(config.gamma_limit)),
            )
            contour.append(ContourPoint(x, r_c[0], r_c[1], f"СА{i + 2} выход"))
            r_prev = r_c

    openings: list[float] = []
    for left, right in zip(contour, contour[1:]):
        dx = right.x - left.x
        gamma_n = math.degrees(math.atan((right.r_outer - left.r_outer) / dx))
        gamma_v = math.degrees(math.atan(-(right.r_inner - left.r_inner) / dx))
        openings.append(gamma_n + gamma_v)
    gamma_max = max(openings)
    dx_t = contour[-1].x
    gamma_turbine = section_opening_angle(
        contour[0].r_outer,
        contour[0].r_inner,
        contour[-1].r_outer,
        contour[-1].r_inner,
        dx_t,
    )

    return MeridionalGeometry(
        contour=tuple(contour),
        geom=tuple(geom),
        gamma_max=gamma_max,
        dx_t=dx_t,
        gamma_turbine=gamma_turbine,
    )


__all__ = [
    "BladeWidths",
    "ContourPoint",
    "MeridionalGeometry",
    "MeridionalGeometryConfig",
    "WIDTH_ITER_MAX",
    "blade_width",
    "calculate_meridional_geometry",
    "outer_inner_radii",
    "section_opening_angle",
]

"""EXPERIMENTAL section geometry, Vinogradov (2010), pp.48–52, table 3.3.

Coordinates are SI (x=U axial, y=V circumferential). Polygon integrals are
exact for the supplied polygon; sampling a curved profile is a numerical
approximation. No density, structural allowable, or stress closure is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SideDistribution:
    """Table 3.3 items 15–22, indexed by curve parameter, not axial x."""

    parameter: np.ndarray
    points_m: np.ndarray
    principal_points_m: np.ndarray
    arc_length_m: np.ndarray
    modulus_u_m3: np.ndarray
    modulus_v_m3: np.ndarray


@dataclass(frozen=True)
class SectionProperties:
    area_m2: float
    centroid_m: np.ndarray
    first_moments_m3: np.ndarray  # [integral x dA (=S_V), integral y dA (=S_U)]
    second_moments_m4: np.ndarray  # [integral y², integral x², integral xy]
    centroidal_moments_m4: np.ndarray  # [Ixx, Iyy, Ixy]
    principal_min_m4: float
    principal_max_m4: float
    principal_angle_rad: float  # minimum-inertia U' axis, from +x
    perimeter_m: float
    suction_length_m: float | None
    pressure_length_m: float | None
    contour_distribution: SideDistribution | None = None
    suction_distribution: SideDistribution | None = None
    pressure_distribution: SideDistribution | None = None
    rounded_suction_distribution: SideDistribution | None = None
    rounded_pressure_distribution: SideDistribution | None = None

    def to_principal(self, points) -> np.ndarray:
        """Table 3.3 items 15–16: translate, then rotate by -gamma_UV."""
        points = np.asarray(points, dtype=float)
        c, s = math.cos(self.principal_angle_rad), math.sin(self.principal_angle_rad)
        return (points - self.centroid_m) @ np.array([[c, -s], [s, c]])

    def distributed_moduli(self, points) -> np.ndarray:
        """Items 17–22: local [W_U', W_V'] in m³; neutral-axis poles are inf.

        These are pointwise I/distance, not a single extreme-fibre section
        modulus, and they do not themselves define a bending stress.
        """
        principal = self.to_principal(points)
        result = np.full(principal.shape, np.inf, dtype=float)
        np.divide(self.principal_min_m4, np.abs(principal[..., 1]),
                  out=result[..., 0], where=principal[..., 1] != 0)
        np.divide(self.principal_max_m4, np.abs(principal[..., 0]),
                  out=result[..., 1], where=principal[..., 0] != 0)
        return result

    def principal_axes(self, half_length_m: float) -> np.ndarray:
        """Items 11–14: two axis segments, shape (2,2,2), without tan poles."""
        if not math.isfinite(half_length_m) or half_length_m <= 0:
            raise ValueError("half_length_m must be finite and positive")
        c, s = math.cos(self.principal_angle_rad), math.sin(self.principal_angle_rad)
        directions = np.array([[c, s], [-s, c]])
        return self.centroid_m + half_length_m * np.array([-1., 1.])[None, :, None] * directions[:, None, :]


def _distribution(properties, points, parameter=None):
    points = np.asarray(points, dtype=float)
    if parameter is None:
        parameter = np.linspace(0., 1., len(points))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    moduli = properties.distributed_moduli(points)
    return SideDistribution(parameter, points, properties.to_principal(points), arc,
                            moduli[:, 0], moduli[:, 1])


def section_properties(profile_or_contour, *, count_per_side: int = 1001) -> SectionProperties:
    """Compute all table 3.3 integral and distributed geometric quantities.

    Accept a SectionProfile or an (N,2) simple polygon, open or closed, either
    orientation. Self-intersection validation belongs to the geometry validator.
    Polygon inputs must have >=3 finite non-collinear vertices. Profile surface
    lengths exclude its separately represented leading/trailing edge arcs;
    perimeter includes both edges. Principal direction is modulo pi; for an
    isotropic section gamma=0 (principal axes are not unique).
    """
    if not isinstance(count_per_side, int) or count_per_side < 3:
        raise ValueError("count_per_side must be an integer >=3")
    is_profile = hasattr(profile_or_contour, "contour")
    points = np.asarray(profile_or_contour.contour(count_per_side=count_per_side)
                        if is_profile else profile_or_contour, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("contour must contain at least three finite 2D vertices")
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    # Translate before integration to retain accuracy for sections far from zero.
    origin = points[0]
    q = points - origin
    qn = np.roll(q, -1, axis=0)
    x, y, xn, yn = q[:, 0], q[:, 1], qn[:, 0], qn[:, 1]
    cross = x * yn - xn * y
    signed_area = float(np.sum(cross) / 2.)
    if signed_area == 0.:
        raise ValueError("contour has zero signed area")
    sign = math.copysign(1., signed_area)
    area = abs(signed_area)
    cross = cross * sign
    mx = float(np.sum((x + xn) * cross) / 6.)
    my = float(np.sum((y + yn) * cross) / 6.)
    cx, cy = mx / area, my / area
    ixx = float(np.sum((y*y + y*yn + yn*yn) * cross) / 12.) - area * cy*cy
    iyy = float(np.sum((x*x + x*xn + xn*xn) * cross) / 12.) - area * cx*cx
    ixy = float(np.sum((2*x*y + x*yn + xn*y + 2*xn*yn) * cross) / 24.) - area * cx*cy
    eigenvalues, eigenvectors = np.linalg.eigh([[ixx, -ixy], [-ixy, iyy]])
    if eigenvalues[0] <= 0.:
        raise ValueError("contour does not define positive section inertias")
    angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle = (angle + math.pi / 2.) % math.pi - math.pi / 2.
    if abs(eigenvalues[1] - eigenvalues[0]) <= 32*np.finfo(float).eps*eigenvalues[1]:
        angle = 0.
    centroid = origin + [cx, cy]
    absolute_second = np.array([ixx + area*centroid[1]**2,
                                iyy + area*centroid[0]**2,
                                ixy + area*centroid[0]*centroid[1]])
    perimeter = float(np.linalg.norm(qn-q, axis=1).sum())
    suction = pressure = None
    if is_profile:
        parameter = np.linspace(0., 1., count_per_side)
        suction = np.asarray(profile_or_contour.suction.evaluate(parameter))
        pressure = np.asarray(profile_or_contour.pressure.evaluate(parameter))
    result = SectionProperties(area, centroid, area*centroid, absolute_second,
                               np.array([ixx, iyy, ixy]), float(eigenvalues[0]),
                               float(eigenvalues[1]), angle, perimeter,
                               None if suction is None else float(np.linalg.norm(np.diff(suction, axis=0), axis=1).sum()),
                               None if pressure is None else float(np.linalg.norm(np.diff(pressure, axis=0), axis=1).sum()))
    # Frozen records prevent reassignment by consumers; fill derived distributions
    # only during construction, without reintegrating the polygon.
    object.__setattr__(result, "contour_distribution", _distribution(result, np.vstack([points, points[0]])))
    if is_profile:
        object.__setattr__(result, "suction_distribution", _distribution(result, suction, parameter))
        object.__setattr__(result, "pressure_distribution", _distribution(result, pressure, parameter))
        if hasattr(profile_or_contour, "leading_edge") and hasattr(profile_or_contour, "trailing_edge"):
            leading = np.asarray(profile_or_contour.leading_edge.evaluate(parameter))
            trailing = np.asarray(profile_or_contour.trailing_edge.evaluate(parameter))
            nose, tail = int(np.argmin(leading[:, 0])), int(np.argmax(trailing[:, 0]))
            # Split the real rounded edges at their axial extrema. This makes
            # the upper/lower graph surfaces used by table 3.3 explicit, without
            # pretending that the separately represented main curves include
            # those edge portions. Sampled extrema have O(parameter_step²) error.
            rounded_suction = np.vstack([leading[nose:], suction[1:], trailing[1:tail+1]])
            rounded_pressure = np.vstack([leading[:nose+1][::-1], pressure[1:],
                                         trailing[tail:-1][::-1]])
            object.__setattr__(result, "rounded_suction_distribution", _distribution(result, rounded_suction))
            object.__setattr__(result, "rounded_pressure_distribution", _distribution(result, rounded_pressure))
    return result

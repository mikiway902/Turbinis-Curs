"""EXPERIMENTAL inscribed-circle channel analysis, Vinogradov pp.52–56.

Table 3.4 used graphically selected circles and extrapolated a cubic spline.
Here circles satisfy both analytic wall normals and equal radii; clearance to
both complete blade contours rejects external/intersecting normal branches.
There are no artificial inlet/outlet walls. Their connectors only flag circles
whose extent leaves the sampled two-wall corridor, never provide contacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, minimize_scalar


@dataclass(frozen=True)
class InscribedCircle:
    center_m: np.ndarray
    radius_m: float
    suction_parameter: float
    pressure_parameter: float
    suction_contact_m: np.ndarray
    pressure_contact_m: np.ndarray
    suction_normal: np.ndarray  # contact -> centre
    pressure_normal: np.ndarray
    tangency_residual_m: float
    normal_residual: float
    clearance_residual_m: float
    endcap_limited: bool
    pressure_contact_region: str = "pressure"


@dataclass(frozen=True)
class ChannelAnalysis:
    circles: tuple[InscribedCircle, ...]
    centers_m: np.ndarray
    radii_m: np.ndarray
    arc_length_m: np.ndarray
    width_m: np.ndarray
    width_gradient: np.ndarray  # d(2r)/ds; positive is diffuser
    throat: InscribedCircle
    throat_width_m: float
    convergence_ratio: float  # throat diameter / first sampled diameter
    diffuser_intervals_m: tuple[tuple[float, float], ...]
    rejected_candidates: tuple[dict, ...]
    throat_at_sample_boundary: bool
    status: str = "EXPERIMENTAL"


@dataclass(frozen=True)
class CircleTableInterpolation:
    """Table 3.4 items 8–14: interpolation only, not certified tangent circles."""
    axial_m: np.ndarray
    center_y_m: np.ndarray
    radii_m: np.ndarray
    arc_length_m: np.ndarray
    radius_gradient: np.ndarray  # dr/ds


def interpolate_circle_table(table_m, *, count: int = 201,
                             boundary_condition: str = "not-a-knot") -> CircleTableInterpolation:
    """Interpolate [centre x, centre y, radius] rows on their measured domain.

    No extrapolation to x=0: p.56 plots outside the measured p.55 interval,
    which is not evidence of an inscribed circle there. Mathcad cspline's
    boundary convention is not specified in the book; expose the choice.
    """
    table = np.asarray(table_m, dtype=float)
    if (table.ndim != 2 or table.shape[1] != 3 or len(table) < 4
            or not np.isfinite(table).all() or np.any(np.diff(table[:, 0]) <= 0)
            or np.any(table[:, 2] <= 0)):
        raise ValueError("table needs >=4 finite, increasing-x rows with positive radii")
    if not isinstance(count, int) or count < 2:
        raise ValueError("count must be an integer >=2")
    y = CubicSpline(table[:, 0], table[:, 1], bc_type=boundary_condition, extrapolate=False)
    r = CubicSpline(table[:, 0], table[:, 2], bc_type=boundary_condition, extrapolate=False)
    x = np.linspace(table[0, 0], table[-1, 0], count)
    arc = np.zeros(count)
    for i in range(1, count):
        arc[i] = arc[i-1]+quad(lambda z: math.hypot(1., float(y(z, 1))), x[i-1], x[i], epsabs=1e-12)[0]
    radii = r(x)
    if np.any(radii <= 0):
        raise ValueError("cubic interpolant produces nonpositive radius")
    return CircleTableInterpolation(x, y(x), radii, arc, r(x, 1)/np.hypot(1., y(x, 1)))


def _normal(curve, t, sign):
    tangent = np.asarray(curve.evaluate(t, derivative=1), dtype=float)
    length = float(np.linalg.norm(tangent))
    if not np.isfinite(length) or length == 0.:
        raise ValueError("wall has a zero or nonfinite tangent")
    return sign*np.array([-tangent[1], tangent[0]])/length


def _segment_distance(point, a, b):
    direction = b-a
    denominator = np.sum(direction*direction, axis=-1)
    t = np.zeros_like(denominator, dtype=float)
    np.divide(np.sum((point-a)*direction, axis=-1), denominator,
              out=t, where=denominator > 0)
    return np.linalg.norm(point-(a+np.clip(t, 0., 1.)[..., None]*direction), axis=-1)


def _inside(point, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    crossings = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    if not crossings.any():
        return False
    a, b = a[crossings], b[crossings]
    hits = a[:, 0]+(point[1]-a[:, 1])*(b[:, 0]-a[:, 0])/(b[:, 1]-a[:, 1])
    return bool(np.count_nonzero(hits > point[0]) % 2)


@dataclass(frozen=True)
class _PressureAndEdge:
    pressure: object
    edge: object

    def evaluate(self, t, derivative=0):
        t = np.asarray(t, dtype=float)
        flat = t.reshape(-1)
        result = np.empty((len(flat), 2))
        pressure_mask = flat <= .5
        result[pressure_mask] = self.pressure.evaluate(2*flat[pressure_mask], derivative=derivative)*2**derivative
        result[~pressure_mask] = self.edge.evaluate(2*(1-flat[~pressure_mask]), derivative=derivative)*(-2)**derivative
        return result.reshape(t.shape+(2,))


def analyze_channel(profile, pitch_m: float, *, samples: int = 81,
                    wall_samples: int = 801, tangency_tolerance_m: float = 1e-8,
                    clearance_tolerance_m: float | None = None,
                    parameter_range=(.001, .999),
                    include_trailing_edge: bool = True) -> ChannelAnalysis:
    """Solve actual two-wall tangent circles and quantify throat/diffuser.

    The lower wall is profile.suction, upper is profile.pressure+(0,pitch),
    continued around the real trailing edge (reversed), when available and
    requested. Its augmented parameter assigns half to pressure, half to the
    trailing edge; pressure_contact_region identifies which was touched.
    Unknowns are upper contact parameter and r; lower contact is swept.
    Outward normals use left on suction, right on the shifted upper wall.
    A circle must have interior contacts, centre inside the
    open channel's connector polygon, and no penetration of either complete
    blade contour (polygon clearance approximation, resolution exposed).

    Endpoint contacts and external/blocked branches are reported, not relabelled
    as channel circles. Endcap-limited circles remain physically two-wall
    tangent circles and are flagged; connectors never determine their radius.
    Throat is minimized within the measured contact interval; if its minimum
    lies at its boundary the caller must not assume a global cascade throat.
    Width is diameter (2r), not axial or vertical wall separation.
    """
    if not math.isfinite(pitch_m) or pitch_m <= 0:
        raise ValueError("pitch_m must be finite and positive")
    if not isinstance(samples, int) or samples < 3 or not isinstance(wall_samples, int) or wall_samples < 31:
        raise ValueError("samples>=3 and wall_samples>=31 must be integers")
    if not math.isfinite(tangency_tolerance_m) or tangency_tolerance_m <= 0:
        raise ValueError("tangency tolerance must be finite and positive")
    lo, hi = parameter_range
    if not 0 < lo < hi < 1:
        raise ValueError("parameter_range must be strictly inside (0,1)")
    lower, upper = profile.suction, profile.pressure
    augmented_upper = include_trailing_edge and hasattr(profile, "trailing_edge")
    if augmented_upper:
        upper = _PressureAndEdge(upper, profile.trailing_edge)
    grid = np.linspace(0., 1., wall_samples)
    lp = np.asarray(lower.evaluate(grid))
    up = np.asarray(upper.evaluate(grid))+[0., pitch_m]
    scale = max(float(np.ptp(np.vstack([lp, up]), axis=0).max()), pitch_m)
    if clearance_tolerance_m is None:
        # Chord sagitta is O(h²); report this numerical, not physical tolerance.
        clearance_tolerance_m = max(tangency_tolerance_m, 8*scale/(wall_samples-1)**2)
    if not math.isfinite(clearance_tolerance_m) or clearance_tolerance_m < 0:
        raise ValueError("clearance tolerance must be finite and nonnegative")
    polygon = np.vstack([lp, up[::-1]])
    blade = np.asarray(profile.contour(count_per_side=wall_samples))
    blades = (blade, blade+[0., pitch_m])
    starts = np.vstack([p[:-1] for p in blades])
    ends = np.vstack([p[1:] for p in blades])
    rejected = []

    def solve(t, *, record=False):
        p = np.asarray(lower.evaluate(t))
        n = _normal(lower, t, 1.)
        nearest = int(np.argmin(np.sum((up-p)**2, axis=1)))
        guesses = (grid[nearest], .25, .75)
        best = None
        reason = "normal_solver_failure"
        for initial in guesses:
            v0 = float(np.clip(initial, 1e-8, 1-1e-8))
            r0 = max(float(np.linalg.norm(up[int(round(v0*(wall_samples-1)))]-p))/2, scale*1e-4)
            def equation(z):
                v, radius = z
                q = np.asarray(upper.evaluate(v))+[0., pitch_m]
                return (p+radius*n-q-radius*_normal(upper, v, -1.))/scale
            # Normalize both residual and radius to keep metre-scale designs
            # numerically identical to geometrically scaled reference cases.
            def normalized(z):
                return equation([z[0], z[1]*scale])
            fit = least_squares(normalized, [v0, min(r0/scale, 2.)],
                                bounds=([1e-10, 1e-10], [1-1e-10, 3.]),
                                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=100)
            v, radius = float(fit.x[0]), float(fit.x[1]*scale)
            residual = float(np.linalg.norm(normalized(fit.x))*scale)
            if residual > tangency_tolerance_m:
                continue
            if v <= 1e-7 or v >= 1-1e-7:
                reason = "endpoint_contact"
                continue
            q = np.asarray(upper.evaluate(v))+[0., pitch_m]
            nu = _normal(upper, v, -1.)
            center = (p+radius*n+q+radius*nu)/2.
            if not _inside(center, polygon):
                reason = "external_normal_branch"
                continue
            clearance = float(_segment_distance(center, starts, ends).min())-radius
            if clearance < -clearance_tolerance_m:
                reason = "circle_intersects_blade"
                continue
            normal_residual = max(abs(float(np.dot(center-p, [-n[1], n[0]]))),
                                  abs(float(np.dot(center-q, [-nu[1], nu[0]]))))/radius
            endcap_distance = min(float(_segment_distance(center, lp[0], up[0])),
                                  float(_segment_distance(center, lp[-1], up[-1])))
            candidate = InscribedCircle(center, radius, float(t), v, p, q, n, nu,
                                        residual, normal_residual, clearance,
                                        endcap_distance < radius,
                                        "trailing_edge" if augmented_upper and v > .5 else "pressure")
            if best is None or candidate.radius_m < best.radius_m:
                best = candidate
        if best is None and record:
            rejected.append({"suction_parameter": float(t), "reason": reason})
        return best

    circles = [c for t in np.linspace(lo, hi, samples) if (c := solve(t, record=True)) is not None]
    if len(circles) < 3:
        raise ValueError("fewer than three valid two-wall circles: " + repr(rejected))
    minimum = min(range(len(circles)), key=lambda i: circles[i].radius_m)
    boundary = minimum in (0, len(circles)-1)
    if not boundary:
        left, right = circles[minimum-1].suction_parameter, circles[minimum+1].suction_parameter
        def objective(t):
            circle = solve(t)
            return 3*scale if circle is None else circle.radius_m
        optimum = minimize_scalar(objective, bounds=(left, right), method="bounded",
                                  options={"xatol": 1e-10})
        refined = solve(float(optimum.x))
        if refined is not None and refined.radius_m < circles[minimum].radius_m:
            circles.append(refined)
            circles.sort(key=lambda c: c.suction_parameter)
    centers = np.array([c.center_m for c in circles])
    radii = np.array([c.radius_m for c in circles])
    increments = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    if np.any(increments == 0):
        raise ValueError("degenerate channel centreline")
    arc = np.r_[0., np.cumsum(increments)]
    gradient = np.gradient(2*radii, arc)
    intervals = tuple((float(arc[i]), float(arc[i+1]))
                      for i in range(len(arc)-1) if radii[i+1]-radii[i] > tangency_tolerance_m)
    throat = min(circles, key=lambda c: c.radius_m)
    return ChannelAnalysis(tuple(circles), centers, radii, arc, 2*radii, gradient,
                           throat, 2*throat.radius_m, throat.radius_m/radii[0],
                           intervals, tuple(rejected), boundary)

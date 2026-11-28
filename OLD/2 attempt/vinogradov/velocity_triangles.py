"""EXPERIMENTAL explicit velocity triangles and elementary-stage layout.

Vinogradov (2010), pp.57–60 tables 3.5–3.6. Velocities m/s, geometry m.
Physical vectors use (axial downstream, circumferential along blade motion).
Book triangle plots instead use (-circumferential,-axial); they are retained
as named plotting matrices, not silently treated as physical vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class VelocityTriangles:
    absolute_inlet_m_s: np.ndarray
    relative_inlet_m_s: np.ndarray
    relative_outlet_m_s: np.ndarray
    absolute_outlet_m_s: np.ndarray
    blade_velocity_m_s: np.ndarray
    inlet_plot_m_s: np.ndarray
    outlet_plot_m_s: np.ndarray
    combined_plot_m_s: np.ndarray
    euler_work_j_kg: float


def velocity_triangles(absolute_inlet_speed_m_s: float, absolute_inlet_angle_deg: float,
                       blade_speed_m_s: float, relative_outlet_speed_m_s: float,
                       relative_outlet_angle_deg: float) -> VelocityTriangles:
    """Table 3.5: explicit c1, alpha1, u, w2, beta2; no flow-metal identity.

    alpha1 is from +circumferential toward +axial; beta2 is from
    -circumferential toward +axial. Blade speed may be signed. Angles (0,180)
    retain positive axial flow and permit counter-swirl. c=w+u at both stations.
    """
    values = [absolute_inlet_speed_m_s, absolute_inlet_angle_deg, blade_speed_m_s,
              relative_outlet_speed_m_s, relative_outlet_angle_deg]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("velocity triangle inputs must be finite")
    if absolute_inlet_speed_m_s <= 0 or relative_outlet_speed_m_s <= 0:
        raise ValueError("speed magnitudes must be positive")
    if not 0 < absolute_inlet_angle_deg < 180 or not 0 < relative_outlet_angle_deg < 180:
        raise ValueError("angles must be in (0,180) degrees")
    alpha, beta = np.radians([absolute_inlet_angle_deg, relative_outlet_angle_deg])
    c1 = absolute_inlet_speed_m_s*np.array([math.sin(alpha), math.cos(alpha)])
    w2 = relative_outlet_speed_m_s*np.array([math.sin(beta), -math.cos(beta)])
    u = np.array([0., blade_speed_m_s])
    w1, c2 = c1-u, w2+u
    to_plot = lambda v: -v[::-1]
    inlet = np.array([[0., 0.], to_plot(c1), to_plot(w1), [0., 0.]])
    outlet = np.array([[0., 0.], to_plot(w2), to_plot(c2), [0., 0.]])
    return VelocityTriangles(c1, w1, w2, c2, u, inlet, outlet,
                             np.vstack([inlet, outlet[1:]]),
                             float(blade_speed_m_s*(c1[1]-c2[1])))


@dataclass(frozen=True)
class ElementaryStageLayout:
    stator_contours_m: tuple[np.ndarray, ...]
    rotor_contours_m: tuple[np.ndarray, ...]
    inlet_triangle_m: np.ndarray | None
    outlet_triangle_m: np.ndarray | None


def elementary_stage_layout(stator, rotor, *, stator_pitch_m: float,
                            rotor_pitch_m: float, rotor_axial_offset_m: float = .110,
                            rotor_reflection_y_m: float = .010,
                            triangles: VelocityTriangles | None = None,
                            triangle_scale_s: float = .00075,
                            inlet_triangle_origin_m=(.100, -.050),
                            outlet_triangle_origin_m=(.200, .020),
                            count_per_side: int = 401) -> ElementaryStageLayout:
    """Table 3.6 display assembly, NOT a stage-flow calculation.

    Stator copies are y and y-t; rotor copies are (x+.110,.010-y)
    and (x+.110,.010-y+t), matching p.60 in SI. Constants are source
    illustration offsets and can be explicitly changed. Reflection is for the
    drawing only, not an implicit flow/metal angle or rotor-frame conversion.
    Triangle display scale .75 mm/(m/s) has units seconds, not dimensionless.
    """
    values = [stator_pitch_m, rotor_pitch_m, rotor_axial_offset_m,
              rotor_reflection_y_m, triangle_scale_s]
    if not all(math.isfinite(v) for v in values) or min(stator_pitch_m, rotor_pitch_m, triangle_scale_s) <= 0:
        raise ValueError("layout requires finite inputs, positive pitches and triangle scale")
    s = np.asarray(stator.contour(count_per_side=count_per_side), dtype=float)
    r = np.asarray(rotor.contour(count_per_side=count_per_side), dtype=float).copy()
    r[:, 0] += rotor_axial_offset_m
    r[:, 1] = rotor_reflection_y_m-r[:, 1]
    inlet = outlet = None
    if triangles is not None:
        def place(matrix, origin):
            origin = np.asarray(origin, dtype=float)
            if origin.shape != (2,) or not np.isfinite(origin).all():
                raise ValueError("triangle origins must be finite 2D coordinates")
            return triangle_scale_s*np.column_stack([-matrix[:, 1], matrix[:, 0]])+origin
        inlet = place(triangles.inlet_plot_m_s, inlet_triangle_origin_m)
        outlet = place(triangles.outlet_plot_m_s, outlet_triangle_origin_m)
    return ElementaryStageLayout((s, s-[0., stator_pitch_m]),
                                 (r, r+[0., rotor_pitch_m]), inlet, outlet)

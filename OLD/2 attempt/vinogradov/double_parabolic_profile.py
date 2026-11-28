"""EXPERIMENTAL Vinogradov (2010), tables 4.1 and 5.1, pp. 61–76.

Lengths are metres. ``relative_thickness`` is diameter/chord (not percent),
``thickness_location`` is the maximum-circle centre's axial x/axial width.
The prescribed circle imposes a *local* common-normal thickness; the book's
claim that this is the global maximum is recorded, not silently assumed.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq

from .curves import BezierCurve, CompositeCurve, arc_between, tangent_intersection
from .models import ConstructionError, ProfileSpec, SectionProfile


def _unit(angle):
    return np.array([math.cos(angle), math.sin(angle)])


def _fail(message, **details):
    raise ConstructionError(message, details=details)


def _quadratic(start, end, angle_start, angle_end):
    """Exact tangent-envelope parabola, table 5.1 (no cubic-spline fitting)."""
    try:
        control = tangent_intersection(start, angle_start, end, angle_end)
    except ValueError as exc:
        _fail('Degenerate parabolic control triangle', reason=str(exc))
    # Forward endpoint tangents must point along the curve, not against it.
    a = float(np.dot(control-start, _unit(angle_start)))
    b = float(np.dot(end-control, _unit(angle_end)))
    if min(a, b) <= 0:
        _fail('Parabola lies outside its source control triangle',
              first_tangent_distance_m=a, last_tangent_distance_m=b)
    return BezierCurve(np.array([start, control, end]))


def _edge_centres(spec):
    b = spec.axial_width_m
    chord = b / math.sin(math.radians(spec.stagger_deg))
    slope = -1 / math.tan(math.radians(spec.stagger_deg))
    c1 = np.array([spec.leading_radius_m,
                   spec.origin_y_m+slope*spec.leading_radius_m])
    c2 = np.array([b-spec.trailing_radius_m,
                   spec.origin_y_m+slope*(b-spec.trailing_radius_m)])
    return chord, c1, c2


def _pressure_from_exit(spec, c1, c2):
    """Table 4.1 items 37–43: pressure circle tangent to both edge circles."""
    angle = math.radians(spec.outlet_metal_deg+spec.trailing_wedge_deg/2)
    h = c2-spec.trailing_radius_m*_unit(angle)
    delta = h-c1
    denominator = 2*(float(np.dot(delta, _unit(angle)))+spec.leading_radius_m)
    if abs(denominator) <= 1e-14*spec.axial_width_m:
        _fail('Pressure-circle construction has zero denominator', denominator_m=denominator)
    radius = (float(np.dot(delta, delta))-spec.leading_radius_m**2)/denominator
    if radius <= 0:
        _fail('Pressure-circle radius is not positive', pressure_radius_m=radius)
    centre = h-radius*_unit(angle)
    g = c1+spec.leading_radius_m*(centre-c1)/np.linalg.norm(centre-c1)
    return centre, radius, g, h


def prescribed_thickness_construction(spec: ProfileSpec) -> dict:
    """Return SI construction data, table 4.1 items 1–65 (pp. 61–74).

    Includes the abandoned mean-line first approximation (items 14–30) and
    the pressure-first solution (items 37–65). Arrays are computational data;
    profile metadata converts them to JSON lists.
    """
    chord, c1, c2 = _edge_centres(spec)
    radius = chord*spec.relative_thickness/2
    location = spec.axial_width_m*spec.thickness_location
    if radius <= 0 or not c1[0] < location < c2[0]:
        _fail('Specified thickness/location is outside the construction domain',
              thickness_radius_m=radius, thickness_location_m=location)
    cp, rp, g, h = _pressure_from_exit(spec, c1, c2)
    # For a circular pressure side the normal-offset locus is exactly another
    # circle. This analytically solves table 4.1 item 49's scalar root equation.
    centre_radius = rp+radius
    radicand = centre_radius**2-(location-cp[0])**2
    if radicand <= 0:
        _fail('Thickness-circle centre is outside pressure normal-offset locus',
              normal_offset_radicand_m2=radicand)
    d = np.array([location, cp[1]+math.sqrt(radicand)])
    normal = (d-cp)/centre_radius
    w = cp+rp*normal
    q = cp+(rp+2*radius)*normal
    if not g[0] < w[0] < h[0]:
        _fail('Specified thickness contacts an edge instead of the pressure arc',
              pressure_contact_x_m=float(w[0]), pressure_start_x_m=float(g[0]),
              pressure_end_x_m=float(h[0]))
    angle_q = math.atan2(normal[1], normal[0])-math.pi/2
    inlet = math.radians(90-spec.inlet_metal_deg)
    pressure_inlet = math.atan2((g-cp)[1], (g-cp)[0])-math.pi/2
    suction_inlet = 2*inlet-pressure_inlet
    l = c1+spec.leading_radius_m*_unit(suction_inlet+math.pi/2)
    exit_radial = math.radians(spec.outlet_metal_deg-spec.trailing_wedge_deg/2)
    f = c2+spec.trailing_radius_m*_unit(exit_radial)
    exit_tangent = exit_radial-math.pi/2
    inlet_curve = _quadratic(l, q, suction_inlet, angle_q)
    outlet_curve = _quadratic(q, f, angle_q, exit_tangent)

    # The book first tries a parabolic mean line and discovers its inlet
    # pressure tangent misses the leading circle (p.66 item 30).
    mean = _quadratic(c1, c2, inlet, math.radians(spec.outlet_metal_deg-90))
    parameter = brentq(lambda t: float(mean.evaluate(t)[0])-location, 0., 1.)
    mean_point = mean.evaluate(parameter)
    tangent = mean.evaluate(parameter, 1)
    mean_angle = math.atan2(tangent[1], tangent[0])
    mean_normal = _unit(mean_angle+math.pi/2)
    first_suction = mean_point+radius*mean_normal
    first_pressure = mean_point-radius*mean_normal
    # Signed normal distance of C1 from the trial pressure tangent minus rLE.
    first_clearance = float(np.dot(c1-first_pressure, mean_normal))-spec.leading_radius_m
    throat = spec.pitch_m*math.sin(math.radians(spec.outlet_metal_deg))/spec.throat_multiplier
    beta = math.radians(spec.outlet_metal_deg-spec.trailing_wedge_deg/2+spec.back_angle_deg)
    adjacent = c2+np.array([0., spec.pitch_m])
    throat_point = adjacent-(throat+spec.trailing_radius_m)*np.array([math.sin(beta), math.cos(beta)])
    return dict(chord_m=chord, leading_center=c1, trailing_center=c2,
                pressure_center=cp, pressure_radius_m=rp, pressure_start=g,
                pressure_end=h, thickness_radius_m=radius, thickness_center=d,
                pressure_contact=w, suction_contact=q, suction_start=l,
                suction_end=f, suction_inlet_angle_rad=suction_inlet,
                thickness_tangent_angle_rad=angle_q, suction_exit_angle_rad=exit_tangent,
                inlet_control=inlet_curve.control_points[1],
                outlet_control=outlet_curve.control_points[1],
                mean_point=mean_point, mean_tangent_angle_rad=mean_angle,
                first_suction_contact=first_suction, first_pressure_contact=first_pressure,
                first_pressure_tangent_clearance_m=first_clearance,
                source_throat_m=throat, source_throat_point=throat_point)


def _build(spec, method):
    data = prescribed_thickness_construction(spec)
    l, q, f = (data[k] for k in ('suction_start', 'suction_contact', 'suction_end'))
    suction = CompositeCurve((
        BezierCurve(np.array([l, data['inlet_control'], q])),
        BezierCurve(np.array([q, data['outlet_control'], f])),
    ))
    pressure = CompositeCurve((arc_between(data['pressure_center'], data['pressure_start'],
                                           data['pressure_end'], clockwise=True),))
    metadata = {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in data.items()}
    metadata.update(method=method, status='EXPERIMENTAL', source='Vinogradov 2010 tables 4.1, 5.1 pp.61–76',
                    thickness_definition='prescribed local common-normal diameter, not proven global maximum',
                    prescribed_thickness_m=2*data['thickness_radius_m'],
                    construction_valid=True, global_maximum_thickness_verified=False,
                    pitch_m=spec.pitch_m)
    return SectionProfile(suction=suction, pressure=pressure,
                          leading_edge=arc_between(data['leading_center'], data['pressure_start'], l, clockwise=True),
                          trailing_edge=arc_between(data['trailing_center'], f, data['pressure_end'], clockwise=True),
                          metadata=metadata)


def build_prescribed_thickness_profile(spec: ProfileSpec) -> SectionProfile:
    """Pressure-first prescribed-thickness scheme, table 4.1; table 5.1 closure."""
    return _build(spec, 'vinogradov_prescribed_thickness')


def build_double_parabolic_profile(spec: ProfileSpec) -> SectionProfile:
    """Two exactly tangent quadratic suction segments, table 5.1 pp.75–76."""
    return _build(spec, 'vinogradov_double_parabolic')


def prescribed_reference_figure_data(spec: ProfileSpec, count: int = 401) -> dict:
    """Plot-ready SI curves for fig4.1 and table5.1 items7–8 (pp.74–76).

    Includes the earlier one-parabola comparison via its own source method,
    rather than approximating that reference by a different fitted curve.
    """
    from .parabolic_profile import build_parabolic_profile

    profile = build_double_parabolic_profile(spec)
    data = profile.metadata
    angles = np.linspace(0., 2*math.pi, count)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    contour = profile.contour(count_per_side=count)
    earlier = build_parabolic_profile(spec)
    return {
        'figure_4_1_leading_circle': np.asarray(data['leading_center'])+spec.leading_radius_m*unit,
        'figure_4_1_trailing_circle': np.asarray(data['trailing_center'])+spec.trailing_radius_m*unit,
        'figure_4_1_thickness_circle': np.asarray(data['thickness_center'])+data['thickness_radius_m']*unit,
        'figure_4_1_inlet_control_triangle': np.asarray([
            data['suction_start'], data['inlet_control'], data['suction_contact']]),
        'figure_4_1_outlet_control_triangle': np.asarray([
            data['suction_contact'], data['outlet_control'], data['suction_end']]),
        'figure_4_1_pressure': profile.pressure.sample(count),
        'table_5_1_item_7_profile': contour,
        'table_5_1_item_7_adjacent_profile': contour+np.array([0., spec.pitch_m]),
        'table_5_1_item_8_single_parabola': earlier.suction.sample(count),
        'table_5_1_item_8_double_parabola': profile.suction.sample(count),
    }

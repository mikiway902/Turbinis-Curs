"""EXPERIMENTAL active subsonic profile, Vinogradov 2010 pp.77–86 table 6.1.

The source variant retains the unsuccessful printed inlet (p.84). The default
constrained variant enforces both supplied metal-angle boundary conditions,
using exact tangent circles or an explicitly identified G1 biarc design.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .curves import BezierCurve, CircularArc, CompositeCurve, arc_between
from .double_parabolic_profile import _edge_centres, _fail, _pressure_from_exit, _unit
from .models import ConstructionError, ProfileSpec, SectionProfile


@dataclass(frozen=True)
class ActiveProfileOptions:
    """Source table 6.1 extra inputs, with explicit dimensional overrides.

    The radius ratio is 15.25/25.61 (p.77), NOT the rounded 0.595 on p.78.
    The exit probe is 0.1 mm at source width 100 mm (p.81 item30); scaling
    it with width preserves geometric similarity instead of fixing units.
    For the constrained builder an explicitly supplied options object fixes
    this radius and must agree with the prescribed outlet metal/wedge.
    Omit options to solve the pressure radius from those boundary conditions.
    """
    pressure_radius_chord_ratio: float = 15.25/25.61
    exit_probe_fraction: float = 0.001


def _external_tangent_circle(c1, r1, c2, r2, radius):
    """Solve table 6.1's omitted Pp_SPR_2C_R_zdn by circle intersection."""
    delta = c2-c1
    distance = float(np.linalg.norm(delta))
    if distance == 0:
        _fail('Coincident active-profile edge centres')
    a, b = radius+r1, radius+r2
    along = (a*a-b*b+distance*distance)/(2*distance)
    height2 = a*a-along*along
    if height2 <= 0:
        _fail('Prescribed pressure circle cannot touch both edge circles',
              circle_intersection_height_squared_m2=height2, pressure_radius_m=radius)
    direction = delta/distance
    perpendicular = np.array([-direction[1], direction[0]])
    middle = c1+along*direction
    centres = (middle+math.sqrt(height2)*perpendicular,
               middle-math.sqrt(height2)*perpendicular)
    centre = min(centres, key=lambda c: c[1])
    g = c1+r1*(centre-c1)/np.linalg.norm(centre-c1)
    h = c2+r2*(centre-c2)/np.linalg.norm(centre-c2)
    return centre, g, h


def _point_circle_blend(point, tangent, target_center, target_radius):
    """Circle through point/tangent, internally tangent to target circle.

    Analytic replacement for Pp_Spr_Okp_P, table 6.1 items38–40,45–48.
    The target circle is inside the joining circle; no fitted constants.
    """
    normal = _unit(tangent+math.pi/2)
    delta = target_center-point
    denominator = 2*(float(np.dot(delta, normal))+target_radius)
    if abs(denominator) < 1e-14*max(target_radius, float(np.linalg.norm(delta))):
        _fail('Active circle/point blend is singular', denominator_m=denominator)
    radius = (target_radius**2-float(np.dot(delta, delta)))/denominator
    if radius <= target_radius:
        _fail('Active circle/point blend has wrong internal-tangency branch',
              blend_radius_m=radius, target_radius_m=target_radius)
    center = point-radius*normal
    direction = (target_center-center)/np.linalg.norm(target_center-center)
    contact = center+radius*direction
    return center, radius, contact


def _active_core_geometry(spec: ProfileSpec, options: ActiveProfileOptions | None = None) -> dict:
    """Pressure/concentric geometry and exit probe, independent of source blends."""
    options = options or ActiveProfileOptions()
    if options.pressure_radius_chord_ratio <= 0 or options.exit_probe_fraction <= 0:
        _fail('Active source radius and probe fraction must be positive')
    chord, c1, c2 = _edge_centres(spec)
    radius = chord*options.pressure_radius_chord_ratio
    center, g, h = _external_tangent_circle(c1, spec.leading_radius_m, c2,
                                          spec.trailing_radius_m, radius)
    throat = spec.pitch_m*math.sin(math.radians(spec.outlet_metal_deg))/spec.throat_multiplier
    suction_radius = radius-throat
    if suction_radius <= 0:
        _fail('Concentric suction radius must exceed zero', suction_radius_m=suction_radius,
              pressure_radius_m=radius, source_throat_m=throat)
    source_h = c2-spec.trailing_radius_m*_unit(math.radians(
        spec.outlet_metal_deg+spec.trailing_wedge_deg/2))
    probe_x = source_h[0]-options.exit_probe_fraction*spec.axial_width_m
    radicand = radius**2-(probe_x-center[0])**2
    if not g[0] < probe_x < h[0] or radicand <= 0:
        _fail('Exit probe must lie on the pressure circle, before its edge join',
              probe_x_m=float(probe_x), pressure_end_x_m=float(h[0]))
    radial = np.array([probe_x-center[0], math.sqrt(radicand)])/radius
    join = center+suction_radius*radial
    angle = math.atan2(radial[1], radial[0])-math.pi/2
    # Item46 does NOT join to the inlet edge circle. It fits a circle through
    # the upstream extreme of the translated pressure side, with vertical
    # tangent. Retaining it is necessary to reproduce the source failure.
    inlet_point = np.array([0., c1[1]-spec.pitch_m])
    pressure_inlet = math.atan2((g-center)[1], (g-center)[0])-math.pi/2
    actual_inlet_angle = (pressure_inlet+math.pi/2)/2
    requested_inlet_angle = math.radians(90-spec.inlet_metal_deg)
    inlet_error = math.degrees(actual_inlet_angle-requested_inlet_angle)
    # Item44's attempted direct inlet match, before the replacement of item46.
    normal_g = (g-center)/radius
    direct_inlet_contact = center+suction_radius*normal_g
    return dict(chord_m=chord, leading_center=c1, trailing_center=c2,
                pressure_center=center, pressure_radius_m=radius,
                pressure_start=g, pressure_end=h, source_h=source_h,
                source_h_contact_residual_m=float(np.linalg.norm(h-source_h)),
                source_throat_m=throat, channel_circle_radius_m=throat/2,
                suction_radius_m=suction_radius, exit_probe_x_m=float(probe_x),
                exit_join=join, exit_tangent_angle_rad=angle,
                inlet_point=inlet_point, direct_inlet_contact=direct_inlet_contact,
                inlet_metal_angle_error_deg=inlet_error,
                actual_inlet_metal_deg=90-math.degrees(actual_inlet_angle),
                source_beta_deg=spec.outlet_metal_deg-spec.trailing_wedge_deg/2+spec.back_angle_deg)


def active_profile_construction(spec: ProfileSpec, options: ActiveProfileOptions | None = None) -> dict:
    """Exact full table6.1 source control, including its optional blend branches.

    ``source_h`` follows the printed numerical result (alpha1+gamma/2), not
    item18's conflicting minus sign. Source blend failures remain errors in
    this explicit source reconstruction, but do not constrain the independent
    boundary-preserving builder.
    """
    d = _active_core_geometry(spec, options)
    shifted_c2 = d['trailing_center']-np.array([0., spec.pitch_m])
    exit_center, exit_radius, exit_contact = _point_circle_blend(
        d['exit_join'], d['exit_tangent_angle_rad'], shifted_c2, spec.trailing_radius_m)
    inlet_center, inlet_radius, inlet_contact = _point_circle_blend(
        d['inlet_point'], math.pi/2, d['pressure_center'], d['suction_radius_m'])
    d.update(exit_blend_center=exit_center, exit_blend_radius_m=exit_radius,
             exit_blend_contact=exit_contact, inlet_blend_center=inlet_center,
             inlet_blend_radius_m=inlet_radius, inlet_blend_contact=inlet_contact)
    d['source_blends_evaluated'] = True
    return d


def _build_source_active_profile(spec: ProfileSpec, options: ActiveProfileOptions | None = None) -> SectionProfile:
    """Return table6.1's complete circular candidate, with honest failure flag.

    Source Spa is on the blade below W: translate Spa by +pitch to pair it
    with this profile's W. Leading/trailing circles remain at their original
    coordinates. This avoids a fake open or pitch-shifted contour.
    The p.84 inlet failure is retained in metadata; reject invalid candidates
    before engineering use, even if their contour is closed and tangent.
    """
    d = active_profile_construction(spec, options)
    shift = np.array([0., spec.pitch_m])
    p0, p1, p2, p3 = (d[key]+shift for key in
                      ('inlet_point', 'inlet_blend_contact', 'exit_join', 'exit_blend_contact'))
    suction = CompositeCurve((
        arc_between(d['inlet_blend_center']+shift, p0, p1, clockwise=True),
        arc_between(d['pressure_center']+shift, p1, p2, clockwise=True),
        arc_between(d['exit_blend_center']+shift, p2, p3, clockwise=True),
    ))
    pressure = CompositeCurve((arc_between(d['pressure_center'], d['pressure_start'],
                                           d['pressure_end'], clockwise=True),))
    metadata = {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in d.items()}
    valid = abs(d['inlet_metal_angle_error_deg']) <= 1e-7
    metadata.update(method='vinogradov_active', variant='source', status='EXPERIMENTAL',
                    source='Vinogradov 2010 table6.1 pp.77–86, figures6.1 and items33,43,49,51',
                    construction_valid=valid, pitch_m=spec.pitch_m,
                    construction_failure='' if valid else 'source inlet replacement violates requested metal angle',
                    nondiffusing_channel_verified=False,
                    source_ambiguities=['item18 minus sign conflicts with numerical h',
                                        'item16 beta uses an unprinted gamma',
                                        'item44 direct inlet join explicitly fails; item46 changes inlet condition'])
    return SectionProfile(suction=suction, pressure=pressure,
                          leading_edge=arc_between(d['leading_center'], d['pressure_start'], p0, clockwise=True),
                          trailing_edge=arc_between(d['trailing_center'], p3, d['pressure_end'], clockwise=True),
                          metadata=metadata)


def active_reference_figure_data(
    spec: ProfileSpec, options: ActiveProfileOptions | None = None, count: int = 401,
) -> dict:
    """Plot-ready SI reconstruction of table6.1 items33,43,49,51.

    Plot coordinates are always (axial,circumferential), even where the book
    swaps figure axes. These arrays preserve the source's failed inlet.
    """
    profile = build_active_profile(spec, options, variant='source')
    data = profile.metadata
    center = np.asarray(data['pressure_center'])
    angles = np.linspace(0., 2*math.pi, count)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    pressure_start = np.asarray(data['pressure_start'])
    radial_inlet = (pressure_start-center)/data['pressure_radius_m']
    radial_exit = _unit(data['exit_tangent_angle_rad']+math.pi/2)
    half_width = data['channel_circle_radius_m']
    channel_inlet = pressure_start-half_width*radial_inlet
    channel_exit = center+(data['pressure_radius_m']-half_width)*radial_exit
    contour = profile.contour(count_per_side=count)
    shift = np.array([0., spec.pitch_m])
    return {
        'figure_6_1_constant_width_limit': center+data['suction_radius_m']*unit,
        'table_6_1_item_33_pressure': profile.pressure.sample(count),
        'table_6_1_item_33_channel_inlet_circle': channel_inlet+half_width*unit,
        'table_6_1_item_33_channel_exit_circle': channel_exit+half_width*unit,
        'table_6_1_item_43_exit_blend': profile.suction.segments[2].sample(count)-shift,
        'table_6_1_item_49_inlet_blend': profile.suction.segments[0].sample(count)-shift,
        'table_6_1_item_51_profile': contour,
        'table_6_1_item_51_adjacent_profile': contour+shift,
    }


def _arc_from_tangent(start, end, tangent):
    """Exact directed minor circular arc through two points and one tangent."""
    chord = end-start
    length2 = float(np.dot(chord, chord))
    if length2 <= 1e-28:
        _fail('Constrained active biarc has coincident endpoints', chord_squared_m2=length2)
    normal = np.array([-tangent[1], tangent[0]])
    denominator = 2*float(np.dot(chord, normal))
    if abs(denominator) <= 1e-12*math.sqrt(length2):
        forward = float(np.dot(chord, tangent))
        if forward <= 0:
            _fail('Constrained active straight limit points backwards', forward_distance_m=forward)
        return BezierCurve(np.array([start, end]))
    signed_radius = length2/denominator
    center = start+signed_radius*normal
    return arc_between(center, start, end, clockwise=signed_radius < 0)


def _forward_arc(curve):
    """Reject loops/backtracking analytically, not by coarse sampling."""
    if isinstance(curve, BezierCurve):
        minimum_dx = float(curve.control_points[-1, 0]-curve.control_points[0, 0])
        scale = float(np.linalg.norm(curve.control_points[-1]-curve.control_points[0]))
    else:
        lo, hi = sorted((curve.start_angle, curve.end_angle))
        sweep = curve.end_angle-curve.start_angle
        candidates = [curve.start_angle, curve.end_angle]
        first = math.ceil((lo-math.pi/2)/math.pi)
        last = math.floor((hi-math.pi/2)/math.pi)
        candidates.extend(math.pi/2+k*math.pi for k in range(first, last+1))
        minimum_dx = min(-curve.radius*sweep*math.sin(theta) for theta in candidates)
        scale = curve.radius*abs(sweep)
    if minimum_dx < -1e-10*scale:
        _fail('Constrained active join backtracks axially',
              minimum_axial_parameter_derivative_m=minimum_dx)


def _g1_biarc(start, tangent_start, end, tangent_end):
    """Equal-tangent-distance analytic G1 biarc (numerical design, not book).

    For v=end-start solve 2(1-t0·t1)d²+2v·(t0+t1)d-|v|²=0,
    and join at (start+end+d(t0-t1))/2. Both endpoint tangents are
    enforced exactly. A straight-arc limiting member is represented by a line.
    """
    delta = end-start
    length2 = float(np.dot(delta, delta))
    a = max(0., 2*(1-float(np.dot(tangent_start, tangent_end))))
    b = 2*float(np.dot(delta, tangent_start+tangent_end))
    if a < 1e-12:
        if b <= 0:
            _fail('Constrained active parallel-tangent biarc has no forward root',
                  tangent_projection_m=b)
        distance = length2/b
    else:
        root = math.sqrt(b*b+4*a*length2)
        distance = 2*length2/(b+root) if b >= 0 else (-b+root)/(2*a)
    middle = (start+end+distance*(tangent_start-tangent_end))/2
    first = _arc_from_tangent(start, middle, tangent_start)
    backward = _arc_from_tangent(end, middle, -tangent_end)
    if isinstance(backward, CircularArc):
        second = CircularArc(backward.center, backward.radius,
                             backward.end_angle, backward.start_angle)
    else:
        second = BezierCurve(backward.control_points[::-1])
    for segment in (first, second):
        _forward_arc(segment)
    u = first.evaluate(1, 1)
    v = second.evaluate(0, 1)
    residual = float(np.linalg.norm(u/np.linalg.norm(u)-v/np.linalg.norm(v)))
    if residual > 1e-9:
        _fail('Constrained active biarc G1 join failed', tangent_residual=residual)
    return (first, second)


def build_active_profile(
    spec: ProfileSpec, options: ActiveProfileOptions | None = None, *,
    variant: str = 'constrained',
) -> SectionProfile:
    """Construct an active profile with actual metal and trailing-wedge limits.

    ``variant='source'`` explicitly requests the known-invalid table6.1
    reconstruction. With no options the constrained default solves the
    pressure radius from the requested outlet metal angle, trailing wedge,
    and both exact edge-circle tangencies (table4.1 items37–43). An explicit
    options radius is a constraint, not a hint, and is rejected if incompatible.
    Reflected endpoint tangents and numerical G1 biarcs impose both actual
    metal boundaries without changing the requested trailing wedge.
    """
    if variant == 'source':
        return _build_source_active_profile(spec, options)
    if variant != 'constrained':
        raise ValueError("active profile variant must be 'constrained' or 'source'")
    chord, c1, c2 = _edge_centres(spec)
    _, solved_radius, _, _ = _pressure_from_exit(spec, c1, c2)
    if options is not None:
        requested_radius = options.pressure_radius_chord_ratio*chord
        residual = requested_radius-solved_radius
        if not math.isfinite(residual) or abs(residual) > 1e-10*max(chord, solved_radius):
            _fail('Explicit active pressure radius conflicts with outlet metal/wedge constraints',
                  requested_pressure_radius_m=requested_radius,
                  required_pressure_radius_m=solved_radius,
                  pressure_radius_residual_m=residual,
                  outlet_metal_deg=spec.outlet_metal_deg,
                  trailing_wedge_deg=spec.trailing_wedge_deg)
        constrained_options = options
    else:
        constrained_options = ActiveProfileOptions(pressure_radius_chord_ratio=solved_radius/chord)
    d = _active_core_geometry(spec, constrained_options)
    d['source_blends_evaluated'] = False
    shift = np.array([0., spec.pitch_m])
    center = d['pressure_center']
    g, h = d['pressure_start'], d['pressure_end']
    pressure_inlet = math.atan2((g-center)[1], (g-center)[0])-math.pi/2
    pressure_outlet = math.atan2((h-center)[1], (h-center)[0])-math.pi/2
    desired_inlet = math.radians(90-spec.inlet_metal_deg)
    desired_outlet = math.radians(spec.outlet_metal_deg-90)
    suction_inlet = 2*desired_inlet-pressure_inlet
    suction_outlet = 2*desired_outlet-pressure_outlet
    p0 = d['leading_center']+spec.leading_radius_m*_unit(suction_inlet+math.pi/2)
    p3 = d['trailing_center']+spec.trailing_radius_m*_unit(suction_outlet+math.pi/2)
    p2 = d['exit_join']+shift
    central_center = center+shift
    central_radius = d['suction_radius_m']
    inlet_failure = {}
    try:
        blend_center, blend_radius, p1 = _point_circle_blend(
            p0, suction_inlet, central_center, central_radius)
        inlet_arc = arc_between(blend_center, p0, p1, clockwise=True)
        _forward_arc(inlet_arc)
        if not p0[0] < p1[0] < p2[0]:
            _fail('Constrained active inlet circle joins outside available wall',
                  inlet_contact_x_m=float(p1[0]), exit_contact_x_m=float(p2[0]))
        inlet_segments = (inlet_arc,)
        inlet_method = 'exact tangent circle'
    except ConstructionError as exc:
        inlet_failure = dict(exc.details)
        inlet_failure['reason'] = str(exc)
        p1 = d['direct_inlet_contact']+shift
        if not p0[0] < p1[0] < p2[0]:
            _fail('Constrained active inlet biarc has no ordered source contact',
                  inlet_start_x_m=float(p0[0]), inlet_join_x_m=float(p1[0]),
                  exit_join_x_m=float(p2[0]), single_circle_failure=inlet_failure)
        inlet_tangent = _unit(math.atan2((p1-central_center)[1],
                                       (p1-central_center)[0])-math.pi/2)
        inlet_segments = _g1_biarc(p0, _unit(suction_inlet), p1, inlet_tangent)
        inlet_method = 'equal-tangent-distance G1 biarc'
    central = arc_between(central_center, p1, p2, clockwise=True)
    _forward_arc(central)
    exit_segments = _g1_biarc(p2, _unit(d['exit_tangent_angle_rad']),
                              p3, _unit(suction_outlet))
    suction = CompositeCurve(inlet_segments+(central,)+exit_segments)
    pressure = CompositeCurve((arc_between(center, g, h, clockwise=True),))
    leading = arc_between(d['leading_center'], g, p0, clockwise=True)
    trailing = arc_between(d['trailing_center'], p3, h, clockwise=True)
    if abs(leading.end_angle-leading.start_angle) >= math.pi or abs(
            trailing.end_angle-trailing.start_angle) >= math.pi:
        _fail('Constrained active metal angles require a reversed or overlapping edge',
              leading_sweep_deg=math.degrees(leading.end_angle-leading.start_angle),
              trailing_sweep_deg=math.degrees(trailing.end_angle-trailing.start_angle))
    actual_inlet = math.atan2(*suction.evaluate(0, 1)[::-1])
    actual_outlet = math.atan2(*suction.evaluate(1, 1)[::-1])
    actual_pressure_outlet = math.atan2(*pressure.evaluate(1, 1)[::-1])
    actual_wedge = math.degrees(actual_pressure_outlet-actual_outlet)
    wedge_error = actual_wedge-spec.trailing_wedge_deg
    inlet_error = math.degrees((pressure_inlet+actual_inlet)/2-desired_inlet)
    outlet_error = math.degrees((actual_pressure_outlet+actual_outlet)/2-desired_outlet)
    if max(abs(inlet_error), abs(outlet_error), abs(wedge_error)) > 1e-7:
        _fail('Constrained active metal/wedge enforcement failed',
              inlet_metal_angle_error_deg=inlet_error, outlet_metal_angle_error_deg=outlet_error,
              trailing_wedge_error_deg=wedge_error)
    metadata = {key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in d.items()}
    metadata.update(
        method='vinogradov_active', variant='constrained', status='EXPERIMENTAL',
        source='Vinogradov 2010 tables4.1/6.1 circle geometry; constrained numerical G1 joins',
        numerical_design_choice='equal-tangent-distance biarcs; reflected boundary tangents',
        inlet_join_method=inlet_method, inlet_single_circle_failure=inlet_failure,
        preconstraint_inlet_metal_angle_error_deg=d['inlet_metal_angle_error_deg'],
        pressure_radius_constraint='solved from outlet metal and trailing wedge' if options is None else 'explicit compatible radius',
        solved_pressure_radius_m=solved_radius, actual_trailing_wedge_deg=actual_wedge,
        trailing_wedge_error_deg=wedge_error,
        inlet_metal_angle_error_deg=inlet_error, outlet_metal_angle_error_deg=outlet_error,
        actual_inlet_metal_deg=90-math.degrees((pressure_inlet+actual_inlet)/2),
        actual_outlet_metal_deg=90+math.degrees((pressure_outlet+actual_outlet)/2),
        constrained_inlet_suction_point=p0.tolist(), constrained_exit_suction_point=p3.tolist(),
        construction_valid=True, pitch_m=spec.pitch_m, nondiffusing_channel_verified=False,
        construction_failure='', full_cascade_clearance_verified=False,
    )
    return SectionProfile(suction=suction, pressure=pressure, leading_edge=leading,
                          trailing_edge=trailing, metadata=metadata)

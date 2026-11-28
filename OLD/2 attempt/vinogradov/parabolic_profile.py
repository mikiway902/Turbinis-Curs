"""EXPERIMENTAL Vinogradov (2010), tables 2.1–3.2, pp. 19–46.

Lengths are metres; angles in the spec are degrees. The exact quadratic
replaces the book's sparse-point cubic spline, not its geometric construction.
No flow/metal-angle identity or empirical radius/pitch default is introduced.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import least_squares

from .curves import BezierCurve, CircularArc, CompositeCurve, arc_between, tangent_intersection, nearest_parameter_to_point
from .models import ConstructionError, ProfileSpec, SectionProfile


def parabolic_envelope(start, end, start_angle_rad, end_angle_rad):
    """Exact table 2.2 half-median/envelope parabola (pp. 29–33).

    For tangent intersection B, Q(t)=(1-t)^2 A+2t(1-t)B+t^2 C.
    Q(1/2)=(B+(A+C)/2)/2 and Q'(1/2)=C-A, exactly the
    printed midpoint and parallel-tangent rules, recursively on both halves.
    Tangent angles specify lines, not necessarily forward-oriented vectors.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    try:
        control = tangent_intersection(start, start_angle_rad, end, end_angle_rad)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ConstructionError('Parallel endpoint tangents do not define a parabola',
                                details={'source': 'table 2.2'}) from exc
    return BezierCurve(np.array([start, control, end]))


def _point(center, radius, angle):
    return center + radius * np.array([math.cos(angle), math.sin(angle)])


def build_scheme(spec: ProfileSpec) -> dict:
    """Table 2.1, items 1–41 (pp. 20–28), including circular pressure side.

    Returns SI points A/O1/O2/O2_pitch/O3/g/h/l/f/B/p, dimensional
    chord_m, pressure_radius_m, design_throat_m, and radian tangent angles.
    p is the *prescribed* throat point, not a forced point on the parabola.
    Item 19's undefined beta_2 is read as outlet metal angle: this uniquely
    reproduces printed p=(91.719,42.793) mm.
    """
    b = spec.axial_width_m
    stagger = math.radians(spec.stagger_deg)
    inlet = math.radians(spec.inlet_metal_deg)
    outlet = math.radians(spec.outlet_metal_deg)
    wedge = math.radians(spec.trailing_wedge_deg)
    chord = b / math.sin(stagger)
    slope = -1. / math.tan(stagger)
    r1, r2 = spec.leading_radius_m, spec.trailing_radius_m
    o1 = np.array([r1, spec.origin_y_m + slope*r1])
    o2 = np.array([b-r2, spec.origin_y_m + slope*(b-r2)])
    o2p = o2 + np.array([0., spec.pitch_m])
    beta_h = outlet + wedge/2
    h = _point(o2, r2, math.pi+beta_h)
    delta = h-o1
    denominator = 2*(np.dot(delta, [math.cos(beta_h), math.sin(beta_h)])+r1)
    if denominator <= 0:
        raise ConstructionError('Circular pressure construction has no positive radius',
                                details={'denominator_m': float(denominator)})
    radius = float((np.dot(delta, delta)-r1*r1)/denominator)
    if radius <= 0:
        raise ConstructionError('Circular pressure radius is nonpositive',
                                details={'radius_m': radius})
    o3 = h-radius*np.array([math.cos(beta_h), math.sin(beta_h)])
    theta_g = math.atan2(*(o3-o1)[::-1]) % (2*math.pi)
    g = _point(o1, r1, theta_g)
    pressure_inlet_angle = theta_g-3*math.pi/2
    # Reflect the pressure tangent about the explicit inlet-metal direction.
    inlet_direction = math.pi/2-inlet
    angle_l = 2*inlet_direction-pressure_inlet_angle
    l = _point(o1, r1, angle_l+math.pi/2)
    theta_f = outlet-wedge/2
    f = _point(o2, r2, theta_f)
    angle_f = theta_f+math.pi/2
    parabola = parabolic_envelope(l, f, angle_l, angle_f)
    throat = spec.pitch_m*math.sin(outlet)/spec.throat_multiplier
    beta_p = outlet-wedge/2+math.radians(spec.back_angle_deg)
    p = o2p-(throat+r2)*np.array([math.sin(beta_p), math.cos(beta_p)])
    return {'A': np.array([0., spec.origin_y_m]), 'O1': o1, 'O2': o2,
            'O2_pitch': o2p, 'O3': o3, 'g': g, 'h': h, 'l': l, 'f': f,
            'B': parabola.control_points[1], 'p': p, 'chord_m': chord,
            'pressure_radius_m': radius, 'design_throat_m': throat,
            'inlet_wedge_deg': math.degrees(2*(angle_l-inlet_direction)),
            'angle_l_rad': angle_l, 'angle_f_rad': angle_f,
            'theta_g_rad': theta_g, 'theta_h_rad': math.pi+beta_h,
            'theta_l_rad': angle_l+math.pi/2, 'theta_f_rad': theta_f,
            'parabola': parabola}




def _throat(parabola, scheme, spec):
    """Table 3.1 items 43–47: normal contact with adjacent trailing circle."""
    center = scheme['O2_pitch']
    t=nearest_parameter_to_point(parabola,center)
    p = parabola.evaluate(t)
    tangent = parabola.evaluate(t, 1)
    angle = math.degrees(math.atan2(tangent[1], tangent[0]))
    distance = float(np.linalg.norm(center-p))
    return {'throat_point_m': p.tolist(), 'throat_parameter': t,
            'throat_circle_radius_m': distance,
            'throat_m': distance-spec.trailing_radius_m,
            'back_angle_deg': angle+90-spec.outlet_metal_deg+spec.trailing_wedge_deg/2,
            'throat_contact_is_interior': bool(0 < t < 1)}


def _assemble(spec, scheme, pressure, parabola, method):
    leading = arc_between(scheme['O1'], pressure.evaluate(0), parabola.evaluate(0), clockwise=True)
    trailing = arc_between(scheme['O2'], parabola.evaluate(1), pressure.evaluate(1), clockwise=True)
    metadata = {'method': method, 'status': 'EXPERIMENTAL', 'source': 'Vinogradov 2010, tables 2.1–3.1',
                'chord_m': float(scheme['chord_m']),
                'pressure_radius_m': float(scheme['pressure_radius_m']),
                'inlet_wedge_deg': float(scheme['inlet_wedge_deg']),
                'design_throat_m': float(scheme['design_throat_m']),
                'points_m': {name: scheme[name].tolist() for name in
                             ('A','O1','O2','O2_pitch','O3','g','h','l','f','B','p')},
                'representation': 'exact quadratic half-median envelope, not book cubic spline',
                **_throat(parabola, scheme, spec)}
    return SectionProfile(CompositeCurve((parabola,)), pressure, leading, trailing, metadata)


def build_parabolic_profile(spec: ProfileSpec) -> SectionProfile:
    """Table 3.1 (pp. 34–43): closed parabolic suction/circular pressure.

    The actual throat/back angle are outputs. As p34 states, the initial
    prescribed throat point generally does not lie on the constructed parabola.
    Radius and pitch are explicit dimensional inputs rather than correlations.
    """
    scheme = build_scheme(spec)
    pressure = CompositeCurve((arc_between(scheme['O3'], scheme['g'], scheme['h'], clockwise=True),))
    return _assemble(spec, scheme, pressure, scheme['parabola'], 'vinogradov_parabolic')


def _arc_from_tangent(start, end, tangent):
    delta = end-start
    normal = np.array([-tangent[1], tangent[0]])
    denominator = 2*np.dot(delta, normal)
    if abs(denominator) < 1e-13*np.linalg.norm(delta):
        raise ConstructionError('Two-circle pressure degenerates to a straight segment',
                                details={'normal_projection_m': float(denominator/2)})
    signed_radius = np.dot(delta, delta)/denominator
    center = start+signed_radius*normal
    return arc_between(center, start, end, clockwise=bool(signed_radius < 0))


def build_two_circle_pressure_profile(spec: ProfileSpec, *, inlet_wedge_deg: float,
                                      tangent_distance_ratio: float = 1.) -> SectionProfile:
    """Fig. 2.3 (p19), two-circle pressure alternative, EXPERIMENTAL.

    The book gives no join/radius equations or numerical example. This explicit
    G1 biarc reconstruction requires an inlet wedge and positive tangent-length
    ratio (software choice, not an empirical/source correlation). No automatic
    adjustment of either input is made to force a desired thickness.
    """
    if not 0 < inlet_wedge_deg < 180 or not math.isfinite(tangent_distance_ratio) or tangent_distance_ratio <= 0:
        raise ConstructionError('Biarc needs an admissible inlet wedge and positive finite ratio', details={})
    scheme = build_scheme(spec)
    inlet_direction = math.pi/2-math.radians(spec.inlet_metal_deg)
    angle_g = inlet_direction-math.radians(inlet_wedge_deg)/2
    angle_l = inlet_direction+math.radians(inlet_wedge_deg)/2
    g = _point(scheme['O1'], spec.leading_radius_m, angle_g-math.pi/2)
    l = _point(scheme['O1'], spec.leading_radius_m, angle_l+math.pi/2)
    h = scheme['h']
    tangent0 = np.array([math.cos(angle_g), math.sin(angle_g)])
    angle_h = math.radians(spec.outlet_metal_deg+spec.trailing_wedge_deg/2)-math.pi/2
    tangent1 = np.array([math.cos(angle_h), math.sin(angle_h)])
    delta = h-g
    ratio = tangent_distance_ratio
    a = 2*ratio*(1-np.dot(tangent0,tangent1))
    b = 2*np.dot(delta, tangent0+ratio*tangent1)
    c = -np.dot(delta,delta)
    if abs(a) < 1e-14:
        distance = -c/b if b > 0 else -1.
    else:
        distance = -2*c/(b+math.sqrt(b*b-4*a*c))
    if distance <= 0 or not math.isfinite(distance):
        raise ConstructionError('No positive biarc tangent distance', details={'distance_m': float(distance)})
    d0, d1 = distance, ratio*distance
    join = (d1*(g+d0*tangent0)+d0*(h-d1*tangent1))/(d0+d1)
    first = _arc_from_tangent(g, join, tangent0)
    backwards = _arc_from_tangent(h, join, -tangent1)
    second = CircularArc(backwards.center, backwards.radius, backwards.end_angle, backwards.start_angle)
    parabola = parabolic_envelope(l, scheme['f'], angle_l, scheme['angle_f_rad'])
    scheme.update(g=g, l=l, B=parabola.control_points[1], inlet_wedge_deg=inlet_wedge_deg)
    result = _assemble(spec, scheme, CompositeCurve((first,second)), parabola,
                       'vinogradov_parabolic_two_circle_pressure')
    result.metadata.update({'source': 'Vinogradov 2010, fig. 2.3, pp.18–19; explicit G1 biarc reconstruction',
                            'biarc_tangent_distance_ratio': float(ratio),
                            'biarc_join_m': join.tolist(),
                            'pressure_radii_m': [float(first.radius), float(second.radius)],
                            'biarc_join_is_software_choice': True})
    return result


def maximum_thickness(profile: SectionProfile) -> dict:
    """Table 3.2 common-normal maximum thickness (pp. 44–46), in SI.

    Solves both normal equations on every analytic side-segment pair, with
    multiple starts and rejects reversed/outside contacts. Unlike the book's
    seven-sample linear interpolation of angular mismatch, contacts are solved
    directly. This is a common-normal thickness, not a guarantee that its circle
    is globally inscribed for nonconvex/arbitrary profiles; metadata says so.
    """
    candidates = []
    scale = max(float(np.ptp(profile.contour(), axis=0).max()), 1e-12)
    for suction in profile.suction.segments:
        for pressure in profile.pressure.segments:
            def equations(parameters):
                ts, tp = parameters
                s, p = suction.evaluate(ts), pressure.evaluate(tp)
                ds, dp = suction.evaluate(ts,1), pressure.evaluate(tp,1)
                delta = s-p
                return np.array([np.dot(delta,ds),np.dot(delta,dp)])/scale**2
            for ts in (.1,.3,.5,.7,.9):
                for tp in (.1,.3,.5,.7,.9):
                    solved = least_squares(equations, [ts,tp], bounds=(0.,1.),
                                           xtol=1e-13, ftol=1e-13, gtol=1e-13)
                    x = solved.x
                    if not solved.success or not np.all((x > 1e-8) & (x < 1-1e-8)):
                        continue
                    residual = float(np.linalg.norm(equations(x)))
                    if residual > 1e-8:
                        continue
                    s, p = suction.evaluate(x[0]), pressure.evaluate(x[1])
                    delta = s-p
                    ds, dp = suction.evaluate(x[0],1), pressure.evaluate(x[1],1)
                    # suction is left of the inlet->outlet pressure direction.
                    if dp[0]*delta[1]-dp[1]*delta[0] <= 0 or np.dot(ds,dp) <= 0:
                        continue
                    candidates.append((float(np.linalg.norm(delta)),s,p,residual))
    if not candidates:
        raise ConstructionError('No interior common-normal thickness contact found',
                                details={'source': 'table 3.2', 'searched_segment_pairs':
                                         len(profile.suction.segments)*len(profile.pressure.segments)})
    diameter, suction, pressure, residual = max(candidates, key=lambda candidate: candidate[0])
    return {'diameter_m': diameter, 'radius_m': diameter/2,
            'suction_contact_m': suction.tolist(), 'pressure_contact_m': pressure.tolist(),
            'center_m': ((suction+pressure)/2).tolist(),
            'normal_equation_residual': residual, 'status': 'EXPERIMENTAL',
            'source': 'Vinogradov 2010, table 3.2, pp.44–46',
            'globally_inscribed_circle_verified': False}

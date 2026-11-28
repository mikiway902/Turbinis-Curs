"""Vinogradov 2010 pp.4–14: definitions and EMPIRICAL starting guidance.

Recommendations are not hard physical constraints. Angles in degrees relative
 to the front, length in metres. No new empirical incidence/deviation law.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import least_squares
from .curves import nearest_parameter_to_point
from .cascade_throat import measured_back_angle


def optimal_relative_pitch(inlet_flow_deg, outlet_flow_deg, inlet_metal_deg,
                           effective_outlet_deg, relative_thickness):
    """Dyshelevsky p.11: .55*[180*K/(180-(beta1+beta2))]**(1/3)*(1-c).

    K=sin(beta1')/sin(beta2_eff). Flow angles in the denominator are distinct
    from metal/effective angles. Formula checked directly against PDF image.
    Applicability: empirical subsonic preliminary design, not a loss model.
    """
    a, b, am, be, c = map(float, (inlet_flow_deg,outlet_flow_deg,inlet_metal_deg,
                                 effective_outlet_deg,relative_thickness))
    if not all(math.isfinite(x) for x in (a,b,am,be,c)) or not 0 <= c < 1:
        raise ValueError('invalid pitch correlation input')
    if not 0 < a < 180 or not 0 < b < 180 or a+b >= 180 or not 0 < am < 180 or not 0 < be < 90:
        raise ValueError('pitch correlation outside positive-turning subsonic domain')
    K = math.sin(math.radians(am))/math.sin(math.radians(be))
    return .55*(180*K/(180-a-b))**(1/3)*(1-c)


def pitch_recommendation(row, active=False):
    """p.12 mean-radius starting ranges, EMPIRICAL, not universal bounds."""
    if row == 'SA':
        return (.8, 1.)
    if row == 'RK':
        return (.5, .75) if active else (.6, .8)
    raise ValueError('row must be SA or RK')


def count_candidates(diameter_m, chord_m, relative_pitch, row, *, minimum=3,
                     half_window=4, previous=0):
    """Z=pi*D/t; recompute real t for every candidate (p.6 geometric identity).

    Odd SA/even RK and increasing counts are EXISTING project decisions, not
    Vinogradov laws. One accepted count is used for all sections of a row.
    """
    if min(diameter_m,chord_m,relative_pitch) <= 0:
        raise ValueError('diameter, chord and relative pitch must be positive')
    if row not in ('SA','RK'):
        raise ValueError('invalid row')
    target = math.pi*diameter_m/(chord_m*relative_pitch)
    center = int(round(target))
    lower = max(minimum,previous+1,center-half_window)
    upper = max(lower+max(4,previous),center+half_window+1)
    parity = 1 if row == 'SA' else 0
    counts = [z for z in range(lower,upper+1) if z%2 == parity]
    counts=sorted(counts,key=lambda z:abs(z-target))[:max(1,half_window+1)]
    return [dict(count=z,pitch_m=math.pi*diameter_m/z,
                 relative_pitch=math.pi*diameter_m/z/chord_m,
                 target_count=target) for z in sorted(counts,key=lambda z:abs(z-target))]


def thickness_recommendation(row, span_fraction):
    """p.13 starting thickness fractions; interpolation is a design assumption."""
    if not 0 <= span_fraction <= 1:
        raise ValueError('span fraction outside [0,1]')
    if row == 'SA':
        return (.10,.10)
    if row != 'RK':
        raise ValueError('invalid row')
    return (float(np.interp(span_fraction,[0,.5,1],[.20,.10,.04])),
            float(np.interp(span_fraction,[0,.5,1],[.30,.15,.05])))


def area_distribution(span_fraction, root_area_m2, tip_area_m2, law='intermediate', blend=.5):
    """p.13 suggests laws but supplies no unique intermediate equation.

    Linear A=(1-s)Ar+sAt; exponential A=Ar*(At/Ar)**s; intermediate is an
    explicit convex blend. These are software parameterizations of AREA,
    never prescriptions for maximum thickness. Endpoints are exact.
    """
    s=np.asarray(span_fraction,float)
    if root_area_m2 <= 0 or tip_area_m2 <= 0 or np.any((s<0)|(s>1)) or not 0 <= blend <= 1:
        raise ValueError('invalid area law input')
    linear=(1-s)*root_area_m2+s*tip_area_m2
    exponential=root_area_m2*(tip_area_m2/root_area_m2)**s
    if law == 'linear': return linear
    if law == 'exponential': return exponential
    if law == 'intermediate': return (1-blend)*linear+blend*exponential
    raise ValueError('unknown area law')


def medial_line(profile, samples=81):
    """p.4/r.1.1 equidistant centers, solved from two analytic wall normals.

    This is a numerical approximation to the continuous medial branch.
    Contacts must lie inside each curve parameter interval; residuals are
    returned, not suppressed. End-circle centers close the physical branch.
    """
    centers=[]; radii=[]; residuals=[]; contacts=[]
    from .channel_convergence import _inside,_segment_distance
    polygon=profile.contour(401)
    clearance_tolerance=max(np.ptp(polygon,axis=0).max()*1e-5,1e-10)
    for t in np.linspace(.001,.999,samples):
        q=profile.suction.evaluate(t)
        dq=profile.suction.evaluate(t,1)
        n=np.array([dq[1],-dq[0]])/np.linalg.norm(dq)
        def residual(vr):
            v,r=vr
            p=profile.pressure.evaluate(v)
            dp=profile.pressure.evaluate(v,1)
            np_=np.array([-dp[1],dp[0]])/np.linalg.norm(dp)
            return (q+r*n)-(p+r*np_)
        length=max(np.linalg.norm(profile.trailing_edge.center-profile.leading_edge.center),1e-6)
        fit=least_squares(residual,[t,length*.05],bounds=([0,0],[1,length]),
                          xtol=1e-11,ftol=1e-11,gtol=1e-11)
        err=float(np.linalg.norm(residual(fit.x)))
        if fit.success and err <= length*1e-6:
            center=q+fit.x[1]*n
            if not _inside(center,polygon):
                continue
            if _segment_distance(center,polygon[:-1],polygon[1:]).min()<fit.x[1]-clearance_tolerance:
                continue
            centers.append(q+fit.x[1]*n);radii.append(fit.x[1]);residuals.append(err)
            contacts.append([t,fit.x[0]])
    return dict(centers_m=np.asarray(centers).reshape(-1,2),radii_m=np.asarray(radii),
                residuals_m=np.asarray(residuals),contacts=np.asarray(contacts).reshape(-1,2),
                requested_samples=samples,accepted_samples=len(centers))


def exit_back_geometry(profile,spec):
    """Table3.1 items43–47, p.43: contact towards the adjacent trailing circle.

    This exit construction is distinct from a global minimum between the
    two complete channel walls, which may occur upstream.
    """
    center=profile.trailing_edge.center+np.array([0.,spec.pitch_m])
    parameter=nearest_parameter_to_point(profile.suction,center)
    contact=profile.suction.evaluate(parameter);tangent=profile.suction.evaluate(parameter,1)
    normal=np.array([-tangent[1],tangent[0]])
    return dict(back_angle_deg=measured_back_angle(normal,spec.outlet_metal_deg,spec.trailing_wedge_deg),
        suction_parameter=parameter,suction_contact_m=contact,
        exit_trailing_circle_gap_m=float(np.linalg.norm(center-contact)-profile.trailing_edge.radius))


def geometric_parameters(profile, spec, *, samples=81):
    """Measured definitions pp.4–7 plus nominal construction chord table2.1.

    Camber uses medial-circle centers, not pointwise parameter averaging.
    Source 'b' in tables2.1+ means axial width, whereas p.5 'b' means chord.
    Both nominal construction chord and measured center chord are returned.
    """
    medial=medial_line(profile,samples)
    centers=medial['centers_m']; radii=medial['radii_m']
    if len(centers)<3:
        raise ValueError('no resolved interior medial branch')
    p0=profile.leading_edge.center;p1=profile.trailing_edge.center
    direction=p1-p0; chord=float(np.linalg.norm(direction));e=direction/chord
    normal=np.array([-e[1],e[0]])
    station=(centers-p0)@e/chord
    camber=(centers-p0)@normal
    imax=int(np.argmax(radii));ifmax=int(np.argmax(np.abs(camber)))
    bend_in=180-spec.inlet_metal_deg-spec.stagger_deg
    bend_out=spec.stagger_deg-spec.outlet_metal_deg
    def wedge(t):
        a=profile.suction.evaluate(t,1);b=profile.pressure.evaluate(t,1)
        return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))
    return dict(chord_m=chord,construction_chord_m=spec.axial_width_m/math.sin(math.radians(spec.stagger_deg)),
                axial_width_m=spec.axial_width_m,maximum_thickness_m=2*float(radii[imax]),
                relative_thickness=2*float(radii[imax])/chord,
                thickness_location=float(station[imax]),maximum_camber_m=float(abs(camber[ifmax])),
                camber_location=float(station[ifmax]),stagger_deg=spec.stagger_deg,
                inlet_bend_deg=bend_in,outlet_bend_deg=bend_out,
                leading_radius_m=profile.leading_edge.radius,trailing_radius_m=profile.trailing_edge.radius,
                incidence_deg=spec.incidence_deg,deviation_deg=spec.deviation_deg,
                flow_turn_deg=180-(spec.inlet_metal_deg-spec.incidence_deg)-(spec.outlet_metal_deg+spec.deviation_deg),
                leading_wedge_deg=wedge(0),trailing_wedge_deg=wedge(1),
                requested_trailing_wedge_deg=spec.trailing_wedge_deg,
                back_angle_deg=exit_back_geometry(profile,spec)['back_angle_deg'],
                requested_back_angle_deg=spec.back_angle_deg,
                pitch_m=spec.pitch_m,solidity=chord/spec.pitch_m,
                medial=medial)

"""Quantified gates; geometry alone never proves aerodynamic efficiency.

Vinogradov pp.8–14: self/neighbor intersection, smooth channel, curvature.
Numerical tolerances below are explicit software accuracy, not book limits.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from .curves import curvature,BezierCurve,CircularArc,CompositeCurve,nearest_parameter_to_point
import math


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None
    limit: float | str
    passed: bool
    classification: str
    source: str
    units: str = '1'
    note: str = ''

    def to_dict(self): return asdict(self)


def _cross(a,b): return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def intersections(a,b=None):
    """Proper crossing and collinear overlap for finite sampled polylines.

    Returns segment index pairs, skipping only connected neighboring segments
    of the same closed contour. Discretization is assessed on a refined mesh.
    """
    a=np.asarray(a,float); same=b is None;b=a if same else np.asarray(b,float)
    p=a[:-1,None,:];r=(a[1:]-a[:-1])[:,None,:]
    q=b[None,:-1,:];s=(b[1:]-b[:-1])[None,:,:]
    det=_cross(r,s);delta=q-p
    scale=max(float(np.max(np.ptp(a,axis=0))),float(np.max(np.ptp(b,axis=0))),1e-6)
    eps=1e-11*scale**2
    valid=np.abs(det)>eps
    tt=np.divide(_cross(delta,s),det,out=np.zeros_like(det),where=valid)
    uu=np.divide(_cross(delta,r),det,out=np.zeros_like(det),where=valid)
    hit=valid&(tt>=-1e-10)&(tt<=1+1e-10)&(uu>=-1e-10)&(uu<=1+1e-10)
    col=(~valid)&(np.abs(_cross(delta,r))<=eps)
    # Ignore degenerate zero-length closure segments; validate joints separately.
    rr=np.sum(r*r,axis=-1)
    t0=np.divide(np.sum(delta*r,axis=-1),rr,out=np.zeros_like(det),where=rr>eps)
    t1=t0+np.divide(np.sum(s*r,axis=-1),rr,out=np.zeros_like(det),where=rr>eps)
    overlap=(np.minimum(np.maximum(t0,t1),1)-np.maximum(np.minimum(t0,t1),0))>1e-10
    hit |= col&overlap&(rr>eps)
    if same:
        i,j=np.indices(hit.shape);n=len(a)-1
        hit &= (j>i+1)&~((i==0)&(j==n-1))
    return np.argwhere(hit)


def _axial_bounds(curve):
    """Analytic extrema; coarse initialization must not miss a short overhang."""
    if isinstance(curve,CompositeCurve):
        bounds=[_axial_bounds(segment) for segment in curve.segments]
        return min(v[0] for v in bounds),max(v[1] for v in bounds)
    if isinstance(curve,CircularArc):
        lo,hi=sorted((curve.start_angle,curve.end_angle))
        angles=[lo,hi,*[k*math.pi for k in range(math.ceil(lo/math.pi),math.floor(hi/math.pi)+1)]]
        x=curve.center[0]+curve.radius*np.cos(angles)
    elif isinstance(curve,BezierCurve):
        P=np.polynomial.Polynomial
        degree=len(curve.control_points)-1
        polynomial=P([0.])
        for i,point in enumerate(curve.control_points):
            polynomial+=point[0]*math.comb(degree,i)*P([0.,1.])**i*P([1.,-1.])**(degree-i)
        roots=polynomial.deriv().roots()
        parameters=[0.,1.,*[float(root.real) for root in roots
            if abs(root.imag)<1e-12 and 0<root.real<1]]
        x=polynomial(parameters)
    else:
        raise TypeError('Axial envelope requires an analytic supported curve')
    return float(min(x)),float(max(x))


def validate_profile(profile, pitch_m, *, samples=201, relative_tolerance=1e-7):
    contour=profile.contour(samples)
    chord=np.linalg.norm(profile.trailing_edge.center-profile.leading_edge.center)
    tol=max(chord*relative_tolerance,1e-12)
    ordered=(*profile.suction.segments,profile.trailing_edge,
             *tuple(reversed(profile.pressure.segments)),profile.leading_edge)
    # Reverse pressure segment parameter direction without changing geometry.
    orientations=([1]*len(profile.suction.segments)+[1]+[-1]*len(profile.pressure.segments)+[1])
    gaps=[];angles=[];jumps=[]
    for i,(a,sa) in enumerate(zip(ordered,orientations)):
        b=ordered[(i+1)%len(ordered)];sb=orientations[(i+1)%len(ordered)]
        ta=1 if sa==1 else 0;tb=0 if sb==1 else 1
        gaps.append(float(np.linalg.norm(a.evaluate(ta)-b.evaluate(tb))))
        da=a.evaluate(ta,1)*sa;db=b.evaluate(tb,1)*sb
        cosine=np.dot(da,db)/(np.linalg.norm(da)*np.linalg.norm(db))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine,-1,1)))))
        jumps.append(float(abs(curvature(a,ta)*sa-curvature(b,tb)*sb)))
    neighbor=contour+np.array([0,pitch_m])
    crossing=intersections(contour);adjacent=intersections(contour,neighbor)
    # Signed area catches collapsed contours, while intersection check rejects folds.
    area=abs(np.sum(_cross(contour[:-1],contour[1:]))/2)
    axial_start=float(profile.leading_edge.center[0]-profile.leading_edge.radius)
    axial_end=float(profile.trailing_edge.center[0]+profile.trailing_edge.radius)
    bounds=[_axial_bounds(c) for c in (profile.suction,profile.pressure,profile.leading_edge,profile.trailing_edge)]
    axial_excess=max(0.,axial_start-min(v[0] for v in bounds),
                     max(v[1] for v in bounds)-axial_end)
    checks=[Check('closed joints',max(gaps),tol,max(gaps)<=tol,'numerical','Vinogradov pp.18–19','m'),
            Check('continuous tangents',max(angles),.01,max(angles)<=.01,'numerical','Vinogradov pp.8–9','deg'),
            Check('positive area',area,0.,area>tol**2,'physical','Vinogradov pp.4–5','m2'),
            Check('self intersections',float(len(crossing)),0.,len(crossing)==0,'physical','Vinogradov p.9'),
            Check('neighbor intersections',float(len(adjacent)),0.,len(adjacent)==0,'physical','Vinogradov p.9'),
            Check('axial envelope boundary',axial_excess,tol,axial_excess<=tol,
                  'numerical','Prescribed axial width between leading/trailing outer edge planes','m')]
    return dict(checks=checks,passed=all(c.passed for c in checks),
                joint_gaps_m=gaps,joint_tangent_angles_deg=angles,curvature_jumps_1_m=jumps,
                self_intersections=crossing,neighbor_intersections=adjacent,
                numerical_samples_per_side=samples,
                warning='Finite mesh intersections, not a continuum proof; repeat at 2x resolution. Curvature jumps reported, not smoothed.')


def _thickness_boundary(profile, tolerance):
    """Linear interpolation with ||C-L|| <= max||C''||/(8*n*n)."""
    def pieces(curve):
        if isinstance(curve,CompositeCurve):
            return [part for segment in curve.segments for part in pieces(segment)]
        if not isinstance(curve,(BezierCurve,CircularArc)):
            raise TypeError('Thickness bounds require Bezier, circular or composite boundaries')
        return [curve]

    ordered=([(c,False) for c in pieces(profile.suction)]
             +[(c,False) for c in pieces(profile.trailing_edge)]
             +[(c,True) for c in reversed(pieces(profile.pressure))]
             +[(c,False) for c in pieces(profile.leading_edge)])
    samples=[];error=0.
    for curve,reverse in ordered:
        if isinstance(curve,CircularArc):
            second=curve.radius*(curve.end_angle-curve.start_angle)**2
        else:
            degree=len(curve.control_points)-1
            second=(degree*(degree-1)*float(np.max(np.linalg.norm(
                np.diff(curve.control_points,n=2,axis=0),axis=1))) if degree>1 else 0.)
        count=max(1,math.ceil(math.sqrt(second/(8*tolerance))))
        error=max(error,second/(8*count*count))
        points=curve.evaluate(np.linspace(0.,1.,count+1))
        samples.append(points[::-1] if reverse else points)
    magnitude=max(float(np.max(np.abs(p))) for p in samples)
    roundoff=128*np.finfo(float).eps*max(magnitude,1e-12)
    gaps=[np.linalg.norm(samples[i][-1]-samples[(i+1)%len(samples)][0])
          for i in range(len(samples))]
    if max(gaps)>roundoff:
        raise ValueError('Largest-circle bounds require a closed analytic contour')
    polygon=np.vstack([p[:-1] for p in samples])
    return polygon,float(error+max(gaps)+roundoff)


def validate_prescribed_thickness(profile, *, diameter_tolerance_m=1e-6,
                                 boundary_tolerance_m=None, max_cells=20001):
    """Optional, bounded largest-inscribed-circle final analysis (SI).

    A best-first rectangle subdivision maximizes the polygon's signed distance,
    a 1-Lipschitz function. Analytic second-derivative interpolation bounds then
    enclose the actual curved-boundary maximum, not just sampled medial points.
    Bounds assume a simple closed contour and its topology-preserving polygonal
    approximation; validate_profile remains the separate, cheap geometry gate.
    Floating-point arithmetic is guarded, not certified interval arithmetic.
    Budget exhaustion returns an unresolved interval, never a global proof.
    """
    import heapq
    if boundary_tolerance_m is None:
        boundary_tolerance_m=diameter_tolerance_m/16
    if (not np.isfinite([diameter_tolerance_m,boundary_tolerance_m]).all()
            or diameter_tolerance_m<=0 or boundary_tolerance_m<=0
            or 4*boundary_tolerance_m>=diameter_tolerance_m):
        raise ValueError('Positive tolerances require 4*boundary_tolerance_m < diameter_tolerance_m')
    if not isinstance(max_cells,int) or max_cells<1:
        raise ValueError('max_cells must be a positive integer')
    polygon,boundary_error=_thickness_boundary(profile,boundary_tolerance_m)
    start=polygon;end=np.roll(polygon,-1,axis=0);delta=end-start
    length2=np.sum(delta*delta,axis=1)
    if np.any(length2==0):
        raise ValueError('Largest-circle bounds require nondegenerate boundary segments')

    def signed_distance(point):
        offset=point-start
        parameter=np.clip(np.sum(offset*delta,axis=1)/length2,0.,1.)
        distance=float(np.min(np.linalg.norm(offset-parameter[:,None]*delta,axis=1)))
        # Half-open horizontal ray: horizontal segments never divide by zero.
        crossing=(start[:,1]>point[1])!=(end[:,1]>point[1])
        selected=np.flatnonzero(crossing)
        x=start[selected,0]+(point[1]-start[selected,1])*delta[selected,0]/delta[selected,1]
        inside=np.count_nonzero(x>point[0])%2
        return distance if inside else -distance

    lo=np.min(polygon,axis=0);hi=np.max(polygon,axis=0)
    if np.any(hi<=lo):
        raise ValueError('Largest-circle bounds require positive contour area')
    prescribed=profile.metadata.get('prescribed_thickness_m')
    requested_center=profile.metadata.get('thickness_center')
    seeds=[(lo+hi)/2,np.mean(polygon,axis=0)]
    if requested_center is not None:
        requested_center=np.asarray(requested_center,float)
        if requested_center.shape!=(2,) or not np.isfinite(requested_center).all():
            raise ValueError('thickness_center must be a finite axial/circumferential point')
        seeds.append(requested_center)
    best_center=max(seeds,key=signed_distance).copy()
    best=signed_distance(best_center)
    local_signed=(signed_distance(requested_center) if requested_center is not None else None)
    local_lower=(2*max(0.,local_signed-boundary_error) if local_signed is not None else None)
    local_upper=(2*max(0.,local_signed+boundary_error) if local_signed is not None else None)
    heap=[];evaluations=0

    def push(center,half):
        nonlocal evaluations,best,best_center
        distance=signed_distance(center)
        if distance>best:
            best=distance;best_center=center.copy()
        evaluations+=1
        upper=distance+float(np.linalg.norm(half))
        heapq.heappush(heap,(-upper,evaluations,center,half))

    push((lo+hi)/2,(hi-lo)/2)
    def needs_refinement():
        lower=2*max(0.,best-boundary_error)
        upper=2*max(0.,-heap[0][0]+boundary_error)
        # Resolve the prescribed centre too: a different tied maximum is valid.
        undecided=(prescribed is not None and local_lower is not None
                   and lower<=local_upper+diameter_tolerance_m
                   and upper>local_lower+diameter_tolerance_m)
        return upper-lower>diameter_tolerance_m or undecided

    while needs_refinement():
        if evaluations+4>max_cells:
            break
        _,_,center,half=heapq.heappop(heap)
        half=half/2
        for sx,sy in ((-1,-1),(-1,1),(1,-1),(1,1)):
            push(center+half*np.array([sx,sy]),half)
    lower=2*max(0.,best-boundary_error)
    upper=2*max(0.,-heap[0][0]+boundary_error)
    error=max(0.,upper-lower)
    converged=error<=diameter_tolerance_m
    boundaries=(profile.suction,profile.pressure,profile.leading_edge,profile.trailing_edge)

    def analytic_diameter(center):
        return 2*min(float(np.linalg.norm(curve.evaluate(
            nearest_parameter_to_point(curve,center))-center)) for curve in boundaries)

    measured=analytic_diameter(best_center) if best>boundary_error else 0.
    source='Vinogradov 2010 pp.4–5; table 4.1 pp.61,69–70; table 5.1 pp.75–76'
    checks=[Check('global maximum thickness convergence',error,diameter_tolerance_m,
                  converged,'numerical',source,'m',
                  'Conditional curved-boundary diameter enclosure; no interval-arithmetic certification.')]
    local=None;status='not_requested'
    if prescribed is not None and requested_center is not None:
        prescribed=float(prescribed)
        if not math.isfinite(prescribed) or prescribed<=0:
            raise ValueError('prescribed_thickness_m must be finite and positive')
        local=analytic_diameter(requested_center) if local_signed>boundary_error else 0.
        local_matches=local_signed>boundary_error and abs(local-prescribed)<=diameter_tolerance_m
        if not local_matches or lower>local_upper+diameter_tolerance_m:
            status='disproved'
        elif upper-local_lower<=diameter_tolerance_m:
            status='within_tolerance'
        else:
            status='unresolved'
        checks.extend([
            Check('prescribed circle is inscribed',abs(local-prescribed),
                  diameter_tolerance_m,local_matches,'source',source,'m',
                  'Nearest distance to all actual boundaries at metadata thickness_center.'),
            Check('prescribed circle attains global maximum',max(0.,lower-local_upper),
                  diameter_tolerance_m,status=='within_tolerance','source',source,'m',
                  'Status: '+status+'. Tied maxima are accepted; axial centre is not a projected chord coordinate.')])
    return dict(maximum_diameter_m=measured,maximum_center_m=best_center.tolist(),
                diameter_lower_m=lower,diameter_upper_m=upper,diameter_error_m=error,
                converged=converged,diameter_tolerance_m=diameter_tolerance_m,
                boundary_error_m=boundary_error,boundary_segments=len(polygon),
                cells_evaluated=evaluations,max_cells=max_cells,
                prescribed_diameter_m=prescribed,prescribed_center_m=(
                    requested_center.tolist() if requested_center is not None else None),
                prescribed_center_diameter_m=local,prescribed_status=status,
                checks=checks,passed=all(check.passed for check in checks),
                assumptions=('Simple closed analytic contour; topology-preserving interpolation. '
                             'Second-derivative and 1-Lipschitz bounds in floating point, '
                             'not a formal interval proof. Diameter bounds do not bound centre location.'))


def stage_balance_checks(stage):
    """Independent meanline mass, Euler and energy evaluation, no core changes."""
    s=stage
    mass1=np.pi*s['D_c']*s['l_c']*s['c1a']/s['v1']
    mass2=np.pi*s['D_л']*s['l_л']*s['c2a']/s['v2']
    # Existing core c2u is magnitude opposite shaft rotation, not signed c_theta.
    euler=s['u1']*s['c1u']+s['u2']*s['c2u']
    energy=s['cp']*(s['T0']-s['T2'])-s['c2']**2/2
    rows=[]
    for name,value in [('inlet mass',mass1/s['G_c']-1),('outlet mass',mass2/s['G_л']-1),
                       ('Euler',euler/s['L_u']-1),('energy',energy/s['L_u']-1),
                       ('inlet triangle',(s['c1a']**2+(s['c1u']-s['u1'])**2)/s['w1']**2-1),
                       ('outlet triangle',(s['c2a']**2+(s['c2u']+s['u2'])**2)/s['w2']**2-1)]:
        rows.append(Check(name,float(abs(value)),1e-6,abs(value)<1e-6,'numerical',
                          'Existing Mikhaltsev stage, points63,82,84,95,98,101,126'))
    return rows

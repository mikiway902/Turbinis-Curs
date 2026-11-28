"""Coupled profile/cascade evaluation, Vinogradov pp.9,11–14,18,52–57.

All empirical guidance remains separately reported from physical checks.
Subsonic construction policy is a design choice, never a universal flow law.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from .models import ProfileSpec, ConstructionError
from .parabolic_profile import build_parabolic_profile,build_two_circle_pressure_profile
from .double_parabolic_profile import build_prescribed_thickness_profile,build_double_parabolic_profile
from .active_profile import build_active_profile
from .validation import Check,validate_profile
from .blade_geometry import geometric_parameters,thickness_recommendation,pitch_recommendation


METHODS={'parabolic':build_parabolic_profile,'prescribed_thickness':build_prescribed_thickness_profile,
         'double_parabolic':build_double_parabolic_profile,'active':build_active_profile}


def select_method(section,*,require_thickness=True,active_reaction=.05,active_angle_difference_deg=10.):
    """p.77 active root allowed; thresholds are explicit search policy only."""
    if max(section.inlet.mach(section.frame_inlet_speed_m_s),section.outlet.mach(section.frame_speed_m_s))>=1:
        raise ConstructionError('Subsonic Vinogradov recommendation not applicable to trans/supersonic section',
            {'Mach_in':section.inlet.mach(section.frame_inlet_speed_m_s),'Mach_out':section.outlet.mach(section.frame_speed_m_s)})
    if section.row=='RK' and (section.reaction<=active_reaction or
        abs(section.flow_inlet_deg-section.flow_outlet_deg)<=active_angle_difference_deg):return 'active'
    return 'double_parabolic' if require_thickness else 'parabolic'


def select_row_methods(sections):
    """Choose one analytic family for a continuous radial loft.

    A binary active/reactive switch at an arbitrary radius creates a geometry
    jump, even with smooth flow. Mixed rows therefore use the common two-
    parabola family; a root section may still be aerodynamically active.
    Fully active rows use the dedicated constant-channel method. This is an
    explicit design policy, not an additional textbook correlation.
    """
    preferred=[select_method(s) for s in sections]
    if len(set(preferred))>1:
        return ['double_parabolic']*len(sections),preferred
    return preferred,preferred


def build_profile(spec,method='parabolic'):
    if method=='two_circle_pressure':
        return build_two_circle_pressure_profile(spec,inlet_wedge_deg=45.)
    if method not in METHODS:raise ValueError('unknown profile method')
    return METHODS[method](spec)


def sampled_throat(profile,pitch_m,samples=101):
    """Cheap finite-curve distance for optimizer screening, not circle analysis.

    Exact tangent inscribed-circle solver is mandatory for final evaluation.
    """
    a=profile.suction.sample(samples)
    b=np.vstack((profile.pressure.sample(samples),profile.trailing_edge.sample(max(17,samples//4))))+[0,pitch_m]
    distance=cdist(a,b);i,j=np.unravel_index(np.argmin(distance),distance.shape)
    return float(distance[i,j]),a[i],b[j]


def refined_throat(profile,pitch_m):
    """Continuous wall clearance for constraints; final circle analysis remains mandatory.

    The mesh only supplies a starting point on each analytic curve pair.
    Unlike point-cloud distance, this does not overestimate a narrow gap
    merely because its contact falls between sample coordinates.
    """
    lower=getattr(profile.suction,'segments',(profile.suction,))
    upper=(*getattr(profile.pressure,'segments',(profile.pressure,)),profile.trailing_edge)
    grid=np.linspace(0.,1.,17);shift=np.array([0.,pitch_m]);best=None
    for a in lower:
        for b in upper:
            distances=cdist(a.evaluate(grid),b.evaluate(grid)+shift)
            i,j=np.unravel_index(np.argmin(distances),distances.shape)
            def distance_and_gradient(v):
                delta=a.evaluate(v[0])-b.evaluate(v[1])-shift
                return (float(delta@delta)/pitch_m**2,
                    2*np.array([delta@a.evaluate(v[0],1),-delta@b.evaluate(v[1],1)])/pitch_m**2)
            fit=minimize(distance_and_gradient,[grid[i],grid[j]],jac=True,
                bounds=[(0.,1.),(0.,1.)],method='L-BFGS-B',
                options={'ftol':1e-15,'gtol':1e-11,'maxiter':100})
            pa=a.evaluate(fit.x[0]);pb=b.evaluate(fit.x[1])+shift
            candidate=(float(np.linalg.norm(pa-pb)),pa,pb)
            if best is None or candidate[0]<best[0]:best=candidate
    return best


def initial_spec(section,axial_width_m,pitch_m,*,incidence_deg,deviation_deg,
                 relative_thickness=None,thickness_location=.3,stagger_deg=None):
    """pp.7,12–13: explicit incidence/deviation; no automatic flow-metal equality."""
    lower,upper=thickness_recommendation(section.row,section.span_fraction)
    thickness=(lower+upper)/2 if relative_thickness is None else relative_thickness
    inlet=section.flow_inlet_deg+incidence_deg
    outlet=section.flow_outlet_deg-deviation_deg
    if stagger_deg is None:
        # Mean velocity direction p17: inlet +y, outlet -y in source frame.
        v0=section.inlet.speed(section.frame_inlet_speed_m_s)
        v1=section.outlet.speed(section.frame_speed_m_s)
        a=math.radians(inlet);b=math.radians(outlet)
        dx=v0*math.sin(a)+v1*math.sin(b);dy=v0*math.cos(a)-v1*math.cos(b)
        stagger_deg=math.degrees(math.atan2(dx,-dy))
    chord=axial_width_m/math.sin(math.radians(stagger_deg))
    max_thickness=chord*thickness
    # p12 LE .2–.3 and interpolated root/tip TE ranges are STARTING guidance.
    rte_ratio=(1-section.span_fraction)*.065+section.span_fraction*.115
    return ProfileSpec(axial_width_m,stagger_deg,inlet,outlet,pitch_m,.25*max_thickness,
                       rte_ratio*max_thickness,origin_y_m=0.,incidence_deg=incidence_deg,
                       deviation_deg=deviation_deg,relative_thickness=thickness,
                       thickness_location=thickness_location)


def edge_radius_sensitivity(spec,method,*,radius_factors=(.8,1.,1.2),
                            incidence_offsets_deg=(-3.,3.),samples=101):
    """Rebuild actual geometry; no claim of an aerodynamic incidence polar.

    pp.12–14 recommendations are not monotonic improvement laws. Report
    independent LE/TE radius perturbations and metal-inlet perturbations at
    fixed flow direction, using actual throat, curvature and boundary checks.
    The finite-curve throat here is screening, not a substitute for the final
    inscribed-circle analysis.
    """
    from .curves import curvature
    cases=[('leading_radius',factor,replace(spec,leading_radius_m=spec.leading_radius_m*factor))
           for factor in radius_factors]
    cases.extend(('trailing_radius',factor,replace(spec,trailing_radius_m=spec.trailing_radius_m*factor))
                 for factor in radius_factors)
    cases.extend(('incidence_deg',offset,replace(spec,
        incidence_deg=spec.incidence_deg+offset,inlet_metal_deg=spec.inlet_metal_deg+offset))
                 for offset in incidence_offsets_deg)
    records=[]
    for variable,change,trial in cases:
        record=dict(variable=variable,change=float(change))
        try:
            profile=build_profile(trial,method)
            geometry=validate_profile(profile,trial.pitch_m,samples=samples)
            throat,_,_=sampled_throat(profile,trial.pitch_m,samples)
            record.update(constructed=True,throat_m=throat,
                trailing_thickness_pitch=2*trial.trailing_radius_m/trial.pitch_m,
                maximum_abs_curvature_m_inv=max(float(np.max(np.abs(curvature(curve,np.linspace(0,1,samples)))))
                    for curve in (profile.suction,profile.pressure,profile.leading_edge,profile.trailing_edge)),
                checks=[c.to_dict() for c in geometry['checks']],
                classification='geometric sensitivity only; incidence losses/separation require gas-dynamic validation')
        except (ConstructionError,ValueError,np.linalg.LinAlgError) as exc:
            record.update(constructed=False,message=str(exc),details=getattr(exc,'details',{}))
        records.append(record)
    return records


def actual_loss_coefficients(profile,throat_m,pitch_m,height_m,flow_outlet_deg,row):
    """Existing Mikhaltsev notebook cell26 model, with actual finite TE and throat.

    ζ_profile=ζ_friction+.2*s2/a; ζ_end=2ζ_profile*t*sin(angle)/height.
    The empirical .017/.019 friction estimates are retained, not re-fitted to
    geometry. Endwall, shock and secondary-flow CFD validation remain required.
    """
    if throat_m<=0 or height_m<=0:raise ValueError('positive throat and height required')
    friction={'SA':.017,'RK':.019}[row]
    profile_loss=friction+.2*(2*profile.trailing_edge.radius)/throat_m
    end_loss=2*profile_loss*pitch_m*math.sin(math.radians(flow_outlet_deg))/height_m
    total=profile_loss+end_loss
    return dict(friction=friction,profile=profile_loss,endwall=end_loss,total=total,
                velocity_coefficient=math.sqrt(1-total) if total<1 else math.nan,
                source='Existing Mikhaltsev Main_Turbinis_Calc_Book cell26, chapter8',
                classification='inherited empirical closure, not a profile CFD validation')


def evaluate_cascade(profile,spec,section,*,channel_samples=61):
    from .channel_convergence import analyze_channel
    geom=validate_profile(profile,spec.pitch_m)
    measured=geometric_parameters(profile,spec)
    channel=analyze_channel(profile,spec.pitch_m,samples=channel_samples)
    return dict(profile=profile,spec=spec,section=section,geometry=geom,parameters=measured,channel=channel)

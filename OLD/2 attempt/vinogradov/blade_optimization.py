"""EXPERIMENTAL bounded mixed discrete/continuous blade design.

Search bounds are recorded numerical design choices, not new universal hard
constraints. Every evaluated candidate and every failed gate is retained.
Legacy gas equations remain owned by the existing cycle/stage core.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
import math
from itertools import product
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from .models import ConstructionError
from .blade_geometry import count_candidates,optimal_relative_pitch,area_distribution,exit_back_geometry
from .cascade_geometry import build_profile,select_row_methods,sampled_throat,refined_throat
from .section_properties import section_properties
from .validation import validate_profile
from .blade_strength import centrifugal_loads,gas_loads,section_stress,recovery_temperature_bound


@dataclass
class RowOptimization:
    count: int
    specs: list
    profiles: list
    method: list[str]
    objective: float
    archive: list[dict]
    search_converged: bool
    area_law: str
    area_blend: float


def _vector(spec):
    chord=spec.axial_width_m/math.sin(math.radians(spec.stagger_deg))
    thickness=chord*spec.relative_thickness
    return [spec.axial_width_m,spec.stagger_deg,spec.relative_thickness,
            spec.thickness_location,spec.leading_radius_m/thickness,spec.trailing_radius_m/thickness,
            spec.trailing_wedge_deg,spec.back_angle_deg,spec.incidence_deg,spec.deviation_deg]


def _rotor_area_law(specs,profiles,span,ratio,law,blend,methods):
    """Eliminate dependent widths in the homothetic table-4/5 family.

    Rebuild with a new design width and proportionate edge radii; do not
    rescale already accepted contours. Metal angles and relative thickness
    remain unchanged. Physical width bounds are checked by the optimizer.
    """
    areas=np.array([section_properties(p,count_per_side=101).area_m2 for p in profiles])
    if np.any(areas<=0) or not np.isfinite(areas).all():
        raise ConstructionError('Positive areas required for rotor area-law parameterization')
    target=area_distribution(span,areas[0],areas[0]/ratio,law,blend)
    result=list(specs);curves=list(profiles)
    for j in range(1,len(specs)):
        scale=math.sqrt(target[j]/areas[j]);p=specs[j]
        result[j]=replace(p,axial_width_m=p.axial_width_m*scale,
            leading_radius_m=p.leading_radius_m*scale,trailing_radius_m=p.trailing_radius_m*scale)
        curves[j]=build_profile(result[j],methods[j])
    return result,curves


def _spec(vector,seed,section,pitch):
    width,stagger,thickness,location,le,te,wedge,back,incidence,deviation=map(float,vector)
    absolute=width/math.sin(math.radians(stagger))*thickness
    return replace(seed,axial_width_m=width,stagger_deg=stagger,relative_thickness=thickness,
                   thickness_location=location,leading_radius_m=le*absolute,trailing_radius_m=te*absolute,
                   trailing_wedge_deg=wedge,back_angle_deg=back,pitch_m=pitch,
                   incidence_deg=incidence,deviation_deg=deviation,
                   inlet_metal_deg=section.flow_inlet_deg+incidence,
                   outlet_metal_deg=section.flow_outlet_deg-deviation)


def _coordinate_bounds(spec):
    return [(max(.005,spec.axial_width_m*.7),max(.0051,spec.axial_width_m*1.4)),
            (5.,175.),(.035,.35),(.10,.70),(.15,.35),(.03,.18),
            (4.,8.),(5.,15.),(-10.,5.),(0.,5.)]


def _constructible_start(vector,seed,section,pitch,method,bounds,count,index,archive,minimum_open_fraction=0.):
    """Deterministic starts avoid a flat invalid-construction plateau.

    Actual design coordinates vary only inside the same recorded search
    bounds; no boundary condition or physical limit is relaxed.
    """
    bounds=np.asarray(bounds,float);best=vector;best_score=math.inf
    # The old cubic-profile stagger is an initializer, not a physical bound
    # on a different analytic family. Enumerate the geometric angular domain.
    staggers=sorted(set([float(vector[1]),*np.linspace(bounds[1,0],bounds[1,1],13)]),
                    key=lambda angle:abs(angle-vector[1]))
    for stagger,location,incidence,thickness in product(staggers,dict.fromkeys((float(vector[3]),.3,.2,.5)),(0.,-5.,3.),(1.,.7,1.3)):
        trial=np.array(vector,copy=True);trial[1]=stagger;trial[3]=location
        trial[8]+=incidence;trial[2]*=thickness
        trial=np.clip(trial,bounds[:,0],bounds[:,1])
        try:
            sp=_spec(trial,seed,section,pitch);profile=build_profile(sp,method)
            checks=validate_profile(profile,pitch,samples=31)['checks']
            throat,_,_=sampled_throat(profile,pitch,41)
            target=pitch*max(math.sin(math.radians(section.flow_outlet_deg))/sp.throat_multiplier,minimum_open_fraction)
            score=10*sum(not c.passed for c in checks)+(throat/target-1)**2
            failures=[c.to_dict() for c in checks if not c.passed]
            status='initialization_constructed'
        except (ConstructionError,ValueError,np.linalg.LinAlgError) as exc:
            score=1e4;status='initialization_failed'
            failures=[dict(name=type(exc).__name__,message=str(exc),details=getattr(exc,'details',{}))]
        archive.append(dict(count=count,section=index,variables=trial.tolist(),objective=float(score),
                            status=status,violations=failures))
        if score<best_score:best,best_score=trial,score
        if score<.01:break
    return best


def optimize_row(sections,seeds,*,previous_count=0,count_half_window=2,max_iterations=8,
                 methods=None,area_law='intermediate',area_blend=.5,archive_callback=None,
                 material=None,life_hours=10000.,screening_limit_Pa=300e6,minimum_open_fractions=None,
                 minimum_trailing_thickness_m=0.,root_tip_area_ratio=None,enforce_course_geometry=False,
                 loss_evaluator=None,strength_evaluator=None,geometry_regularization=1.,
                 maximum_back_angle_deg=None):
    """Integer count shared at all radii, with real t(r)=2pi*r/Z.

    Ten continuous coordinates per radial section: axial width (and derived
    chord), stagger, thickness, location, both edge ratios, wedge, back angle,
    incidence, deviation. With native callbacks the objective is actual row
    loss plus explicitly weighted geometric guidance; strength constraints
    use actual-contour LM and normal stress. Without callbacks this is a
    geometric search, not a qualified turbine. Final full-channel/radial
    gates remain mandatory.
    """
    if len(sections)!=len(seeds) or len(sections)<3:raise ValueError('matching >=3 radial sections required')
    if not math.isfinite(geometry_regularization) or geometry_regularization<0:
        raise ValueError('nonnegative finite geometric regularization required')
    if maximum_back_angle_deg is not None and (not math.isfinite(maximum_back_angle_deg) or maximum_back_angle_deg<=0):
        raise ValueError('positive finite back-angle limit required')
    if len({(s.stage,s.row) for s in sections})!=1:raise ValueError('one row per optimization')
    minimum_open_fractions=(np.zeros(len(sections)) if minimum_open_fractions is None
                            else np.asarray(minimum_open_fractions,float))
    if minimum_open_fractions.shape!=(len(sections),) or not np.isfinite(minimum_open_fractions).all() or np.any(minimum_open_fractions<0):
        raise ValueError('finite nonnegative minimum open fraction per section required')
    if np.any(minimum_open_fractions>1):
        raise ConstructionError('Required critical area exceeds even a blade-free annular strip',
                                {'minimum_open_fractions':minimum_open_fractions.tolist()})
    if minimum_trailing_thickness_m<0 or (root_tip_area_ratio is not None and root_tip_area_ratio<=1):
        raise ValueError('nonnegative edge thickness and rotor area ratio >1 required')
    methods=list(methods) if methods is not None else select_row_methods(sections)[0]
    if len(methods)!=len(sections):raise ValueError('method required for each radial station')
    middle=len(sections)//2;s=sections[middle];seed=seeds[middle]
    seed_bounds=[limit for p in seeds for limit in _coordinate_bounds(p)]
    initial_vector=np.clip(np.array([_vector(p) for p in seeds]).ravel(),
                           np.array(seed_bounds)[:,0],np.array(seed_bounds)[:,1])
    reference_trials=[]
    reference_vector=_constructible_start(initial_vector.reshape(-1,10)[middle],seed,s,seed.pitch_m,
        methods[middle],seed_bounds[10*middle:10*middle+10],round(2*np.pi*s.radius_m/seed.pitch_m),
        middle,reference_trials,minimum_open_fractions[middle])
    reference=_spec(reference_vector,seed,s,seed.pitch_m)
    chord=reference.axial_width_m/math.sin(math.radians(reference.stagger_deg))
    thickness=chord*reference.relative_thickness
    reference_valid=False
    try:
        profile=build_profile(reference,methods[middle])
        reference_valid=validate_profile(profile,reference.pitch_m,samples=31)['passed']
        if reference_valid:
            chord=float(np.linalg.norm(profile.trailing_edge.center-profile.leading_edge.center))
    except (ConstructionError,ValueError,np.linalg.LinAlgError):
        pass  # Count proposal only; no failed reference is accepted as geometry.
    effective=math.degrees(math.asin(math.sin(math.radians(s.flow_outlet_deg))/reference.throat_multiplier))
    ratio=optimal_relative_pitch(s.flow_inlet_deg,s.flow_outlet_deg,reference.inlet_metal_deg,
                                 effective,thickness/chord)
    candidates=count_candidates(2*s.radius_m,chord,ratio,s.row,half_window=count_half_window,previous=previous_count)
    for candidate in candidates:candidate['origin']='empirical reference'
    # An empirical recommendation must not exclude a coherent existing count,
    # especially a feasible warm start outside the narrow proposal window.
    implied=np.array([2*np.pi*section.radius_m/spec.pitch_m for section,spec in zip(sections,seeds)])
    seed_count=round(float(implied[middle]))
    coherent=bool(np.allclose(implied,seed_count,rtol=0.,atol=1e-8))
    eligible=coherent and seed_count>=3 and seed_count>previous_count and seed_count%2==(1 if s.row=='SA' else 0)
    if eligible and seed_count not in [v['count'] for v in candidates]:
        candidates.append(dict(count=seed_count,pitch_m=2*np.pi*s.radius_m/seed_count,
            relative_pitch=2*np.pi*s.radius_m/(seed_count*chord),
            target_count=candidates[0]['target_count'],origin='coherent existing count'))
    span=np.array([s.span_fraction for s in sections])
    archive=[dict(status='count_reference',constructed_reference_valid=reference_valid,
        chord_m=chord,local_prescribed_relative_thickness=thickness/chord,
        thickness_scope='Local prescription; global maximum is checked after optimization',
        trials=reference_trials,existing_count=seed_count,existing_count_eligible=bool(eligible))]
    best=None;best_rank=None
    for candidate in candidates:
        count=candidate['count'];pitch=[2*np.pi*s.radius_m/count for s in sections]
        archive.append(dict(status='count_candidate',**candidate))
        bounds=list(seed_bounds);x0=initial_vector.copy()
        seed_vector=np.r_[x0.copy(),area_blend]
        initial=x0.reshape(-1,10)
        for j,(p,s,t) in enumerate(zip(seeds,sections,pitch)):
            initial[j]=_constructible_start(initial[j],p,s,t,methods[j],bounds[10*j:10*j+10],count,j,archive,minimum_open_fractions[j])
        x0=np.r_[x0,area_blend];bounds.append((0.,1.))
        gas=gas_loads(sections,count) if material is not None and sections[0].row=='RK' else None
        parameterize_area=False
        evaluations={};best_feasible_x=None;best_feasible_cost=math.inf
        @lru_cache(maxsize=128)
        def constructed(values,enforce_area):
            # Objective and constraints share finite-difference points.
            specs=[_spec(v,p,s,t) for v,p,s,t in zip(np.asarray(values[:-1]).reshape(-1,10),seeds,sections,pitch)]
            profiles=[build_profile(p,m) for p,m in zip(specs,methods)]
            if enforce_area:
                return _rotor_area_law(specs,profiles,span,root_tip_area_ratio,area_law,float(values[-1]),methods)
            return specs,profiles
        @lru_cache(maxsize=128)
        def clearances(values,enforce_area):
            specs,profiles=constructed(values,enforce_area)
            return tuple(refined_throat(p,s.pitch_m) for p,s in zip(profiles,specs))
        @lru_cache(maxsize=128)
        def measured_geometry(values,enforce_area):
            specs,profiles=constructed(values,enforce_area)
            return tuple(validate_profile(p,s.pitch_m,samples=31) for p,s in zip(profiles,specs))
        @lru_cache(maxsize=128)
        def physical_results(values,enforce_area):
            specs,profiles=constructed(values,enforce_area);gaps=clearances(values,enforce_area)
            # Allocation overshoot is still a hard constraint. It does not
            # erase meaningful local loads on an otherwise valid contour.
            invalid=[c.to_dict() for g in measured_geometry(values,enforce_area) for c in g['checks']
                     if not c.passed and c.name!='axial envelope boundary']
            if invalid:
                raise ConstructionError('Native local models require nonintersecting closed profiles',{'checks':invalid})
            loss=None if loss_evaluator is None else float(loss_evaluator(count,specs,profiles,gaps))
            if loss is not None and not 0<loss<1:
                raise ConstructionError('Native row loss outside (0,1)',{'loss':loss})
            strength=None if strength_evaluator is None else strength_evaluator(count,specs,profiles,gaps)
            if strength is not None:
                values_=[strength[k] for k in ('minimum_lm_margin','required_lm_margin','maximum_absolute_normal_Pa')]
                if not np.isfinite(values_).all() or values_[1]<=0 or values_[2]<0:
                    raise ConstructionError('Actual-contour strength unavailable',strength)
            return loss,strength
        def objective(x,*,retain=True):
            profiles=[];specs=[];violations=[];cost=0.;geometry_valid=True;native_loss=None;native_strength=None
            try:
                specs,profiles=constructed(tuple(x),parameterize_area)
                gaps=clearances(tuple(x),parameterize_area)
                for j,(sp,profile,s) in enumerate(zip(specs,profiles,sections)):
                    throat=gaps[j][0]
                    actual_back=exit_back_geometry(profile,sp)['back_angle_deg']
                    cost+=((actual_back-sp.back_angle_deg)/10.)**2
                    target=pitch[j]*max(math.sin(math.radians(s.flow_outlet_deg))/sp.throat_multiplier,minimum_open_fractions[j])
                    chord=float(np.linalg.norm(profile.trailing_edge.center-profile.leading_edge.center))
                    prescribed=sp.axial_width_m/math.sin(math.radians(sp.stagger_deg))*sp.relative_thickness
                    if not 0<throat<=pitch[j]:
                        raise ConstructionError('geometric throat/pitch outside (0,1]',{'throat_m':throat,'pitch_m':pitch[j]})
                    effective=math.degrees(math.asin(throat/pitch[j]))
                    optimum=optimal_relative_pitch(s.flow_inlet_deg,s.flow_outlet_deg,sp.inlet_metal_deg,effective,prescribed/chord)
                    cost+=.05*(pitch[j]/(chord*optimum)-1)**2
                    residual=(throat/target-1)
                    cost+=residual**2
                    capacity_deficit=max(0.,minimum_open_fractions[j]*pitch[j]/throat-1.)
                    cost+=capacity_deficit**2
                    if capacity_deficit:
                        violations.append(dict(section=j,name='required critical open area',
                            value=throat/pitch[j],limit=float(minimum_open_fractions[j])))
                    if profile.metadata.get('construction_valid',True) is False:
                        violations.append(dict(section=j,name='source construction boundary',value=1.))
                        cost+=10.
                        geometry_valid=False
                    geometry=measured_geometry(tuple(x),parameterize_area)[j]
                    for check in geometry['checks']:
                        if not check.passed:
                            violations.append(dict(section=j,**check.to_dict()));cost+=10.;geometry_valid=False
                properties=[section_properties(p,count_per_side=101) for p in profiles]
                areas=np.array([p.area_m2 for p in properties])
                target_area=area_distribution(span,areas[0],areas[-1],area_law,float(x[-1]))
                cost+=float(np.mean((areas/target_area-1)**2))
                if areas[-1]>=areas[0] and sections[0].row=='RK':
                    # Source p13 decreases rotor area, not automatically chord.
                    error=areas[-1]/areas[0]-1;cost+=error**2
                    violations.append(dict(name='rotor area decrease guidance',value=float(error),classification='recommendation'))
                if gas is not None:
                    loads=centrifugal_loads([s.radius_m for s in sections],areas,sections[0].speed_rpm,material.density_kg_m3)
                    for j,(profile,prop,section,N,M) in enumerate(zip(profiles,properties,sections,loads['force_N'],gas['moments_N_m'])):
                        stress=section_stress(profile,N,M,properties=prop,contour=profile.contour(101))
                        maximum=max(float(stress.max()),0.)
                        screening_excess=max(0.,maximum/screening_limit_Pa-1)
                        cost+=screening_excess**2
                        if screening_excess:
                            violations.append(dict(section=j,name='300MPa screening',value=maximum,limit=screening_limit_Pa,units='Pa'))
                        recovery=recovery_temperature_bound(section)
                        # A 1.2 reserve is a SOFT exploration target only;
                        # final physical comparison uses the configured margin.
                        allowable=float(material.rupture_allowable(recovery,life_hours))/1.2
                        if not np.isfinite(allowable):
                            violations.append(dict(section=j,name='unknown LM allowable',temperature_K=recovery));cost+=10.
                        else:
                            excess=max(0.,maximum/allowable-1);cost+=excess**2
                            if excess:violations.append(dict(section=j,name='creep optimization screening',value=maximum,limit=allowable,units='Pa'))
                if loss_evaluator is not None or strength_evaluator is not None:
                    native_loss,native_strength=physical_results(tuple(x),parameterize_area)
                    if native_loss is not None:cost=native_loss+geometry_regularization*cost
                status='evaluated'
            except (ConstructionError,ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                cost=1e4;status='construction_failed';geometry_valid=False
                violations.append(dict(name=type(exc).__name__,message=str(exc),details=getattr(exc,'details',{})))
            evaluations[(parameterize_area,tuple(x))]=(float(cost),geometry_valid)
            if retain:
                record=dict(count=count,variables=np.asarray(x).tolist(),objective=float(cost),status=status,
                            violations=violations,area_law=area_law,area_blend=float(x[-1]),
                            area_parameterized=parameterize_area,geometry_valid=geometry_valid,
                            native_row_loss=native_loss,native_strength=native_strength)
                archive.append(record)
                if archive_callback is not None:archive_callback(record)
            return float(cost)
        def feasibility(x):
            """Prescribed course constraints, not extra empirical shape guides."""
            nonlocal best_feasible_x,best_feasible_cost
            from .validation import _axial_bounds
            n=len(sections);residual=[]
            try:
                specs,profiles=constructed(tuple(x),parameterize_area)
                gaps=clearances(tuple(x),parameterize_area)
                areas=[];chords=[]
                for j,(profile,sp) in enumerate(zip(profiles,specs)):
                    bounds_=[_axial_bounds(c) for c in (profile.suction,profile.pressure,profile.leading_edge,profile.trailing_edge)]
                    left=profile.leading_edge.center[0]-profile.leading_edge.radius
                    right=profile.trailing_edge.center[0]+profile.trailing_edge.radius
                    excess=max(left-min(v[0] for v in bounds_),max(v[1] for v in bounds_)-right)
                    throat=gaps[j][0]
                    chord=np.linalg.norm(profile.trailing_edge.center-profile.leading_edge.center)
                    tolerance=max(chord*1e-7,1e-12)  # same numerical accuracy as validate_profile
                    residual.extend([2*sp.trailing_radius_m/sp.axial_width_m-minimum_trailing_thickness_m/sp.axial_width_m,
                        throat/sp.pitch_m-minimum_open_fractions[j],(tolerance-excess)/sp.axial_width_m,
                        (sp.axial_width_m-bounds[10*j][0])/sp.axial_width_m,
                        (bounds[10*j][1]-sp.axial_width_m)/sp.axial_width_m])
                    areas.append(section_properties(profile,count_per_side=101).area_m2)
                    chords.append(np.linalg.norm(profile.trailing_edge.center-profile.leading_edge.center))
                rotor=sections[0].row=='RK'
                residual.extend((np.diff(areas)*(-1 if rotor else 0)/areas[0]).tolist())
                ratio=areas[0]/areas[-1]
                residual.extend(([ratio/root_tip_area_ratio-.995,1.005-ratio/root_tip_area_ratio]
                                 if root_tip_area_ratio is not None else [1.,1.]))
                residual.append((chords[0]-chords[-1])*(1 if rotor else -1)/chords[0])
                if maximum_back_angle_deg is not None:
                    residual.extend((maximum_back_angle_deg-exit_back_geometry(p,s)['back_angle_deg'])/maximum_back_angle_deg
                                    for p,s in zip(profiles,specs))
                if strength_evaluator is not None:
                    _,strength=physical_results(tuple(x),parameterize_area)
                    residual.extend([strength['minimum_lm_margin']/strength['required_lm_margin']-1.,
                                     1.-strength['maximum_absolute_normal_Pa']/screening_limit_Pa])
            except (ConstructionError,ValueError,np.linalg.LinAlgError,FloatingPointError):
                return np.full(5*n+n-1+3+(n if maximum_back_angle_deg is not None else 0)
                               +(2 if strength_evaluator is not None else 0),-1e3)
            result=np.asarray(residual)
            if np.isfinite(result).all() and np.all(result>=0):
                key=(parameterize_area,tuple(x))
                if key not in evaluations:objective(x,retain=False)
                value,valid=evaluations[key]
                if valid and value<best_feasible_cost:
                    best_feasible_x=np.asarray(x).copy();best_feasible_cost=value
            return result
        fit=minimize(objective,x0,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':max_iterations,'maxfun':max(100,20*len(x0)), 'ftol':1e-9})
        if enforce_course_geometry:
            # Explicit constraints prevent a small loss/step benefit from
            # trading away manufacturing, taper or mass conservation.
            parameterize_area=(sections[0].row=='RK' and root_tip_area_ratio is not None
                and all(m in ('double_parabolic','prescribed_thickness') for m in methods))
            for initial_candidate in (seed_vector,x0,fit.x):
                feasibility(initial_candidate)
            lower=np.array([v[0] for v in bounds]);extent=np.array([v[1]-v[0] for v in bounds])
            # Metres, degrees and dimensionless shape parameters must not
            # share an unscaled finite-difference step.
            active=np.array([j for j in range(len(bounds)) if not (parameterize_area and j>0 and j%10==0 and j<len(bounds)-1)])
            base=(fit.x if best_feasible_x is None else best_feasible_x).copy()
            def physical(z):
                result=base.copy();result[active]=lower[active]+extent[active]*z
                return result
            refined=minimize(lambda z:objective(physical(z)),(base[active]-lower[active])/extent[active],
                method='SLSQP',bounds=[(0.,1.)]*len(active),
                constraints=[dict(type='ineq',fun=lambda z:feasibility(physical(z)))],
                options={'maxiter':max_iterations*5,'ftol':1e-12})
            refined.x=physical(refined.x)
            archive.append(dict(count=count,status='course_constraint_solve',
                converged=bool(refined.success),message=str(refined.message),
                minimum_constraint=float(np.min(feasibility(refined.x))),
                area_parameterized=parameterize_area,variables=refined.x.tolist()))
            if np.min(feasibility(refined.x))>=np.min(feasibility(fit.x)):
                fit=refined
            if best_feasible_x is not None:
                if not np.array_equal(best_feasible_x,fit.x):
                    fit.x=best_feasible_x;fit.success=False
                archive.append(dict(count=count,status='best_feasible_candidate',
                    objective=best_feasible_cost,variables=fit.x.tolist(),area_parameterized=parameterize_area))
        cost=objective(fit.x)
        try:
            specs,profiles=constructed(tuple(fit.x),parameterize_area)
        except (ConstructionError,ValueError,np.linalg.LinAlgError):continue
        rank=(max(0.,-float(np.min(feasibility(fit.x)))) if enforce_course_geometry else 0.,cost)
        if best_rank is None or rank<best_rank:
            best_rank=rank
            best=RowOptimization(count,specs,profiles,methods,cost,archive,bool(fit.success),area_law,float(fit.x[-1]))
    if best is None:raise ConstructionError('No constructible row in the explicit search domain',{'archive':archive})
    best.archive=archive
    return best


def coupled_optimization(gas_evaluator,geometry_evaluator,initial_gas_options,*,max_iterations=6,
                         coefficient_tolerance=1e-3,relaxation=.5):
    """Eight-level feedback: existing gas -> radial -> profiles -> channels ->
    section properties -> strength -> real-geometry losses -> existing gas.

    Callbacks preserve ownership of gas equations. An unconverged/failed model
    never becomes a passing result. Every gas and geometry iteration persists.
    """
    if not 0<relaxation<=1 or max_iterations<1:raise ValueError('invalid coupling controls')
    options=dict(initial_gas_options);history=[];last=None
    for iteration in range(max_iterations):
        gas=gas_evaluator(dict(options));geometry=geometry_evaluator(gas)
        target=geometry['velocity_coefficients']
        changes={name:abs(float(target[name])-float(options[name])) for name in ('phi','psi')}
        error=max(changes.values());last=dict(gas=gas,geometry=geometry)
        history.append(dict(iteration=iteration,gas_options=dict(options),target_coefficients=target,
                            coefficient_error=error,violations=geometry['violations']))
        if not all(np.isfinite(list(target.values()))):
            return dict(**last,history=history,converged=False,passed=False,reason='row loss coefficients unavailable or nonphysical; see construction/applicability violations')
        if error<=coefficient_tolerance:
            return dict(**last,history=history,converged=True,passed=not geometry['violations'])
        for name in ('phi','psi'):options[name]=(1-relaxation)*options[name]+relaxation*float(target[name])
    return dict(**last,history=history,converged=False,passed=False,reason='coupled coefficient tolerance not reached')


def optimize_axial_gaps(row_bounds,*,ratio_bounds=(.20,.25),maximum_opening_deg=25.):
    """Optimize real gaps separately from row widths, inherited Mikhaltsev ch.6.

    row_bounds: (width_m,root_radius_m,tip_radius_m) along the turbine.
    .20–.25 is the EXISTING gap/width recommendation, not a new physical law.
    The inherited signed sum of wall angles is retained and both individual
    wall slopes are reported. Height monotonicity is not imposed.
    """
    data=np.asarray(row_bounds,float)
    if data.ndim!=2 or data.shape[1]!=3 or len(data)<2 or np.any(data[:,0]<=0):
        raise ValueError('need >=2 width/root/tip rows')
    lo,hi=ratio_bounds
    if not 0<lo<=hi:raise ValueError('invalid gap search domain')
    widths=data[:-1,0]
    def angles(gaps):
        dx=gaps+data[1:,0]
        outer=np.degrees(np.arctan(np.diff(data[:,2])/dx))
        inner=np.degrees(np.arctan(-np.diff(data[:,1])/dx))
        return outer,inner
    def constraint(gaps):
        outer,inner=angles(gaps)
        return maximum_opening_deg-(outer+inner)
    fit=minimize(lambda gaps:float(np.sum(gaps)),widths*(lo+hi)/2,
                 method='SLSQP',bounds=list(zip(widths*lo,widths*hi)),
                 constraints={'type':'ineq','fun':constraint},
                 options={'maxiter':50,'ftol':1e-12})
    outer,inner=angles(fit.x)
    return dict(gaps_m=fit.x,outer_wall_deg=outer,inner_wall_deg=inner,
                opening_deg=outer+inner,passed=bool(np.all(constraint(fit.x)>=-1e-7)),
                converged=bool(fit.success),ratio_bounds=ratio_bounds,
                maximum_opening_deg=maximum_opening_deg,
                message=str(fit.message))

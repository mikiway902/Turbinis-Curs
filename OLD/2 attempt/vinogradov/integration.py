"""Actual TWO-stage course gas/radial states with analytical Vinogradov blades.

Original resource notebook functions own thermodynamics, losses and the two
strength discretizations. Failed rows remain quantified and are never replaced
by historical calculations or cosmetically reconstructed contours.
"""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
import math
from functools import lru_cache
from copy import deepcopy
import numpy as np
from scipy.interpolate import PchipInterpolator
from .models import ConstructionError,ProfileSpec
from .course_adapter import CourseRadialSampler
from .blade_geometry import optimal_relative_pitch,geometric_parameters,area_distribution
from .cascade_geometry import initial_spec,select_row_methods,build_profile,edge_radius_sensitivity,refined_throat
from .blade_optimization import optimize_row
from .channel_convergence import analyze_channel
from .cascade_throat import effective_outlet_angle,integrate_annular_capacity
from .section_properties import section_properties
from .validation import validate_profile,validate_prescribed_thickness
from .blade_stacking import stack_sections,adaptive_sections,centroid_scatter
from .blade_strength import IN738C,beam_modes,campbell_screening
from .stator_strength import assess_stator
from .cad_export import export_csv,export_json,export_dxf,export_step


def _boundary_angles(profile):
    def mean_tangent(t):
        a=profile.suction.evaluate(t,1);b=profile.pressure.evaluate(t,1)
        a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);d=a+b
        return math.degrees(math.atan2(d[1],d[0]))
    return 90-mean_tangent(0),90+mean_tangent(1)




def _source_profiles(sampler):
    coupled=sampler.ns.get('COUPLED',{})
    return coupled['profiles'] if coupled.get('result') is sampler.result else {}


def _source_specs(sampler,index,row,sections,source_profiles,options):
    name=sampler.name(index,row);registry=sampler.design[name]
    count=int(registry['Z']);specs=[]
    warm=options.get('initial_specifications',{}).get(f'{row}{index+1}')
    if warm is not None:
        if len(warm)!=len(sections):
            raise ValueError('Warm-start specifications must match the radial station count')
        for raw,section in zip(warm,sections):
            seed=ProfileSpec(**raw)
            specs.append(replace(seed,pitch_m=2*np.pi*section.radius_m/count,
                inlet_metal_deg=section.flow_inlet_deg+seed.incidence_deg,
                outlet_metal_deg=section.flow_outlet_deg-seed.deviation_deg))
        return specs
    for j,section in enumerate(sections):
        source=source_profiles.get((name,j))
        if source is None:
            part=sampler.inputs[name][j]
            shape=dict(registry['sections'][j])
            width=float(shape.pop('bax_m',registry['bax_m']))
            source=sampler.ns['preliminary_profile'](part['beta_in'],part['beta_out'],
                width,part['kind'],sampler.ns['BOOK_DEFAULTS']['trailing_edge_m'],
                part['deviation'],outlet_mach=part['mach'],**shape)
        incidence=float(source['beta_in_metal'])-section.flow_inlet_deg
        deviation=float(source['deviation'])
        seed=initial_spec(section,float(source['bax']),2*np.pi*section.radius_m/count,
            incidence_deg=float(options.get('incidence_deg',incidence)),deviation_deg=deviation,
            relative_thickness=float(source['relative_max_thickness']),
            stagger_deg=float(source['gamma']))
        specs.append(replace(seed,leading_radius_m=float(source['rLE']),
            trailing_radius_m=float(source['te'])/2,
            trailing_wedge_deg=float(source['trailing_wedge_deg'])))
    return specs


def _source_profile_record(profile,spec,section,throat_m,count,mass_fraction):
    """Translate actual analytic surfaces to the original loss/strength API."""
    return dict(polygon=profile.contour(401),radius=section.radius_m,Z=count,
        suction=profile.suction.sample(401),pressure=profile.pressure.sample(401),
        throat_m=throat_m,intersections=[],self_intersection=False,
        te=2*profile.trailing_edge.radius,pitch=spec.pitch_m,
        beta_out_flow=section.flow_outlet_deg,height_out=section.height_m,
        Re_times_mu=section.outlet.density_kg_m3*section.outlet.speed(section.frame_speed_m_s)*
            np.linalg.norm(profile.trailing_edge.center-profile.leading_edge.center),
        mass_fraction=float(mass_fraction))


def _source_row_evaluators(sampler,index,row,sections):
    """Use unchanged native loss/UKR equations on a real candidate.

    The optimizer validates closed, nonintersecting local surfaces first.
    Axial allocation and full-radial validation remain separate constraints.
    """
    name=sampler.name(index,row)
    def records(count,specs,profiles,gaps):
        return {(name,j):_source_profile_record(profile,spec,section,gap[0],count,j/(len(sections)-1))
                for j,(spec,profile,section,gap) in enumerate(zip(specs,profiles,sections,gaps))}
    def loss(count,specs,profiles,gaps):
        _,mean=sampler.ns['cascade_loss_review'](sampler.result,sampler.radials,sampler.channel,
            records(count,specs,profiles,gaps),row_names=(name,))
        return float(mean.iloc[0]['ζ сумма'])
    def strength(count,specs,profiles,gaps):
        data=(sampler.strength(records(count,specs,profiles,gaps),rotors=(index+1,))
              if row=='RK' else assess_stator(sampler,index,sections,profiles,count))
        table=data['table']
        if table.empty:
            raise ConstructionError('Requested-row strength unavailable',{'reasons':data['reasons']})
        stresses=table[['sigma_normal_min_MPa','sigma_normal_max_MPa']].to_numpy(dtype=float)
        if not np.isfinite(stresses).all():
            raise ConstructionError('Nonfinite native normal stresses',{'reasons':data['reasons']})
        return dict(minimum_lm_margin=float(data['min_margin']),
            required_lm_margin=float(sampler.requirements['required_margin']),
            maximum_absolute_normal_Pa=float(np.max(np.abs(stresses)))*1e6)
    return loss,strength


def _source_channel_checks(sampler,index,row,j,profile,spec,section,channel,values):
    """Compare the preceding course geometry guides, not competing gas laws."""
    part=sampler.inputs[sampler.name(index,row)][j]
    height_in,height_out=part['l_in'],part['l_out']
    x=channel.centers_m[:,0]
    heights=height_in+(height_out-height_in)*x/spec.axial_width_m
    reduced=channel.width_m*heights/height_in
    expansion=float(np.max(reduced/np.minimum.accumulate(reduced))) if np.all(reduced>0) else math.inf
    throat_x=float(channel.throat.center_m[0])
    throat_height=height_in+(height_out-height_in)*throat_x/spec.axial_width_m
    # Areas scale sin(angle), not the angle in degrees. This area-equivalent
    # comparison assumes equal mass flux; it is NOT the actual exit-flow angle.
    area_ratio=channel.throat_width_m/spec.pitch_m*throat_height/height_out
    angle=(effective_outlet_angle(channel.throat_width_m*throat_height/height_out,spec.pitch_m)
           if 0<=area_ratio<=1 else None)
    polygon=profile.contour(401);neighbor=polygon+[0.,spec.pitch_m]
    crossing=sampler.ns['_book_segments_intersect']
    intersects=bool(crossing(polygon,neighbor) or sampler.ns['_book_inside'](polygon[0],neighbor)
        or sampler.ns['_book_inside'](neighbor[0],polygon))
    self_intersects=bool(crossing(polygon,polygon,same=True))
    # Inlet/outlet cuts are open boundaries, not solid walls. Crossing a
    # plotting cut is not a blade collision; analytic wall clearance and
    # contact residuals have already been checked by analyze_channel.
    geometry=dict(geometry_valid=not(intersects or self_intersects) and channel.throat_width_m>0,
        circle_crosses_open_station_cut=bool(channel.throat.endcap_limited),
        intersections=intersects,self_intersection=self_intersects,
        area_equivalent_angle_deg=angle,area_ratio=area_ratio,
        angle_comparison='uniform-flux geometric comparison, not actual gas exit angle',
        angle_error_deg=angle-section.flow_outlet_deg if angle is not None else math.inf,
        te_over_throat=2*profile.trailing_edge.radius/channel.throat_width_m,
        book_te_over_throat=sampler.ns['BOOK_DEFAULTS']['stator_te_over_throat' if row=='SA' else 'rotor_te_over_throat'],
        max_reduced_expansion=expansion)
    measured=dict(kind=part['kind'],relative_max_thickness=values['relative_thickness'],
        maximum_thickness_m=values['maximum_thickness_m'],pitch=spec.pitch_m,chord=values['chord_m'],
        suction=profile.suction.sample(401),pressure=profile.pressure.sample(401))
    criteria=sampler.ns['_profile_geometry_criteria'](measured,geometry,j)
    return criteria,geometry


def gas_preflight(namespace):
    """Source variable-property radial and subsonic applicability diagnostics."""
    sampler=CourseRadialSampler(namespace);rows=[];checks=[]
    for i in range(len(sampler.stages)):
        for row in ('SA','RK'):
            sections=sampler.row(i,row,5)
            maximum=max(max(s.inlet.mach(s.frame_inlet_speed_m_s),
                            s.outlet.mach(s.frame_speed_m_s)) for s in sections)
            audit=sampler.audit(i,row)
            record=dict(id=f'{row}{i+1}',maximum_frame_mach=maximum,radial_audit=audit)
            rows.append(record)
            checks.append(dict(name=f'{record["id"]}: subsonic source applicability',
                value=maximum,limit='<1',passed=maximum<1,amount=max(0.,maximum-1),
                source='Vinogradov subsonic method domain; not a new universal turbine Mach limit'))
            for c in audit['checks']:
                checks.append(dict(name=f'{record["id"]}: {c.name}',value=c.value,limit=c.limit,
                    passed=c.passed,amount=max(0.,float(c.value)-float(c.limit)),source=c.source))
    return dict(rows=rows,checks=checks,passed=all(c['passed'] for c in checks))


def _parameters(section,spec,profile,channel,thickness_analysis=None):
    values=geometric_parameters(profile,spec,samples=41)
    values.pop('medial')
    props=section_properties(profile)
    inlet_metal,outlet_metal=_boundary_angles(profile)
    normal=channel.throat.suction_normal
    values['global_throat_suction_normal_deg_from_axial']=math.degrees(math.atan2(normal[1],normal[0]))
    values['back_angle_source']='adjacent trailing-circle exit contact; Vinogradov table3.1 item43, distinct from global channel minimum'
    values.update(stage=section.stage,row=section.row,radius_m=section.radius_m,
        root_diameter_m=2*section.root_radius_m,mean_diameter_m=2*section.mean_radius_m,tip_diameter_m=2*section.tip_radius_m,
        height_m=section.height_m,speed_rpm=section.speed_rpm,blade_speed_m_s=section.blade_speed_m_s,
        reaction=section.reaction,alpha0_deg=section.alpha0_deg,alpha1_deg=section.alpha1_deg,alpha2_deg=section.alpha2_deg,
        beta1_deg=section.beta1_deg,beta2_deg=section.beta2_deg,
        actual_inlet_metal_deg=inlet_metal,actual_outlet_metal_deg=outlet_metal,
        frame='absolute' if section.row=='SA' else 'relative',
        c_in_m_s=section.inlet.speed(),c_out_m_s=section.outlet.speed(),
        w_in_m_s=section.inlet.speed((section.inlet_radius_m or section.radius_m)*section.speed_rpm*2*np.pi/60),
        w_out_m_s=section.outlet.speed(section.blade_speed_m_s),
        c_in_axial_m_s=section.inlet.axial_m_s,c_in_theta_m_s=section.inlet.circumferential_m_s,
        c_out_axial_m_s=section.outlet.axial_m_s,c_out_theta_m_s=section.outlet.circumferential_m_s,
        Mach_in=section.inlet.mach(section.frame_inlet_speed_m_s),Mach_out=section.outlet.mach(section.frame_speed_m_s),
        Mach_absolute_in=section.inlet.mach(),Mach_absolute_out=section.outlet.mach(),
        Mach_relative_in=section.inlet.mach((section.inlet_radius_m or section.radius_m)*section.speed_rpm*2*np.pi/60),
        Mach_relative_out=section.outlet.mach(section.blade_speed_m_s),
        Reynolds_in=section.inlet.reynolds(values['chord_m'],section.frame_inlet_speed_m_s),
        Reynolds_out=section.outlet.reynolds(values['chord_m'],section.frame_speed_m_s),
        temperature_in_K=section.inlet.temperature_K,temperature_out_K=section.outlet.temperature_K,
        pressure_in_Pa=section.inlet.pressure_Pa,pressure_out_Pa=section.outlet.pressure_Pa,
        density_in_kg_m3=section.inlet.density_kg_m3,density_out_kg_m3=section.outlet.density_kg_m3,
        mass_flow_kg_s=section.mass_flow_kg_s,required_mass_flux_kg_m2_s=section.required_mass_flux_kg_m2_s,
        area_m2=props.area_m2,perimeter_m=props.perimeter_m,centroid_m=props.centroid_m,
        first_moments_m3=props.first_moments_m3,centroidal_moments_m4=props.centroidal_moments_m4,
        principal_min_m4=props.principal_min_m4,principal_max_m4=props.principal_max_m4,principal_angle_rad=props.principal_angle_rad,
        throat_m=channel.throat_width_m,effective_angle_deg=effective_outlet_angle(channel.throat_width_m,spec.pitch_m),
        channel_inlet_width_m=float(channel.width_m[0]),convergence_ratio=channel.convergence_ratio,
        channel_max_to_inlet=float(max(channel.width_m)/channel.width_m[0]),
        trailing_blockage_fraction=2*profile.trailing_edge.radius/spec.pitch_m,
        source=profile.metadata.get('source'),method=profile.metadata.get('method'))
    if thickness_analysis is not None:
        values['sampled_medial_maximum_thickness_m']=values['maximum_thickness_m']
        values['sampled_medial_thickness_location']=values['thickness_location']
        values['maximum_thickness_m']=thickness_analysis['maximum_diameter_m']
        values['maximum_thickness_lower_m']=thickness_analysis['diameter_lower_m']
        values['maximum_thickness_upper_m']=thickness_analysis['diameter_upper_m']
        values['maximum_thickness_error_m']=thickness_analysis['diameter_error_m']
        values['maximum_thickness_center_m']=thickness_analysis['maximum_center_m']
        values['relative_thickness']=values['maximum_thickness_m']/values['chord_m']
        direction=profile.trailing_edge.center-profile.leading_edge.center
        values['thickness_location']=float((np.asarray(thickness_analysis['maximum_center_m'])-
            profile.leading_edge.center)@direction/(values['chord_m']**2))
        values['thickness_location_scope']='Projected chord coordinate of best numerical circle; diameter bounds do not bound centre location or prove uniqueness'
        values['prescribed_axial_thickness_location']=spec.thickness_location
        values['prescribed_thickness_status']=thickness_analysis['prescribed_status']
    values['optimal_relative_pitch']=optimal_relative_pitch(section.flow_inlet_deg,section.flow_outlet_deg,
        spec.inlet_metal_deg,values['effective_angle_deg'],values['relative_thickness'])
    values['actual_relative_pitch']=spec.pitch_m/values['chord_m']
    return values


def design_from_namespace(namespace,options=None):
    """Attempt all four rows using actual source angular/count/taper settings.

    Five source mass-fraction sections are mandatory for five-trapezoid and
    three-quadratic strength; root/mid/tip alone are drawn. Incidence may be
    explicitly overridden, otherwise it is read from the source metal angles.
    """
    options=dict(options or {})
    sampler=CourseRadialSampler(namespace)
    out=Path(options.get('output_dir',Path(__file__).resolve().parents[1]/
        'exports'/'vinogradov'/f'{sampler.resource_hours:g}h'))
    out.mkdir(parents=True,exist_ok=True)
    options['output_dir']=str(out)
    for key,value in [('life_hours',sampler.resource_hours),
                      ('resource_hours',sampler.resource_hours),
                      ('required_lm_margin',float(sampler.requirements['required_margin']))]:
        if key in options and float(options[key])!=value:
            raise ValueError(f'{key} must match the selected source resource ({value:g})')
    material=IN738C(density_kg_m3=float(sampler.requirements['density_kg_m3']),
        lm_constant=float(sampler.requirements['C_LM']))
    report={'status':'EXPERIMENTAL','passed':False,'rows':[],'violations':[],'warnings':[],'parameters':[],
            'source':'Vinogradov 2010; actual two-stage course resource calculation',
            'resource_hours':sampler.resource_hours,'source_notebook':sampler.source_notebook,
            'stage_count':2,'speed_rpm':sampler.speed_rpm,
            'fixed_speed_rpm':sampler.speed_rpm,
            'source_settings':dict(config=asdict(sampler.cfg),geometry=np.array(sampler.result['geometry'],copy=True),
                profile_row_design=deepcopy(sampler.design),strength_requirements=deepcopy(sampler.requirements)),
            'source_gas':dict(power_after_gearbox_W=sampler.result['power_W'],
                air_mass_flow_kg_s=sampler.result['G'],fuel_mass_flow_kg_s=sampler.result['m_fuel'],
                stages=sampler.stages),
            'material':sampler.requirements,'gas_units':{'pressure':'Pa','temperature':'K',
                'mass_flow':'kg/s','specific_enthalpy':'J/kg','length':'m','fuel_display':'kg/h'},
            'options':options,'velocity_coefficients':{},'gas_balances':[],
            'limitations':['Geometry is not CFD/experimental proof of no separation or efficiency.',
            'Reynolds uses a documented air-viscosity proxy, not measured mixture transport.',
            'Source variable-cp thermodynamics and independent pressure-equilibrium audits are retained.',
            'IN738C/LM data are preliminary, not certified supply allowables.',
            'HCF NOT_CHECKED: grade-specific fatigue curve, forcing and damping are unavailable.']}
    row_objects=[];previous_count=0;axial=0.;all_parameters=[];actual_profiles={}
    count_sections=int(options.get('radial_sections',5))
    if count_sections!=5:raise ValueError('five source mass-fraction sections required')
    source_profiles=_source_profiles(sampler)
    report['gas_balances']=namespace['balances'](sampler.result).to_dict('records')
    report['source_design_checks']=namespace['design_checks'](sampler.result,
        namespace.get('RADIAL_LAW','alpha_const')).to_dict('records')
    report['violations'].extend(namespace.get('VINOGRADOV_SOURCE_ERRORS',[]))
    for check in report['source_design_checks']:
        if not check['Выполнено']:
            report['violations'].append(dict(name=check['Условие'],value=check['Значение'],
                limit=check['Ориентир'],source=check['Основание']))
    for i,stage in enumerate(sampler.stages):
        for row in ('SA','RK'):
            row_id=f'{row}{i+1}';record={'id':row_id,'stage':i+1,'row':row,'checks':[],'violations':[]}
            report['rows'].append(record)
            try:
                sections=sampler.row(i,row,count_sections)
                record['gas_sections']=[asdict(s) for s in sections]
                source_name=sampler.name(i,row)
                registry=sampler.design[source_name]
                record['upstream_radial_law']=namespace.get('RADIAL_LAW','alpha_const')
                record['source_geometry']=dict(name=source_name,count=int(registry['Z']),
                    row_width_m=float(registry['bax_m']),sections=registry['sections'],
                    root_tip_area_ratio=registry.get('root_tip_area_ratio'))
                audit=sampler.audit(i,row)
                for c in audit['checks']:
                    record['checks'].append(c.to_dict())
                    if not c.passed:record['violations'].append(c.to_dict())
                middle=sections[count_sections//2]
                provisional=_source_specs(sampler,i,row,sections,source_profiles,options)
                methods,preferred=select_row_methods(sections)
                record['individually_preferred_methods']=preferred
                record['radial_method_policy']='one analytic family; no binary radial geometry splice'
                minimum_open_fractions=[]
                for section in sections:
                    critical=sampler.throat_state(i,section.outlet,section.frame_speed_m_s,0.)['critical_flux_kg_m2_s']
                    minimum_open_fractions.append(section.outlet.density_kg_m3*section.outlet.axial_m_s/critical)
                record['mass_conservation_minimum_a_over_t']=minimum_open_fractions
                loss_evaluator,strength_evaluator=_source_row_evaluators(sampler,i,row,sections)
                if options.get('optimize_geometry',True):
                    fit=optimize_row(sections,provisional,previous_count=previous_count,
                        count_half_window=int(options.get('count_half_window',1)),max_iterations=int(options.get('max_iterations',4)),
                        methods=methods,area_law=options.get('area_law','intermediate'),area_blend=float(options.get('area_blend',.5)),
                        material=None,life_hours=sampler.resource_hours,minimum_open_fractions=minimum_open_fractions,
                        minimum_trailing_thickness_m=float(namespace['BOOK_DEFAULTS']['trailing_edge_m']),
                        root_tip_area_ratio=registry.get('root_tip_area_ratio'),
                        enforce_course_geometry=True,loss_evaluator=loss_evaluator,
                        strength_evaluator=strength_evaluator,maximum_back_angle_deg=15.,
                        geometry_regularization=float(options.get('geometry_regularization',1.)))
                    count,specs,profiles=fit.count,fit.specs,fit.profiles
                    record['optimized_area_law']={'law':fit.area_law,'blend':fit.area_blend}
                    record['area_parameterized']=any(item.get('area_parameterized',False)
                        for item in fit.archive if item.get('count')==count and item.get('status')=='course_constraint_solve')
                    export_json(out/f'{row_id}_search.json',fit.archive)
                    record.update(search_converged=fit.search_converged,search_objective=fit.objective,
                                  evaluated_candidates=len(fit.archive),archive_path=str(out/f'{row_id}_search.json'))
                    record['objective_model']='native mean row loss plus geometric guidance; actual native UKR/LM constraints'
                else:
                    count=int(registry['Z'])
                    specs=provisional
                    profiles=[build_profile(p,m) for p,m in zip(specs,methods)]
                neighbor_count=report['rows'][-2].get('count') if row=='RK' else None
                previous_row_count=previous_count
                record['count']=int(count);previous_count=count
                record['local_pitch_m']=[p.pitch_m for p in specs]
                record['methods']=methods;record['specifications']=[asdict(p) for p in specs]
                if previous_row_count and math.gcd(previous_row_count,count)!=1:
                    record['violations'].append(dict(name='source neighboring counts coprime',value=count,
                        previous=previous_row_count,gcd=math.gcd(previous_row_count,count),limit=1))
                if count%2!=(1 if row=='SA' else 0) or count<=previous_row_count:
                    record['violations'].append(dict(name='source count parity/order',value=count,
                        previous=previous_row_count,limit='odd SA, even RK; strictly increasing unique Z'))
                evaluations=[];capacities=[];row_parameters=[]
                for j,(profile,spec,section) in enumerate(zip(profiles,specs,sections)):
                    geom=validate_profile(profile,spec.pitch_m,samples=151)
                    refined=validate_profile(profile,spec.pitch_m,samples=301)
                    for c in [*geom['checks'],*refined['checks']]:
                        record['checks'].append(dict(section=j,**c.to_dict()))
                        if not c.passed:record['violations'].append(dict(section=j,**c.to_dict()))
                    if not profile.metadata.get('construction_valid',True):
                        record['violations'].append(dict(section=j,name='construction boundary failure',metadata=profile.metadata))
                    channel=analyze_channel(profile,spec.pitch_m,samples=int(options.get('channel_samples',41)))
                    thickness_analysis=None
                    if all(c.passed for c in [*geom['checks'],*refined['checks']]
                           if c.name in ('closed joints','positive area','self intersections')):
                        thickness_analysis=validate_prescribed_thickness(profile,
                            diameter_tolerance_m=float(options.get('thickness_tolerance_m',1e-6)),
                            max_cells=int(options.get('thickness_max_cells',20001)))
                        record.setdefault('thickness_validation',[]).append(dict(section=j,
                            **{k:v for k,v in thickness_analysis.items() if k!='checks'},
                            checks=[c.to_dict() for c in thickness_analysis['checks']]))
                        for check in thickness_analysis['checks']:
                            record['checks'].append(dict(section=j,**check.to_dict()))
                            if not check.passed:record['violations'].append(dict(section=j,**check.to_dict()))
                    values=_parameters(section,spec,profile,channel,thickness_analysis);values.update(count=count,section=j)
                    if values['back_angle_deg']>15:
                        record['violations'].append(dict(section=j,name='back angle beyond selected method domain',
                            value=values['back_angle_deg'],limit=15.,units='deg',
                            source='User specification §7: maximum 15 degrees within this method'))
                    elif not 5<=values['back_angle_deg']<=10:
                        report['warnings'].append(dict(row=row_id,section=j,name='measured back angle outside recommendation',
                            value=values['back_angle_deg'],recommended_range_deg=[5.,10.],
                            classification='empirical recommendation; user-selected method ceiling is separately 15 degrees'))
                    row_parameters.append(values);all_parameters.append(values)
                    actual_in,actual_out=_boundary_angles(profile)
                    for name,error in [('metal inlet angle',actual_in-spec.inlet_metal_deg),('metal outlet angle',actual_out-spec.outlet_metal_deg),
                                       ('trailing wedge',values['trailing_wedge_deg']-spec.trailing_wedge_deg)]:
                        if abs(error)>1e-5:record['violations'].append(dict(section=j,name=name,value=error,limit=1e-5,units='deg',classification='numerical'))
                    if channel.throat_at_sample_boundary:
                        record['violations'].append(dict(section=j,name='global throat not bracketed',value=channel.throat_width_m,units='m'))
                    expansion=float(max(channel.width_m)/channel.width_m[0])
                    if expansion>1.1:
                        report['warnings'].append(dict(row=row_id,section=j,name='subsonic entrance expansion guidance',value=expansion,limit=1.1,classification='empirical recommendation; CFD required'))
                    lower,upper=(.1,.1) if row=='SA' else (float(np.interp(section.span_fraction,[0,.5,1],[.20,.10,.04])),float(np.interp(section.span_fraction,[0,.5,1],[.30,.15,.05])))
                    if not lower<=values['relative_thickness']<=upper:
                        report['warnings'].append(dict(row=row_id,section=j,name='relative thickness starting guidance',value=values['relative_thickness'],range=[lower,upper],classification='empirical, p13'))
                    if row=='RK' and j==len(sections)-1 and values['maximum_thickness_m']<.002:
                        report['warnings'].append(dict(row=row_id,section=j,name='2–3mm tip guidance is not scale compatible',actual_m=values['maximum_thickness_m'],lower_m=.002,
                            chord_required_at_5percent_m=.002/.05,classification='manufacturing recommendation requires process-specific qualification, p13'))
                    if section.row=='RK' and abs(section.flow_inlet_deg-section.flow_outlet_deg)<=10:
                        variation=float(np.ptp(channel.width_m)/channel.width_m.mean())
                        record.setdefault('active_channel_variation',[]).append(variation)
                    area=count*channel.throat_width_m*section.radial_weight_m
                    streamtube_mass=2*np.pi*section.radius_m*section.radial_weight_m*section.outlet.density_kg_m3*section.outlet.axial_m_s
                    capacity_check=sampler.throat_state(i,section.outlet,section.frame_speed_m_s,streamtube_mass/area)
                    capacities.append(capacity_check['critical_flux_kg_m2_s']*area)
                    values['critical_capacity_kg_s']=capacities[-1]
                    values['throat_area_strip_m2']=area
                    values['required_throat_state']=capacity_check
                    values['streamtube_mass_kg_s']=streamtube_mass
                    if not capacity_check['possible']:
                        record['violations'].append(dict(section=j,name='local streamtube choke conflict',**capacity_check))
                    evaluations.append(dict(profile=profile,spec=spec,section=section,geometry=geom,channel=channel,parameters=values))
                    actual_profiles[(source_name,j)]=_source_profile_record(profile,spec,section,
                        channel.throat_width_m,count,j/(count_sections-1))
                    criteria,source_channel=_source_channel_checks(sampler,i,row,j,profile,spec,section,channel,values)
                    values['historical_source_profile_comparison']=dict(criteria=criteria,measurements=source_channel)
                    actual_profiles[(source_name,j)]['intersections']=source_channel['intersections']
                    actual_profiles[(source_name,j)]['self_intersection']=source_channel['self_intersection']
                    for name,passed in criteria.items():
                        if not passed:
                            comparison=dict(section=j,name='prior profile guide: '+name,
                                measurements=source_channel,relative_thickness=values['relative_thickness'],
                                actual_relative_pitch=values['actual_relative_pitch'])
                            if name in ('вписанные круги','нет пересечений'):
                                record['violations'].append(comparison)
                            else:
                                comparison['classification']='historical empirical profile guide; Vinogradov uses its own recommendations'
                                report['warnings'].append(dict(row=row_id,**comparison))
                record['loss_sample_count']=len(evaluations)
                record['edge_radius_sensitivity']=edge_radius_sensitivity(specs[len(specs)//2],methods[len(methods)//2])
                record['coarse_critical_capacity_kg_s']=float(sum(capacities))
                record['required_mass_flow_kg_s']=sections[0].mass_flow_kg_s
                record['area_laws']={law:area_distribution(np.linspace(0,1,count_sections),row_parameters[0]['area_m2'],row_parameters[-1]['area_m2'],law).tolist()
                                     for law in ('linear','exponential','intermediate')}
                if row=='RK':
                    bound=namespace['candidate_temperature_bound'](sampler.radials[i],stage,sampler.cfg)
                    record['temperature_bound']=bound
                    props=[section_properties(p) for p in profiles]
                    modulus=material.elastic_modulus([bound['upper_K']]*len(sections))
                    if np.isfinite(modulus).all():
                        modes=beam_modes([s.radius_m for s in sections],[p.area_m2 for p in props],
                                         [p.centroidal_moments_m4[1] for p in props],modulus,material.density_kg_m3,sections[0].speed_rpm)
                        record['axial_bending_frequencies_Hz']=modes['frequencies_Hz'].tolist()
                        record['Campbell']=campbell_screening(modes['frequencies_Hz'],sections[0].speed_rpm,neighbor_count)
                else:
                    static=assess_stator(sampler,i,sections,profiles,count)
                    source_rows=static['table'].to_dict('records')
                    record['temperature_bound']=static['temperature_bound']
                    record['strength']={'rows':[dict(radius_m=v['r_m'],
                        normal_min_Pa=v['sigma_normal_min_MPa']*1e6,
                        normal_max_Pa=v['sigma_normal_max_MPa']*1e6,
                        lm_allowable_Pa=v['allowable_MPa']*1e6,method=v['method'],
                        temperature_K=v['T_bound_K'],lm_margin=v['margin']) for v in source_rows],
                        'source_rows':source_rows,'required_lm_margin':sampler.requirements['required_margin'],
                        'mechanical_speed_rpm':0.,'limitations':static['limitations']}
                    export_csv(out/f'{row_id}_strength.csv',source_rows)
                    for value in source_rows:
                        if not value['passed']:
                            record['violations'].append(dict(name='stationary SA normal-stress/LM assessment',**value))
                record['vibration_strength_status']='NOT_CHECKED: no grade-specific fatigue curve, damping or actual harmonic forcing; elastic modes where available are screening only'
                record['violations'].append(dict(name='vibration strength',status='NOT_CHECKED',reason=record['vibration_strength_status']))
                radii=[s.radius_m for s in sections]
                blade=stack_sections(profiles,radii,row,count_per_side=81,axial_origin_m=axial,mesh_converged=not options.get('adaptive_sections',True))
                record['centroid_scatter']=centroid_scatter(blade)
                record['stacking_basis']='leading edge' if row=='SA' else 'centroid'
                row_object=dict(id=row_id,evaluations=evaluations,blade=blade,count=count)
                row_objects.append(row_object)
                export_csv(out/f'{row_id}_parameters.csv',row_parameters)
                export_dxf(out/f'{row_id}.dxf',profiles,[p.pitch_m for p in specs])
                record['parameters_path']=str(out/f'{row_id}_parameters.csv')
                frac=np.array([s.span_fraction for s in sections])
                interpolators={name:PchipInterpolator(frac,[getattr(p,name) for p in specs]) for name in (
                    'axial_width_m','stagger_deg','relative_thickness','thickness_location',
                    'leading_radius_m','trailing_radius_m','incidence_deg','deviation_deg',
                    'trailing_wedge_deg','back_angle_deg','throat_multiplier','origin_y_m')}
                area_ends=[section_properties(p,count_per_side=101).area_m2 for p in (profiles[0],profiles[-1])]
                @lru_cache(maxsize=512)
                def radial_section(radius):
                    f=(radius-radii[0])/(radii[-1]-radii[0])
                    return sampler.section(i,row,float(f),sections[0].height_m/(count_sections-1))
                @lru_cache(maxsize=128)
                def builder(radius):
                    f=(radius-radii[0])/(radii[-1]-radii[0]);s=radial_section(radius)
                    coordinates={name:float(interpolator(f)) for name,interpolator in interpolators.items()}
                    p=replace(specs[0],**coordinates,
                        inlet_metal_deg=s.flow_inlet_deg+coordinates['incidence_deg'],
                        outlet_metal_deg=s.flow_outlet_deg-coordinates['deviation_deg'],pitch_m=2*np.pi*radius/count)
                    profile=build_profile(p,methods[0])
                    if record.get('area_parameterized'):
                        law=record['optimized_area_law']
                        target=float(area_distribution(f,*area_ends,law['law'],law['blend']))
                        actual=section_properties(profile,count_per_side=101).area_m2
                        scale=math.sqrt(target/actual)
                        p=replace(p,axial_width_m=p.axial_width_m*scale,
                            leading_radius_m=p.leading_radius_m*scale,trailing_radius_m=p.trailing_radius_m*scale)
                        profile=build_profile(p,methods[0])
                    check=validate_profile(profile,p.pitch_m,samples=81)
                    if not check['passed']:
                        raise ConstructionError('Intermediate radial profile fails geometry validation',
                            {'radius_m':radius,'checks':[c.to_dict() for c in check['checks'] if not c.passed]})
                    return profile
                def strip_at_radius(radius):
                    s=radial_section(radius)
                    throat=refined_throat(builder(radius),2*np.pi*radius/count)[0]
                    critical=sampler.throat_state(i,s.outlet,s.frame_speed_m_s,0.)['critical_flux_kg_m2_s']
                    return s.outlet,throat,critical
                capacity_review=integrate_annular_capacity(radii,count,strip_at_radius)
                record['annular_capacity']=capacity_review
                capacity=capacity_review['critical_capacity_kg_s']
                error=capacity_review['estimated_absolute_error_kg_s']
                record['critical_capacity_kg_s']=capacity
                if not capacity_review['converged']:
                    record['violations'].append(dict(name='annular capacity quadrature unconverged',**capacity_review))
                required=sections[0].mass_flow_kg_s
                if capacity-error<required:
                    record['violations'].append(dict(name='insufficient choked capacity' if capacity+error<required else 'annular capacity numerically unresolved',
                        value=capacity,estimated_error_kg_s=error,limit=required,units='kg/s'))
                if capacity_review['minimum_sampled_local_margin']<0:
                    record['violations'].append(dict(name='radial local choke conflict',
                        value=capacity_review['minimum_sampled_local_margin'],
                        radius_m=capacity_review['minimum_sampled_margin_radius_m'],limit=0.))
                if options.get('adaptive_sections',True):
                    rr,pp,error,converged=adaptive_sections(builder,radii[0],radii[-1],tolerance_m=float(options.get('cad_tolerance_m',2e-5)),max_sections=int(options.get('max_radial_sections',65)),row=row)
                    blade=stack_sections(pp,rr,row,count_per_side=81,axial_origin_m=axial,interpolation_error_m=error,mesh_converged=converged)
                    record['adaptive_radial_mesh']={'count':len(rr),'error_m':error,'converged':converged}
                    if not converged:record['violations'].append(dict(name='radial CAD mesh unconverged',value=error,limit=options.get('cad_tolerance_m',2e-5)))
                row_object['blade']=blade
                width_actual=max(p.axial_width_m for p in specs)
                record['axial_width_m']=width_actual
                axial+=width_actual
            except (ConstructionError,ValueError,KeyError,np.linalg.LinAlgError,FloatingPointError) as exc:
                details=dict(getattr(exc,'details',{}))
                if 'archive' in details:
                    archive=details.pop('archive');archive_path=out/f'{row_id}_search.json'
                    export_json(archive_path,archive)
                    details.update(archive_path=str(archive_path),attempts=len(archive))
                record['violations'].append(dict(name='construction/calculation failure',message=str(exc),details=details))
            report['violations'].extend(dict(row=row_id,**v) for v in record['violations'])
    # The original meridional function owns row planes, free gaps, post-loss
    # mixing and its 15-degree wall constraint. Do not transplant old gap rules.
    row_widths={}
    for record in report['rows']:
        obj=next((v for v in row_objects if v['id']==record['id']),None)
        source_name=sampler.name(record['stage']-1,record['row'])
        row_widths[source_name]=(max(float(np.ptp(p[:,0])) for p in obj['blade'].contours_m)
            if obj else float(sampler.design[source_name]['bax_m']))
    try:
        channel=namespace['meridional_channel'](sampler.result,sampler.radials,row_widths=row_widths)
        report['meridional_channel']=dict(max_angle_deg=channel['max_angle_deg'],
            blade_rows=channel['blade_rows'].to_dict('records'),gaps=channel['gaps'].to_dict('records'),
            stations=channel['stations'].to_dict('records'))
        report['meridional_channel']['basis']='source meridional solver with actual analytic row envelopes'
        report['meridional_channel']['contour']={key:channel['contour'][key].to_numpy()
            for key in ('x','rhub','rtip')}
    except (ValueError,ArithmeticError) as exc:
        channel=sampler.channel
        report['violations'].append(dict(name='source meridional envelope constraint',
            message=str(exc),attempted_row_widths_m=row_widths))
        report['meridional_channel']=dict(basis='original source layout ONLY; analytic envelopes rejected',
            max_angle_deg=channel['max_angle_deg'],
            contour={key:channel['contour'][key].to_numpy() for key in ('x','rhub','rtip')})
    channel_rows=channel['blade_rows'].set_index('name')
    for record in report['rows']:
        blade_row=channel_rows.loc[sampler.name(record['stage']-1,record['row'])]
        record['axial_start_m']=float(blade_row.x_le)
        record['axial_envelope_m']=float(blade_row.bax)
        record['axial_gap_m']=float(blade_row.gap_after) if blade_row.gap_after is not None else 0.
        obj=next((v for v in row_objects if v['id']==record['id']),None)
        if obj is not None:
            blade=obj['blade'];origin=min(float(p[:,0].min()) for p in blade.contours_m)
            shift=np.array([record['axial_start_m']-origin,0.])
            obj['blade']=replace(blade,contours_m=tuple(p+shift for p in blade.contours_m),
                offsets_m=blade.offsets_m+shift,centroids_m=blade.centroids_m+shift,
                leading_points_m=blade.leading_points_m+shift)
    report['parameters']=all_parameters
    report['velocity_coefficients']={'phi':[None,None],'psi':[None,None]}
    if len(actual_profiles)==20:
        try:
            local,mean=namespace['cascade_loss_review'](sampler.result,sampler.radials,
                sampler.channel,actual_profiles)
            report['losses']=local.to_dict('records')
            report['row_loss_coefficients']=mean.to_dict('records')
            by_name=mean.set_index('Венец')['Коэффициент по потерям']
            report['velocity_coefficients']={
                'phi':[float(by_name[f'СА{i}']) for i in (1,2)],
                'psi':[float(by_name[f'РК{i}']) for i in (1,2)]}
        except (ValueError,KeyError,ArithmeticError) as exc:
            report['violations'].append(dict(name='source loss evaluation',message=str(exc)))
    strength=sampler.strength(actual_profiles)
    report['strength']={key:value for key,value in strength.items() if key!='rotor_results'}
    report['strength']['table']=strength['table'].to_dict('records')
    report['strength']['project_normal_stress_limit_MPa']=300.0  # explicit user design cap, not material allowable
    for record in report['rows']:
        if record['row']!='RK':continue
        table=strength['table']
        rows=(table.loc[table.rotor==record['stage']].to_dict('records') if not table.empty else [])
        record['strength']={'rows':[dict(radius_m=v['r_m'],
            normal_min_Pa=v['sigma_normal_min_MPa']*1e6,
            normal_max_Pa=v['sigma_normal_max_MPa']*1e6,lm_allowable_Pa=v['allowable_MPa']*1e6,
            method=v['method'],temperature_K=v['T_bound_K'],lm_margin=v['margin']) for v in rows],
            'source_rows':rows,'required_lm_margin':sampler.requirements['required_margin']}
        for value in rows:
            peak=max(abs(value['sigma_normal_min_MPa']),abs(value['sigma_normal_max_MPa']))
            if peak>300.0:
                violation=dict(name='project actual absolute normal stress',value=peak,
                    limit=300.0,units='MPa',section=value['section'],method=value['method'],
                    source='User specification: maximum 300 MPa, separate from material/LM')
                record['violations'].append(violation)
                report['violations'].append(dict(row=record['id'],**violation))
            if not value['passed']:
                violation=dict(name='source normal-stress LM margin',**value)
                record['violations'].append(violation)
                report['violations'].append(dict(row=record['id'],**violation))
    if not strength['passed']:
        report['violations'].append(dict(name='source strength assessment',reasons=strength['reasons']))
    for record in report['rows']:
        params=[p for p in all_parameters if p['stage']==record['stage'] and p['row']==record['row']]
        if len(params)!=5:continue
        if record['row']=='RK':
            ratio=params[0]['area_m2']/params[-1]['area_m2']
            target=sampler.design[sampler.name(record['stage']-1,'RK')].get('root_tip_area_ratio')
            if target is not None and not math.isclose(ratio,float(target),rel_tol=.005):
                report['violations'].append(dict(row=record['id'],name='source rotor taper mismatch',
                    value=ratio,limit=target))
        chord_ok=(params[0]['chord_m']>params[-1]['chord_m'] if record['row']=='RK'
                  else params[0]['chord_m']<params[-1]['chord_m'])
        if not chord_ok:
            report['violations'].append(dict(row=record['id'],name='source chord direction',
                root_m=params[0]['chord_m'],tip_m=params[-1]['chord_m'],
                limit='root>tip RK; root<tip SA'))
        if record['row']=='RK' and not all(a['area_m2']>b['area_m2'] for a,b in zip(params,params[1:])):
            report['violations'].append(dict(row=record['id'],name='source rotor areas must decrease',
                value=[p['area_m2'] for p in params],limit='strict decrease root to tip'))
        for p in params:
            spec=record['specifications'][p['section']]
            checks=[('source finite trailing edge',2*spec['trailing_radius_m'],.0005,
                     2*spec['trailing_radius_m']>=.0005),
                    ('source leading radius/axial width',spec['leading_radius_m']/spec['axial_width_m'],
                     '.03… .04',.03<=spec['leading_radius_m']/spec['axial_width_m']<=.04)]
            for name,value,limit,passed in checks:
                if not passed:
                    check=dict(row=record['id'],section=p['section'],name=name,value=value,limit=limit)
                    if name=='source leading radius/axial width':
                        check['classification']='historical empirical nose guide, not a Vinogradov restriction'
                        report['warnings'].append(check)
                    else:report['violations'].append(check)
        for p in params:
            wedge=record['specifications'][p['section']]['trailing_wedge_deg']
            if not 2.<=wedge<=5.:
                report['warnings'].append(dict(row=record['id'],section=p['section'],
                    name='prior profile trailing wedge guide',value=wedge,limit='2…5 deg',
                    classification='historical empirical guide; Vinogradov p12 recommends 4…8 deg'))
    for c in report['gas_balances']:
        if not c['passed']:report['violations'].append(c)
    for record in report['rows']:
        record['passed']=not(record['violations'] or any(v.get('row')==record['id'] for v in report['violations']))
        record['status']='FAILED' if not record['passed'] else 'CONSTRUCTED'
    report['passed']=bool(len(row_objects)==2*len(sampler.stages) and not report['violations'])
    report['status']='PRELIMINARY_PASS' if report['passed'] else 'FAILED'
    if options.get('export_step',True) and row_objects:
        try:report['STEP']=export_step(out/'turbine_rows.step',[v['blade'] for v in row_objects],[v['count'] for v in row_objects])
        except (ValueError,RuntimeError) as exc:report['violations'].append(dict(name='STEP export rejected',message=str(exc)));report['passed']=False
    if not report['passed']:report['status']='FAILED'
    from .reporting import render_report
    render_report(report,row_objects,out,namespace)
    export_json(out/'report.json',report)
    export_csv(out/'violations.csv',report['violations'] or [{'name':'none','status':'no violations'}])
    return report

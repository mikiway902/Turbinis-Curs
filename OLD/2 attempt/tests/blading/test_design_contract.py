"""Consumer-visible frame, geometry, mechanics and real CAD regression checks."""
import math
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import trapezoid
from vinogradov.models import FlowStation,RadialSectionInput,ProfileSpec
from vinogradov.blade_geometry import (optimal_relative_pitch,count_candidates,area_distribution,geometric_parameters)
from vinogradov.parabolic_profile import build_parabolic_profile
from vinogradov.validation import validate_profile
from vinogradov.blade_stacking import stack_sections,adaptive_sections
from vinogradov.blade_strength import (IN738C,centrifugal_loads,section_stress,beam_modes,vibration_response)
from vinogradov.radial_equilibrium import audit_radial_state,resample_sections
from vinogradov.cascade_throat import mass_flux_at_mach,perfect_gas_throat_flow
from vinogradov.course_adapter import air_viscosity_proxy
from vinogradov.cad_export import export_json,export_csv,export_dxf,export_step


def profile_spec(scale=1.):
    chord=.1/math.sin(math.radians(40))
    return ProfileSpec(.1*scale,40,90,17,.69*chord*scale,.024*chord*scale,.006*chord*scale,origin_y_m=.1*scale)


def section(radius=.15):
    def state(ca,cu):
        T=1000.;p=2e5;R=287.;g=1.33;cp=g*R/(g-1)
        Tt=T+(ca*ca+cu*cu)/(2*cp)
        return FlowStation(ca,cu,T,p,p/R/T,Tt,p*(Tt/T)**(g/(g-1)),4e-5,R,g)
    return RadialSectionInput(1,'RK',radius,.1,.15,.2,30000,.4,.7,state(100,350),state(110,-20),90,17,80,30,35,200,.025,inlet_radius_m=radius*.95)


def test_pitch_uses_three_distinct_angle_concepts_and_cube_root_scope():
    got=optimal_relative_pitch(80,25,83,23,.12)
    expected=.55*((180/(180-80-25))*(math.sin(math.radians(83))/math.sin(math.radians(23))))**(1/3)*.88
    assert got==pytest.approx(expected,rel=1e-14)
    assert got!=pytest.approx(optimal_relative_pitch(80,25,80,25,.12))
    with pytest.raises(ValueError):optimal_relative_pitch(90,90,90,15,.1)


def test_single_integer_count_has_local_radial_pitch_and_order():
    previous=0
    for row in ['SA','RK']*3:
        candidates=count_candidates(.2,.015,.8,row,previous=previous)
        adopted=candidates[0];Z=adopted['count']
        assert Z>previous and Z%2==(row=='SA')
        radii=np.array([.07,.1,.13]);t=2*np.pi*radii/Z
        np.testing.assert_allclose(2*np.pi*radii/t,Z)
        assert adopted['pitch_m']==pytest.approx(t[1])
        previous=Z


def test_area_law_is_area_not_thickness_and_has_exact_endpoints():
    s=np.array([0,.25,.5,.75,1])
    linear=area_distribution(s,4e-5,1e-5,'linear')
    exponential=area_distribution(s,4e-5,1e-5,'exponential')
    mixed=area_distribution(s,4e-5,1e-5,'intermediate',.3)
    np.testing.assert_allclose(mixed,.7*linear+.3*exponential)
    np.testing.assert_allclose(mixed[[0,-1]],[4e-5,1e-5],rtol=1e-14)
    assert np.all(exponential<=linear)


def test_real_profile_tangency_and_bend_angles_close():
    spec=profile_spec();p=build_parabolic_profile(spec)
    checks=validate_profile(p,spec.pitch_m,samples=151)
    assert checks['passed'],checks
    parameters=geometric_parameters(p,spec,samples=31)
    assert parameters['inlet_bend_deg']+parameters['outlet_bend_deg']==pytest.approx(180-90-17)
    assert parameters['maximum_thickness_m']>0
    assert parameters['construction_chord_m']==pytest.approx(.1/math.sin(math.radians(40)))
    assert parameters['chord_m']!=pytest.approx(parameters['axial_width_m'])
    assert not validate_profile(p,spec.pitch_m*.1,samples=151)['passed']


def test_stacking_respects_sa_leading_edge_rk_centroid_and_physical_mirror():
    profiles=[build_parabolic_profile(profile_spec(v)) for v in (1.,.95,.9)]
    rk=stack_sections(profiles,[.2,.21,.22],'RK')
    sa=stack_sections(profiles,[.2,.21,.22],'SA')
    np.testing.assert_allclose(rk.centroids_m,0.,atol=1e-14)
    np.testing.assert_allclose(sa.leading_points_m,0.,atol=1e-14)
    # Source profiles turn toward -V; physical SA turns toward +shaft theta.
    assert sa.contours_m[0][-20,1]>sa.contours_m[0][0,1]
    points=rk.ring(4)
    assert len(points)==4
    for blade in points:
        for r,p in zip(rk.radii_m,blade):np.testing.assert_allclose(np.hypot(p[:,1],p[:,2]),r,atol=1e-14)


def test_adaptive_radial_grid_measures_nonlinear_shape_error():
    def builder(r):return build_parabolic_profile(profile_spec(1+2*(r-.2)**2))
    r,p,error,converged=adaptive_sections(builder,.2,.3,tolerance_m=1e-5,max_sections=17)
    assert converged and error<=1e-5
    assert r[0]==.2 and r[-1]==.3 and .25 in r
    # A tighter tolerance with a three-section cap must not report convergence.
    _,_,error,converged=adaptive_sections(builder,.2,.3,tolerance_m=1e-10,max_sections=3)
    assert not converged and error>1e-10


def test_centrifugal_constant_area_exact_and_quadratic_positive_gate():
    r=np.array([.1,.15,.2]);A=np.full(3,2e-5);n=30000.;density=8110.
    expected=density*(2*np.pi*n/60)**2*(r[-1]**2-r**2)/2
    for method in ('trapezoid','quadratic'):
        result=centrifugal_loads(r,A,n,density,method=method)
        np.testing.assert_allclose(result['stress_Pa'],expected,rtol=1e-14,atol=1e-7)
        assert result['force_N'][-1]==0
    # Positive endpoint/midpoint areas can still create a negative quadratic.
    with pytest.raises(ValueError):centrifugal_loads(r,[1e-5,1e-8,1e-8],n,density,method='quadratic')


def test_real_section_unsymmetric_bending_not_rectangle_substitution():
    p=build_parabolic_profile(profile_spec())
    from vinogradov.section_properties import section_properties
    prop=section_properties(p)
    stress=section_stress(p,1000,[2.,-3.])
    xy=p.contour()-prop.centroid_m
    # Independent signed-triangle Gaussian integration recovers the APPLIED
    # resultant force and bending moments from the returned stress field.
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    force=mx=my=0.
    for j in range(len(xy)-1):
        vertices=np.array([[0.,0.],xy[j],xy[j+1]])
        values=np.array([1000/prop.area_m2,stress[j],stress[j+1]])
        samples=bary@vertices;sigma=bary@values
        area=(xy[j,0]*xy[j+1,1]-xy[j,1]*xy[j+1,0])/2
        force+=area*sigma.mean();mx+=area*(samples[:,1]*sigma).mean()
        my-=area*(samples[:,0]*sigma).mean()
    if force<0:force,mx,my=-force,-mx,-my
    np.testing.assert_allclose([force,mx,my],[1000,2,-3],rtol=1e-4)


def test_material_temperature_resource_and_no_extrapolation():
    material=IN738C()
    T=(1500-32)*5/9+273.15
    # Independent literal Table IX IN738C 1000h, NOT LC 10000h.
    assert material.rupture_allowable(T,1000)==pytest.approx(48000*6894.757293168)
    assert material.rupture_allowable(T,10000)<material.rupture_allowable(T,300)
    assert np.isnan(material.rupture_allowable(1800,10000))
    assert np.isnan(material.yield_strength(1800))
    assert material.elastic_modulus((1400-32)*5/9+273.15)==pytest.approx(23.2e6*6894.757293168)


def test_beam_frequency_analytic_limit_and_centrifugal_stiffening():
    r=np.linspace(.1,.2,31);A=np.full_like(r,3e-4);I=np.full_like(r,1e-9);E=np.full_like(r,150e9)
    stationary=beam_modes(r,A,I,E,8110.,0.)
    expected=1.875104068711961**2/(2*np.pi*.1**2)*math.sqrt(150e9*1e-9/(8110*3e-4))
    assert stationary['frequencies_Hz'][0]==pytest.approx(expected,rel=1e-4)
    rotating=beam_modes(r,A,I,E,8110.,30000.)
    assert rotating['frequencies_Hz'][0]>stationary['frequencies_Hz'][0]
    with pytest.raises(ValueError):beam_modes(r,A,I,E*np.nan,8110.)


def test_modal_response_requires_actual_harmonic_amplitude_and_fatigue_data():
    r=np.linspace(.1,.2,11);a=np.full_like(r,3e-4);I=np.full_like(r,1e-9);E=np.full_like(r,150e9)
    beam=beam_modes(r,a,I,E,8110.)
    force=np.zeros(beam['mass'].shape[0]-2);force[-2]=1.
    derivative=np.ones((1,len(beam['frequencies_Hz'])))*1e5
    off=vibration_response(beam,force,0.,.02,endurance_limit_Pa=1e8,stress_per_unit_mode_Pa=derivative)
    on=vibration_response(beam,force,float(beam['frequencies_Hz'][0]),.02,endurance_limit_Pa=1e8,stress_per_unit_mode_Pa=derivative)
    assert on['amplitude_Pa'][0]>off['amplitude_Pa'][0]
    with pytest.raises(ValueError):vibration_response(beam,force,0.,0.,endurance_limit_Pa=1e8,stress_per_unit_mode_Pa=derivative)


def test_radial_audit_accepted_pressure_equation_and_mass():
    r=np.linspace(.1,.2,201);R=287.;T=np.full_like(r,1000.);cu=20./r;ca=np.full_like(r,100.)
    p=2e5*np.exp(-20.**2/(2*R*1000)*(1/r**2-1/.15**2))
    mass=trapezoid(2*np.pi*r*p/(R*T)*ca,r)
    audit=audit_radial_state(r,ca,cu,T,p,R,1150.,mass)
    assert audit['passed'],audit['checks']
    bad=audit_radial_state(r,ca,cu,T,np.full_like(r,2e5),R,1150.,mass)
    assert not bad['passed']


def test_mapped_rotor_inlet_uses_its_own_radius():
    s=section()
    assert s.frame_inlet_speed_m_s==pytest.approx(.95*s.frame_speed_m_s)
    assert s.inlet.speed(s.frame_inlet_speed_m_s)!=pytest.approx(s.inlet.speed(s.frame_speed_m_s))
    states=[section(v) for v in (.1,.15,.2)]
    samples=resample_sections(states,[.1,.125,.15,.175,.2])
    assert samples[1].inlet_radius_m==pytest.approx(.125*.95)


def test_flow_mass_flux_no_double_discharge_or_throat_multiplier():
    flux=mass_flux_at_mach(2e5,1000.,1.33,287.,1.)
    flow=perfect_gas_throat_flow(stagnation_pressure_pa=2e5,stagnation_temperature_k=1000.,
        back_pressure_pa=0,gamma=1.33,gas_constant_j_kg_k=287.,area_m2=.001)
    assert flow.mass_flow_kg_s==pytest.approx(flux*.001)
    assert mass_flux_at_mach(2e5,1000.,1.33,287.,.2)<flux
    assert 1e-5<air_viscosity_proxy(300)<3e-5


def test_actual_dxf_json_csv_and_step_roundtrip(tmp_path):
    import json
    import ezdxf
    import cadquery as cq
    p=build_parabolic_profile(profile_spec());blade=stack_sections([p,p,p],[.2,.21,.22],'RK',count_per_side=41)
    export_json(tmp_path/'data.json',{'finite':1.,'unknown':float('nan'),'array':np.array([2,3])})
    assert json.loads((tmp_path/'data.json').read_text())=={'finite':1.,'unknown':None,'array':[2,3]}
    export_csv(tmp_path/'parameters.csv',[{'area_m2':blade.areas_m2[0]}])
    export_dxf(tmp_path/'blade.dxf',[p],[profile_spec().pitch_m])
    dxf=ezdxf.readfile(tmp_path/'blade.dxf')
    assert dxf.units==ezdxf.units.MM
    assert len(dxf.modelspace().query('SPLINE'))==2
    assert len(dxf.modelspace().query('ARC'))==6
    result=export_step(tmp_path/'blade.step',[blade],[3])
    imported=cq.importers.importStep(str(tmp_path/'blade.step')).val()
    assert imported.isValid() and len(imported.Solids())==3
    expected=3*blade.areas_m2[0]*.02
    assert imported.Volume()*1e-9==pytest.approx(expected,rel=.02)
    assert result['blade_count']==3


def test_axial_gap_optimization_keeps_widths_separate_and_reports_conflicts():
    from vinogradov.blade_optimization import optimize_axial_gaps
    rows=np.array([[.01,.10,.12],[.012,.10,.121],[.009,.10,.122]])
    result=optimize_axial_gaps(rows)
    assert result['passed'] and result['converged']
    np.testing.assert_allclose(result['gaps_m'],rows[:-1,0]*.2,atol=1e-9)
    expected=np.degrees(np.arctan(np.diff(rows[:,2])/(result['gaps_m']+rows[1:,0])))
    np.testing.assert_allclose(result['opening_deg'],expected)
    impossible=optimize_axial_gaps([[.005,.1,.12],[.005,.1,.22]])
    assert not impossible['passed']
    assert impossible['opening_deg'][0]>25.


def test_required_throat_mass_conservation_and_choked_failure():
    from vinogradov.cascade_throat import required_throat_mach
    flux=mass_flux_at_mach(2e5,900,1.33,287,.6)
    state=required_throat_mach(mass_flux_kg_m2_s=flux,stagnation_pressure_pa=2e5,
        stagnation_temperature_k=900,gamma=1.33,gas_constant_j_kg_k=287)
    assert state['possible'] and state['mach']==pytest.approx(.6,abs=1e-12)
    critical=mass_flux_at_mach(2e5,900,1.33,287,1.)
    rejected=required_throat_mach(mass_flux_kg_m2_s=critical*1.01,stagnation_pressure_pa=2e5,
        stagnation_temperature_k=900,gamma=1.33,gas_constant_j_kg_k=287)
    assert not rejected['possible'] and rejected['mach'] is None
    assert rejected['excess_fraction']==pytest.approx(.01)


def test_row_selection_does_not_create_a_radial_geometry_discontinuity():
    from vinogradov.cascade_geometry import select_row_methods,select_method
    sections=[replace(section(r),speed_rpm=6000.,reaction=reaction,beta1_deg=angle,beta2_deg=25.)
              for r,reaction,angle in [(.1,.02,26.),(.15,.3,45.),(.2,.6,65.)]]
    methods,preferred=select_row_methods(sections)
    assert preferred[0]=='active' and preferred[-1]=='double_parabolic'
    assert methods==['double_parabolic']*3
    too_fast=replace(sections[0],outlet=replace(sections[0].outlet,axial_m_s=1000.))
    with pytest.raises(ValueError):select_method(too_fast)


def test_gas_strip_torque_preserves_angular_momentum_at_unequal_radii():
    from vinogradov.blade_strength import gas_loads
    sections=[section(r) for r in (.1,.15,.2)]
    count=42
    loads=gas_loads(sections,count)
    for s,q in zip(sections,loads['force_per_span_N_m']):
        mass_per_span=2*np.pi*s.radius_m*s.outlet.density_kg_m3*s.outlet.axial_m_s
        specific_torque=s.inlet_radius_m*s.inlet.circumferential_m_s-s.radius_m*s.outlet.circumferential_m_s
        assert q[1]*count*s.radius_m==pytest.approx(mass_per_span*specific_torque)


def test_missing_stator_count_cannot_use_a_previous_rotor_excitation():
    from vinogradov.blade_strength import campbell_screening
    actual=campbell_screening([40000.,40150.],33000.,73)
    assert actual['relative_frequency_separation'][1]==pytest.approx(0.)
    assert actual['relative_frequency_separation'][0]==pytest.approx(150/40150)
    missing=campbell_screening([40000.,40150.],33000.,None)
    assert missing['status']=='NOT_CHECKED' and 'neighbor_blade_passing_Hz' not in missing


def test_diagnostic_missing_row_does_not_impose_unsourced_coprimality():
    next_rotor=count_candidates(.2,.015,.8,'RK',previous=70)
    assert next_rotor and next_rotor[0]['count']>70
    assert next_rotor[0]['count']%2==0


def test_modal_stress_matches_continuum_cantilever_root_curvature():
    from scipy.integrate import quad
    from vinogradov.blade_strength import modal_section_stress
    from vinogradov.section_properties import section_properties
    p=build_parabolic_profile(profile_spec());points=p.contour(101);properties=section_properties(points)
    r=np.linspace(.1,.2,33);L=r[-1]-r[0];E=2e11;density=8110.
    beam=beam_modes(r,np.full(len(r),properties.area_m2),
        np.full(len(r),properties.centroidal_moments_m4[1]),np.full(len(r),E),density,modes=1)
    result=modal_section_stress(beam,r,[points]*len(r),E)
    z=1.875104068711961;beta=z/L
    k=(np.cosh(z)+np.cos(z))/(np.sinh(z)+np.sin(z))
    def shape(x):
        return np.cosh(beta*x)-np.cos(beta*x)-k*(np.sinh(beta*x)-np.sin(beta*x))
    normalization=np.sqrt(density*properties.area_m2*quad(lambda x:shape(x)**2,0,L)[0])
    expected=E*np.max(np.abs(points[:,0]-properties.centroid_m[0]))*2*beta**2/normalization
    actual=np.max(np.abs(result['stress_per_unit_mode_Pa'][result['section_index']==0]))
    assert actual==pytest.approx(expected,rel=.002)


def test_closed_tangent_profile_still_must_fit_the_prescribed_axial_width():
    from vinogradov.double_parabolic_profile import build_double_parabolic_profile
    spec=ProfileSpec(.02236309352291337,44.07866116608724,32.36385724199586,
        27.260756849807482,.02104125693721863,.000782708273301968,
        .0002500000000000014,trailing_wedge_deg=3.,origin_y_m=0.,
        relative_thickness=.12282186549577312,thickness_location=.3)
    oversized=build_double_parabolic_profile(spec)
    actual_width=np.ptp(oversized.contour(401)[:,0])
    assert actual_width>5*spec.axial_width_m
    checks={c.name:c for c in validate_profile(oversized,spec.pitch_m)['checks']}
    assert checks['closed joints'].passed and checks['continuous tangents'].passed
    assert not checks['axial envelope boundary'].passed
    # A distinct constructible stagger keeps the same input/output metal,
    # edge radii and required axial width; no cosmetic rescaling is applied.
    feasible=build_double_parabolic_profile(replace(spec,stagger_deg=75.))
    np.testing.assert_allclose(np.ptp(feasible.contour(401)[:,0]),spec.axial_width_m,rtol=1e-4)
    assert all(c.passed for c in validate_profile(feasible,spec.pitch_m)['checks'])

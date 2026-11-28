"""Preliminary real-section mechanics, NOT a certified blade-life assessment.

Sources separate from Vinogradov: supplied UKR homework sec.2 (centrifugal
integral, distributed gas force and unsymmetric bending); INCO bulletin497
pp.3,4 TablesV,VII,IX (IN-738C); NASA TM105997 pp.3–4 geometric stiffness.
LM definition: Mines Albi SciMat 'Larson-Miller parameters'. C=20 is an explicit
engineering assumption, not a fit or a certified lower-bound material curve.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.linalg import eigh
from .section_properties import section_properties

PSI_TO_PA=6894.757293168
MATERIAL_SOURCE='INCO 497 pp.3–4,8 Tables V,VII,IX; https://www.scribd.com/document/664099561/ni-inco-497-alloy738'


def _kelvin(f):return (np.asarray(f,float)-32)*5/9+273.15


def _bounded_interp(x,xp,yp):
    x=np.asarray(x,float)
    return np.where((x>=xp[0])&(x<=xp[-1]),np.interp(x,xp,yp),np.nan)


@dataclass(frozen=True)
class IN738C:
    name: str='IN-738C vacuum cast, INCO497 heat treatment p.3'
    density_kg_m3: float=8110.
    lm_constant: float=20.
    source: str=MATERIAL_SOURCE

    def elastic_modulus(self,T_K):
        return _bounded_interp(T_K,_kelvin([75,200,400,600,800,1000,1200,1400,1600,1800]),
            np.array([29.1,28.3,27.6,26.8,26.,25.4,24.3,23.2,21.9,20.3])*1e6*PSI_TO_PA)

    def yield_strength(self,T_K):
        return _bounded_interp(T_K,_kelvin([70,1200,1400,1600,1800]),
                               np.array([138000,132000,115000,80000,50000])*PSI_TO_PA)

    def rupture_curve(self):
        # Table IX high-carbon columns ONLY; LC 10000h is not substituted.
        T=np.repeat(_kelvin([1350,1500,1700,1800]),2)
        hours=np.tile([100,1000],4)
        stress=np.array([96000,78000,66000,48000,31000,20000,19000,12000])*PSI_TO_PA
        P=T*(self.lm_constant+np.log10(hours))
        order=np.argsort(P)
        return P[order],stress[order]

    def rupture_allowable(self,T_K,life_hours):
        if not math.isfinite(life_hours) or life_hours<=0:raise ValueError('life must be positive hours')
        P,stress=self.rupture_curve()
        return np.exp(_bounded_interp(np.asarray(T_K)*(self.lm_constant+np.log10(life_hours)),P,np.log(stress)))

    def rupture_life(self,T_K,stress_Pa):
        P,stress=self.rupture_curve()
        sigma=np.asarray(stress_Pa,float)
        inverse=_bounded_interp(np.log(np.where(sigma>0,sigma,np.nan)),np.log(stress[::-1]),P[::-1])
        return 10**(inverse/np.asarray(T_K)-self.lm_constant)


def _tail(r,f):
    integral=PchipInterpolator(r,f).antiderivative()
    return integral(r[-1])-integral(r)


def centrifugal_loads(radii_m,areas_m2,speed_rpm,density_kg_m3,*,method='trapezoid'):
    """UKR sec.2: N(r)=rho*omega² integral_r^tip R*A(R)dR, zero at tip.

    Quadratic mode requires exactly three true areas, checks the polynomial's
    internal extrema for positive area, and integrates r*A analytically.
    """
    r,A=np.asarray(radii_m,float),np.asarray(areas_m2,float)
    if r.ndim!=1 or A.shape!=r.shape or len(r)<3 or np.any(np.diff(r)<=0) or np.any(A<=0) or r[0]<=0 or speed_rpm<0 or density_kg_m3<=0:
        raise ValueError('invalid centrifugal integration input')
    omega=2*np.pi*speed_rpm/60
    if method=='trapezoid':
        N=-cumulative_trapezoid((density_kg_m3*omega**2*r*A)[::-1],r[::-1],initial=0)[::-1]
    elif method=='quadratic':
        if len(r)!=3:raise ValueError('quadratic area requires exactly three sections')
        law=np.polynomial.Polynomial.fit(r,A,2).convert()
        roots=law.deriv().roots();probe=np.r_[r,roots[(roots>r[0])&(roots<r[-1])]]
        if np.any(law(probe)<=0):raise ValueError('quadratic area negative between sections')
        primitive=(law*np.polynomial.Polynomial([0,1])).integ()
        N=density_kg_m3*omega**2*(primitive(r[-1])-primitive(r))
    else: raise ValueError('unknown area integration method')
    return dict(force_N=N,stress_Pa=N/A,area_m2=A,radius_m=r)


def gas_loads(sections,count):
    """UKR sec.2 annular-strip momentum; c_theta is SIGNED in both states.

    Output force on one blade per unit OUTLET radial length, N/m. Outlet
    continuity defines that mass strip; upstream swirl is transferred through
    r_in*c_theta_in/r_out so torque obeys angular momentum for unequal radii.
    Pressure difference uses the UKR strip approximation, not a 3D surface
    pressure integral. No repeated blockage or extra loss multiplier.
    """
    if count<3:raise ValueError('blade count must be >=3')
    r=np.array([s.radius_m for s in sections])
    q=[]
    for s in sections:
        area_per_span=2*np.pi*s.radius_m/count
        dm=area_per_span*s.outlet.density_kg_m3*s.outlet.axial_m_s
        inlet_radius=s.radius_m if s.inlet_radius_m is None else s.inlet_radius_m
        equivalent_inlet_swirl=inlet_radius/s.radius_m*s.inlet.circumferential_m_s
        q.append([area_per_span*(s.inlet.pressure_Pa-s.outlet.pressure_Pa)
                  +dm*(s.inlet.axial_m_s-s.outlet.axial_m_s),
                  dm*(equivalent_inlet_swirl-s.outlet.circumferential_m_s)])
    q=np.asarray(q)
    Mx=-(_tail(r,r*q[:,1])-r*_tail(r,q[:,1]))
    My=_tail(r,r*q[:,0])-r*_tail(r,q[:,0])
    return dict(force_per_span_N_m=q,moments_N_m=np.column_stack((Mx,My)))


def section_stress(profile,normal_force_N,moments_N_m,*,properties=None,contour=None):
    properties=section_properties(profile) if properties is None else properties
    Ixx,Iyy,Ixy=properties.centroidal_moments_m4
    gradient=np.linalg.solve([[Ixy,Ixx],[-Iyy,-Ixy]],moments_N_m)
    points=(profile.contour() if contour is None else contour)-properties.centroid_m
    bending=points@gradient
    return normal_force_N/properties.area_m2+bending


def recovery_temperature_bound(section):
    """Recovery=1 at BOTH row planes; an upper estimate, not a thermal solution."""
    values=[]
    for station,speed in ((section.inlet,section.frame_inlet_speed_m_s),
                          (section.outlet,section.frame_speed_m_s)):
        cp=station.gamma*station.gas_constant_J_kg_K/(station.gamma-1)
        values.append(station.temperature_K+station.speed(speed)**2/(2*cp))
    return max(values)


def strength_assessment(profiles,sections,count,material,life_hours,*,screening_limit_Pa=300e6,
                        required_lm_margin=1.0,metal_temperature_K=None):
    """Real contour stress + gas bending; temperature recovery bound if unknown.

    No thermal stress, root fillet concentration, disk, coating or HCF claim.
    Unknown material allowance is failure, never green. 300 MPa is a project
    screening value checked independently against temperature-dependent data.
    A rupture margin of one is the mathematical comparison with the selected
    typical curve, NOT certification. Higher mandatory reserve must be an
    explicit project input; it is not invented from Vinogradov's geometry text.
    """
    if len(profiles)!=len(sections) or len(profiles)<3:raise ValueError('need >=3 matching real sections')
    r=np.array([s.radius_m for s in sections]);props=[section_properties(p) for p in profiles]
    A=np.array([p.area_m2 for p in props])
    loads=centrifugal_loads(r,A,sections[0].speed_rpm,material.density_kg_m3)
    gas=gas_loads(sections,count)
    recovery=np.array([recovery_temperature_bound(s) for s in sections])
    T=recovery if metal_temperature_K is None else np.broadcast_to(metal_temperature_K,r.shape)
    allowed=material.rupture_allowable(T,life_hours);yield_=material.yield_strength(T)
    rows=[];distributions=[]
    for j,(p,N,M) in enumerate(zip(profiles,loads['force_N'],gas['moments_N_m'])):
        stress=section_stress(p,N,M);maximum=float(max(stress.max(),0.))
        margin=float(allowed[j]/maximum) if maximum>0 and np.isfinite(allowed[j]) else (math.inf if np.isfinite(allowed[j]) else math.nan)
        rows.append(dict(radius_m=r[j],area_m2=A[j],normal_force_N=N,gas_moment_N_m=M.tolist(),
            tension_Pa=loads['stress_Pa'][j],normal_min_Pa=float(stress.min()),normal_max_Pa=maximum,
            temperature_K=float(T[j]),yield_Pa=float(yield_[j]),lm_allowable_Pa=float(allowed[j]),lm_margin=margin,
            required_lm_margin=float(required_lm_margin),
            screening_passed=maximum<=screening_limit_Pa,
            material_passed=bool(np.isfinite(yield_[j]) and maximum<=yield_[j]),
            creep_passed=bool(np.isfinite(allowed[j]) and margin>=required_lm_margin)))
        distributions.append(stress)
    indices=np.array([0,len(r)//2,len(r)-1])
    quadratic=centrifugal_loads(r[indices],A[indices],sections[0].speed_rpm,material.density_kg_m3,method='quadratic')
    return dict(rows=rows,contour_stress_Pa=distributions,loads=loads,gas=gas,quadratic_three=quadratic,
        passed=all(v['screening_passed'] and v['material_passed'] and v['creep_passed'] for v in rows),
        temperature_basis='relative total recovery=1 upper estimate, not solved metal temperature' if metal_temperature_K is None else 'explicit supplied metal temperature',
        material_source=material.source,limitations=['no thermal-gradient stress','no root attachment/fillet concentration',
        'no certified minimum material allowable','no creep strain prediction from rupture alone','vibration requires separate harmonic loading/SN data'])


def beam_modes(radii_m,areas_m2,inertia_m4,elastic_modulus_Pa,density_kg_m3,speed_rpm=0.,modes=3):
    """Axial/out-of-rotation-plane Euler–Bernoulli screening with centrifugal K_g.

    NASA TM105997 pp.3–4 describes geometric stiffness from centrifugal load.
    Hermite FE follows stationary energy integral E I v''²+N v'². Only this
    polarization has no gyroscopic/centrifugal-softening terms; twisted blades
    need coupled 3D modes. No damping or fatigue strength is inferred.
    """
    r,A,I,E=map(lambda x:np.asarray(x,float),(radii_m,areas_m2,inertia_m4,elastic_modulus_Pa))
    if any(x.shape!=r.shape for x in (A,I,E)) or np.any(I<=0) or np.any(E<=0) or not np.isfinite(E).all():raise ValueError('invalid beam property data')
    N=centrifugal_loads(r,A,speed_rpm,density_kg_m3)['force_N']
    n=len(r);K=np.zeros((2*n,2*n));M=K.copy()
    for j,L in enumerate(np.diff(r)):
        EI=(E[j]*I[j]+E[j+1]*I[j+1])/2; mass=density_kg_m3*(A[j]+A[j+1])/2
        ke=EI/L**3*np.array([[12,6*L,-12,6*L],[6*L,4*L*L,-6*L,2*L*L],[-12,-6*L,12,-6*L],[6*L,2*L*L,-6*L,4*L*L]])
        me=mass*L/420*np.array([[156,22*L,54,-13*L],[22*L,4*L*L,13*L,-3*L*L],[54,13*L,156,-22*L],[-13*L,-3*L*L,-22*L,4*L*L]])
        kg=(N[j]+N[j+1])/2/(30*L)*np.array([[36,3*L,-36,3*L],[3*L,4*L*L,-3*L,-L*L],[-36,-3*L,36,-3*L],[3*L,-L*L,-3*L,4*L*L]])
        sl=slice(2*j,2*j+4);K[sl,sl]+=ke+kg;M[sl,sl]+=me
    lam,shapes=eigh(K[2:,2:],M[2:,2:],subset_by_index=(0,min(modes,2*n-2)-1))
    return dict(frequencies_Hz=np.sqrt(lam)/(2*np.pi),mass_normalized_modes=shapes,
                stiffness=K,mass=M,centrifugal_force_N=N,
                scope='1D axial bending screening only; not full vibrational clearance')


def vibration_response(beam,force_vector_N,excitation_Hz,damping_ratio,*,endurance_limit_Pa,
                       stress_per_unit_mode_Pa):
    """Explicit harmonic modal stress and fatigue margin; no fabricated amplitude.

    A material- and temperature-specific endurance limit must be supplied;
    INCO TableXII is LC, so it is NEVER silently used for IN-738C.
    """
    if damping_ratio<=0 or endurance_limit_Pa<=0 or excitation_Hz<0:raise ValueError('explicit positive damping/endurance required')
    phi=beam['mass_normalized_modes'];f=beam['frequencies_Hz'];w=2*np.pi*excitation_Hz;wn=2*np.pi*f
    force=np.asarray(force_vector_N,float)
    if force.shape!=(phi.shape[0],):raise ValueError('force vector must contain free beam DOFs')
    q=(phi.T@force)/(wn**2-w**2+2j*damping_ratio*wn*w)
    stress=np.abs(np.asarray(stress_per_unit_mode_Pa)@q)
    maximum=float(np.max(stress))
    return dict(amplitude_Pa=stress,margin=endurance_limit_Pa/maximum if maximum else math.inf,
                excitation_Hz=excitation_Hz,damping_ratio=damping_ratio,passed=maximum<=endurance_limit_Pa)


def campbell_screening(frequencies_Hz,speed_rpm,neighbor_blade_count):
    """Blade-passing excitation requires the actual preceding row, not a
    previous successfully built rotor elsewhere in a partly failed turbine."""
    if neighbor_blade_count is None:
        return dict(status='NOT_CHECKED',reason='preceding stator blade count unavailable',
                    shaft_frequency_Hz=speed_rpm/60)
    if neighbor_blade_count<1 or speed_rpm<=0:
        raise ValueError('positive actual neighbor count and shaft speed required')
    excitation=neighbor_blade_count*speed_rpm/60
    return dict(status='SCREENING_ONLY',shaft_frequency_Hz=speed_rpm/60,
                neighbor_blade_count=int(neighbor_blade_count),
                neighbor_blade_passing_Hz=excitation,
                relative_frequency_separation=(np.abs(np.asarray(frequencies_Hz)-excitation)/excitation).tolist())


def modal_section_stress(beam,radii_m,section_contours,elastic_modulus_Pa):
    """Recover real-contour axial bending stress from Hermite modal curvature.

    Same ONE axial polarization as beam_modes, not a 3D torsional/HCF model.
    Includes the two clamped root DOFs omitted from the eigenproblem. Output
    feeds vibration_response directly; no manually invented stress multiplier.
    """
    r=np.asarray(radii_m,float);n=len(r)
    if n<2 or np.any(np.diff(r)<=0) or len(section_contours)!=n:
        raise ValueError('matching increasing radial mesh and actual contours required')
    modes=np.asarray(beam['mass_normalized_modes'],float)
    if modes.ndim!=2 or modes.shape[0]!=2*(n-1):
        raise ValueError('modal DOFs do not match the clamped radial mesh')
    E=np.broadcast_to(np.asarray(elastic_modulus_Pa,float),(n,))
    if np.any(E<=0) or not np.isfinite(E).all():
        raise ValueError('positive finite temperature-dependent moduli required')
    full=np.vstack((np.zeros((2,modes.shape[1])),modes))
    stresses=[];stations=[]
    for j,contour in enumerate(section_contours):
        element=min(j,n-2);xi=0. if j<n-1 else 1.;L=r[element+1]-r[element]
        d2=np.array([(-6+12*xi)/L**2,(-4+6*xi)/L,
                     (6-12*xi)/L**2,(-2+6*xi)/L])
        curvature=d2@full[2*element:2*element+4]
        points=np.asarray(contour,float);properties=section_properties(points)
        stress=E[j]*(points[:,0]-properties.centroid_m[0])[:,None]*curvature
        stresses.append(stress);stations.extend([j]*len(points))
    return dict(stress_per_unit_mode_Pa=np.vstack(stresses),section_index=np.asarray(stations),
                scope='actual contours; one-axis Euler-Bernoulli modal stress, no torsion or harmonic load assumption')

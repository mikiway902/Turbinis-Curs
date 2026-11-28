"""EXPERIMENTAL cascade throat geometry and explicit perfect-gas choking.

Vinogradov (2010), p.18: a=t*sin(beta_2)/m, m=1.05..1.1 is an
EMPIRICAL construction correction, not a compressible-flow choking law.
All lengths SI. Angles returned and accepted here are degrees.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad_vec

def _positive(**values):
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("quantities must be finite and positive: " + ", ".join(values))


def effective_outlet_angle(throat_m: float, pitch_m: float) -> float:
    """Geometric effective angle asin(a/t), not metal angle or flow deviation."""
    _positive(throat_m=throat_m, pitch_m=pitch_m)
    if throat_m > pitch_m:
        raise ValueError("throat/pitch must be <=1")
    return math.degrees(math.asin(throat_m/pitch_m))


def throat_from_effective_angle(pitch_m: float, effective_angle_deg: float) -> float:
    _positive(pitch_m=pitch_m)
    if not math.isfinite(effective_angle_deg) or not 0 < effective_angle_deg <= 90:
        raise ValueError("effective angle must be in (0,90] degrees")
    return pitch_m*math.sin(math.radians(effective_angle_deg))


def source_throat_width(pitch_m: float, outlet_metal_deg: float, *, throat_multiplier: float = 1.05) -> float:
    """p.18 empirical design width; do not identify beta_2 with flow angle."""
    _positive(throat_multiplier=throat_multiplier)
    return throat_from_effective_angle(pitch_m, outlet_metal_deg)/throat_multiplier


@dataclass(frozen=True)
class ChokingResult:
    choked: bool
    critical_pressure_ratio: float
    exit_mach: float
    mass_flux_kg_m2_s: float
    mass_flow_kg_s: float


def mass_flux_at_mach(stagnation_pressure_pa,stagnation_temperature_k,gamma,
                      gas_constant_j_kg_k,mach,discharge_coefficient=1.):
    """Constant-gamma isentropic mass flux, kg/(m² s); no repeated loss factor."""
    _positive(stagnation_pressure_pa=stagnation_pressure_pa,
              stagnation_temperature_k=stagnation_temperature_k,
              gas_constant_j_kg_k=gas_constant_j_kg_k,
              discharge_coefficient=discharge_coefficient)
    if not math.isfinite(mach) or mach<0 or not math.isfinite(gamma) or gamma<=1 or discharge_coefficient>1:
        raise ValueError("invalid Mach, gamma or discharge coefficient")
    return (discharge_coefficient*stagnation_pressure_pa/math.sqrt(stagnation_temperature_k)
            *math.sqrt(gamma/gas_constant_j_kg_k)*mach
            *(1+(gamma-1)*mach*mach/2)**(-(gamma+1)/(2*(gamma-1))))


def perfect_gas_throat_flow(*, stagnation_pressure_pa: float,
                           stagnation_temperature_k: float, back_pressure_pa: float,
                           gamma: float, gas_constant_j_kg_k: float,
                           area_m2: float, discharge_coefficient: float = 1.) -> ChokingResult:
    """Isentropic 1D ideal-gas reservoir-to-throat flow, NOT a book correlation.

    Requires constant gamma,R and explicit total conditions in the throat's
    reference frame (relative for rotor, absolute for stator). No shocks, heat,
    real-gas effects or downstream diffuser closure. Back pressure may be zero.
    Cd is explicit/empirical when different from 1; area is geometric flow area.
    """
    _positive(stagnation_pressure_pa=stagnation_pressure_pa,
              stagnation_temperature_k=stagnation_temperature_k,
              gas_constant_j_kg_k=gas_constant_j_kg_k, area_m2=area_m2,
              discharge_coefficient=discharge_coefficient)
    if not math.isfinite(gamma) or gamma <= 1:
        raise ValueError("gamma must exceed 1")
    if not math.isfinite(back_pressure_pa) or not 0 <= back_pressure_pa <= stagnation_pressure_pa:
        raise ValueError("back pressure must lie between zero and stagnation pressure")
    if discharge_coefficient > 1:
        raise ValueError("discharge_coefficient must be <=1")
    ratio = back_pressure_pa/stagnation_pressure_pa
    critical = (2/(gamma+1))**(gamma/(gamma-1))
    choked = ratio <= critical
    mach = 1. if choked else math.sqrt(2/(gamma-1)*(ratio**(-(gamma-1)/gamma)-1))
    flux=mass_flux_at_mach(stagnation_pressure_pa,stagnation_temperature_k,gamma,
                          gas_constant_j_kg_k,mach,discharge_coefficient)
    return ChokingResult(choked, critical, mach, flux, flux*area_m2)


def required_throat_mach(*,mass_flux_kg_m2_s,stagnation_pressure_pa,
                          stagnation_temperature_k,gamma,gas_constant_j_kg_k):
    """Subsonic throat Mach required by actual flow area; no fake flow fallback."""
    from scipy.optimize import brentq
    if not math.isfinite(mass_flux_kg_m2_s) or mass_flux_kg_m2_s<0:
        raise ValueError("mass flux must be finite and nonnegative")
    args=(stagnation_pressure_pa,stagnation_temperature_k,gamma,gas_constant_j_kg_k)
    critical=mass_flux_at_mach(*args,1.)
    if mass_flux_kg_m2_s>critical:
        return dict(possible=False,required_flux=mass_flux_kg_m2_s,critical_flux=critical,
                    excess_fraction=mass_flux_kg_m2_s/critical-1,mach=None)
    M=brentq(lambda m:mass_flux_at_mach(*args,m)-mass_flux_kg_m2_s,0.,1.)
    static_pressure=stagnation_pressure_pa*(1+(gamma-1)*M*M/2)**(-gamma/(gamma-1))
    return dict(possible=True,required_flux=mass_flux_kg_m2_s,critical_flux=critical,
                excess_fraction=0.,mach=M,static_pressure_Pa=static_pressure)


def integrate_annular_capacity(radii_m,blade_count,strip_at_radius,*,absolute_tolerance_kg_s=1e-8):
    """Integrate Z*a(r)*g_critical(r), independently of sparse strength sections.

    Vinogradov p.11: t=2*pi*r/Z; continuity: dG=2*pi*r*rho*c_a*dr.
    ``strip_at_radius`` returns (outlet FlowStation, actual throat in metres,
    critical mass flux in the appropriate absolute/relative frame).
    The same quadrature evaluates the gas mass integral, exposing sparse
    discretization error instead of attributing it to geometric choking.
    Error is the integrator's estimate, not a rigorous continuum certificate.
    """
    radii=np.asarray(radii_m,float)
    if radii.ndim!=1 or len(radii)<2 or not np.isfinite(radii).all() or np.any(np.diff(radii)<=0):
        raise ValueError('strictly increasing finite radial integration stations required')
    _positive(root_radius_m=float(radii[0]),blade_count=blade_count,
              absolute_tolerance_kg_s=absolute_tolerance_kg_s)
    if int(blade_count)!=blade_count:raise ValueError('integer blade count required')
    samples={}
    def integrand(radius):
        outlet,throat,critical=strip_at_radius(float(radius))
        _positive(throat_m=throat,critical_flux=critical,
                  density=outlet.density_kg_m3,axial_speed=outlet.axial_m_s)
        capacity=blade_count*throat*critical
        mass=2*math.pi*radius*outlet.density_kg_m3*outlet.axial_m_s
        samples[float(radius)]=(capacity,mass)
        return np.array([capacity,mass])
    value,error,info=quad_vec(integrand,radii[0],radii[-1],points=radii[1:-1],
        epsabs=absolute_tolerance_kg_s,epsrel=0.,norm='max',full_output=True)
    limiting=min(samples,key=lambda r:samples[r][0]/samples[r][1])
    return dict(critical_capacity_kg_s=float(value[0]),integrated_gas_mass_kg_s=float(value[1]),
        estimated_absolute_error_kg_s=float(error),converged=bool(info.success),
        evaluations=int(info.neval),minimum_sampled_local_margin=samples[limiting][0]/samples[limiting][1]-1,
        minimum_sampled_margin_radius_m=limiting,
        scope='adaptive annular quadrature; local margin sampled, not a continuum proof')


def measured_back_angle(suction_normal_xy,outlet_metal_deg,trailing_wedge_deg):
    """Table3.1 item43, p.43: epsilon=tangent+90-beta2+omega/2.

    The oriented normal points from suction into the actual adjacent channel.
    Its angle FROM +AXIAL x equals tangent+90; swapping atan2 arguments would
    silently replace this angle by its complement.
    """
    normal=np.asarray(suction_normal_xy,float)
    if normal.shape!=(2,) or not np.isfinite(normal).all() or np.linalg.norm(normal)==0:
        raise ValueError('finite nonzero oriented suction normal required')
    return math.degrees(math.atan2(normal[1],normal[0]))-outlet_metal_deg+trailing_wedge_deg/2

"""Bridge to accepted radial residuals; does NOT replace Mikhaltsev closures.

Authoritative equations: docs/engineering/EQUATIONS.md pp.3–54 (line register),
Mikhaltsev part1 ch.7. New interpolation is a numerical approximation only.
"""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator
from .radial_equilibrium_core import pressure_residual, general_residual, entropy_from_state
from .models import FlowStation, RadialSectionInput
from .validation import Check


def audit_radial_state(radius_m, axial_m_s, circumferential_m_s, temperature_K,
                       pressure_Pa, gas_constant, cp, mass_flow_kg_s, tolerance=.01):
    """Accepted static-pressure and general residual, plus integral mass.

    One-percent is explicit diagnostic numerical tolerance, not promotion of
    a closure to ACCEPTED. Failed pressure momentum cannot be hidden by entropy.
    """
    r,ca,cu,T,p=map(lambda x:np.asarray(x,float),(radius_m,axial_m_s,circumferential_m_s,temperature_K,pressure_Pa))
    rho=p/(gas_constant*T)
    if np.any(rho<=0) or np.any(T<=0): raise ValueError('nonphysical radial state')
    rp=pressure_residual(r,p,T,cu,gas_constant=gas_constant)
    entropy=entropy_from_state(T,p,cp=cp,gas_constant=gas_constant)
    rg=general_residual(r,T,entropy,ca,cu,cp=cp)
    scale=max(float(np.max(cu*cu/r)),float(np.max(ca*ca/r)),1.)
    actual_mass=float(trapezoid(2*np.pi*r*rho*ca,r))
    rows=[Check('radial pressure momentum',float(np.max(abs(rp))/scale),tolerance,
                bool(np.max(abs(rp))/scale<=tolerance),'numerical','Accepted radial/equilibrium.py'),
          Check('general radial equilibrium',float(np.max(abs(rg))/scale),tolerance,
                bool(np.max(abs(rg))/scale<=tolerance),'numerical','Accepted radial/equilibrium.py'),
          Check('integral mass',abs(actual_mass/mass_flow_kg_s-1),tolerance,
                abs(actual_mass/mass_flow_kg_s-1)<=tolerance,'numerical','Annular continuity')]
    return dict(checks=rows,pressure_residual_m_s2=rp,general_residual_m_s2=rg,
                actual_mass_kg_s=actual_mass,passed=all(c.passed for c in rows))


def resample_sections(sections, radius_m):
    """Shape-preserving interpolation of explicit input states, no silent extrapolation.

    Increasing this mesh improves geometry interpolation, NOT the resolution
    of the upstream gas solver. The integration report retains input grid size.
    """
    rr=np.array([s.radius_m for s in sections]);r=np.asarray(radius_m,float)
    if len(rr)<3 or np.any(np.diff(rr)<=0) or np.min(r)<rr[0] or np.max(r)>rr[-1]:
        raise ValueError('need >=3 increasing sections and no extrapolation')
    def interpolate(values): return PchipInterpolator(rr,values)(r)
    result=[]
    states={}
    for side in ('inlet','outlet'):
        states[side]={name:interpolate([getattr(getattr(s,side),name) for s in sections])
                      for name in FlowStation.__dataclass_fields__}
    scalars={name:interpolate([getattr(s,name) for s in sections]) for name in
             ('reaction','alpha0_deg','alpha1_deg','alpha2_deg','beta1_deg','beta2_deg','required_mass_flux_kg_m2_s')}
    inlet_radii=interpolate([s.inlet_radius_m if s.inlet_radius_m is not None else s.radius_m for s in sections])
    boundaries=np.r_[r[0],(r[:-1]+r[1:])/2,r[-1]]
    for j,radius in enumerate(r):
        result.append(replace(sections[0],radius_m=float(radius),radial_weight_m=float(boundaries[j+1]-boundaries[j]),
            inlet_radius_m=float(inlet_radii[j]),
            inlet=FlowStation(**{k:float(v[j]) for k,v in states['inlet'].items()}),
            outlet=FlowStation(**{k:float(v[j]) for k,v in states['outlet'].items()}),
            **{k:float(v[j]) for k,v in scalars.items()}))
    return result

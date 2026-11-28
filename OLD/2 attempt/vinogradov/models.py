"""SI interfaces and source distinctions for Vinogradov blading (2010 pp.4–14).

Experimental geometry only; a completed construction is not aerodynamic proof.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np
from .curves import CompositeCurve, CircularArc


class ConstructionError(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = dict(details or {})


@dataclass(frozen=True)
class ProfileSpec:
    axial_width_m: float
    stagger_deg: float
    inlet_metal_deg: float
    outlet_metal_deg: float
    pitch_m: float
    leading_radius_m: float
    trailing_radius_m: float
    trailing_wedge_deg: float = 7.
    throat_multiplier: float = 1.05
    origin_y_m: float = .1
    incidence_deg: float = 0.
    deviation_deg: float = 0.
    back_angle_deg: float = 7.
    relative_thickness: float = .15
    thickness_location: float = .4

    def __post_init__(self):
        values = vars(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('profile specification must be finite')
        for name in ('axial_width_m','pitch_m','leading_radius_m','trailing_radius_m'):
            if values[name] <= 0:
                raise ValueError(name+' must be positive')
        if self.leading_radius_m+self.trailing_radius_m >= self.axial_width_m:
            raise ValueError('edge circles do not fit axial width')
        if not 0 < self.stagger_deg < 180 or not 0 < self.inlet_metal_deg < 180 or not 0 < self.outlet_metal_deg < 180:
            raise ValueError('profile angles must be strictly between 0 and 180 degrees')
        if not 0 < self.trailing_wedge_deg < 90 or self.throat_multiplier <= 0:
            raise ValueError('invalid wedge or throat multiplier')
        if not 0 < self.relative_thickness < 1 or not 0 < self.thickness_location < 1:
            raise ValueError('thickness and thickness position must be fractions in (0,1)')


@dataclass(frozen=True)
class SectionProfile:
    suction: CompositeCurve
    pressure: CompositeCurve
    leading_edge: CircularArc
    trailing_edge: CircularArc
    metadata: dict[str, Any] = field(default_factory=dict)

    def contour(self, count_per_side=401):
        if count_per_side < 3:
            raise ValueError('need at least three points per side')
        edge_count = max(17, count_per_side//8)
        pieces = (self.suction.sample(count_per_side), self.trailing_edge.sample(edge_count),
                  self.pressure.sample(count_per_side)[::-1], self.leading_edge.sample(edge_count))
        points = np.vstack([p[:-1] for p in pieces])
        return np.vstack((points, points[0]))


@dataclass(frozen=True)
class FlowStation:
    """Static gas state and velocity in SI; c=(axial,circumferential).

    T,p,rho are STATIC. T0,p0 are explicit stagnation quantities. Keeping both
    avoids substituting total density into annular continuity.
    """
    axial_m_s: float
    circumferential_m_s: float
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    total_temperature_K: float
    total_pressure_Pa: float
    dynamic_viscosity_Pa_s: float
    gas_constant_J_kg_K: float
    gamma: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('nonfinite flow station')
        if self.axial_m_s <= 0 or min(self.temperature_K,self.pressure_Pa,self.density_kg_m3,
                                      self.total_temperature_K,self.total_pressure_Pa,
                                      self.dynamic_viscosity_Pa_s,self.gas_constant_J_kg_K) <= 0 or self.gamma <= 1:
            raise ValueError('nonphysical gas station')

    @property
    def velocity(self):
        return np.array([self.axial_m_s, self.circumferential_m_s])

    def speed(self, blade_speed_m_s=0.):
        return math.hypot(self.axial_m_s, self.circumferential_m_s-blade_speed_m_s)

    def mach(self, blade_speed_m_s=0.):
        return self.speed(blade_speed_m_s)/math.sqrt(self.gamma*self.gas_constant_J_kg_K*self.temperature_K)

    def reynolds(self, chord_m, blade_speed_m_s=0.):
        return self.density_kg_m3*self.speed(blade_speed_m_s)*chord_m/self.dynamic_viscosity_Pa_s


@dataclass(frozen=True)
class RadialSectionInput:
    stage: int
    row: str
    radius_m: float
    root_radius_m: float
    mean_radius_m: float
    tip_radius_m: float
    speed_rpm: float
    reaction: float
    mass_flow_kg_s: float
    inlet: FlowStation
    outlet: FlowStation
    alpha0_deg: float
    alpha1_deg: float
    alpha2_deg: float
    beta1_deg: float
    beta2_deg: float
    required_mass_flux_kg_m2_s: float
    radial_weight_m: float
    inlet_radius_m: float | None = None

    def __post_init__(self):
        if self.row not in ('SA','RK') or self.stage < 1:
            raise ValueError('row must be SA or RK and stage >=1')
        if not 0 < self.root_radius_m <= self.radius_m <= self.tip_radius_m or not self.root_radius_m <= self.mean_radius_m <= self.tip_radius_m:
            raise ValueError('inconsistent radial station')
        if self.speed_rpm < 0 or self.mass_flow_kg_s <= 0 or self.required_mass_flux_kg_m2_s <= 0 or self.radial_weight_m <= 0:
            raise ValueError('invalid speed/flow/section weight')

    @property
    def blade_speed_m_s(self):
        return self.radius_m*self.speed_rpm*2*math.pi/60

    @property
    def frame_speed_m_s(self):
        return self.blade_speed_m_s if self.row == 'RK' else 0.

    @property
    def frame_inlet_speed_m_s(self):
        radius=self.radius_m if self.inlet_radius_m is None else self.inlet_radius_m
        return radius*self.speed_rpm*2*math.pi/60 if self.row == 'RK' else 0.

    @property
    def height_m(self):
        return self.tip_radius_m-self.root_radius_m

    @property
    def span_fraction(self):
        return (self.radius_m-self.root_radius_m)/self.height_m

    @property
    def flow_inlet_deg(self):
        return self.alpha0_deg if self.row == 'SA' else self.beta1_deg

    @property
    def flow_outlet_deg(self):
        return self.alpha1_deg if self.row == 'SA' else self.beta2_deg

"""Back angle follows the exit trailing-circle construction, not global throat."""
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from vinogradov.models import FlowStation, RadialSectionInput, ProfileSpec
from vinogradov.cascade_geometry import build_profile
from vinogradov.channel_convergence import analyze_channel
from vinogradov.integration import _parameters
from vinogradov.blade_geometry import geometric_parameters


def _reference_exit_angle(profile,spec):
    centre=profile.trailing_edge.center+np.array([0.,spec.pitch_m])
    candidates=[]
    for curve in profile.suction.segments:
        fit=minimize_scalar(lambda t:np.sum((curve.evaluate(t)-centre)**2),
            bounds=(0.,1.),method='bounded',options={'xatol':1e-14})
        for parameter in (0.,1.,fit.x):
            candidates.append((float(np.linalg.norm(curve.evaluate(parameter)-centre)),curve,parameter))
    _,curve,parameter=min(candidates,key=lambda v:v[0])
    tangent=curve.evaluate(parameter,1)
    return math.degrees(math.atan2(tangent[1],tangent[0]))+90-spec.outlet_metal_deg+spec.trailing_wedge_deg/2


def test_requested_back_angle_cannot_change_measured_angle_without_changing_curves():
    width = .02
    chord = width/math.sin(math.radians(40))
    spec = ProfileSpec(width, 40., 90., 17., .0191354,
        .0243*chord, .006025*chord, relative_thickness=.106,
        thickness_location=.245, origin_y_m=.02, back_angle_deg=7.)
    other = replace(spec, back_angle_deg=11.)
    profile = build_profile(spec, 'double_parabolic')
    alternate = build_profile(other, 'double_parabolic')
    np.testing.assert_array_equal(profile.contour(101), alternate.contour(101))
    channel = analyze_channel(profile, spec.pitch_m, samples=21)

    def state(angle):
        ca = 100.
        cu = ca/math.tan(math.radians(angle))
        temperature, pressure, gas_constant, gamma = 1000., 2e5, 287., 1.33
        cp = gamma*gas_constant/(gamma-1)
        total = temperature+(ca*ca+cu*cu)/(2*cp)
        return FlowStation(ca, cu, temperature, pressure, pressure/gas_constant/temperature,
            total, pressure*(total/temperature)**(gamma/(gamma-1)), 4e-5, gas_constant, gamma)

    section = RadialSectionInput(1, 'SA', .15, .1, .15, .2, 30000., .4, .7,
        state(90.), state(17.), 90., 17., 20., 30., 35., 200., .025)
    first = _parameters(section, spec, profile, channel)
    second = _parameters(section, other, alternate, channel)
    expected = _reference_exit_angle(profile,spec)
    assert first['back_angle_deg'] == pytest.approx(expected, abs=1e-5)
    assert second['back_angle_deg'] == pytest.approx(expected, abs=1e-5)
    assert first['requested_back_angle_deg'] == 7.
    assert second['requested_back_angle_deg'] == 11.


def test_exit_back_angle_is_not_replaced_by_global_channel_minimum():
    spec=ProfileSpec(.02,70.,40.,20.,.015,.0005,.0002,
        relative_thickness=.106,thickness_location=.3)
    profile=build_profile(spec,'parabolic')
    channel=analyze_channel(profile,spec.pitch_m,samples=81)
    expected=_reference_exit_angle(profile,spec)
    values=geometric_parameters(profile,spec,samples=41)
    assert values['back_angle_deg']==pytest.approx(expected,abs=1e-5)
    tangent=profile.suction.evaluate(channel.throat.suction_parameter,1)
    global_value=math.degrees(math.atan2(tangent[1],tangent[0]))+90-spec.outlet_metal_deg+spec.trailing_wedge_deg/2
    assert abs(values['back_angle_deg']-global_value)>1.

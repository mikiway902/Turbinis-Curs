"""Actual course gas states, analytic cubic load integrals and triangle inertia."""
import json
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from vinogradov.course_adapter import CourseRadialSampler, SOURCE_CELLS
from vinogradov.models import ProfileSpec
from vinogradov.parabolic_profile import build_parabolic_profile
from vinogradov.stator_strength import assess_stator


@pytest.fixture(scope='module', params=(300, 2000, 10000))
def sampler(request):
    source = Path(__file__).resolve().parents[2] / f'TVD_1CBK_2st_{request.param}h.ipynb'
    notebook = json.loads(source.read_text(encoding='utf-8'))
    module = ModuleType(f'_stator_strength_source_{request.param}')
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    ns = module.__dict__
    try:
        for index in SOURCE_CELLS:
            if index != 42:
                exec(compile(''.join(notebook['cells'][index]['source']),
                             f'{source.name}:cell{index}', 'exec'), ns)
        if request.param == 300:
            # Use the recorded coupled source settings, not the 300 h
            # notebook's uncoupled starting point (its RK2 wall angle fails).
            report = json.loads((source.parent/'exports/vinogradov/300h/report.json').read_text())
            settings = report['source_settings']
            config = dict(settings['config'])
            config['phi'], config['psi'] = tuple(config['phi']), tuple(config['psi'])
            ns['C'] = ns['Config'](**config)
            ns['GEOMETRY'] = tuple(settings['geometry'])
            ns['PROFILE_ROW_DESIGN'] = settings['profile_row_design']
        ns['P0'] = ns['evaluate'](ns['GEOMETRY'], ns['C'])
        yield CourseRadialSampler(ns)
    finally:
        sys.modules.pop(module.__name__, None)


def actual_profiles(scale=1.):
    result = []
    for j in range(5):
        width = .02*scale*(1+.04*j)
        chord = width/math.sin(math.radians(40))
        result.append(build_parabolic_profile(ProfileSpec(width, 40, 90, 17,
            .69*chord, .024*chord, .006*chord, origin_y_m=.1*scale)))
    return result


def cubic_tail(radii, values):
    """Analytically integrate each cubic's monomials, no source tail helper."""
    coefficients = PchipInterpolator(radii, values).c
    powers = np.arange(4, 0, -1)[:, None]
    interval = np.sum(coefficients*np.diff(radii)[None, :]**powers/powers, axis=0)
    return np.r_[np.cumsum(interval[::-1])[::-1], 0.]


def triangle_inertia(contour):
    """Degree-two exact three-point triangle quadrature, independent of Green sums."""
    origin = contour[0]
    points = contour-origin
    a, b = points[:-1], points[1:]
    signed_area = (a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])/2
    if signed_area.sum() < 0:
        signed_area = -signed_area
    area = signed_area.sum()
    centroid = np.sum(signed_area[:, None]*(a+b)/3, axis=0)/area
    quadrature = np.stack(((a+b)/6, 2*a/3+b/6, a/6+2*b/3), axis=1)-centroid
    x, y = quadrature[:, :, 0], quadrature[:, :, 1]
    ixx = np.sum(signed_area*np.mean(y*y, axis=1))
    iyy = np.sum(signed_area*np.mean(x*x, axis=1))
    ixy = np.sum(signed_area*np.mean(x*y, axis=1))
    return area, origin+centroid, ixx, iyy, ixy


@pytest.mark.parametrize('index', (0, 1))
def test_absolute_gas_force_actual_mirrored_inertia_and_stationary_beam(sampler, index):
    sections = sampler.row(index, 'SA', 5)
    profiles = actual_profiles()
    count = 31+2*index
    result = assess_stator(sampler, index, sections, profiles, count)
    r = np.array([s.radius_m for s in sections])
    # Independent annular momentum balance: actual ABSOLUTE inlet/outlet,
    # including unequal inlet/outlet radii. Machine angular velocity never enters.
    expected_force = []
    for s in sections:
        strip = 2*math.pi*s.radius_m/count
        mass = strip*s.outlet.density_kg_m3*s.outlet.axial_m_s
        expected_force.append((strip*(s.inlet.pressure_Pa-s.outlet.pressure_Pa)
            + mass*(s.inlet.axial_m_s-s.outlet.axial_m_s),
            mass*(s.inlet_radius_m/s.radius_m*s.inlet.circumferential_m_s
                  - s.outlet.circumferential_m_s)))
    force = np.array(expected_force)
    moments = np.column_stack((-(cubic_tail(r, r*force[:, 1])-r*cubic_tail(r, force[:, 1])),
                               cubic_tail(r, r*force[:, 0])-r*cubic_tail(r, force[:, 0])))
    assert np.linalg.norm(force) > 1 and np.linalg.norm(moments) > .001
    np.testing.assert_allclose(result['force_per_span_N_m'], force, rtol=1e-12)
    np.testing.assert_allclose(result['moments_N_m'], moments, rtol=1e-10, atol=1e-11)
    np.testing.assert_array_equal(result['normal_force_N'], np.zeros(5))
    assert sampler.speed_rpm > 30000 and result['mechanical_speed_rpm'] == 0
    wrong_mirror_difference = []
    for j, (contour, stress, moment) in enumerate(zip(result['contours_m'], result['contour_stress_Pa'], moments)):
        area, center, ixx, iyy, ixy = triangle_inertia(contour)
        determinant = ixx*iyy-ixy*ixy
        mx, my = moment
        gradient = np.array((-(ixx*my+ixy*mx)/determinant,
                              (iyy*mx+ixy*my)/determinant))
        expected_stress = (contour-center)@gradient
        np.testing.assert_allclose(stress, expected_stress, rtol=2e-10, atol=1e-5)
        prop = result['properties'][j]
        assert prop.area_m2 == pytest.approx(area, rel=2e-12)
        np.testing.assert_allclose(prop.centroidal_moments_m4, [ixx, iyy, ixy], rtol=2e-10)
        row = result['table'].iloc[j]
        assert row.sigma_normal_min_MPa == pytest.approx(expected_stress.min()/1e6, abs=1e-10)
        assert row.sigma_normal_max_MPa == pytest.approx(expected_stress.max()/1e6, abs=1e-10)
        assert row.sigma_normal_abs_max_MPa == pytest.approx(np.max(np.abs(expected_stress))/1e6, abs=1e-10)
        assert row.normal_limit_passed == (np.max(np.abs(expected_stress)) <= 300e6)
        # Missing the physical y reflection changes both Ixy and coordinates.
        wrong_gradient = np.array((-(ixx*my-ixy*mx)/determinant,
                                    (iyy*mx-ixy*my)/determinant))
        wrong = ((contour-center)*[1, -1])@wrong_gradient
        wrong_mirror_difference.append(np.max(np.abs(wrong-expected_stress)))
    assert max(wrong_mirror_difference) > 1e5
    np.testing.assert_array_equal(result['contour_stress_Pa'][-1], 0.)
    assert set(result['table'].row) == {'SA'} and set(result['table'].stage) == {index+1}
    assert 'rotor' not in result['table']
    # The input retains fixed shaft speed, but cannot acquire centrifugal load.
    assert all(s.speed_rpm == sampler.speed_rpm for s in sections)
    assert np.all(result['table'].sigma_tension_MPa == 0)


def test_native_material_life_and_absolute_thermal_bound(sampler):
    sections = sampler.row(1, 'SA', 5)
    result = assess_stator(sampler, 1, sections, actual_profiles(), 33)
    expected_temperature = max(station.total_temperature_K for s in sections
                               for station in (s.inlet, s.outlet))
    assert result['temperature_bound']['upper_K'] == pytest.approx(expected_temperature, abs=1e-10)
    points = np.asarray(sampler.requirements['curve_points'])
    constant = sampler.requirements['C_LM']
    life = sampler.requirements['life_hours']
    parameter = expected_temperature*(constant+math.log10(life))
    assert points[0, 0] <= parameter/1000 <= points[-1, 0]
    allowable = math.exp(np.interp(parameter/1000, points[:, 0], np.log(points[:, 1])))
    table = result['table']
    for row in table.itertuples():
        assert row.life_hours == life and row.required_margin == 1.2
        assert row.allowable_MPa == pytest.approx(allowable, rel=1e-12)
        expected_margin = allowable/row.sigma_normal_max_MPa if row.sigma_normal_max_MPa else math.inf
        assert row.margin == pytest.approx(expected_margin, rel=1e-12)
        if points[-1, 1] <= row.sigma_normal_max_MPa <= points[0, 1]:
            inverse = 1000*np.interp(math.log(row.sigma_normal_max_MPa),
                                    np.log(points[::-1, 1]), points[::-1, 0])
            assert row.rupture_hours == pytest.approx(10**(inverse/expected_temperature-constant), rel=1e-11)
        else:
            assert math.isnan(row.rupture_hours)
    assert result['min_margin'] == pytest.approx(table.margin.min(), rel=1e-12)
    assert result['passed'] == bool(table.passed.all())
    # Independently use original variable-property enthalpy; relative rotor
    # stagnation temperature is numerically distinct and is not admissible here.
    s = sections[0]
    absolute = sampler.total_state(1, s.outlet, 0.)[0]
    relative = sampler.total_state(1, s.outlet, s.blade_speed_m_s)[0]
    assert abs(absolute-relative) > 1
    assert result['temperature_bound']['sides']['outlet']['absolute_total_K'][0] == pytest.approx(absolute)


def test_unknown_curve_keeps_real_stresses_and_fails_even_zero_tip(sampler):
    sections = sampler.row(0, 'SA', 5)
    original = sampler.requirements
    try:
        sampler.requirements = dict(original, curve_points=None)
        result = assess_stator(sampler, 0, sections, actual_profiles(), 31)
        assert not result['passed'] and math.isnan(result['min_margin'])
        assert result['table'].margin.isna().all()
        assert not result['table'].passed.any()
        assert all('unknown' in reason for reason in result['table'].reason)
        assert np.max(np.abs(result['contour_stress_Pa'][0])) > 1e5
        np.testing.assert_array_equal(result['contour_stress_Pa'][-1], 0.)
        # Retain the original bounded C curve, move P beyond its documented
        # range using life alone; there must be no interpolation extrapolation.
        sampler.requirements = dict(original, life_hours=1e10)
        outside = assess_stator(sampler, 0, sections, actual_profiles(), 31)
        assert not outside['passed'] and math.isnan(outside['min_margin'])
        assert outside['table'].allowable_MPa.isna().all()
        assert not outside['table'].lm_passed.any()
    finally:
        sampler.requirements = original


def test_absolute_normal_cap_includes_compression(sampler):
    sections = sampler.row(0, 'SA', 5)
    baseline = assess_stator(sampler, 0, sections, actual_profiles(), 31)
    root = baseline['table'].iloc[0]
    # Fixed loads and geometrically similar real curved profiles: bending
    # stress scales exactly with inverse length cubed, not rectangular area.
    scale = (root.sigma_normal_abs_max_MPa/310.)**(1/3)
    changed = assess_stator(sampler, 0, sections, actual_profiles(scale), 31)
    np.testing.assert_allclose(changed['contour_stress_Pa'][0],
                               baseline['contour_stress_Pa'][0]/scale**3, rtol=2e-9, atol=1e-4)
    assert changed['table'].iloc[0].sigma_normal_abs_max_MPa == pytest.approx(310., rel=2e-9)
    assert not changed['table'].iloc[0].normal_limit_passed
    assert not changed['table'].iloc[0].passed

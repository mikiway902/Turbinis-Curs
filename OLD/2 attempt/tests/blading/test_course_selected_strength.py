"""Requested-rotor strength retains analytical loads and genuine gas bending."""
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


@pytest.fixture(scope='module')
def strength_source():
    source = Path(__file__).resolve().parents[2] / 'TVD_1CBK_2st_10000h.ipynb'
    notebook = json.loads(source.read_text(encoding='utf-8'))
    module = ModuleType('_vinogradov_selected_strength_source')
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    ns = module.__dict__
    try:
        for index in SOURCE_CELLS:
            if index != 42:
                exec(compile(''.join(notebook['cells'][index]['source']),
                             f'{source.name}:cell{index}', 'exec'), ns)
        ns['P0'] = ns['evaluate'](ns['GEOMETRY'], ns['C'])
        yield ns
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize('rotor', (1, 2))
def test_selected_rotor_preserves_analytical_load_stress_and_margin(strength_source, rotor):
    sampler = CourseRadialSampler(strength_source)
    radii = np.array([s.radius_m for s in sampler.row(rotor-1, 'RK', 5)])
    k = (1-1/1.6)/(radii[-1]-radii[0])
    chord = .02/math.sin(math.radians(40))
    profile = build_parabolic_profile(ProfileSpec(.02, 40, 90, 17,
        .69*chord, .024*chord, .006*chord))
    polygon = profile.contour(101)
    profiles = {(f'РК{rotor}', j): dict(radius=r, polygon=polygon*np.sqrt(1-k*(r-radii[0])), Z=26)
                for j, r in enumerate(radii)}
    strength = sampler.strength(profiles, rotors=(rotor,))
    table = strength['table']
    assert len(table) == 8 and set(table.rotor) == {rotor}
    assert strength['computed_rotors'] == [rotor] and strength['incomplete_rotors'] == []
    assert set(strength['rotor_results']) == set(strength['temperature_bounds']) == {rotor}
    assert math.isfinite(strength['min_margin'])
    assert strength['min_margin'] == pytest.approx(table.margin.min(), rel=1e-12)
    assert strength['passed'] == bool(table.passed.all())

    # A(r)=Aroot*(1-k*(r-rroot)): exact integral and composite-trapezoid error.
    root_area = abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                           - np.roll(polygon[:, 0], -1)*polygon[:, 1]))/2
    integral = ((1+k*radii[0])*(radii[-1]**2-radii[0]**2)/2
                - k*(radii[-1]**3-radii[0]**3)/3)
    factor = sampler.requirements['density_kg_m3']*(2*math.pi*sampler.speed_rpm/60)**2
    force = factor*root_area*integral
    trapezoid_force = factor*root_area*(integral-k*np.sum(np.diff(radii)**3)/6)
    native = strength['rotor_results'][rotor]
    root = native['sections'][0]
    assert root['N'] == pytest.approx(force, rel=1e-9)
    assert native['trapezoid_table'].iloc[0].N == pytest.approx(trapezoid_force, rel=1e-9)
    quadratic = table.loc[(table.method == 'quadratic_3_tension_gas_bending') & (table.section == 0)].iloc[0]
    tension = table.loc[(table.method == 'trapezoid_5_tension') & (table.section == 0)].iloc[0]
    assert quadratic.sigma_tension_MPa == pytest.approx(force/root_area/1e6, rel=1e-9)
    assert tension.sigma_tension_MPa == pytest.approx(trapezoid_force/root_area/1e6, rel=1e-9)

    # Independently integrate actual native gas loads, then solve the Cartesian
    # inertia tensor rather than reusing native principal-axis stress results.
    span = native['span']
    r = span.r.to_numpy()
    # Native loads are PCHIP-integrated, not composite-trapezoid loads.
    def moment(load):
        return (PchipInterpolator(r, r*load).integrate(r[0], r[-1])
                - r[0]*PchipInterpolator(r, load).integrate(r[0], r[-1]))
    Mxi = -moment(span.qy.to_numpy())
    Meta = moment(span.qx.to_numpy())
    assert abs(Mxi)+abs(Meta) > 0
    assert root['Mxi'] == pytest.approx(Mxi, rel=1e-10, abs=1e-12)
    assert root['Meta'] == pytest.approx(Meta, rel=1e-10, abs=1e-12)
    gradient = np.linalg.solve([[root['Ieta'], root['Ixy']], [root['Ixy'], root['Ixi']]], [-Meta, Mxi])
    bending = (polygon-root['centroid'])@gradient/1e6
    expected_normal = force/root_area/1e6+bending.max()
    assert quadratic.sigma_bending_max_MPa == pytest.approx(bending.max(), rel=1e-9)
    assert quadratic.sigma_normal_max_MPa == pytest.approx(expected_normal, rel=1e-9)
    points = np.asarray(sampler.requirements['curve_points'])
    P = quadratic.T_bound_K*(sampler.requirements['C_LM']+math.log10(sampler.resource_hours))
    assert points[0, 0] <= P/1000 <= points[-1, 0]
    allowed = math.exp(np.interp(P/1000, points[:, 0], np.log(points[:, 1])))
    assert quadratic.margin == pytest.approx(allowed/expected_normal, rel=1e-9)

    # Candidate geometry must still be recomputed while gas temperatures cache.
    enlarged = {key: dict(section, polygon=section['polygon']*1.1) for key, section in profiles.items()}
    changed = sampler.strength(enlarged, rotors=(rotor,))
    changed_native = changed['rotor_results'][rotor]
    assert changed['temperature_bounds'][rotor] is strength['temperature_bounds'][rotor]
    assert changed_native['sections'][0]['N'] == pytest.approx(force*1.1**2, rel=1e-9)
    assert changed_native['sections'][0]['sigma_bending'].max()/1e6 == pytest.approx(bending.max()/1.1**3, rel=1e-9)
    fresh = CourseRadialSampler(strength_source).strength(profiles, rotors=(rotor,))
    assert fresh['temperature_bounds'][rotor] is not strength['temperature_bounds'][rotor]

    partial = sampler.strength(profiles)
    assert partial['computed_rotors'] == [rotor] and partial['incomplete_rotors'] == [3-rotor]
    assert not partial['passed'] and math.isnan(partial['min_margin'])
    np.testing.assert_allclose(partial['table'].margin, table.margin)
    broken = dict(profiles)
    broken[(f'РК{rotor}', 0)] = dict(profiles[(f'РК{rotor}', 0)], polygon=polygon*np.nan)
    failed = sampler.strength(broken, rotors=(rotor,))
    assert not failed['passed'] and math.isnan(failed['min_margin'])
    assert failed['computed_rotors'] == [] and failed['reasons']

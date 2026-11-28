"""An unavailable RK1 must not erase real RK2 loads or fabricate RK1 geometry."""
import json
import math
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from vinogradov.course_adapter import CourseRadialSampler, SOURCE_CELLS
from vinogradov.models import ProfileSpec
from vinogradov.parabolic_profile import build_parabolic_profile


def test_available_last_rotor_preserves_both_real_centrifugal_integrals():
    source = Path(__file__).resolve().parents[2] / 'TVD_1CBK_2st_10000h.ipynb'
    notebook = json.loads(source.read_text(encoding='utf-8'))
    module = ModuleType('_vinogradov_partial_strength_source')
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    ns = module.__dict__
    try:
        for index in SOURCE_CELLS:
            if index == 42:
                continue  # no repeated legacy cubic-profile optimization
            exec(compile(''.join(notebook['cells'][index]['source']),
                         f'{source.name}:cell{index}', 'exec'), ns)
        ns['P0'] = ns['evaluate'](ns['GEOMETRY'], ns['C'])
        sampler = CourseRadialSampler(ns)
        stations = sampler.row(1, 'RK', 5)
        r = np.array([s.radius_m for s in stations])
        k = (1-1/1.6)/(r[-1]-r[0])
        chord = .02/math.sin(math.radians(40))
        p = build_parabolic_profile(ProfileSpec(.02,40,90,17,.69*chord,.024*chord,.006*chord))
        polygon = p.contour(101)
        profiles = {('РК2',j): dict(radius=radius,polygon=polygon*np.sqrt(1-k*(radius-r[0])),Z=26)
                    for j,radius in enumerate(r)}
        strength = sampler.strength(profiles)
        rows = strength['table']
        quadratic = rows.loc[(rows.method=='quadratic_3_tension_gas_bending') & (rows.section==0)].iloc[0]
        trapezoid = rows.loc[(rows.method=='trapezoid_5_tension') & (rows.section==0)].iloc[0]
        integral = (1+k*r[0])*(r[-1]**2-r[0]**2)/2-k*(r[-1]**3-r[0]**3)/3
        multiplier = sampler.requirements['density_kg_m3']*(2*math.pi*sampler.speed_rpm/60)**2/1e6
        assert quadratic.sigma_tension_MPa == pytest.approx(multiplier*integral, rel=1e-9)
        assert trapezoid.sigma_tension_MPa == pytest.approx(
            multiplier*(integral-k*np.sum(np.diff(r)**3)/6), rel=1e-9)
        assert strength['computed_rotors']==[2] and strength['incomplete_rotors']==[1]
        assert not strength['passed'] and math.isnan(strength['min_margin'])
    finally:
        sys.modules.pop(module.__name__, None)

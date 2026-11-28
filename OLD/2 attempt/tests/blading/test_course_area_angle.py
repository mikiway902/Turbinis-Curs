"""Area-equivalent channel angle must convert sine, not multiply degrees."""
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vinogradov.blade_geometry import geometric_parameters
from vinogradov.channel_convergence import analyze_channel
from vinogradov.course_adapter import CourseRadialSampler
from vinogradov.integration import _source_channel_checks
from vinogradov.models import ProfileSpec
from vinogradov.parabolic_profile import build_parabolic_profile
from test_design_contract import section


def test_source_comparison_converts_free_area_not_the_angle_in_degrees():
    source = Path(__file__).resolve().parents[2] / 'TVD_1CBK_2st_10000h.ipynb'
    notebook = json.loads(source.read_text(encoding='utf-8'))
    module = ModuleType('_vinogradov_area_angle_source')
    sys.modules[module.__name__] = module
    try:
        # Real source geometry helpers, not replacement/mocked predicates.
        for index in (3, 5, 11, 36):
            exec(compile(''.join(notebook['cells'][index]['source']),
                         f'{source.name}:cell{index}', 'exec'), module.__dict__)
        chord = .1 / math.sin(math.radians(40))
        spec = ProfileSpec(.1, 40, 90, 17, .69*chord, .024*chord, .006*chord)
        profile = build_parabolic_profile(spec)
        passage = analyze_channel(profile, spec.pitch_m, samples=31, wall_samples=401)
        station = replace(section(), row='SA', alpha1_deg=17.)
        height_in, height_out = .04, .02
        sampler = SimpleNamespace(inputs={'СА1': [{'kind': 'СА', 'l_in': height_in, 'l_out': height_out}]},
            name=CourseRadialSampler.name, ns=module.__dict__)
        _, measured = _source_channel_checks(sampler, 0, 'SA', 0, profile,
            spec, station, passage, geometric_parameters(profile, spec, samples=41))
        throat_height = height_in + (height_out-height_in)*passage.throat.center_m[0]/spec.axial_width_m
        open_area = passage.throat_width_m * throat_height
        pitch_area = spec.pitch_m * height_out
        expected = math.degrees(math.asin(open_area/pitch_area))
        assert measured['angle_error_deg'] + station.flow_outlet_deg == pytest.approx(expected, abs=1e-10)
    finally:
        sys.modules.pop(module.__name__, None)

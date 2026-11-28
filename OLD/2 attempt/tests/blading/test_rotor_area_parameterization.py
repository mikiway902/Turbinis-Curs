"""The optimizer must enforce actual section area, not a thickness proxy."""
import math
from dataclasses import replace

import numpy as np
import pytest

from vinogradov.models import ProfileSpec
from vinogradov.blade_geometry import area_distribution
from vinogradov.blade_optimization import _rotor_area_law
from vinogradov.cascade_geometry import build_profile
from vinogradov.section_properties import section_properties


@pytest.mark.parametrize('law', ['linear', 'exponential', 'intermediate'])
def test_rotor_area_law_rebuilds_physical_profiles_preserving_metal_and_local_pitch(law):
    width = .02
    chord = width/math.sin(math.radians(40))
    base = ProfileSpec(width, 40., 90., 17., .0191354,
        .0243*chord, .006025*chord, relative_thickness=.106,
        thickness_location=.245, origin_y_m=.02)
    span = np.linspace(0., 1., 5)
    specs = [replace(base, pitch_m=2*np.pi*r/36) for r in np.linspace(.1, .2, 5)]
    methods = ['double_parabolic']*5
    profiles = [build_profile(s, m) for s, m in zip(specs, methods)]
    actual_specs, actual_profiles = _rotor_area_law(specs, profiles, span,
        1.6, law, .37, methods)
    areas = np.array([section_properties(p, count_per_side=1001).area_m2
                      for p in actual_profiles])
    expected = area_distribution(span, areas[0], areas[0]/1.6, law, .37)
    np.testing.assert_allclose(areas, expected, rtol=1e-11)
    assert np.all(np.diff(areas) < 0)
    for old, new in zip(specs, actual_specs):
        assert new.pitch_m == old.pitch_m
        assert new.inlet_metal_deg == old.inlet_metal_deg
        assert new.outlet_metal_deg == old.outlet_metal_deg
        assert new.stagger_deg == old.stagger_deg
        assert new.relative_thickness == old.relative_thickness
        scale = new.axial_width_m/old.axial_width_m
        assert new.leading_radius_m == pytest.approx(old.leading_radius_m*scale)
        assert new.trailing_radius_m == pytest.approx(old.trailing_radius_m*scale)

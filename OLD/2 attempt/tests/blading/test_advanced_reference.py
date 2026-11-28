"""Vinogradov 2010 pp.61–88: printed values AND quantified contradictions.

Source dimensions below are mm; the production API is SI. These fixtures are
new experimental references, not changes to the immutable Mikhaltsev baselines.
"""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.optimize import brentq

from vinogradov.models import ConstructionError, ProfileSpec
from vinogradov.double_parabolic_profile import (
    build_double_parabolic_profile,
    build_prescribed_thickness_profile,
    prescribed_thickness_construction,
)
from vinogradov.active_profile import (
    ActiveProfileOptions, active_profile_construction, build_active_profile,
)


def prescribed_spec(scale=1.):
    chord = .1/math.sin(math.radians(40))
    return ProfileSpec(axial_width_m=.1*scale, stagger_deg=40,
                       inlet_metal_deg=90, outlet_metal_deg=17,
                       pitch_m=.095677*scale, leading_radius_m=.0243*chord*scale,
                       trailing_radius_m=.006025*chord*scale, origin_y_m=.1*scale,
                       relative_thickness=.106, thickness_location=.245)


def active_spec(scale=1.):
    chord = .1/math.sin(math.radians(78))
    return ProfileSpec(axial_width_m=.1*scale, stagger_deg=78,
                       inlet_metal_deg=40, outlet_metal_deg=20,
                       pitch_m=.58*chord*scale, leading_radius_m=.53/25.61*chord*scale,
                       trailing_radius_m=.2/25.61*chord*scale, origin_y_m=.1*scale)


def _mm(actual, printed, tolerance=.0006):
    np.testing.assert_allclose(np.asarray(actual)*1000, printed, atol=tolerance, rtol=0)


def _unit_tangent(curve, t):
    v = curve.evaluate(t, 1)
    return v/np.linalg.norm(v)


def _assert_closed_tangent(profile):
    suction, pressure = profile.suction, profile.pressure
    np.testing.assert_allclose(profile.leading_edge.evaluate(0), pressure.evaluate(0), atol=1e-12)
    np.testing.assert_allclose(profile.leading_edge.evaluate(1), suction.evaluate(0), atol=1e-12)
    np.testing.assert_allclose(profile.trailing_edge.evaluate(0), suction.evaluate(1), atol=1e-12)
    np.testing.assert_allclose(profile.trailing_edge.evaluate(1), pressure.evaluate(1), atol=1e-12)
    np.testing.assert_allclose(_unit_tangent(profile.leading_edge, 0), -_unit_tangent(pressure, 0), atol=1e-10)
    np.testing.assert_allclose(_unit_tangent(profile.leading_edge, 1), _unit_tangent(suction, 0), atol=1e-10)
    np.testing.assert_allclose(_unit_tangent(profile.trailing_edge, 0), _unit_tangent(suction, 1), atol=1e-10)
    np.testing.assert_allclose(_unit_tangent(profile.trailing_edge, 1), -_unit_tangent(pressure, 1), atol=1e-10)
    for side in (suction, pressure):
        for a, b in zip(side.segments, side.segments[1:]):
            np.testing.assert_allclose(a.evaluate(1), b.evaluate(0), atol=1e-12)
            np.testing.assert_allclose(_unit_tangent(a, 1), _unit_tangent(b, 0), atol=1e-10)


def test_table_4_1_pressure_and_thickness_reference():
    d = prescribed_thickness_construction(prescribed_spec())
    _mm(d['chord_m'], 155.572)
    _mm(d['leading_center'], [3.780, 95.495])
    _mm(d['trailing_center'], [99.063, -18.058])
    _mm(d['pressure_radius_m'], 208.955)
    _mm(d['pressure_center'], [-97.538, -91.564])
    _mm(d['pressure_end'], [98.185, -18.387])
    _mm(d['pressure_start'][0], 1.980)
    _mm(d['thickness_radius_m'], 8.245)
    _mm(d['thickness_center'], [24.500, 88.110])
    _mm(d['suction_contact'], [29.133, 94.931])
    _mm(d['suction_start'], [1.980, 98.819])
    _mm(d['suction_end'], [99.974, -17.839])
    _mm(d['inlet_control'], [13.902, 105.276])
    _mm(d['source_throat_m'], 26.641)
    _mm(d['source_throat_point'], [89.404, 51.787], tolerance=.001)


def test_table_4_1_printed_root_is_not_a_solution():
    """p.69 Z_D=20.527 contradicts p.70 centre and contact coordinates."""
    d = prescribed_thickness_construction(prescribed_spec())
    _mm(d['pressure_contact'][0], 19.8672212929, tolerance=1e-8)
    printed_z = .020527
    cp, rp, rd = d['pressure_center'], d['pressure_radius_m'], d['thickness_radius_m']
    residual = printed_z+rd*(printed_z-cp[0])/rp-.0245
    _mm(residual, .68581349334, tolerance=1e-8)
    # Do not loosen a tolerance or substitute the contradictory printed root.
    assert abs(residual) > .00068
    np.testing.assert_allclose((d['pressure_contact']+d['suction_contact'])/2,
                               d['thickness_center'], atol=1e-14)
    assert np.linalg.norm(d['suction_contact']-d['pressure_contact']) == pytest.approx(2*rd)


def test_mean_line_trial_records_source_failure():
    """p.65–66 items22–30 explicitly abandon the common mean-line trial."""
    d = prescribed_thickness_construction(prescribed_spec())
    _mm(d['mean_point'], [24.5, 91.904], tolerance=.003)
    assert math.degrees(d['mean_tangent_angle_rad']) == pytest.approx(-19.841, abs=.01)
    assert d['first_pressure_tangent_clearance_m'] > 0


@pytest.mark.parametrize('builder', [build_prescribed_thickness_profile, build_double_parabolic_profile])
def test_two_parabolas_are_exact_and_closed(builder):
    p = builder(prescribed_spec())
    assert len(p.suction.segments) == 2
    assert all(segment.control_points.shape == (3, 2) for segment in p.suction.segments)
    _assert_closed_tangent(p)
    assert p.metadata['status'] == 'EXPERIMENTAL'
    assert p.metadata['prescribed_thickness_m'] == pytest.approx(.01649067256472037)


@pytest.mark.parametrize('x_mm,y_mm', [(2.728, 99.206), (3.483, 99.560),
                                       (5.012, 100.171), (8.148, 100.997),
                                       (14.729, 101.075)])
def test_table_5_1_first_coordinate_matrix(x_mm, y_mm):
    curve = build_double_parabolic_profile(prescribed_spec()).suction.segments[0]
    t = brentq(lambda t: curve.evaluate(t)[0]-.001*x_mm, 0., 1.)
    _mm(curve.evaluate(t)[1], y_mm, tolerance=.002)


def test_table_5_1_second_matrix_is_not_the_same_endpoint():
    """p.75 item3 starts 2.392,99.011,0.376; item2 starts 1.980,98.819,0.496."""
    p = build_double_parabolic_profile(prescribed_spec())
    start = p.suction.evaluate(0)
    assert .00040 < .002392-start[0] < .00042
    assert .00018 < .099011-start[1] < .00020
    actual = math.atan2(*p.suction.evaluate(0, 1)[::-1])
    assert .119 < actual-.376 < .121


def test_table_6_1_circle_reference():
    d = active_profile_construction(active_spec())
    _mm(d['chord_m'], 102.234)
    _mm(d['pressure_radius_m'], 60.877)
    _mm(d['leading_center'], [2.116, 99.550])
    _mm(d['trailing_center'], [99.202, 78.914])
    _mm(d['pressure_center'], [43.626, 52.169])
    _mm(d['pressure_start'], [3.510, 97.959])
    _mm(d['pressure_end'], [98.482, 78.568])
    _mm(d['source_throat_m'], 19.315)
    _mm(d['exit_join'], [81.00108, 70.35133], tolerance=.00002)
    assert math.degrees(d['exit_tangent_angle_rad']) == pytest.approx(-64.058, abs=.001)
    _mm(d['exit_blend_radius_m'], 288.761, tolerance=.003)
    _mm(d['exit_blend_contact'], [99.972, 19.828], tolerance=.001)
    _mm(d['exit_blend_center'], [-178.663, -55.972], tolerance=.003)
    _mm(d['inlet_blend_radius_m'], 76.988, tolerance=.002)
    _mm(d['inlet_blend_center'], [76.988, 40.255], tolerance=.002)
    _mm(d['inlet_blend_contact'], [4.485, 66.148], tolerance=.002)
    _mm(d['direct_inlet_contact'], [16.238, 83.431], tolerance=.002)


def test_active_source_failure_is_not_promoted_to_pass():
    p = build_active_profile(active_spec(), variant='source')
    _assert_closed_tangent(p)
    assert not p.metadata['construction_valid']
    assert not p.metadata['nondiffusing_channel_verified']
    assert p.metadata['inlet_metal_angle_error_deg'] == pytest.approx(15.61076965, abs=1e-7)
    assert p.metadata['actual_inlet_metal_deg'] == pytest.approx(24.38923035, abs=1e-7)
    # Source beta=17.077 requires gamma=.577, whereas an explicit gamma=7 gives23.5.
    assert p.metadata['source_beta_deg']-17.077 == pytest.approx(6.423)
    assert len(p.metadata['source_ambiguities']) == 3
    _mm(p.metadata['source_h_contact_residual_m'], .0306377579, tolerance=1e-8)


@pytest.mark.parametrize('builder,factory', [(build_double_parabolic_profile, prescribed_spec),
                                             (build_active_profile, active_spec)])
def test_advanced_construction_similarity(builder, factory):
    base, scaled = builder(factory()), builder(factory(.2))
    for name in ('suction', 'pressure', 'leading_edge', 'trailing_edge'):
        np.testing.assert_allclose(getattr(scaled, name).evaluate(np.linspace(0, 1, 57)),
                                   .2*getattr(base, name).evaluate(np.linspace(0, 1, 57)), atol=1e-13)


def test_invalid_construction_domains_have_quantified_errors():
    with pytest.raises(ConstructionError) as exc:
        build_double_parabolic_profile(replace(prescribed_spec(), thickness_location=.001))
    assert 'thickness_location_m' in exc.value.details
    with pytest.raises(ConstructionError) as exc:
        build_active_profile(active_spec(), ActiveProfileOptions(pressure_radius_chord_ratio=.01), variant='source')
    assert 'circle_intersection_height_squared_m2' in exc.value.details
    with pytest.raises(ConstructionError) as exc:
        build_active_profile(replace(active_spec(), pitch_m=1.))
    assert exc.value.details['suction_radius_m'] < 0


def test_reference_figure_reconstructions_preserve_source_pitch_and_circles():
    from vinogradov.double_parabolic_profile import prescribed_reference_figure_data
    from vinogradov.active_profile import active_reference_figure_data

    spec = prescribed_spec()
    figures = prescribed_reference_figure_data(spec, count=81)
    np.testing.assert_allclose(
        figures['table_5_1_item_7_adjacent_profile']-figures['table_5_1_item_7_profile'],
        np.broadcast_to([0., spec.pitch_m], figures['table_5_1_item_7_profile'].shape), atol=1e-14)
    d = prescribed_thickness_construction(spec)
    radii = np.linalg.norm(figures['figure_4_1_thickness_circle']-d['thickness_center'], axis=1)
    np.testing.assert_allclose(radii, d['thickness_radius_m'], atol=1e-14)
    spec = active_spec()
    figures = active_reference_figure_data(spec, count=81)
    np.testing.assert_allclose(
        figures['table_6_1_item_51_adjacent_profile']-figures['table_6_1_item_51_profile'],
        np.broadcast_to([0., spec.pitch_m], figures['table_6_1_item_51_profile'].shape), atol=1e-14)
    d = active_profile_construction(spec)
    radii = np.linalg.norm(figures['figure_6_1_constant_width_limit']-d['pressure_center'], axis=1)
    np.testing.assert_allclose(radii, d['suction_radius_m'], atol=1e-14)


def test_constrained_active_enforces_actual_inlet_and_outlet_metal_angles():
    spec = active_spec()
    p = build_active_profile(spec)
    _assert_closed_tangent(p)
    assert p.metadata['variant'] == 'constrained'
    assert p.metadata['inlet_join_method'] == 'equal-tangent-distance G1 biarc'
    assert p.metadata['inlet_single_circle_failure']['blend_radius_m'] < 0
    assert p.metadata['construction_valid']
    assert p.metadata['actual_inlet_metal_deg'] == pytest.approx(spec.inlet_metal_deg, abs=1e-9)
    assert p.metadata['actual_outlet_metal_deg'] == pytest.approx(spec.outlet_metal_deg, abs=1e-9)
    # Check derivatives themselves, not only the metadata assertion.
    pressure_in = math.atan2(*p.pressure.evaluate(0, 1)[::-1])
    suction_in = math.atan2(*p.suction.evaluate(0, 1)[::-1])
    pressure_out = math.atan2(*p.pressure.evaluate(1, 1)[::-1])
    suction_out = math.atan2(*p.suction.evaluate(1, 1)[::-1])
    assert 90-math.degrees((pressure_in+suction_in)/2) == pytest.approx(40, abs=1e-9)
    assert 90+math.degrees((pressure_out+suction_out)/2) == pytest.approx(20, abs=1e-9)
    assert not p.metadata['full_cascade_clearance_verified']
    assert not p.metadata['nondiffusing_channel_verified']


def test_constrained_biarc_rejects_an_impossible_forward_join():
    from vinogradov.active_profile import _g1_biarc

    with pytest.raises(ConstructionError) as exc:
        _g1_biarc(np.array([1., 0.]), np.array([-1., 0.]),
                   np.array([0., 0.]), np.array([-1., 0.]))
    assert 'minimum_axial_parameter_derivative_m' in exc.value.details


def test_active_source_variant_is_explicit_and_selection_is_strict():
    source = build_active_profile(active_spec(), variant='source')
    actual = build_active_profile(active_spec())
    assert source.metadata['variant'] == 'source'
    assert not source.metadata['construction_valid']
    assert actual.metadata['preconstraint_inlet_metal_angle_error_deg'] != 0
    assert source.metadata['inlet_metal_angle_error_deg'] == pytest.approx(15.610769652, abs=1e-8)
    with pytest.raises(ValueError, match='variant'):
        build_active_profile(active_spec(), variant='unknown')


def test_constrained_active_enforces_both_actual_trailing_side_tangents_and_wedge():
    spec = active_spec()
    p = build_active_profile(spec)
    pressure = _unit_tangent(p.pressure, 1)
    suction = _unit_tangent(p.suction, 1)
    measured = math.degrees(math.atan2(
        suction[0]*pressure[1]-suction[1]*pressure[0], float(np.dot(suction, pressure))))
    assert measured == pytest.approx(spec.trailing_wedge_deg, abs=1e-9)
    assert math.degrees(math.atan2(pressure[1], pressure[0])) == pytest.approx(-66.5, abs=1e-9)
    assert math.degrees(math.atan2(suction[1], suction[0])) == pytest.approx(-73.5, abs=1e-9)
    assert p.pressure.segments[0].radius == pytest.approx(.05917187371317375, abs=1e-13)
    assert p.metadata['actual_trailing_wedge_deg'] == pytest.approx(measured, abs=1e-9)
    assert abs(p.metadata['trailing_wedge_error_deg']) < 1e-9


def test_explicit_active_pressure_radius_is_a_constraint_not_an_overridden_hint():
    spec = active_spec()
    with pytest.raises(ConstructionError, match='radius conflicts') as exc:
        build_active_profile(spec, ActiveProfileOptions())
    _mm(exc.value.details['requested_pressure_radius_m'], 60.8773684955, tolerance=1e-8)
    _mm(exc.value.details['required_pressure_radius_m'], 59.1718737132, tolerance=1e-8)
    _mm(exc.value.details['pressure_radius_residual_m'], 1.7054947823, tolerance=1e-8)
    compatible = ActiveProfileOptions(pressure_radius_chord_ratio=.5787882630346465)
    p = build_active_profile(spec, compatible)
    assert p.metadata['pressure_radius_constraint'] == 'explicit compatible radius'
    assert p.metadata['actual_trailing_wedge_deg'] == pytest.approx(7, abs=1e-9)


def test_active_core_has_no_dependency_on_source_inlet_or_exit_blends():
    from vinogradov.active_profile import _active_core_geometry

    spec = active_spec()
    core = _active_core_geometry(spec)
    source = active_profile_construction(spec)
    for key, value in core.items():
        np.testing.assert_equal(value, source[key])
    assert 'inlet_blend_radius_m' not in core
    assert 'exit_blend_radius_m' not in core
    assert source['source_blends_evaluated']
    constrained = build_active_profile(spec)
    assert not constrained.metadata['source_blends_evaluated']
    assert 'exit_blend_radius_m' not in constrained.metadata
    assert constrained.metadata['actual_trailing_wedge_deg'] == pytest.approx(7, abs=1e-9)


def test_valid_constrained_domain_does_not_require_source_inlet_blend_branch():
    """Deliberate 40mm-pitch case, not a fabricated old RK3 midpoint fixture."""
    from vinogradov.active_profile import _active_core_geometry, _point_circle_blend

    spec = replace(active_spec(), pitch_m=.04)
    options = ActiveProfileOptions(pressure_radius_chord_ratio=.5787882630346465)
    core = _active_core_geometry(spec, options)
    assert core['suction_radius_m'] > 0
    with pytest.raises(ConstructionError, match='internal-tangency branch') as exc:
        _point_circle_blend(core['inlet_point'], math.pi/2,
                            core['pressure_center'], core['suction_radius_m'])
    # A real positive radius on the wrong source-only internal branch.
    assert 0 < exc.value.details['blend_radius_m'] < exc.value.details['target_radius_m']
    constrained = build_active_profile(spec, options)
    _assert_closed_tangent(constrained)
    assert constrained.metadata['construction_valid']
    assert not constrained.metadata['source_blends_evaluated']
    assert constrained.metadata['actual_inlet_metal_deg'] == pytest.approx(40, abs=1e-9)
    assert constrained.metadata['actual_outlet_metal_deg'] == pytest.approx(20, abs=1e-9)
    pressure = _unit_tangent(constrained.pressure, 1)
    suction = _unit_tangent(constrained.suction, 1)
    wedge = math.degrees(math.atan2(
        suction[0]*pressure[1]-suction[1]*pressure[0], float(np.dot(suction, pressure))))
    assert wedge == pytest.approx(7, abs=1e-9)

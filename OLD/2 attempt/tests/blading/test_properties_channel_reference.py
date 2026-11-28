"""EXPERIMENTAL geometry checks; printed refs from Vinogradov pp.48–60.

Source tolerances include graphical construction, printed rounding and the
book's branchwise surface integration. No accepted baseline is changed.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from vinogradov.cascade_throat import (
    effective_outlet_angle, perfect_gas_throat_flow, source_throat_width,
    throat_from_effective_angle,
)
from vinogradov.channel_convergence import analyze_channel, interpolate_circle_table
from vinogradov.curves import BezierCurve, CompositeCurve
from vinogradov.section_properties import section_properties
from vinogradov.velocity_triangles import elementary_stage_layout, velocity_triangles


# Literal table 3.4 item 9, p.55, mm -> m. These are NOT solver-generated refs.
SOURCE_CIRCLES_M = np.array([
    [43.37, 129.662, 27.8], [49.31, 125.268, 25.8],
    [55.18, 120.138, 24.], [60.966, 114.184, 22.4],
    [66.654, 107.276, 21.], [72.364, 99.282, 19.65],
    [78.138, 89.873, 18.3], [81.39, 83.208, 17.75],
])*1e-3


def _rectangle_profile(length=.1, thickness=.02):
    suction = CompositeCurve((BezierCurve([[0., 0.], [length, 0.]]),))
    pressure = CompositeCurve((BezierCurve([[0., -thickness], [length, -thickness]]),))
    contour = np.array([[0., 0.], [length, 0.], [length, -thickness], [0., -thickness], [0., 0.]])
    return SimpleNamespace(suction=suction, pressure=pressure,
                           contour=lambda count_per_side=401: contour)


def test_polygon_integrals_principal_axes_and_distributed_moduli():
    # Independent rectangular formulas A=bh, Ixx=bh^3/12, Iyy=hb^3/12.
    points = np.array([[0., 0.], [.1, 0.], [.1, .02], [0., .02]])
    properties = section_properties(points)
    assert properties.area_m2 == pytest.approx(.002)
    np.testing.assert_allclose(properties.centroid_m, [.05, .01], atol=1e-15)
    np.testing.assert_allclose(properties.centroidal_moments_m4,
                               [.1*.02**3/12, .02*.1**3/12, 0.], atol=1e-20)
    assert properties.principal_angle_rad == pytest.approx(0., abs=1e-14)
    assert properties.principal_min_m4 == pytest.approx(.1*.02**3/12)
    moduli = properties.distributed_moduli(np.vstack([[.1, .02], properties.centroid_m]))
    np.testing.assert_allclose(moduli[0], [.1*.02**2/6, .02*.1**2/6])
    assert np.isinf(moduli[1]).all()
    np.testing.assert_allclose(properties.to_principal(points), points-[.05, .01])
    assert properties.contour_distribution.arc_length_m[-1] == pytest.approx(.24)
    reverse = section_properties(points[::-1])
    np.testing.assert_allclose(reverse.centroidal_moments_m4, properties.centroidal_moments_m4, atol=1e-20)


def test_rotated_translated_section_and_scale_laws():
    points = np.array([[0., 0.], [.1, 0.], [.1, .02], [0., .02]])
    angle = -.7
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    transformed = section_properties(points@rotation.T+[1., 2.])
    assert transformed.principal_angle_rad == pytest.approx(angle, abs=1e-12)
    assert transformed.principal_min_m4 == pytest.approx(.1*.02**3/12, rel=1e-11)
    base, scaled = section_properties(points), section_properties(3*points)
    assert scaled.area_m2/base.area_m2 == pytest.approx(9.)
    assert scaled.principal_max_m4/base.principal_max_m4 == pytest.approx(81.)
    np.testing.assert_allclose(scaled.distributed_moduli(3*points), 27*base.distributed_moduli(points))


def test_profile_side_distributions_are_not_only_extreme_fibre_moduli():
    properties = section_properties(_rectangle_profile())
    assert properties.suction_length_m == pytest.approx(.1)
    assert properties.pressure_length_m == pytest.approx(.1)
    for side in (properties.suction_distribution, properties.pressure_distribution):
        assert len(side.points_m) == 1001
        assert side.arc_length_m[-1] == pytest.approx(.1)
        np.testing.assert_allclose(side.modulus_u_m3, .1*.02**2/6)
        assert side.modulus_v_m3[500] > 1e6


@pytest.mark.parametrize("bad", [np.zeros((3, 2)), [[0, 0], [1, 0]], [[0, 0], [1, 0], [0, float('nan')]]])
def test_invalid_contours(bad):
    with pytest.raises(ValueError):
        section_properties(bad)


def test_true_two_wall_tangency_parallel_channel():
    channel = analyze_channel(_rectangle_profile(), .06, samples=9, wall_samples=101,
                              parameter_range=(.1, .9), include_trailing_edge=False)
    # Gap .06-.02=.04, not the pitch, and not an axial separation.
    np.testing.assert_allclose(channel.radii_m, .02, atol=1e-10)
    np.testing.assert_allclose(channel.centers_m[:, 1], .02, atol=1e-10)
    assert channel.throat_width_m == pytest.approx(.04, abs=1e-9)
    assert channel.convergence_ratio == pytest.approx(1., abs=1e-9)
    assert not channel.diffuser_intervals_m
    for circle in channel.circles:
        np.testing.assert_allclose(circle.center_m-circle.suction_contact_m,
                                   circle.radius_m*circle.suction_normal, atol=1e-9)
        np.testing.assert_allclose(circle.center_m-circle.pressure_contact_m,
                                   circle.radius_m*circle.pressure_normal, atol=1e-9)
        assert circle.normal_residual < 1e-8
        assert circle.clearance_residual_m >= -1e-8
    assert any(c.endcap_limited for c in channel.circles)
    assert any(not c.endcap_limited for c in channel.circles)


def test_table34_spline_domain_and_printed_data():
    interpolation = interpolate_circle_table(SOURCE_CIRCLES_M)
    assert interpolation.axial_m[0] == .04337
    assert interpolation.axial_m[-1] == .08139
    assert interpolation.center_y_m[0] == pytest.approx(.129662)
    assert interpolation.center_y_m[-1] == pytest.approx(.083208)
    assert interpolation.radii_m[0] == pytest.approx(.0278)
    assert interpolation.radii_m[-1] == pytest.approx(.01775)
    assert interpolation.arc_length_m[0] == 0.
    assert np.all(np.diff(interpolation.arc_length_m) > 0)
    assert np.all(interpolation.radius_gradient < 0)
    # Independent lower bound: centreline length >= straight endpoint distance.
    assert interpolation.arc_length_m[-1] >= np.linalg.norm(SOURCE_CIRCLES_M[-1, :2]-SOURCE_CIRCLES_M[0, :2])
    with pytest.raises(ValueError):
        interpolate_circle_table(SOURCE_CIRCLES_M[::-1])


def _c9017():
    from vinogradov.models import ProfileSpec
    from vinogradov.parabolic_profile import build_parabolic_profile
    chord = .1/math.sin(math.radians(40.))
    spec = ProfileSpec(axial_width_m=.1, stagger_deg=40., inlet_metal_deg=90.,
                       outlet_metal_deg=17., pitch_m=.69*chord,
                       leading_radius_m=.024*chord, trailing_radius_m=.006*chord,
                       trailing_wedge_deg=7., throat_multiplier=1.05, origin_y_m=.1)
    return build_parabolic_profile(spec), spec


def test_table33_c9017_printed_integrals():
    profile, _ = _c9017()
    properties = section_properties(profile, count_per_side=2001)
    # Source numerical/graphical reference, pp.48–50; 1% integral tolerance,
    # 0.5 mm centroid and 0.5 degree principal-axis orientation tolerances.
    assert properties.area_m2 == pytest.approx(2.659e-3, rel=.01)
    np.testing.assert_allclose(properties.first_moments_m3, [1.368e-4, 1.755e-4], rtol=.01)
    np.testing.assert_allclose(properties.second_moments_m4, [1.363e-5, 8.506e-6, 7.488e-6], rtol=.01)
    np.testing.assert_allclose(properties.centroid_m, [.051457, .066011], atol=.0005)
    np.testing.assert_allclose(properties.centroidal_moments_m4, [2.04e-6, 1.466e-6, -1.543e-6], rtol=.02)
    assert properties.principal_min_m4 == pytest.approx(1.841e-7, rel=.03)
    assert properties.principal_max_m4 == pytest.approx(3.323e-6, rel=.02)
    assert math.degrees(properties.principal_angle_rad) == pytest.approx(-50.268, abs=.5)
    # Source items 3–4 stop short of the tail's vertical tangent, at unequal
    # axial limits. Compare the same limits, not only the main curve lengths.
    suction, pressure = properties.rounded_suction_distribution, properties.rounded_pressure_distribution
    assert np.interp(.0995, suction.points_m[:, 0], suction.arc_length_m) == pytest.approx(.178126, abs=.001)
    assert np.interp(.09945, pressure.points_m[:, 0], pressure.arc_length_m) == pytest.approx(.155260, abs=.001)


def test_table34_graphical_circles_against_solved_normals():
    profile, spec = _c9017()
    channel = analyze_channel(profile, spec.pitch_m, samples=61, wall_samples=1001)
    order = np.argsort(channel.centers_m[:, 0])
    x = channel.centers_m[order, 0]
    # p.55 radii were chosen graphically (0.05–0.2 mm increments), not exact roots.
    assert x[0] <= SOURCE_CIRCLES_M[0, 0]
    assert x[-1] >= SOURCE_CIRCLES_M[-1, 0]
    y = np.interp(SOURCE_CIRCLES_M[:, 0], x, channel.centers_m[order, 1])
    r = np.interp(SOURCE_CIRCLES_M[:, 0], x, channel.radii_m[order])
    np.testing.assert_allclose(y, SOURCE_CIRCLES_M[:, 1], atol=.0005)
    np.testing.assert_allclose(r, SOURCE_CIRCLES_M[:, 2], atol=.0005)
    assert max(c.normal_residual for c in channel.circles) < 1e-6
    # p.54 item 6 uses the prescribed a_th from p.18, not the achieved
    # two-wall throat: p.18 explicitly warns the parabola may miss point p.
    design_throat = source_throat_width(spec.pitch_m, spec.outlet_metal_deg,
                                        throat_multiplier=spec.throat_multiplier)
    assert design_throat/.066 == pytest.approx(.453, abs=.0005)
    # Independently check the achieved throat by dense nearest-wall distance,
    # without calling the normal-circle solver or re-pinning its output.
    from scipy.spatial import cKDTree
    parameter = np.linspace(0., 1., 4001)
    lower_wall = profile.suction.evaluate(parameter)
    upper_wall = np.vstack([profile.pressure.evaluate(parameter),
                            profile.trailing_edge.evaluate(parameter)])+[0., spec.pitch_m]
    sampled_gap = float(cKDTree(upper_wall).query(lower_wall)[0].min())
    assert channel.throat_width_m == pytest.approx(sampled_gap, abs=2e-6)


def test_table35_velocity_vectors_and_source_plot():
    triangles = velocity_triangles(200., 17., 100., 150., 35.)
    np.testing.assert_allclose(triangles.absolute_inlet_m_s, [58.47434094, 191.26095119], atol=1e-8)
    np.testing.assert_allclose(triangles.relative_outlet_m_s, [86.03646545, -122.87280664], atol=1e-8)
    np.testing.assert_allclose(triangles.relative_inlet_m_s+triangles.blade_velocity_m_s, triangles.absolute_inlet_m_s)
    np.testing.assert_allclose(triangles.relative_outlet_m_s+triangles.blade_velocity_m_s, triangles.absolute_outlet_m_s)
    np.testing.assert_allclose(triangles.inlet_plot_m_s,
                              [[0, 0], [-191.26095119, -58.47434094], [-91.26095119, -58.47434094], [0, 0]], atol=1e-8)
    np.testing.assert_allclose(triangles.outlet_plot_m_s,
                              [[0, 0], [122.87280664, -86.03646545], [22.87280664, -86.03646545], [0, 0]], atol=1e-8)
    assert triangles.euler_work_j_kg == pytest.approx(21413.375783, abs=1e-6)
    assert triangles.combined_plot_m_s.shape == (7, 2)


def test_table36_elementary_stage_transform():
    profile = _rectangle_profile()
    triangles = velocity_triangles(200., 17., 100., 150., 35.)
    layout = elementary_stage_layout(profile, profile, stator_pitch_m=.06,
                                     rotor_pitch_m=.07, triangles=triangles)
    np.testing.assert_allclose(layout.stator_contours_m[1], layout.stator_contours_m[0]-[0, .06])
    np.testing.assert_allclose(layout.rotor_contours_m[0][0], [.11, .01])
    np.testing.assert_allclose(layout.rotor_contours_m[1], layout.rotor_contours_m[0]+[0, .07])
    np.testing.assert_allclose(layout.inlet_triangle_m[0], [.1, -.05])
    np.testing.assert_allclose(layout.outlet_triangle_m[0], [.2, .02])


def test_throat_relation_and_separate_choking_thermodynamics():
    width = source_throat_width(.1, 30., throat_multiplier=1.05)
    assert width == pytest.approx(.05/1.05)
    assert throat_from_effective_angle(.1, effective_outlet_angle(width, .1)) == pytest.approx(width)
    flow = perfect_gas_throat_flow(stagnation_pressure_pa=100000., stagnation_temperature_k=300.,
                                   back_pressure_pa=20000., gamma=1.4, gas_constant_j_kg_k=287., area_m2=.001)
    assert flow.choked
    assert flow.exit_mach == 1.
    assert flow.critical_pressure_ratio == pytest.approx(.5282817877, abs=1e-10)
    assert flow.mass_flux_kg_m2_s == pytest.approx(233.355856, abs=1e-5)
    assert flow.mass_flow_kg_s == pytest.approx(.233355856, abs=1e-8)
    stationary = perfect_gas_throat_flow(stagnation_pressure_pa=100000., stagnation_temperature_k=300.,
                                         back_pressure_pa=100000., gamma=1.4, gas_constant_j_kg_k=287., area_m2=.001)
    assert stationary.mass_flow_kg_s == 0.
    assert not stationary.choked
    with pytest.raises(ValueError):
        effective_outlet_angle(.11, .1)

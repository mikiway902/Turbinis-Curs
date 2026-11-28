"""Source references, not acceptance of the EXPERIMENTAL blading model.

Printed mm are converted explicitly to SI. No legacy baseline is imported or
changed. See docs/vinogradov/source_19_44.md for source arithmetic mismatches.
"""
import math
import unittest
from dataclasses import replace

import numpy as np

from vinogradov.curves import BezierCurve, CircularArc, CompositeCurve
from vinogradov.models import ConstructionError, ProfileSpec, SectionProfile
from vinogradov.parabolic_profile import (
    build_parabolic_profile, build_scheme, build_two_circle_pressure_profile,
    maximum_thickness, parabolic_envelope,
)


def reference_spec(outlet=17., pitch_coefficient=.685):
    chord = .1/math.sin(math.radians(36.33))
    return ProfileSpec(axial_width_m=.1, stagger_deg=36.33,
                       inlet_metal_deg=90., outlet_metal_deg=outlet,
                       pitch_m=pitch_coefficient*chord,
                       leading_radius_m=.0243*chord,
                       trailing_radius_m=.006025*chord)


class ParabolicSourceReferences(unittest.TestCase):
    def test_table_2_1_construction(self):
        """pp21–28, printed lengths at .001 mm resolution."""
        scheme = build_scheme(reference_spec(14., .595))
        tolerance = .0006e-3
        self.assertAlmostEqual(scheme['chord_m'], 168.795e-3, delta=tolerance)
        self.assertAlmostEqual(scheme['pressure_radius_m'], 233.025e-3, delta=tolerance)
        for name, expected in {
            'O1': [4.102,94.422], 'O2': [98.983,-34.601],
            'O2_pitch': [98.983,65.832], 'O3': [-124.227,-104.979],
            'p': [91.719,42.793], 'h': [98.013,-34.907],
            'l': [1.882,97.871], 'f': [99.983,-34.416],
            'B': [67.623,140.181],
        }.items():
            np.testing.assert_allclose(scheme[name], np.array(expected)*1e-3,
                                       atol=tolerance, rtol=0, err_msg=name)
        self.assertAlmostEqual(scheme['inlet_wedge_deg'], 65.528, delta=.001)
        self.assertAlmostEqual(scheme['design_throat_m'], 23.14e-3, delta=.0051e-3)

    def test_table_2_2_all_four_envelopes(self):
        """p32 inputs, p33 plots; vertical tangents are not slope singularities."""
        cases = [([0,0],[100,0],math.pi/3,-math.pi/6),
                 ([0,0],[100,10],2*math.pi/3,-math.pi/12),
                 ([0,0],[100,-10],math.pi/2,-math.pi/4),
                 ([-10,10],[100,10],-math.pi/4,-math.pi/2)]
        for start,end,a0,a1 in cases:
            with self.subTest(start=start,end=end,a0=a0):
                curve = parabolic_envelope(np.array(start)*1e-3,
                                            np.array(end)*1e-3,a0,a1)
                a,b,c = curve.control_points
                np.testing.assert_allclose(curve.evaluate(.5), (a+2*b+c)/4,
                                           atol=1e-15)
                np.testing.assert_allclose(curve.evaluate(.5,1), c-a, atol=1e-15)
                for t,angle in [(0,a0),(1,a1)]:
                    tangent = curve.evaluate(t,1)
                    self.assertAlmostEqual(tangent[0]*math.sin(angle)-
                                           tangent[1]*math.cos(angle),0.,delta=1e-14)
                # Half-median remains exact recursively on both subarcs.
                for lo,hi in [(0,.5),(.5,1),(0,.25),(.75,1)]:
                    p0,p1 = curve.evaluate(lo),curve.evaluate(hi)
                    d0,d1 = curve.evaluate(lo,1),curve.evaluate(hi,1)
                    subcurve = parabolic_envelope(p0,p1,math.atan2(d0[1],d0[0]),
                                                 math.atan2(d1[1],d1[0]))
                    np.testing.assert_allclose(subcurve.evaluate(.5),
                                               curve.evaluate((lo+hi)/2),atol=1e-14)
        vertical = parabolic_envelope([0,0],[.1,-.01],math.pi/2,-math.pi/4)
        np.testing.assert_allclose(vertical.evaluate(.5),[.025,.0425],atol=1e-14)
        last = parabolic_envelope([-.01,.01],[.1,.01],-math.pi/4,-math.pi/2)
        np.testing.assert_allclose(last.evaluate(.5),[.0725,-.045],atol=1e-14)

    def test_table_3_1_closed_profile_and_dyadic_points(self):
        """pp37–43; exact envelope replaces the printed spline only."""
        spec = reference_spec()
        scheme = build_scheme(spec)
        profile = build_parabolic_profile(spec)
        np.testing.assert_allclose(scheme['B'],[.058454,.138571],atol=.0006e-3,rtol=0)
        self.assertAlmostEqual(scheme['pressure_radius_m'],.273077,delta=.0006e-3)
        np.testing.assert_allclose(scheme['O3'],[-.157753,-.130591],atol=.0006e-3,rtol=0)
        # Table3.1 item37 first seven output points are dyadic, not uniform.
        ts = [0,1/1024,1/512,1/256,1/128,1/64,1/32]
        expected = np.array([[1.707,97.752],[1.817,97.832],[1.928,97.911],
                             [2.15,98.068],[2.592,98.377],[3.476,98.975],
                             [5.238,100.095]])*1e-3
        np.testing.assert_allclose(profile.suction.evaluate(ts),expected,atol=.0006e-3,rtol=0)
        np.testing.assert_allclose(profile.leading_edge.evaluate(0),profile.pressure.evaluate(0),atol=1e-14)
        np.testing.assert_allclose(profile.leading_edge.evaluate(1),profile.suction.evaluate(0),atol=1e-14)
        np.testing.assert_allclose(profile.trailing_edge.evaluate(0),profile.suction.evaluate(1),atol=1e-14)
        np.testing.assert_allclose(profile.trailing_edge.evaluate(1),profile.pressure.evaluate(1),atol=1e-14)
        contour = profile.contour()
        np.testing.assert_array_equal(contour[0],contour[-1])
        np.testing.assert_allclose(profile.metadata['throat_point_m'],[.066932,.064816],atol=.006e-3,rtol=0)
        self.assertAlmostEqual(profile.metadata['back_angle_deg'],13.325,delta=.005)
        # Direct exact-quadratic normal distance, not the inconsistent printed R.
        self.assertAlmostEqual(profile.metadata['throat_circle_radius_m'],.0359208394592493,delta=1e-12)
        self.assertAlmostEqual(profile.metadata['throat_m'],.0349038503921967,delta=1e-10)

    def test_table_3_2_reconstructed_table_3_1_common_normal(self):
        """Additional table3.1 control; table3.2's fixture is printed on p47."""
        profile = build_parabolic_profile(reference_spec())
        result = maximum_thickness(profile)
        self.assertAlmostEqual(result['diameter_m'],.0309166171701376,delta=1e-10)
        np.testing.assert_allclose(result['suction_contact_m'],[.0449542223020968,.0959521357622643],atol=1e-10)
        np.testing.assert_allclose(result['pressure_contact_m'],[.0243385744310064,.0729123380179252],atol=1e-10)
        self.assertLess(result['normal_equation_residual'],1e-8)
        self.assertFalse(result['globally_inscribed_circle_verified'])

    def test_table_3_2_published_fixture(self):
        """p47 inputs; pp45–46 outputs include spline/interpolation error."""
        chord = .1/math.sin(math.radians(47.))
        spec = ProfileSpec(axial_width_m=.1, stagger_deg=47.,
                           inlet_metal_deg=101., outlet_metal_deg=25.,
                           pitch_m=.63*chord, leading_radius_m=.031*chord,
                           trailing_radius_m=.0095*chord,
                           trailing_wedge_deg=7., throat_multiplier=1.05,
                           origin_y_m=.1)
        profile = build_parabolic_profile(spec)
        result = maximum_thickness(profile)
        # Printed contacts come from interpolating seven angular residuals.
        np.testing.assert_allclose(result['pressure_contact_m'],[.036035,.074523],
                                   atol=.12e-3,rtol=0)
        np.testing.assert_allclose(result['suction_contact_m'],[.044876,.087862],
                                   atol=.12e-3,rtol=0)
        self.assertAlmostEqual(result['diameter_m'],16.003e-3,delta=.006e-3)
        # Independently reconstructed exact envelope/common-normal solution.
        self.assertAlmostEqual(result['diameter_m'],.0159981355319271,delta=1e-10)
        np.testing.assert_allclose(result['pressure_contact_m'],
                                   [.0359393150875714,.0745864422938332],atol=1e-10)
        np.testing.assert_allclose(result['suction_contact_m'],
                                   [.0447692462401117,.0879270815715930],atol=1e-10)
        # The printed pressure ordinate does reproduce this exact circle.
        scheme = build_scheme(spec)
        center, radius = scheme['O3'],scheme['pressure_radius_m']
        ordinate = center[1]+math.sqrt(radius**2-(.036035-center[0])**2)
        self.assertAlmostEqual(ordinate,.074523,delta=.0006e-3)

    def test_thickness_analytical_limit(self):
        suction = CompositeCurve((BezierCurve([[0,.01],[.05,.03],[.1,.01]]),))
        pressure = CompositeCurve((BezierCurve([[0,0],[.1,0]]),))
        leading = CircularArc([0,.005],.005,-math.pi/2,-3*math.pi/2)
        trailing = CircularArc([.1,.005],.005,math.pi/2,-math.pi/2)
        profile = SectionProfile(suction,pressure,leading,trailing,{})
        result = maximum_thickness(profile)
        self.assertAlmostEqual(result['diameter_m'],.02,delta=1e-10)
        np.testing.assert_allclose(result['center_m'],[.05,.01],atol=1e-9)

    def test_table_3_2_printed_control_internal_consistency(self):
        """pp45–46 printed identities, with distinct fixture supplied on p47."""
        # Rounding the printed angular residuals limits the interpolated root.
        x = np.array([10.,17.,24.,31.,38.,45.,52.])
        mismatch = np.array([-.284,-.204,-.124,-.049,.019,.075,.117])
        interpolated = x[3]-mismatch[3]*(x[4]-x[3])/(mismatch[4]-mismatch[3])
        self.assertAlmostEqual(interpolated,36.035,delta=.06)
        pressure = np.array([36.035,74.523])*1e-3
        suction = np.array([44.876,87.862])*1e-3
        self.assertAlmostEqual(np.linalg.norm(suction-pressure),16.003e-3,delta=.0015e-3)


    def test_two_circle_schematic_alternative(self):
        profile = build_two_circle_pressure_profile(reference_spec(),inlet_wedge_deg=45.)
        first,second = profile.pressure.segments
        np.testing.assert_allclose(first.evaluate(1),second.evaluate(0),atol=1e-13)
        a,b = first.evaluate(1,1),second.evaluate(0,1)
        np.testing.assert_allclose(a/np.linalg.norm(a),b/np.linalg.norm(b),atol=1e-12)
        np.testing.assert_allclose(profile.leading_edge.evaluate(0),first.evaluate(0),atol=1e-13)
        np.testing.assert_allclose(profile.trailing_edge.evaluate(1),second.evaluate(1),atol=1e-13)
        self.assertTrue(profile.metadata['biarc_join_is_software_choice'])
        self.assertEqual(profile.metadata['inlet_wedge_deg'],45.)

    def test_scaling_and_explicit_metal_angles(self):
        spec = reference_spec()
        profile = build_parabolic_profile(spec)
        scaled = replace(spec,axial_width_m=spec.axial_width_m*2,pitch_m=spec.pitch_m*2,
                         leading_radius_m=spec.leading_radius_m*2,
                         trailing_radius_m=spec.trailing_radius_m*2,origin_y_m=spec.origin_y_m*2)
        other = build_parabolic_profile(scaled)
        np.testing.assert_allclose(other.suction.evaluate([0,.5,1]),
                                   2*profile.suction.evaluate([0,.5,1]),atol=1e-14)
        # Flow corrections are recorded in spec but never applied again to metal.
        corrected = build_parabolic_profile(replace(spec,incidence_deg=3.,deviation_deg=2.))
        np.testing.assert_array_equal(corrected.suction.evaluate(.5),profile.suction.evaluate(.5))

    def test_degenerate_envelope_and_invalid_biarc(self):
        with self.assertRaises(ConstructionError):
            parabolic_envelope([0,0],[1,0],0.,0.)
        with self.assertRaises(ConstructionError):
            build_two_circle_pressure_profile(reference_spec(),inlet_wedge_deg=0.)
        with self.assertRaises(ConstructionError):
            build_two_circle_pressure_profile(reference_spec(),inlet_wedge_deg=45.,tangent_distance_ratio=-1.)


if __name__ == '__main__':
    unittest.main()

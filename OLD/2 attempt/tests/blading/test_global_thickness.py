"""Actual maximal inscribed circles, not vertical or equal-parameter gaps."""
import math

import numpy as np
import pytest

from vinogradov.curves import BezierCurve, CircularArc, CompositeCurve
from vinogradov.double_parabolic_profile import build_double_parabolic_profile
from vinogradov.models import ProfileSpec, SectionProfile
from vinogradov.validation import validate_prescribed_thickness, validate_profile


def circle_profile(radius=.01):
    arc=lambda a,b: CircularArc(np.zeros(2),radius,a,b)
    return SectionProfile(CompositeCurve((arc(math.pi,math.pi/2),)),
                          CompositeCurve((arc(-math.pi/2,0.),)),
                          arc(-math.pi/2,-math.pi),arc(math.pi/2,0.),
                          {'prescribed_thickness_m':2*radius,'thickness_center':[0.,0.]})


def tapered_capsule(small=.002,large=.006,length=.03):
    """Convex hull of two circles; its exact inradius is the larger radius.

    Both circles are contained; the complete hull lies in |y| <= large,
    independently establishing the matching upper bound.
    """
    nx=(small-large)/length
    normal=np.array([nx,math.sqrt(1-nx*nx)])
    first=np.zeros(2);last=np.array([length,0.])
    top=np.array([small*normal,last+large*normal])
    bottom=top*np.array([1.,-1.])
    angle=math.atan2(normal[1],normal[0])
    return SectionProfile(CompositeCurve((BezierCurve(top),)),
                          CompositeCurve((BezierCurve(bottom),)),
                          CircularArc(first,small,-angle,angle-2*math.pi),
                          CircularArc(last,large,angle,-angle),
                          {'prescribed_thickness_m':2*small,'thickness_center':[0.,0.]})


def reference_spec():
    chord=.1/math.sin(math.radians(40))
    return ProfileSpec(.1,40,90,17,.095677,.0243*chord,.006025*chord,
                       relative_thickness=.106,thickness_location=.245)


def course_spec():
    # Actual course-derived dimensions and metal angles from the axial-envelope
    # regression, with its constructible 75-degree stagger; no contour scaling.
    return ProfileSpec(.02236309352291337,75.,32.36385724199586,
        27.260756849807482,.02104125693721863,.000782708273301968,
        .0002500000000000014,trailing_wedge_deg=3.,origin_y_m=0.,
        relative_thickness=.12282186549577312,thickness_location=.3)


def assert_enclosure(result,known_diameter,tolerance):
    assert result['diameter_lower_m'] <= known_diameter <= result['diameter_upper_m']
    assert result['diameter_error_m'] == pytest.approx(
        result['diameter_upper_m']-result['diameter_lower_m'])
    assert result['diameter_error_m'] <= tolerance
    assert abs(result['maximum_diameter_m']-known_diameter) <= tolerance
    assert result['converged']


def test_known_circle_radius_and_independent_error_enclosure():
    result=validate_prescribed_thickness(circle_profile(),diameter_tolerance_m=2e-6)
    assert_enclosure(result,.02,2e-6)
    np.testing.assert_allclose(result['maximum_center_m'],[0.,0.],atol=1e-12)
    assert result['prescribed_status']=='within_tolerance'
    assert result['passed']


def test_remote_bulge_invalidates_a_genuinely_inscribed_local_target():
    result=validate_prescribed_thickness(tapered_capsule(),diameter_tolerance_m=2e-6)
    assert_enclosure(result,.012,2e-6)
    assert result['prescribed_center_diameter_m']==pytest.approx(.004,abs=1e-12)
    assert result['maximum_center_m'][0]>.025
    assert result['prescribed_status']=='disproved'
    checks={check.name:check for check in result['checks']}
    assert checks['prescribed circle is inscribed'].passed
    assert not checks['prescribed circle attains global maximum'].passed
    assert checks['prescribed circle attains global maximum'].value>.0079


def test_tied_global_maximum_does_not_invalidate_requested_axial_centre():
    result=validate_prescribed_thickness(
        tapered_capsule(small=.006,large=.006,length=.02),
        diameter_tolerance_m=1e-4)
    assert_enclosure(result,.012,1e-4)
    assert result['maximum_center_m'][0]>.005
    assert result['prescribed_center_m']==[0.,0.]
    assert result['prescribed_status']=='within_tolerance'
    assert result['passed']


def test_budget_exhaustion_exposes_unresolved_physical_error():
    result=validate_prescribed_thickness(circle_profile(),max_cells=1)
    assert result['diameter_lower_m']<=.02<=result['diameter_upper_m']
    assert result['diameter_error_m']>result['diameter_tolerance_m']
    assert result['cells_evaluated']==1
    assert not result['converged']
    assert result['prescribed_status']=='unresolved'
    assert not result['passed']


def test_source_two_parabola_circle_attains_maximum_within_reported_accuracy():
    profile=build_double_parabolic_profile(reference_spec())
    result=validate_prescribed_thickness(profile)
    assert_enclosure(result,.016490672564720373,1e-6)
    assert result['prescribed_status']=='within_tolerance'
    assert result['prescribed_center_m'][0]==pytest.approx(.0245)
    # The input is axial x/b_axial, not the unrelated chord projection.
    assert result['prescribed_center_m'][0]/reference_spec().axial_width_m==pytest.approx(.245)


def test_course_two_parabola_has_a_larger_remote_maximum_without_axial_violation():
    spec=course_spec()
    profile=build_double_parabolic_profile(spec)
    assert all(check.passed for check in validate_profile(profile,spec.pitch_m)['checks'])
    result=validate_prescribed_thickness(profile)
    assert result['converged']
    # Independent nearest-boundary optimization gives ~2.86024 mm; bounded
    # final analysis must resolve the 16.6 micrometre excess, not just a flag.
    assert 0.002859 < result['diameter_lower_m'] < 0.002861
    assert result['diameter_upper_m'] < .002862
    assert result['diameter_error_m']<=1e-6
    assert result['diameter_lower_m']-result['prescribed_diameter_m']>15e-6
    assert result['prescribed_center_diameter_m']==pytest.approx(.0028435691333494537,abs=1e-12)
    assert abs(result['maximum_center_m'][0]-result['prescribed_center_m'][0])>.002
    assert result['prescribed_status']=='disproved'

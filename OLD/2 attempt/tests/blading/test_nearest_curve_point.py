"""Exit-edge contacts must resolve sub-grid roots and clockwise arc wraps."""
import math

import numpy as np
import pytest

from vinogradov.curves import BezierCurve, CircularArc, CompositeCurve, nearest_parameter_to_point


def test_nearest_composite_parabola_uses_actual_stationary_point():
    length, bend, contact, clearance = .04, .01, .437, .003
    parabola = BezierCurve(np.array([
        [0., bend*contact**2],
        [length/2, bend*contact**2-bend*contact],
        [length, bend*(1-contact)**2],
    ]))
    remote = BezierCurve(np.array([[0., 1.], [.02, 1.], [.04, 1.]]))
    curve = CompositeCurve((remote, parabola))
    point = np.array([length*contact, -clearance])
    parameter = nearest_parameter_to_point(curve, point)
    assert parameter == pytest.approx((1+contact)/2, abs=1e-12)
    assert np.linalg.norm(curve.evaluate(parameter)-point) == pytest.approx(clearance, abs=1e-12)


def test_nearest_clockwise_arc_handles_angle_crossing_full_turn():
    centre, radius = np.array([.2, .3]), .04
    arc = CircularArc(centre, radius, math.radians(370), math.radians(190))
    direction = np.array([math.cos(math.radians(280)), math.sin(math.radians(280))])
    parameter = nearest_parameter_to_point(arc, centre+2*radius*direction)
    assert parameter == pytest.approx(.5, abs=1e-12)
    np.testing.assert_allclose(arc.evaluate(parameter), centre+radius*direction, atol=1e-12)


def test_finite_arc_does_not_choose_nonexistent_part_of_full_circle():
    arc = CircularArc(np.zeros(2), 1., 0., math.pi/2)
    parameter = nearest_parameter_to_point(arc, np.array([-2., -1.]))
    assert parameter == pytest.approx(1.)
    assert np.linalg.norm(arc.evaluate(parameter)-[-2., -1.]) == pytest.approx(math.sqrt(8.))

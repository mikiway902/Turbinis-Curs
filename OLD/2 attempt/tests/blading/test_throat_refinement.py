"""Sub-grid contact must not manufacture extra choked mass capacity."""
from types import SimpleNamespace

import numpy as np
import pytest

from vinogradov.curves import BezierCurve, CircularArc
from vinogradov.cascade_geometry import sampled_throat, refined_throat


def test_continuous_gap_does_not_inherit_point_cloud_capacity_overestimate():
    length, pitch, gap, curvature, contact = .04, .02, .0037, .01, .437
    lower = BezierCurve(np.array([
        [0., -curvature*contact**2],
        [length/2, -curvature*contact**2+curvature*contact],
        [length, -curvature*(1-contact)**2],
    ]))
    upper = BezierCurve(np.array([
        [0., gap-pitch+curvature*contact**2],
        [length/2, gap-pitch+curvature*contact**2-curvature*contact],
        [length, gap-pitch+curvature*(1-contact)**2],
    ]))
    # Remote endcap cannot limit this two-wall test channel.
    profile = SimpleNamespace(suction=lower, pressure=upper,
        trailing_edge=CircularArc(np.array([length, .1]), .001, 0., np.pi))
    sampled, _, _ = sampled_throat(profile, pitch, samples=41)
    actual, lower_contact, upper_contact = refined_throat(profile, pitch)
    assert sampled > gap+1e-6
    assert actual == pytest.approx(gap, abs=1e-12)
    assert lower_contact[0] == pytest.approx(length*contact, abs=1e-8)
    assert upper_contact[0] == pytest.approx(length*contact, abs=1e-8)

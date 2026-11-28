"""Analytic curves for Vinogradov's constructions (2010, pp. 18–19, 29–33).

Quadratic Bezier is the exact envelope/half-median parabola, not a spline fit.
Coordinates are SI axial x / circumferential y. Parameters are dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import math
import numpy as np
from scipy.special import comb


class Curve(Protocol):
    def evaluate(self, t, derivative: int = 0) -> np.ndarray: ...


@dataclass(frozen=True)
class BezierCurve:
    control_points: np.ndarray

    def __post_init__(self):
        points = np.array(self.control_points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('control points must be a finite (n>=2,2) array')
        points.setflags(write=False)
        object.__setattr__(self, 'control_points', points)

    def evaluate(self, t, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('only derivatives 0,1,2 are supported')
        t = np.asarray(t, dtype=float)
        points = self.control_points
        factor = 1
        degree = len(points) - 1
        for k in range(derivative):
            factor *= degree - k
            points = np.diff(points, axis=0)
        if not len(points):
            return np.zeros(t.shape + (2,))
        degree = len(points) - 1
        indices = np.arange(degree + 1)
        basis = comb(degree, indices) * t[..., None]**indices * (1 - t[..., None])**(degree-indices)
        return factor * (basis @ points)

    def sample(self, count=401):
        return self.evaluate(np.linspace(0., 1., count))


@dataclass(frozen=True)
class CircularArc:
    center: np.ndarray
    radius: float
    start_angle: float
    end_angle: float

    def __post_init__(self):
        c = np.array(self.center, dtype=float, copy=True)
        if c.shape != (2,) or not np.isfinite(c).all() or not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError('circle needs finite center and positive radius')
        if not np.isfinite([self.start_angle, self.end_angle]).all() or self.start_angle == self.end_angle:
            raise ValueError('arc needs a finite nonzero sweep')
        c.setflags(write=False)
        object.__setattr__(self, 'center', c)

    def evaluate(self, t, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError('only derivatives 0,1,2 are supported')
        sweep = self.end_angle-self.start_angle
        theta = self.start_angle + np.asarray(t, dtype=float)*sweep + derivative*np.pi/2
        value = self.radius*sweep**derivative*np.stack((np.cos(theta), np.sin(theta)), axis=-1)
        return value+self.center if derivative == 0 else value

    def sample(self, count=401):
        return self.evaluate(np.linspace(0., 1., count))


@dataclass(frozen=True)
class CompositeCurve:
    segments: tuple[Curve, ...]

    def __post_init__(self):
        object.__setattr__(self, 'segments', tuple(self.segments))
        if not self.segments:
            raise ValueError('empty composite curve')

    def evaluate(self, t, derivative=0):
        t = np.asarray(t, dtype=float)
        flat = t.reshape(-1)
        n = len(self.segments)
        indices = np.minimum((flat*n).astype(int), n-1)
        if (flat < 0).any() or (flat > 1).any():
            raise ValueError('curve parameter outside [0,1]')
        result = np.empty((len(flat), 2))
        for i, segment in enumerate(self.segments):
            mask = indices == i
            if mask.any():
                result[mask] = segment.evaluate(flat[mask]*n-i, derivative)*n**derivative
        return result.reshape(t.shape+(2,))

    def sample(self, count=401):
        return self.evaluate(np.linspace(0., 1., count))


def tangent_intersection(p0, angle0, p1, angle1):
    """Intersection of endpoint tangent lines, angles from +x in radians."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d0 = np.array([np.cos(angle0), np.sin(angle0)])
    d1 = np.array([np.cos(angle1), np.sin(angle1)])
    matrix = np.column_stack((d0, -d1))
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('parallel tangent lines do not define a finite parabola')
    return p0 + np.linalg.solve(matrix, p1-p0)[0]*d0


def arc_between(center, start, end, clockwise=False):
    center = np.asarray(center, float)
    a, b = np.asarray(start, float)-center, np.asarray(end, float)-center
    radius = float(np.linalg.norm(a))
    if not np.isclose(np.linalg.norm(b), radius, rtol=1e-8, atol=1e-12):
        raise ValueError('arc endpoints have different radii')
    theta0, theta1 = math.atan2(a[1], a[0]), math.atan2(b[1], b[0])
    sweep = (theta1-theta0) % (2*np.pi)
    if clockwise:
        sweep -= 2*np.pi
    return CircularArc(center, radius, theta0, theta0+sweep)


def curvature(curve: Curve, t):
    """Signed analytic curvature, 1/m; undefined stationary points are NaN."""
    d1, d2 = curve.evaluate(t, 1), curve.evaluate(t, 2)
    norm = np.linalg.norm(d1, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (d1[...,0]*d2[...,1]-d1[...,1]*d2[...,0])/norm**3


def nearest_parameter_to_point(curve,point):
    """Global finite-curve nearest point: polynomial stationary roots or arcs.

    For a Bezier curve solve (Q(t)-P)·Q'(t)=0 and include both endpoints.
    Composite parameters retain the same segment scaling as evaluate().
    No point-cloud minimum or cosmetic spline is introduced.
    """
    point=np.asarray(point,float)
    if point.shape!=(2,) or not np.isfinite(point).all():
        raise ValueError('finite two-dimensional point required')
    candidates=[0.,1.]
    if isinstance(curve,CompositeCurve):
        count=len(curve.segments)
        candidates.extend((j+nearest_parameter_to_point(segment,point))/count
                          for j,segment in enumerate(curve.segments))
    elif isinstance(curve,BezierCurve):
        degree=len(curve.control_points)-1
        coefficients=np.zeros((degree+1,2))
        for j,p in enumerate(curve.control_points):
            for k in range(degree-j+1):
                coefficients[j+k]+=comb(degree,j)*comb(degree-j,k)*(-1)**k*p
        from numpy.polynomial import Polynomial
        coordinates=[Polynomial(coefficients[:,j]) for j in range(2)]
        stationary=sum((q-point[j])*q.deriv() for j,q in enumerate(coordinates))
        candidates.extend(float(z.real) for z in stationary.roots()
                          if abs(z.imag)<1e-10 and 0.<z.real<1.)
    elif isinstance(curve,CircularArc):
        delta=point-curve.center
        angle=math.atan2(delta[1],delta[0]);lo,hi=sorted((curve.start_angle,curve.end_angle))
        for turn in range(math.ceil((lo-angle)/(2*math.pi)),math.floor((hi-angle)/(2*math.pi))+1):
            candidates.append(float(np.clip((angle+turn*2*math.pi-curve.start_angle)/(curve.end_angle-curve.start_angle),0.,1.)))
    else:
        raise TypeError('Analytic nearest point requires Bezier, circular or composite curve')
    return min(candidates,key=lambda t:float(np.sum((curve.evaluate(t)-point)**2)))

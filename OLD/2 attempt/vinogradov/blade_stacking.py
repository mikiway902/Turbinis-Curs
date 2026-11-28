"""Vinogradov p.13 fig.1.4 p.14: SA leading edge / RK mass-centroid stack.

Axial x and unwrapped circumferential y in metres; radial z is actual radius.
Adaptive refinement is a declared CAD approximation, not a gas-model closure.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .section_properties import section_properties


@dataclass(frozen=True)
class StackedBlade:
    row: str
    radii_m: np.ndarray
    contours_m: tuple[np.ndarray,...]
    offsets_m: np.ndarray
    centroids_m: np.ndarray
    leading_points_m: np.ndarray
    areas_m2: np.ndarray
    interpolation_error_m: float
    mesh_converged: bool

    def ring(self,count):
        if not isinstance(count,int) or count<3:
            raise ValueError('count must be integer >=3')
        xyz=[]
        for k in range(count):
            blade=[]
            for r,p in zip(self.radii_m,self.contours_m):
                theta=p[:,1]/r+2*np.pi*k/count
                blade.append(np.column_stack((p[:,0],r*np.cos(theta),r*np.sin(theta))))
            xyz.append(blade)
        return xyz


def leading_point(profile):
    arc=profile.leading_edge
    candidates=[arc.evaluate(0),arc.evaluate(1)]
    for angle in (math.pi-2*math.pi,math.pi,math.pi+2*math.pi,math.pi+4*math.pi):
        t=(angle-arc.start_angle)/(arc.end_angle-arc.start_angle)
        if 0<=t<=1:candidates.append(arc.evaluate(t))
    return min(candidates,key=lambda point:point[0])


def stack_sections(profiles,radii_m,row,*,count_per_side=151,axial_origin_m=0.,
                   interpolation_error_m=0.,mesh_converged=True):
    if row not in ('SA','RK'): raise ValueError('row must be SA or RK')
    r=np.asarray(radii_m,float)
    if len(profiles)!=len(r) or len(r)<3 or np.any(np.diff(r)<=0):
        raise ValueError('need >=3 profiles on strictly increasing radii')
    properties=[section_properties(p) for p in profiles]
    centers=np.array([p.centroid_m for p in properties])
    # p13: nose points, not circle centers, coincide for SA.
    leading=np.array([leading_point(p) for p in profiles])
    anchors=leading if row=='SA' else centers
    offsets=-anchors+np.array([axial_origin_m,0.])
    mirror=np.array([1.,-1.]) if row=='SA' else np.ones(2)
    contours=tuple((p.contour(count_per_side)+d)*mirror for p,d in zip(profiles,offsets))
    return StackedBlade(row,r,contours,offsets*mirror,(centers+offsets)*mirror,(leading+offsets)*mirror,
                        np.array([p.area_m2 for p in properties]),interpolation_error_m,mesh_converged)


def centroid_scatter(blade,triangle_side_m=.002):
    """p13's 2–3mm is scale-dependent guidance, not a hard turbine limit.

    Report exact maximum pair separation and bounding equilateral triangle
    sufficient side 2*sqrt(3)*enclosing-circle radius about mean. This is a
    conservative fit, not falsely a minimum enclosing triangle.
    """
    points=blade.centroids_m
    diameter=float(np.max(np.linalg.norm(points[:,None]-points[None,:],axis=2)))
    radius=float(np.max(np.linalg.norm(points-points.mean(axis=0),axis=1)))
    return dict(max_pair_distance_m=diameter,sufficient_triangle_side_m=2*math.sqrt(3)*radius,
                guidance_triangle_side_m=triangle_side_m,
                guidance_relative_to_height=triangle_side_m/np.ptp(blade.radii_m),
                classification='empirical scale-dependent guidance, p.13')


def adaptive_sections(builder,root_radius_m,tip_radius_m,*,tolerance_m=2e-5,
                      max_sections=33,count_per_side=81,row='RK'):
    """Build real midpoint profiles and compare with linear loft interpolation.

    Always retain root, middle, tip; stop only with measured midpoint error or
    report mesh_converged=False at the cap. Invalid midpoint is not skipped.
    """
    if not 0<root_radius_m<tip_radius_m or tolerance_m<=0 or max_sections<3:
        raise ValueError('invalid radial refinement controls')
    radii=[root_radius_m,(root_radius_m+tip_radius_m)/2,tip_radius_m]
    cache={r:builder(r) for r in radii}
    def aligned(profile):
        p=profile.contour(count_per_side)
        anchor=leading_point(profile) if row=='SA' else section_properties(p).centroid_m
        return p-anchor
    while True:
        errors=[];add=[]
        for left,right in zip(radii[:-1],radii[1:]):
            mid=(left+right)/2
            if mid not in cache:cache[mid]=builder(mid)
            error=float(np.max(np.linalg.norm(aligned(cache[mid])-(aligned(cache[left])+aligned(cache[right]))/2,axis=1)))
            errors.append(error)
            if error>tolerance_m:add.append(mid)
        largest=max(errors)
        if not add:return radii,[cache[r] for r in radii],largest,True
        if len(radii)+len(add)>max_sections:
            return radii,[cache[r] for r in radii],largest,False
        radii=sorted(radii+add)

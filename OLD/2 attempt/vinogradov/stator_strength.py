"""Stationary, actual-contour SA gas bending and bounded native IN-738C LM.

This is a normal-stress screening calculation, not full blade qualification.
See docs/vinogradov/STATOR_STRENGTH.md for frames, sources and exclusions.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .blade_stacking import leading_point
from .blade_strength import gas_loads, section_stress
from .course_adapter import CourseRadialSampler
from .models import RadialSectionInput, SectionProfile
from .section_properties import section_properties


NORMAL_LIMIT_MPA = 300.


def _material_points(requirements):
    """Validate inputs only; native notebook functions own all LM mathematics."""
    required = ('material', 'life_hours', 'C_LM', 'curve_points', 'required_margin')
    missing = [key for key in required if requirements.get(key) is None]
    if missing:
        raise ValueError('Missing material requirements: ' + ', '.join(missing))
    material = str(requirements['material']).upper().replace('-', '').replace(' ', '')
    if 'IN738C' not in material or 'IN738LC' in material:
        raise ValueError('Original IN-738C material required; no LC substitution')
    life = float(requirements['life_hours'])
    constant = float(requirements['C_LM'])
    margin = float(requirements['required_margin'])
    if not math.isfinite(life) or life <= 0 or not math.isfinite(constant) or constant + math.log10(life) <= 0:
        raise ValueError('Invalid source life or LM constant')
    if margin != 1.2:
        raise ValueError('Source required LM margin must be 1.2')
    points = np.asarray(requirements['curve_points'], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or
            not np.isfinite(points).all() or np.any(points <= 0) or
            np.any(np.diff(points[:, 0]) <= 0) or np.any(np.diff(points[:, 1]) >= 0)):
        raise ValueError('Invalid bounded source LM curve')
    return points


def assess_stator(sampler, index, sections, profiles, count):
    """Assess one actual five-section SA, index 0/1, with a fixed blade count.

    Machine RPM is metadata only: SA mechanical rotation and radial normal
    force are zero. Gas momentum uses signed ABSOLUTE velocities. Reflected
    contours follow blade_stacking's physical SA frame. Loads act through
    section centroids in a root-clamped/free-tip radial beam approximation.
    """
    if not isinstance(sampler, CourseRadialSampler):
        raise TypeError('sampler must be an actual CourseRadialSampler')
    if index not in (0, 1):
        raise ValueError('index must be 0 or 1')
    if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count < 3:
        raise ValueError('count must be an integer >=3')
    sections, profiles = tuple(sections), tuple(profiles)
    if len(sections) != 5 or len(profiles) != 5:
        raise ValueError('Need five matching source sections and actual profiles')
    if any(not isinstance(s, RadialSectionInput) or s.row != 'SA' or s.stage != index+1
           for s in sections):
        raise ValueError('All sections must belong to the selected source SA')
    if any(not isinstance(p, SectionProfile) for p in profiles):
        raise TypeError('profiles must be actual SectionProfile objects')
    radii = np.array([s.radius_m for s in sections])
    if not np.isfinite(radii).all() or np.any(np.diff(radii) <= 0):
        raise ValueError('Five section radii must be finite and strictly increasing')

    gas = gas_loads(sections, int(count))
    # Use one contour discretization for both inertia integration and stress.
    # Translation aligns the leading points, exactly as physical SA stacking;
    # reflection changes Ixy and must precede the unsymmetric bending solve.
    contours = tuple((p.contour(1001)-leading_point(p))*[1., -1.] for p in profiles)
    properties = tuple(section_properties(c) for c in contours)
    distributions = tuple(section_stress(p, 0., moment, properties=prop, contour=c)
                          for p, moment, prop, c in zip(profiles, gas['moments_N_m'], properties, contours))
    if any(not np.isfinite(stress).all() for stress in distributions):
        raise ValueError('Nonfinite actual-contour normal stress')

    # FlowStation totals are computed by the original variable-cp h/inv_h,
    # not constant-cp recovery or the rotor-relative temperature bound.
    sides = {}
    for side in ('inlet', 'outlet'):
        total = np.array([getattr(s, side).total_temperature_K for s in sections])
        sides[side] = dict(r_m=np.array([
            s.radius_m if side == 'outlet' or s.inlet_radius_m is None else s.inlet_radius_m
            for s in sections]), absolute_total_K=total,
            min_K=float(total.min()), max_K=float(total.max()))
    bound = dict(upper_K=max(side['max_K'] for side in sides.values()), sides=sides,
        origin='Maximum source absolute stagnation temperature at both SA planes, five mass-fraction sections; recovery=1, uncooled upper estimate, not solved metal temperature',
        sampling='source five mass-fraction sections; source SA total enthalpy is spanwise uniform')
    stresses = [dict(method='stationary_5_gas_bending', section=j, section_name=f'Section {j}',
        r_m=float(r), area_m2=prop.area_m2, normal_force_N=0., sigma_tension_MPa=0.,
        sigma_bending_min_MPa=float(stress.min())/1e6,
        sigma_bending_max_MPa=float(stress.max())/1e6,
        sigma_normal_min_MPa=float(stress.min())/1e6,
        sigma_normal_max_MPa=max(0., float(stress.max()))/1e6,
        sigma_normal_abs_max_MPa=float(np.max(np.abs(stress)))/1e6,
        gas_force_x_N_m=float(force[0]), gas_force_y_N_m=float(force[1]),
        gas_moment_x_N_m=float(moment[0]), gas_moment_y_N_m=float(moment[1]))
        for j, (r, prop, stress, force, moment) in enumerate(zip(
            radii, properties, distributions, gas['force_per_span_N_m'], gas['moments_N_m']))]
    requirements = sampler.requirements
    try:
        points = _material_points(requirements)
        # The native helper is scalar and rotation-independent. None is NOT a
        # rotor identifier; discard that helper's historical column below.
        rows = sampler.ns['_candidate_lm_rows'](None, stresses, bound, requirements, points)
    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError, OverflowError) as exc:
        rows = [dict(stress, T_bound_K=bound['upper_K'],
            life_hours=requirements.get('life_hours', math.nan), P_K=math.nan,
            allowable_MPa=math.nan, margin=math.nan,
            required_margin=requirements.get('required_margin', math.nan),
            rupture_hours=math.nan, design_life_hours=math.nan, passed=False,
            material=requirements.get('material'), reason=f'Material allowance unknown: {exc}')
            for stress in stresses]
    for row in rows:
        row.pop('rotor', None)
        row.update(row='SA', stage=index+1, machine_speed_rpm=sampler.speed_rpm,
                   mechanical_speed_rpm=0., normal_limit_MPa=NORMAL_LIMIT_MPA,
                   curve_source=requirements.get('curve_source'),
                   material_source_url=requirements.get('source_url'),
                   scope='Stationary SA gas-bending normal stress and bounded LM only; not full qualification')
        row['lm_passed'] = bool(row['passed'])
        row['normal_limit_passed'] = row['sigma_normal_abs_max_MPa'] <= NORMAL_LIMIT_MPA
        row['passed'] = bool(row['lm_passed'] and row['normal_limit_passed'])
        if not row['normal_limit_passed']:
            row['reason'] = '; '.join(filter(None, (row['reason'], 'Absolute normal stress exceeds 300 MPa')))
    table = pd.DataFrame(rows)
    margins = table.margin.to_numpy(dtype=float)
    return dict(table=table, min_margin=float(np.min(margins)) if not np.isnan(margins).any() else math.nan,
        passed=bool(table.passed.all()), temperature_bound=bound, gas=gas,
        force_per_span_N_m=gas['force_per_span_N_m'], moments_N_m=gas['moments_N_m'],
        normal_force_N=np.zeros(5), contours_m=contours, contour_stress_Pa=distributions,
        properties=properties, row='SA', stage=index+1, count=int(count),
        machine_speed_rpm=sampler.speed_rpm, mechanical_speed_rpm=0.,
        reasons=[row['reason'] for row in rows if not row['passed']],
        limitations=['Centroid gas-resultant, radial root-clamped/free-tip beam approximation',
            'No eccentric-resultant torsion or transverse shear assessment',
            'No thermal-gradient stress or solved metal temperature',
            'No attachment/fillet stress concentration or certified material lower bound',
            'HCF NOT_CHECKED: actual excitation, damping and fatigue data absent'])

# Stationary stator normal-stress screening

`vinogradov.stator_strength.assess_stator(sampler, index, sections, profiles, count)`
assesses **SA1 or SA2** (`index=0/1`) of the actual two-stage course. `sampler`
is a `CourseRadialSampler`; `sections` are its five increasing-radius,
equal-mass-fraction `RadialSectionInput` objects (`sampler.row(index, 'SA', 5)`),
and `profiles` are the corresponding five actual `SectionProfile` objects.
`count` is the fixed integer number of blades in this row. Geometry is never
replaced by an equivalent rectangle, nor is an SA sent to a rotor geometry API.

## Mechanics and frame

- `blade_strength.gas_loads` supplies annular-strip pressure and momentum forces
  per blade, using the **signed absolute** velocities and mapped inlet radius.
  Its PCHIP tail integrals supply gas bending moments at all five sections.
  These source equations are reused, not copied into another implementation.
- Coordinates are axial `x`, physical circumferential `y`, radial `r`. The local
  textbook SA contour is translated to its leading point and reflected by
  `(x,y) -> (x,-y)`, exactly as `blade_stacking.stack_sections(..., row='SA')`.
  The gas forces are already in the physical frame and are **not** reflected.
- One sampled actual contour (1001 points per side, plus rounded edge samples)
  supplies both `section_properties` polygon inertia and `section_stress`
  unsymmetric bending. Reflection is applied before computing the product of
  inertia. Polygon integrals are exact for that discretization; the curved
  contour itself remains a numerical approximation.
- Machine resource RPM is retained only as metadata. **Mechanical SA RPM,
  centrifugal force, and radial direct normal stress are zero.** No fictitious
  rotating stator or balancing centrifugal moment is introduced.
- The beam is root-clamped and free at the tip. Gas resultants act through the
  section centroids. This deliberately neglects eccentric-resultant torsion,
  transverse shear, and additional moments from centroid offsets in a fully
  swept 3D beam. It is not an attachment or shrouded-vane load-path solution.

## Thermal and material gate

Both SA planes' **absolute stagnation temperatures** are taken from the actual
source `FlowStation` objects. The course computes these with its variable-property
`h`/`inv_h`, not fixed `cp` and not rotor-relative velocity. The largest of the
ten values is used uniformly as a recovery=1, uncooled thermal upper estimate.
The source SA inlet total enthalpy and SA outlet total enthalpy are spanwise
uniform by construction; the reported bound explicitly identifies its five
mass-fraction samples. It is not a solved metal temperature or a general bound
for an unrelated spanwise-nonuniform inlet model.

The native notebook `_candidate_lm_rows` is scalar and rotation-independent;
it is reused with no rotor identifier. Its historical `rotor` column is removed
and replaced by truthful `row='SA'`, `stage=index+1` metadata. The original
`sampler.requirements` supplies **IN-738C (not LC)**, life, bounded Table IX curve,
`C_LM` and the required **1.2** margin. Native log-stress interpolation and its
inverse determine allowable, rupture life, design life, and tensile-normal-stress
LM margin. There is no extrapolation, including inverse life: an unavailable
inverse remains NaN. Missing/invalid material requirements or out-of-bounds LM
parameter produce unknown margin and **failure, even at the unloaded tip**.
A known allowable and zero tip stress correctly produce infinite stress margin,
not an invented rupture life.

A separate gate checks **maximum absolute contour normal stress <=300 MPa**,
including compression. `passed` requires both gates at every section. Typical
rupture data and the project reserve do not constitute a certified material
lower bound or a full qualification claim. With a 1250 K absolute bound and
10000 h life the first SA may lie outside the published LM domain; an unknown
result is intentional, never replaced by a cooler rotor-relative bound.

## Returned dictionary

- `table`: pandas DataFrame, five rows. Core columns: `row`, `stage`, `section`,
  `r_m`, `sigma_normal_min_MPa`, `sigma_normal_max_MPa`, `T_bound_K`, `margin`,
  `required_margin`, `passed`, `reason`. Additional fields include actual area,
  `sigma_normal_abs_max_MPa`, zero tension/normal force, two gas force components
  (N/m), two moment components (N m), `allowable_MPa`, `P_K`, `life_hours`,
  `rupture_hours`, `design_life_hours`, material/source, `lm_passed`,
  `normal_limit_passed`, machine/mechanical RPM, method and scope.
- `min_margin`: minimum LM margin, NaN if any margin is unknown. This is not
  silently reduced to a minimum over known rows and is not the 300 MPa gate.
- `passed`, `reasons`, `temperature_bound` (`upper_K`, both sampled `sides`,
  origin and sampling basis).
- `gas`, `force_per_span_N_m` (5x2), `moments_N_m` (5x2), `normal_force_N`
  (five zeros), `contours_m`, `contour_stress_Pa` (matching point ordering), and
  `properties`: direct plot-ready numerical arrays and actual section inertias.
- `row`, `stage`, `count`, `machine_speed_rpm`, `mechanical_speed_rpm`,
  `limitations`. No global cache: callers may cache gas-only fields on the
  unchanged source sampler; candidate geometry is always recomputed.

Bad API/geometry inputs raise `TypeError`/`ValueError`; unavailable material
bounds retain the real gas loads/stresses and return failed rows.

## Sources and exclusions

1. Supplied **`ДЗ 1 - УКР/ДЗ 1/УКРФ ДЗ1 Сладков.docx`, section 2**:
   strip gas loads, bending moments, and unsymmetric beam normal stress. The
   implementation is shared with `blade_strength`, not the numerical worked
   example's internally inconsistent arithmetic. See `SOURCE_MAP.md`, M02.
2. **Vinogradov (2010), pp.47–52, Table 3.3**: actual contour area, centroid and
   inertia; **pp.13–14, Fig.1.4**: SA leading-edge stacking. Physical SA reflection
   follows this repository's `blade_stacking` convention, not an extra load sign.
3. **INCO, Alloy IN-738, bulletin 497**, p.3 (vacuum casting/heat treatment),
   **p.8, Table IX, original IN-738C columns, 100/1000 h**:
   [publisher](https://nickelinstitute.org/en/resources/technical-guides/in-738-alloy-preliminary-data-497),
   [available copy](https://www.scribd.com/document/664099561/ni-inco-497-alloy738).
   Native course source cell 37 records the data, unit conversions and bounded
   LM assessment. `C=20` is the source's explicit engineering closure, not a
   certified fit; 1.2 and 300 MPa are separate project screening requirements.
4. Actual `TVD_1CBK_2st_{300,2000,10000}h.ipynb` gas functions and
   `CourseRadialSampler`: absolute gas states, original IN-738C requirements and
   fixed resource shaft speeds. No archived OLD physics is imported.

**Unassessed:** torsion, transverse shear, thermal gradients, creep strain,
root/fillet/attachment concentrations, coating effects, multiaxial creep,
certified supply allowables, and HCF. HCF remains **NOT_CHECKED** without measured
or justified excitation, damping, temperature-dependent fatigue properties and
a validated dynamic model. A green result here is normal-stress screening only.

`tests/blading/test_stator_strength.py` uses all three actual source resource
notebooks, independent analytical cubic integration of gas moments, independent
triangle quadrature for contour inertia, pointwise unsymmetric beam stress,
physical-mirror discrimination, the zero-centrifugal-force condition, actual
bounded material/life response, and unknown-data/absolute-normal-cap cases.

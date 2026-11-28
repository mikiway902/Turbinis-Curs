# Vinogradov 2010: section, channel, triangles and elementary stage

Status: **EXPERIMENTAL**. No equation promotion, Mikhaltsev change or baseline
change. Construction multiplier `m` is **EMPIRICAL**. Polygon/sampling/spline
operations are numerical approximations. Ideal-gas choking is an explicitly
separate physical assumption, not an equation attributed to this book.

## Images inspected

All original PDF page images 45–60 were inspected, not merely OCR:

- 45: table 3.2 normal-angle difference samples and its zero-crossing graph.
- 46: table 3.2 common-normal solution and maximum-thickness circle centre.
- 47: table 3.2 inscribed circle illustration, its stated test inputs, and
  introduction to section geometry.
- 48: table 3.3 C9017 profile, lengths, area and first moments.
- 49: table 3.3 raw, centroidal and principal second moments.
- 50: table 3.3 principal axes, coordinate transformation and local moduli.
- 51: table 3.3 moduli against axial/principal coordinates and rotated profile.
- 52: table 3.3 cumulative surface lengths/moduli and channel introduction.
- 53: channel procedure, table 3.4 inputs and contact-normal circle equations.
- 54: trial-circle illustration, contraction ratio, and cascade circle drawing.
- 55: table 3.4 contact choices and all eight printed circle rows.
- 56: table 3.4 cubic interpolation, centreline length and radius plots.
- 57: table 3.5 five velocity inputs and seven-point combined plot matrix.
- 58: table 3.5 separate inlet/outlet matrices and inlet triangle drawing.
- 59: outlet triangle, explanation of elementary-stage assembly, table 3.6 start.
- 60: table 3.6 stator/rotor renaming, reflection, translations and final drawing.

Page 18 was additionally inspected as an image for the throat relation.

## Public interfaces and SI conventions

### `section_properties.py`

`section_properties(profile_or_contour, *, count_per_side=1001)` accepts a
`SectionProfile` or a simple `(N,2)` polygon (open/closed, either orientation).
The returned `SectionProperties` contains:

- `area_m2`, `centroid_m`, `perimeter_m`;
- `first_moments_m3 = [integral x dA, integral y dA] = [S_V,S_U]`;
- `second_moments_m4 = [integral y² dA, integral x² dA, integral xy dA]`;
- `centroidal_moments_m4` in the same `[Ixx,Iyy,Ixy]` order;
- `principal_min_m4`, `principal_max_m4`, `principal_angle_rad`;
- `principal_axes(half_length_m)` gives both finite axis segments;
- `to_principal(points)` implements table 3.3 items 15–16;
- `distributed_moduli(points)` returns `[Imin/abs(v'), Imax/abs(u')]` in m³.
  These are local functions, not just two extreme-fibre numbers. Neutral-axis
  poles remain infinity, not zero or a clipped finite number.

`contour_distribution`, `suction_distribution`, `pressure_distribution` expose
points, principal coordinates, cumulative arclength and both local modulus
functions, covering items 18–22 without tying them to a plotting library.
`suction_length_m`/`pressure_length_m` are the lengths of the explicit main
curves, excluding separately represented edges. For profiles with real edges,
`rounded_suction_distribution` and `rounded_pressure_distribution` also include
edge portions split at the leading minimum-x/trailing maximum-x points. These
are the graph-surface counterparts of the source's piecewise S_p and W.
Their terminal `arc_length_m` is the full rounded-surface length. Sampled extrema
have second-order parameter error. Source-specific unequal integration cutoffs
are not silently made universal defaults.

Polygon area moments are orientation independent and translated before
integration to reduce cancellation. Central moments use exact polygon boundary
formulas. The minimum-inertia principal axis is obtained from the symmetric
bending-inertia matrix; its direction is modulo pi in [-pi/2,pi/2). This avoids
`atan` branch failures and the source's slope singularity for vertical axes.
For equal principal moments an arbitrary but explicit zero direction is used.
Self-intersection certification remains the parent geometry validator's job.

### `channel_convergence.py`

`analyze_channel(profile, pitch_m, *, samples=81, wall_samples=801,
tangency_tolerance_m=1e-8, clearance_tolerance_m=None,
parameter_range=(.001,.999), include_trailing_edge=True)` returns
`ChannelAnalysis` with actual circles, centres, radii, diameters, centreline
arclength, diameter gradient, throat, contraction ratio, diffuser intervals,
rejected-candidate reasons and a boundary-minimum flag.

For each lower suction contact p, solve simultaneously

`p + r*n_s = q + r*n_p`,

where q is on the adjacent upper pressure wall (translated by positive pitch)
and both normals point from their contacts into the channel. The radius is
solved, not set to half an axial/vertical distance. Residuals include centre
coincidence, contact-normal orthogonality and clearance to the two complete
blade polygons. Multiple seeds permit different normal branches to be found;
non-interior centres, blade-intersecting circles and endpoint contacts are
rejected and labelled. At least three valid circles are required.

The upper wall normally continues along its **real** reversed trailing-edge
arc: the cascade throat can touch this arc rather than the pressure main
curve. `pressure_contact_region` distinguishes these contacts. With this option,
`pressure_parameter` parametrizes the augmented wall, assigning the first half
to pressure and the second half to reversed trailing edge. Set
`include_trailing_edge=False` for pressure-only investigations. The reported
parameter is not an axial coordinate.

Artificial inlet/outlet connectors only define the interior-centre test and
`endcap_limited` flag. They never generate circle contacts or limit radii.
Endcap-limited actual two-wall circles are retained and identified. Clearance
is a polygon approximation with exposed resolution; the default allowance is
`max(tangency_tolerance_m, 8*scale/(wall_samples-1)**2)`, not a physical allowance.
The throat receives local bounded refinement when the sampled minimum has
neighbours. A minimum at the sampled boundary is explicitly not certified as
the global cascade throat. `convergence_ratio` is throat diameter divided by
first **computed** diameter; the first-circle location must therefore accompany
comparisons. Positive diameter slope/intervals indicate diffuser sections.

`interpolate_circle_table(table_m, *, count=201,
boundary_condition='not-a-knot')` implements table 3.4 items 8–14 for rows
`[centre_x, centre_y, radius]`, using cubic splines and integrated centreline
arclength. Its result is interpolation, not a claim that intermediate circles
are still exactly tangent. It deliberately never extrapolates beyond measured
centre coordinates. The book does not specify Mathcad `cspline` boundary
conditions; the choice is exposed rather than asserted identical.

### `velocity_triangles.py`

`velocity_triangles(absolute_inlet_speed_m_s, absolute_inlet_angle_deg,
blade_speed_m_s, relative_outlet_speed_m_s, relative_outlet_angle_deg)` returns
explicit c1,w1,w2,c2,u vectors and source inlet/outlet/combined plotting matrices.
Physical coordinates are `(axial downstream, circumferential blade motion)`:

`c1=(C1*sin(alpha1), C1*cos(alpha1))`,
`w2=(W2*sin(beta2), -W2*cos(beta2))`, `w1=c1-u`, `c2=w2+u`.

Book plotting coordinates are separately `(-circumferential,-axial)`. The seven
combined vertices reproduce table 3.5, including the repeated origin. Euler
work is the explicit identity `u*(c1_circ-c2_circ)`, not a stage efficiency model.
Angles are flow angles; none is automatically a metal angle.

`elementary_stage_layout(stator, rotor, *, stator_pitch_m, rotor_pitch_m,
rotor_axial_offset_m=.110, rotor_reflection_y_m=.010, triangles=None,
triangle_scale_s=.00075, inlet_triangle_origin_m=(.100,-.050),
outlet_triangle_origin_m=(.200,.020), count_per_side=401)` returns two stator
contours, two reflected/translated rotor contours and optional triangle
polylines. Table 3.6 is a graphical assembly, not a thermodynamic stage solver.
Its 110 mm axial and 10 mm reflection offsets and triangle translations are
source illustration defaults; pitches are explicit and independent. Triangle
scale is .75 mm/(m/s), correctly represented with seconds as units. No graphics
library or mutable shared source variable names are required.

### `cascade_throat.py`

`effective_outlet_angle(throat_m,pitch_m)` is geometric `asin(a/t)` in degrees.
`throat_from_effective_angle(pitch_m,effective_angle_deg)` is its inverse.
`source_throat_width(pitch_m,outlet_metal_deg,*,throat_multiplier=1.05)` is the
p.18 empirical construction `a=t*sin(beta2)/m`; its metal-angle argument is
explicitly not a flow-angle identification. The geometric effective angle of
this width is not beta2 when m differs from one.

`perfect_gas_throat_flow(*, stagnation_pressure_pa,
stagnation_temperature_k, back_pressure_pa, gamma, gas_constant_j_kg_k,
area_m2, discharge_coefficient=1)` requires all thermodynamic inputs explicitly.
It implements the constant-gamma 1D isentropic reservoir-to-throat law and
critical ratio `(2/(gamma+1))**(gamma/(gamma-1))`. Total conditions must be in the
relevant reference frame (relative for rotor). It returns choked flag, critical
ratio, Mach, mass flux and mass flow. It does not infer gas properties, total
conditions, span, shock losses, or a book-specific choking correlation.

## Printed reference cases and tolerances

### Table 3.2, pages 45–47 (maximum-thickness ownership is slice A)

Printed x samples in mm: `[10,17,24,31,38,45,52]`; normal-angle differences in
radians: `[-.284,-.204,-.124,-.049,.019,.075,.117]`. Linear interpolation gives
`x_D=36.035 mm`, `W(x_D)=74.523 mm`; suction contact
`(44.876,87.862) mm`; `Dmax=16.003 mm`. The page 47 stated test inputs are
stagger 47°, inlet 101°, outlet 25°, relative radii .031/.0095, Kt=.63,
trailing wedge 7°, m=1.05. These are not the C9017 table 3.3 fixture, and the
adjacent tables must not be silently treated as one unchanged calculation.
The normal-line expression on p.46 appears to apply an additional atan to an
already defined tangent angle; physical common-normal solving should not copy
that dimensionally unnecessary operation. The independently owned slice A
implements maximum thickness and records its own reference ambiguity.

### Table 3.3, pages 48–52

C9017: axial width b=100 mm; stagger 40°, inlet 90°, outlet 17°; Kt=.69;
relative leading/trailing radii .024/.006; trailing wedge 7°; m=1.05. Dimensional
radii are the quoted fractions of **chord** `b/sin(40°)`; pitch is Kt times chord.
The source coordinate origin is y=100 mm. Literal outputs:

| Quantity | Printed value (source mm-based units) |
|---|---:|
| Suction length, integrated x=0..99.5 mm | 178.126 mm |
| Pressure length, integrated x=0..99.45 mm | 155.260 mm |
| F | 2659 mm² |
| S_V, S_U | 136800, 175500 mm³ |
| J_V, J_U, J_UV | 8506000, 13630000, 7488000 mm⁴ |
| Uc, Vc | 51.457, 66.011 mm |
| J_V1, J_U1, J_U1V1 | 1466000, 2040000, -1543000 mm⁴ |
| Jmax, Jmin | 3323000, 184100 mm⁴ |
| phi, gamma_UV | -50.268°, -0.877 rad |

Principal axes, coordinates, modulus distributions and cumulative lengths
implement all items 11–22, not merely the scalar items 1–10. Curved results are
compared against this independently transcribed table, not re-pinned generated
fixtures. Proposed reference tolerances: 1% area/first/raw second moments,
0.5 mm centroid, 2% central/Jmax, 3% Jmin, 0.5° axis orientation, 1 mm branch
length at the source's unequal cutoffs. These include source graphical and
rounded intermediate arithmetic, not solver precision. For example, recomputing
the small principal moment from the rounded printed central moments gives
approximately 183536 mm⁴, about 0.31% below the printed 184100 mm⁴. It is therefore
incorrect to demand machine agreement with every printed rounded quantity.
Analytical rectangle, reversal, rotation, translation and scale-law tests use
much tighter floating-point tolerances independently of these source limits.

### Table 3.4, pages 53–56

Same C9017 inputs. Upper pressure contacts x in mm: `65,70,75,80,85,90,95,98`.
The graphically selected radii at these contacts are
`27.8,25.8,24,22.4,21,19.65,18.3,17.75 mm`. All printed centre rows:

| centre x mm | centre y mm | radius mm |
|---:|---:|---:|
|43.370|129.662|27.800|
|49.310|125.268|25.800|
|55.180|120.138|24.000|
|60.966|114.184|22.400|
|66.654|107.276|21.000|
|72.364|99.282|19.650|
|78.138|89.873|18.300|
|81.390|83.208|17.750|

The circle-table interpolation test checks these literal endpoints and their
measured domain. Actual analytically tangent channel circles are compared with
all eight source y/radius values with a 0.5 mm graphical tolerance; their own
normal residual must remain below 1e-6 dimensionless. Table item 6 prints
`K=a_th/(2*33)=.453`, implying a_th=29.898 mm (±.033 mm from rounding K alone).
This is the **prescribed design width**, not a demonstrated achieved geometric
throat. Applying p.18 directly to the stated C9017 inputs gives
`a_design=29.890118368 mm` and `a_design/66=.452880581`, which rounds to the
printed .453. Page 18 explicitly warns that the tangent-defined parabola may
not pass through the prescribed point p; the source construction does not
enforce that target width on the final suction curve.

Parent's first verification measured an actual tangent-circle throat of
`34.996752113 mm`, while all eight printed circle rows and their normal checks
passed. The achieved-minus-design residual is **+5.106633745 mm (+17.084689%)**.
Using the same 66 mm inlet diameter for the achieved throat gives .530253820,
not .453. The last printed circle diameter is already 35.5 mm, consistent with
the achieved throat being near 35 mm rather than proving a 29.9 mm passage.
The former assertion equating the achieved throat to .453*66 was therefore
incorrect and has been replaced by two separate checks: printed K against the
literal p.18 design equation, and the achieved tangent-circle throat against
an independently sampled minimum wall-to-wall distance (2 micrometre sampling
allowance). No achieved-throat output was pinned as a new reference.

The independently documented table 3.1 radius inconsistency is also relevant
as a warning against treating every printed `a_th` as interchangeable:
`source_19_44.md` records a -0.904160541 mm residual between its printed throat
circle radius and its own Euclidean-distance equation. This is a different
fixture/discrepancy, not a correction applied to the C9017 channel.

The denominator's 33 mm inlet radius is **not** the first tabulated radius
(27.8 mm): dividing the prescribed width by that first diameter instead gives
about .538, not .453. Computed contraction ratios must identify both their
inlet section and whether their numerator is prescribed or achieved, rather
than tuning circles to .453. Page 56 plots outside the eight-row x interval
and integrates from zero; no tangent-circle evidence is given for that
extrapolated section. Implementation starts arclength at the first measured
circle and labels the restriction honestly.

### Tables 3.5 and 3.6, pages 57–60

Inputs C1=200 m/s, alpha1=17°, u=100 m/s, W2=150 m/s, beta2=35°.
Independent trigonometric values:

- c1=(58.47434094,191.26095119) m/s;
- w1=(58.47434094,91.26095119) m/s;
- w2=(86.03646545,-122.87280664) m/s;
- c2=(86.03646545,-22.87280664) m/s;
- Euler identity gives 21413.375783 J/kg.

The source matrices are checked against their literal signs, closure and the
above independently evaluated values (1e-8 m/s coordinate tolerance). Table
3.6 stator uses the C9017 inputs; rotor lists stagger45°, inlet79°, relative
radii .04/.008, wedge7°, m=1.05. Direct reinspection of p60 shows sequential
assignments: alpha1=17° and Kt=.69 from items1–2 are not reassigned in item3,
so the reproducible sequential-program interpretation retains those values.
Width=100mm and VA=100mm are likewise inherited construction inputs. This
dependency is explicit in `reference_cases.control_report`, rather than
called an independent rotor-input list. The resulting full rotor/stator and
triangle drawing is `controls/elementary_stage_source.png`. A separately
initialized rotor program would require these inputs to be supplied anew.
The generic layout API still accepts explicit profiles; rectangle tests
independently check its reflection and translations.

Separate ideal-gas analytical check: p0=100000 Pa, T0=300 K, gamma=1.4,
R=287 J/(kg K), back pressure=20000 Pa, area=.001 m², Cd=1 gives critical ratio
.5282817877, Mach=1, mass flux about 233.355856 kg/(m² s), mass flow .233355856
kg/s. At back pressure=p0 mass flow is zero. This is not a Vinogradov fixture.

## Verification handoff

No tests, builds, linters, formatters or smoke executions were run by this
slice, per the shared integration contract. In particular the proposed
C9017 source-comparison tolerances have not been asserted as observed pass
results. Parent should run `pytest tests/blading/test_properties_channel_reference.py`
after all modules land, then the project's mandated verification workflow.
Any source discrepancy must be reported numerically rather than re-pinning
these transcribed references or silently widening tolerances.

Parent's initial integrated reference run reported 42 passes and one failure:
the incorrect achieved/design-throat equality described above. This slice did
not rerun any checks after correcting that assertion; parent owns the focused
verification of the replacement checks.

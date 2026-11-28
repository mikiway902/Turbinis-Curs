# Vinogradov 2010: parabolic construction source record

Status: **EXPERIMENTAL** geometry; not aerodynamic or structural acceptance.
Source: Л. В. Виноградов, *Автоматизированное проектирование лопаток турбин*,
2010. Printed and PDF page numbers agree. Images inspected individually:
**19–44 inclusive**, plus p18 for the two-circle description and pp45–47 for
actual table 3.2, which starts beyond the assigned p44 introduction. This report
owns only the slice-A construction; existing Mikhaltsev references are unchanged.

## Image review inventory

- p19: figs 2.2 (parabolic envelope of crossing lines) and 2.3 (two-circle pressure).
- p20: incidence discussion, nozzle analogy, construction-program introduction,
  table 2.1 item 1 and list of inputs.
- p21: table 2.1 inputs, chord line, chord, edge radii, centers, leading circle.
- p22: leading/trailing circle figures; inlet-direction line.
- p23: outlet-direction line, pitch, prescribed throat, translated trailing center.
- p24: prescribed throat contact p and trailing pressure contact h.
- p25: pressure radius/center, pressure-circle equation, leading contact g.
- p26: leading wedge, suction inlet contact l and tangent.
- p27: suction outlet contact f, tangent, control-triangle base.
- p28: tangent intersection B and completed control-triangle figure.
- p29: fig 2.4, half-median construction, claimed absence of analytic expression.
- p30: recursive left-half algorithm CRDL, point clustering and N=10 discussion.
- p31: completion of CRDL and first half of right recursion CRDR.
- p32: completion of CRDR; four explicitly printed test configurations.
- p33: four plotted test parabolas and section-3 introduction.
- p34: output back angle versus prescribed pitch, recommended 5–15 degree back-angle
  range, and table 3.1 inputs/starting chord line.
- p35: chord, edge radii/centers and leading circle.
- p36: trailing circle and inlet-direction figure.
- p37: pitch/translated center, h, circular pressure radius.
- p38: pressure center/contact g and inlet-tangent reflection/wedge.
- p39: l, f, endpoint tangents and base inclination.
- p40: base length, B and call to the parabolic subroutine.
- p41: dyadic output coordinates, control triangle, cubic-spline interpolation.
- p42: complete suction equation, cascade plot, throat-contact normal equation.
- p43: contact P, output back angle, printed throat circle radius and throat width.
- p44: common-normal maximum-thickness derivation and fig 3.1.
- p45: table 3.2 normal-angle residual and seven pressure-side abscissae.
- p46: linearly interpolated contact, diameter, normal, circle center/radius.
- p47: table 3.2 circle parametrization/plot and the distinct test-fixture inputs.

## Interfaces and conventions

`turbine/blading/parabolic_profile.py` exports:

- `parabolic_envelope(start, end, start_angle_rad, end_angle_rad) -> BezierCurve`
- `build_scheme(spec: ProfileSpec) -> dict`
- `build_parabolic_profile(spec: ProfileSpec) -> SectionProfile`
- `build_two_circle_pressure_profile(spec, *, inlet_wedge_deg,
  tangent_distance_ratio=1.) -> SectionProfile`
- `maximum_thickness(profile: SectionProfile) -> dict`

Coordinates are `(x,y)=(U,V)` in **metres**, axial downstream then circumferential.
Pitch translates by `(0,+pitch)`. Both sides run inlet to outlet; leading edge
runs pressure to suction and trailing edge suction to pressure. Edge circles
close the contour without splines, smoothing, fitted edge radii, or inferred
flow/metal-angle identity. Inputs are explicit metal angles. Incidence and
deviation must already have been applied in converting flow to metal angles.

Source examples specify radii and pitch relative to chord, not axial width:
`chord=b/sin(stagger)`, `r_leading=r_vx*chord`,
`r_trailing=r_vux*chord`, `pitch=Kt*chord`. The public spec instead requires
all these dimensional quantities explicitly. The 5–15 degree back-angle range
on p34 is source guidance, not an automatic clamp or a guarantee of validity.
The empirical throat multiplier is retained as an explicit spec input.

`build_scheme` exposes points `A,O1,O2,O2_pitch,O3,g,h,l,f,B,p`, SI chord,
pressure radius and design throat; angles ending `_rad` are radians. `B` is the
endpoint tangent intersection (not the far end of the chord). `p` is the
prescribed point of table 2.1; it is not forced onto the final quadratic.
The parabolic profile metadata contains the actual normal-contact throat
point/parameter/radius/width and back angle from table 3.1 items 43–47.
An endpoint distance minimum is explicitly labelled non-interior.

### Exact envelope proof

Let A and C be the endpoints and B their tangent-line intersection. Define

`Q(t)=(1-t)^2*A + 2*t*(1-t)*B + t^2*C`.

Then `Q'(0)=2(B-A)`, `Q'(1)=2(C-B)`,
`Q(1/2)=(B+(A+C)/2)/2`, and `Q'(1/2)=C-A`.
Thus Q satisfies exactly p29's half-median point and parallel-to-base tangent
conditions. De Casteljau subdivision gives the same identities on either half,
so repeated CRDL/CRDR midpoint construction produces precisely the dyadic
samples of Q, not merely a fitted approximation. Equivalently, the envelope
of the crossing straight lines of fig 2.2 has this quadratic parametrization.
The p29 claim that there is no analytic expression is not a mathematical
obstruction. Using line directions avoids the printed slope method's vertical
singularity, including the third and fourth p32 tests. Parallel endpoint
lines do not define a unique finite tangent intersection and raise a
quantified `ConstructionError` rather than selecting a fake parabola.

The book later interpolates sparse points with a cubic spline (table 3.1,
item40). This implementation uses the exact envelope throughout. No
approximation is silently added to imitate that spline's small errors.

## Printed controls and tolerances

All numbers below are the **printed mm/degrees**, not SI inputs.
`tests/blading/test_parabolic_reference.py` converts mm to m explicitly.
Lengths printed to 0.001 mm use 0.0006 mm absolute tolerance; quantities
printed with fewer digits have their own rounding tolerance.

### Table 2.1 (pp21–28)

Inputs: `VA=100,b=100,stagger=36.33,inlet=90,outlet=14,
r_vx=.0243,r_vux=.006025,Kt=.595,m=1.05,wedge_out=7,back_angle=7`.

| Printed quantity | Value |
|---|---:|
| chord | 168.795 |
| inlet/trailing radii | 4.102 / 1.017 |
| O1 | (4.102,94.422) |
| O2 | (98.983,-34.601) |
| pitch / design throat | 100.433 / 23.14 |
| O2' | (98.983,65.832) |
| prescribed p | (91.719,42.793) |
| h | (98.013,-34.907) |
| pressure radius | 233.025 |
| O3 | (-124.227,-104.979) |
| leading pressure angle theta_g | 4.141 rad |
| g axial coordinate | 1.882 |
| leading wedge | 65.528 degrees |
| l | (1.882,97.871) |
| f | (99.983,-34.416) |
| base angle / length | -53.44 degrees / 164.693 |
| B | (67.623,140.181) |

Item19 uses `beta_2` although the program input is called `alpha_1`.
Reading it as the outlet angle 14 degrees reproduces the prescribed p;
using the complementary 76-degree tangent angle does not. The code records
this notational ambiguity rather than pretending another angle was supplied.
Leading inlet angle is obtained by reflecting the pressure tangent about the
explicit metal-inlet direction; this generalizes p26 beyond its 90-degree case.

### Table 2.2 (pp32–33)

The four controls, `(A,C,angleA,angleC)`, are:

1. `(0,0),(100,0), pi/3,-pi/6`;
2. `(0,0),(100,10), 2*pi/3,-pi/12`;
3. `(0,0),(100,-10), pi/2,-pi/4`;
4. `(-10,10),(100,10), -pi/4,-pi/2`.

The plotted third midpoint is `(25,42.5)` and fourth midpoint `(72.5,-45)`.
Tests cover all four inputs, exact endpoint tangency and recursive half-median
identities (1e-14 m tolerance), rather than invent numeric coordinates absent
from the printed graph. A tangent line can have the opposite directed angle
from the forward parametrization; only line equality is required.

### Table 3.1 (pp34–43)

Inputs differ from table 2.1 only in `outlet=17,Kt=.685`; back angle is now
an output. Printed pitch is 115.624; O2' is `(98.983,81.023)`.
Pressure `R3=273.077`, `O3=(-157.753,-130.591)`,
`g.x=l.x=1.707`, `l.y=97.752`, `f=(99.972,-34.364)`,
`B=(58.454,138.571)`, inlet wedge 71.456 degrees.

Item37's first seven printed suction points are:

| t (reconstructed dyadic parameter) | U | V |
|---|---:|---:|
| 0 | 1.707 | 97.752 |
| 1/1024 | 1.817 | 97.832 |
| 1/512 | 1.928 | 97.911 |
| 1/256 | 2.150 | 98.068 |
| 1/128 | 2.592 | 98.377 |
| 1/64 | 3.476 | 98.975 |
| 1/32 | 5.238 | 100.095 |

Tests use 0.0006 mm for these coordinates. Closure and scale invariance use
analytic floating-point tolerances, not visual resemblance to a graph.

**Quantified discrepancies, not tuned away:**

- p43 prints `P=(66.932,64.816)`, tangent angle `-63.175` degrees and back angle
  `13.325` degrees. Direct exact-quadratic source reconstruction yields
  `P=(66.928229597,64.812176256)` and back angle `13.327052500` degrees.
  Reference tolerances are 0.006 mm and 0.005 degrees, accounting for the
  disclosed replacement of the book's sparse cubic interpolation.
- p43 prints `R_ath=36.825`, `ath=35.808`. Those do **not** satisfy the printed
  Euclidean-distance equation with the printed P and O2'. The exact quadratic
  distance is **35.920839459 mm**, hence `ath=34.903850392 mm`. The radius
  discrepancy against the printed 36.825 is **-0.904160541 mm**, much larger
  than spline/rounding error. The code and regression assert the distance
  equation, not the inconsistent printed radius. No throat multiplier is
  fitted to make the mistaken result agree.

### Maximum thickness, table 3.2 (pp44–47)

Source definition is the common normal CK: parallel side tangents, both
perpendicular to CK; diameter `|C-K|` and center `(C+K)/2`. The book samples
normal-angle mismatch at pressure abscissae `(10,17,24,31,38,45,52)` mm,
prints `(-.284,-.204,-.124,-.049,.019,.075,.117)` rad, linearly interpolates
the zero and obtains `xD=36.035`, `W(xD)=74.523`, suction contact
`(44.876,87.862)`, and `Dmax=16.003` mm. Radius and center follow algebraically.
The table's final normal-line expression nests atan on an already computed
angle; the actual definition is tangent slope angle plus pi/2. Code uses
vector orthogonality instead, eliminating this transcription ambiguity.

**Resolved fixture switch:** table 3.2 prints its inputs at its end on **p47**:
`stagger=47,inlet=101,outlet=25,r_vx=.031,r_vux=.0095,Kt=.63,
wedge_out=7,m=1.05`, retaining `b=100,VA=100` mm. Radii and pitch again scale
with chord. This is not the preceding table 3.1 profile. Reconstructing the
proper fixture gives `R3=188.508868397` and
`O3=(-68.105329756,-82.608676407)` mm; its exact pressure ordinate at
the printed `xD=36.035` is **74.523068261 mm**, agreeing with the printed
74.523 to rounding. Thus the initial apparent pressure mismatch is fully
resolved, not a source error.

The exact quadratic and direct common-normal solution give:

- suction contact `(44.769246240,87.927081572)` mm;
- pressure contact `(35.939315088,74.586442294)` mm;
- diameter **15.998135532 mm**.

Residuals against the printed contact coordinates are at most 0.107 mm and
the diameter residual is **-0.004864468 mm**. Reference tests use **0.12 mm**
for the printed contacts and **0.006 mm** for the printed diameter, explicitly
allowing the book's seven-sample residual interpolation and suction spline.
They separately assert the independently reconstructed exact contact solution
at 1e-10 m. Using the exact quadratic with the book's seven-point linear
angular-mismatch interpolation gives `xD=36.051297465` and diameter
15.998060128 mm; the exact quadratic angular mismatches are
`(-.283855,-.203681,-.123563,-.048187,.018590,.074274,.117735)` radians.
The printed angular values differ by at most 0.000813 rad, consistent with
the disclosed spline/numerical representation difference, not a changed input.

For an additional control on the fully specified table 3.1 profile,
independent exact-quadratic algebra (reducing the normal condition against its
circular pressure side to a cubic) gives:

- suction contact `(44.954222302,95.952135762)` mm;
- pressure contact `(24.338574431,72.912338018)` mm;
- common-normal diameter **30.916617170 mm**.

The implementation solves the two vector normal equations on every side segment
pair, with bounded multiple starts and a dimensionless residual threshold
1e-8. This is an EXPERIMENTAL numerical search, not a proof of global root
completeness for arbitrary profiles. It returns the largest admissible common
normal found, rejects reversed tangents and wrong-side contacts, and raises
`ConstructionError` if none exists. Returned metadata explicitly says global
inscription of the candidate circle has **not** been verified. Parent geometry
validation must not equate this routine with a universal medial-axis algorithm.
An independent analytical limit uses horizontal pressure `y=0` and quadratic
suction with controls `(0,10),(50,30),(100,10)` mm: normal thickness is exactly
20 mm at x=50 mm. The published table 3.2 fixture, this analytical limit and
the additional reconstructed table 3.1 case are all tests.

## Two-circle pressure alternative (fig 2.3)

P18 says two pressure radii may reduce an excessive inlet wedge or maximum
thickness; p19 sketches g–S and S–h arcs. Neither page states a numerical
join rule, radius ratio or test example. Consequently a unique numerical
construction cannot be attributed to the source.

The additional builder is an explicit **geometric reconstruction**: the caller
supplies the inlet wedge, and endpoint positions/tangents at g and h define a
G1 biarc with positive tangent-distance ratio. Ratio 1 is a documented software
choice, not an empirical Vinogradov correlation. Changing it is explicit.
The same metal inlet reflection sets suction contact l, then the source
quadratic continues to f. Metadata identifies the join and the two actual
radii and labels the ratio as a software choice. Original single-circle O3
and pressure radius remain construction-scheme provenance, not claimed actual
biarc centers/radii. Tests assert join position/tangent and edge closure; no
numeric source example is invented. Global geometric admissibility remains
for parent validation, as with all experimental profile candidates.

## Checks and integration

Added tests cover printed scheme values, all four envelope cases, vertical
and parallel-tangent limits, p41 dyadic coordinates, closed contour orientation,
actual throat geometry, common-normal analytical/reconstructed cases, biarc G1
continuity, dimensional scaling, and no second incidence/deviation conversion.
No tests, builds, formatters or smoke runs were run in this slice, per the
integration contract. Parent should run the focused tests after all slices land.

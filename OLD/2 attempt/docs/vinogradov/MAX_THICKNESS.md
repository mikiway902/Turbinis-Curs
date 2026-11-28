# Maximum inscribed-circle thickness

## Source and the distinction being checked

Primary source: L. V. Vinogradov, *Автоматизированное проектирование лопаток
турбин* (2010), the checked-in
[PDF](../../Виноградов_Л_В_Автоматизированное_проектирование_лопаток_турбин.pdf).
Page numbers below are the printed pages, also the PDF page indices.

- [Page 4](../../Виноградов_Л_В_Автоматизированное_проектирование_лопаток_турбин.pdf#page=4)
  defines the medial line as the locus of inscribed-circle centres and the chord
  through its farthest points.
- [Page 5, figure 1.1](../../Виноградов_Л_В_Автоматизированное_проектирование_лопаток_турбин.pdf#page=5)
  identifies maximum diameter `Dmax` and relative thickness `Dmax/b`. Vertical
  side differences and equal-parameter side differences are not this diameter.
- [Table 4.1, page 61](../../Виноградов_Л_В_Автоматизированное_проектирование_лопаток_турбин.pdf#page=61)
  describes a prescribed maximum and its location. Items 47–49 on
  [page 69](../../Виноградов_Л_В_Автоматизированное_проектирование_лопаток_турбин.pdf#page=69)
  offset the pressure-side normal to obtain the circle centre. Items 50–53 on
  [page 70](../../Виноградов_Л_В_Автоматизированное_проектирование_лопаток_турбин.pdf#page=70)
  construct the common-normal suction contact. This ensures a **local** tangent
  circle, not a global inequality on every other part of the eventual contour.
- [Figure 4.1, page 74](../../Виноградов_Л_В_Автоматизированное_проектирование_лопаток_турбин.pdf#page=74)
  is the construction diagram, not a largest-circle verification. Table 5.1 on
  [pages 75–76](../../Виноградов_Л_В_Автоматизированное_проектирование_лопаток_турбин.pdf#page=75)
  closes the suction side with two tangent parabolas. Page 77 starts a different,
  active-profile construction; its constant-width channel envelope is not a
  proof about the prescribed-thickness two-parabola profile.

Here the actual boundary remains the existing exact quadratic Beziers, finite
pressure arc and finite edge arcs. No source formula, smoothing, spline fitting
or geometric scaling is changed.

`ProfileSpec.thickness_location` in this construction means **axial**
`centre_x / axial_width_m`: the reference is `24.5/100 = .245`. Metadata
`thickness_center` is the actual two-dimensional centre and
`prescribed_thickness_m` is the requested diameter. Neither should be compared
with a projected-chord thickness coordinate from another reporting API.
Another circle with the same maximum radius does not invalidate the requested
centre: that centre itself must attain the maximum, within declared accuracy.

## Optional final-analysis API

```python
from vinogradov.validation import validate_prescribed_thickness

result = validate_prescribed_thickness(
    profile,
    diameter_tolerance_m=1e-6,
    boundary_tolerance_m=None,  # defaults to diameter_tolerance_m / 16
    max_cells=20001,
)
```

The existing `validate_profile` implementation and its axial-envelope checks
are unchanged. Run that geometry gate separately, before interpreting the
maximum on a purported simple closed blade section. This optional analysis is
not inserted in an inner optimization loop.

The result includes:

- `maximum_diameter_m`, `maximum_center_m`: actual analytic nearest-boundary
  diameter at the best numerical centre; not an exact maximizer claim.
- `diameter_lower_m`, `diameter_upper_m`, `diameter_error_m`: an enclosure for
  the **global diameter**, with error equal to upper minus lower, not a radius
  error. These do not bound uncertainty in centre location.
- `converged`, tolerances, `boundary_error_m`, `boundary_segments`,
  `cells_evaluated`, `max_cells`: reproducible accuracy and effort information.
- Requested diameter/centre, `prescribed_center_diameter_m`, and
  `prescribed_status`: `within_tolerance`, `disproved`, `unresolved`, or
  `not_requested` when complete prescribed-circle metadata is absent.
- Source-cited `Check` objects: global numerical convergence, the local
  circle being inscribed, and that circle attaining the global maximum.
  `passed` requires every applicable check to pass. An unresolved result is
  **not** accepted. Metadata on the input profile is not mutated.

The local check uses distance to **all** actual boundaries: merely touching the
pressure and suction sides is insufficient if another boundary cuts the disk.
The global check compares diameter bounds at the requested centre, never
requiring it to equal the particular argmax returned by the search.

## Bounds, convergence, and assumptions

For a curve interval of parameter length `h`, linear interpolation has vector
position error at most `max(norm(C'')) * h**2 / 8`. For Beziers the second
control-point differences bound this derivative by the convex-hull property;
for a circular arc it is exactly `radius * sweep**2`. Composite curves are split
at their actual joints. The maximum per-interval bound, floating-point guard
and joint closure residual give `boundary_error_m = delta`.

For the resulting polygon, signed distance `s(x)` is 1-Lipschitz. A rectangular
cell centred at `c` with half-diagonal `h` therefore has upper radius bound
`s(c)+h`. A deterministic best-first four-child subdivision covers the entire
polygon bounding box. If the best sampled signed distance is `r` and largest
remaining cell upper bound is `u`, the curved profile has diameter bounds
`[2*max(0,r-delta), 2*max(0,u+delta)]`. The polygon maximum is inside that box;
outside it the polygon signed distance is nonpositive. No medial samples are
silently promoted to an exhaustive global search. Analytic nearest parameters
reuse `curves.nearest_parameter_to_point` (polynomial stationary roots and finite
arc candidates), rather than introducing new distance formulas.

**The enclosure is conditional numerical analysis, not a formal proof.** It
assumes a simple closed analytic contour and topology-preserving interpolation.
The linear homotopy then keeps winding signs unchanged outside the boundary
error tube; signed distances differ by at most `delta`. Self-intersections,
unresolved sub-tolerance loops, and invalid contour topology invalidate that
assumption. Floating-point root solving, ray crossing, and derivative bounds
are not interval-arithmetic certified. A scale-aware roundoff guard is included,
but cannot establish a universal floating-point proof. Unsupported curve types,
nonclosed joints, and degenerate segments are rejected, not replaced by guessed
fallback geometry. The existing finite-mesh geometry validation is itself not
a continuum topology certificate.

Subdivision stops when the global error meets the requested diameter tolerance
and the prescribed-centre comparison is resolved, or the cell budget is
exhausted. It may thus refine further after `converged` becomes possible; if
budget-limited, global convergence and an unresolved prescribed comparison can
coexist. Unresolved accuracy is returned explicitly. Positive tolerances must
satisfy `4*boundary_tolerance_m < diameter_tolerance_m`.

There is no new dependency. Work is approximately `O(N_boundary * N_cells)`
with vectorized segment distances and `O(N_cells)` heap storage. Broad nearly
constant-thickness regions are expensive for this conservative Lipschitz bound.
On the measured source/reference and course examples below, default analysis
used 13,329 and 8,769 cell evaluations, about 1.8 s and 0.9 s respectively in the
science container. These are observations, not guaranteed time limits. The
20,001-cell default is an explicit effort cap, not a promise of convergence at
arbitrary geometry scales.

## Actual two-parabola results

All numbers below are SI outputs converted to mm, using the defaults above.
Independent local Nelder–Mead refinement of analytic nearest-boundary distance
was also performed near each reported maximum. That refinement is corroborating
local evidence, **not** the global upper-bound argument.

### Printed reference

Inputs: axial width 100 mm, stagger 40 degrees, inlet/outlet metal 90/17 degrees,
pitch 95.677 mm, edge radii `.0243*c` and `.006025*c`,
`c = 100/sin(40 degrees)` mm, relative thickness `.106`, axial location `.245`.

| Quantity | mm |
|---|---:|
| Requested diameter | 16.490672564720373 |
| Actual diameter at requested centre | 16.490672564720380 |
| Requested centre `(x,y)` | `(24.5, 88.11000670201666)` |
| Best subdivision centre `(x,y)` | `(24.497076255321163, 88.11196805445305)` |
| Analytic diameter at that centre | 16.49063208392682 |
| Global diameter lower bound | 16.49055099965955 |
| Global diameter upper bound | 16.491534730628452 |
| Enclosure width | 0.0009837309689030016 |

Status: `within_tolerance` at 0.001 mm diameter tolerance, not a claim of exact
uniqueness or global equality. Local analytic refinement finds diameter
16.49067256472039 mm at approximately `(24.4999993145,88.1100071676)` mm.
The returned numerical centre need not coincide exactly with the prescribed
centre, and this is not a location failure.

### Non-reference course-derived section: local does not imply global

The existing axial-envelope regression supplies actual course-derived metal
angles/dimensions. Its constructible 75-degree stagger is used, with no contour
rescaling. Full inputs (metres unless angles):

```python
ProfileSpec(
    .02236309352291337, 75., 32.36385724199586, 27.260756849807482,
    .02104125693721863, .000782708273301968, .0002500000000000014,
    trailing_wedge_deg=3., origin_y_m=0.,
    relative_thickness=.12282186549577312, thickness_location=.3,
)
```

| Quantity | mm |
|---|---:|
| Requested diameter | 2.8435691333494537 |
| Actual diameter at requested centre | 2.8435691333494563 |
| Requested centre `(x,y)` | `(6.708928056874011, 2.7297009371821702)` |
| Best subdivision centre `(x,y)` | `(4.531281097223588, 2.2428424888211894)` |
| Analytic diameter at that centre | 2.8601535349604555 |
| Global diameter lower bound | 2.8601457112070136 |
| Global diameter upper bound | 2.861145548469142 |
| Enclosure width | 0.0009998372621283433 |

Status: `disproved`. Even the global **lower bound** exceeds the requested
circle by 0.0165765778575599 mm, well beyond the 0.001 mm tolerance. The requested
circle is genuinely inscribed; a different part of the actual two-parabola
contour admits a larger disk. Local analytic refinement gives 2.8602369431229434
mm at `(4.526415850507102,2.241414168715517)` mm. This profile passes all existing
`validate_profile` gates, including axial envelope, so that failure is not a
substitute explanation. Conversely the original 44.078661166-degree stagger in
the separate axial-envelope regression still fails that unchanged gate.

This is a numerical counterexample to interpreting the local table 4.1 circle
as automatically the global `Dmax` for every two-parabola closure. It is not a
claim that the reference is wrong, or permission to change the source formulas.
The previously documented printed root inconsistency at page 69 (20.527 mm)
and conflicting first inlet matrices at page 75 remain separate source issues;
this analysis uses the governing analytic construction, not a forced fit to
those inconsistent printed entries.

## Independent physical regressions

`tests/blading/test_global_thickness.py` checks:

1. An exact 10 mm-radius circle, represented by four actual finite arcs: global
   diameter 20 mm must lie inside the reported enclosure.
2. The convex hull of 2 mm and 6 mm-radius circles, centres 30 mm apart, with
   exact common external tangent lines and finite edge arcs. Its maximum
   diameter is independently exactly 12 mm: the large disk is contained, while
   the whole hull lies in the strip `-6 <= y <= 6` mm. A valid 4 mm target disk
   at the other end must fail the global check.
3. Equal 6 mm-radius ends produce a capsule with tied maxima. A returned centre
   different from the requested one must not cause rejection.
4. An intentionally one-cell budget must return a genuine unresolved physical
   diameter interval, not a fabricated convergence flag.
5. The actual reference and non-reference two-parabola profiles above, with
   quantitative diameter/error assertions and preserved axial-envelope checks.

# Vinogradov 2010: prescribed thickness, two parabolas, active profile

Status: **EXPERIMENTAL**. These constructions do not replace Mikhaltsev or promote any equation. Metres are used by the API; all printed quantities below are millimetres unless angles are specified. Source: L. V. Vinogradov, *Автоматизированное проектирование лопаток турбин*, 2010, pp.61–88. Every page image 61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88 was inspected, not only extracted text.

## Interfaces and definitions

- `build_prescribed_thickness_profile(spec: ProfileSpec) -> SectionProfile`: table4.1 construction completed by table5.1's two-parabola suction side.
- `build_double_parabolic_profile(spec: ProfileSpec) -> SectionProfile`: the same source construction, explicitly identified as table5.1.
- `prescribed_thickness_construction(spec) -> dict`: all defining centres, contacts, control points, the abandoned mean-line approximation and source throat point.
- `build_active_profile(spec, options: ActiveProfileOptions | None = None, *, variant='constrained') -> SectionProfile`: default solves the pressure radius from the outlet metal angle, trailing wedge and both edge-circle tangencies, then preserves the actual inlet/outlet metal and wedge with constrained G1 joins. An explicitly supplied options radius must be compatible or raises a quantified error. Explicit `variant='source'` retains all original table6.1 input-radius behavior and its unsuccessful inlet condition.
- `active_profile_construction(spec, options=None) -> dict`: both the exact pressure-circle joins and contradictory printed `h`, exit probe, blends, direct unsuccessful inlet contact and angle residual.
- `ActiveProfileOptions(pressure_radius_chord_ratio=15.25/25.61, exit_probe_fraction=.001)`: extra table6.1 data. The first ratio is an explicit source radius input, not an empirical radius fallback. A caller can replace it. The second is source 0.1mm/100mm.

`relative_thickness = Dmax/100` in the book's percentage notation: source Dmax=10.6 means `.106`, not `.0106`. Its physical diameter is chord × `.106` =16.490672565mm. `thickness_location` refers to the circle centre's axial coordinate divided by axial width: source bD=24.5, b=100 means `.245`. It is **not** the suction contact's x, not pressure contact x and not distance along chord. Both side curves run inlet→outlet. Edge arcs run pressure→suction at the inlet and suction→pressure at the outlet. Table6.1 Spa is on the pitch-translated blade below W; the implementation shifts Spa by +pitch to pair it with the unshifted W. This translation is essential for a closed contour.

`ConstructionError.details` records quantified domain failures (negative/intersection radicands, inadmissible tangent control triangles, pressure contact outside the pressure arc, impossible or axially backtracking G1 joins). Metadata is JSON-compatible and carries method, source, status and construction validity. The explicit active `variant='source'` is a returned **invalid candidate**, not a raised opaque error or a falsely passing blade. The default constrained variant replaces its failing boundary joins; it is not claimed to reproduce the source's final figure.

## Table4.1, pp.61–74

Inputs: b=100; VA=100; stagger=40°; inlet metal=90°; outlet metal=17°; leading radius/chord=.0243; trailing radius/chord=.006025; trailing wedge=7°; back-angle gamma=7°; throat multiplier m=1.05; bD=24.5; thickness/chord=.106; pitch=95.677 (printed table4.1 item31). Pitch must be explicit; no hidden Kt is inferred.

The first mean-line approach (items14–30) fits a parabola between edge centres with the inlet/outlet tangents, then offsets its common normal by the prescribed radius. Its pressure tangent misses the leading circle; item30 explicitly states that this approach need not have a solution. We retain its coordinates and signed circle/tangent clearance. The final construction (items37–65) instead constructs the single-radius pressure circle first, then its normal-offset thickness circle and two suction control triangles.

For pressure centre C and radius R, prescribed radius r and centre axial position xD, the centre locus is exactly the circle of radius R+r. Thus

`D=(xD, Cy+sqrt((R+r)^2-(xD-Cx)^2))`,
`n=(D-C)/(R+r)`, `W=C+R*n`, `Q=C+(R+2r)*n`.

This is an analytic solution of item49's root equation, not a different physical closure. Q and W have exactly common normals and separation 2r. The common tangent at Q defines both parabolas. The inlet suction tangent is the reflection of the pressure inlet tangent around the inlet metal direction. The exit tangent includes half the trailing wedge. Each parabola is an exact quadratic Bezier whose middle control point is the intersection of its endpoint tangent lines; the geometric envelope is unchanged, but no sampled cubic-spline approximation is needed.

| Quantity | Printed | Analytical reconstruction |
|---|---:|---:|
| chord |155.572|155.572382686|
| leading radius |3.780|3.780408899|
| trailing radius |.937|.937323606|
| pressure radius |208.955|208.954944590|
| pressure centre |(-97.538,-91.564)|(-97.537573962,-91.564122076)|
| thickness radius |8.245|8.245336282|
| thickness centre |(24.5,88.11)|(24.5,88.110006702)|
| suction thickness contact |(29.133,94.931)|(29.132778707,94.930778789)|
| pressure thickness contact x, item49 |20.527|**19.867221293**|
| inlet suction control point |(13.902,105.276)|within displayed rounding|
| throat and construction P |26.641; (89.404,51.787)|within displayed rounding|

**Quantified source inconsistency:** the printed item49 ZD=20.527 is not a root. Substitution gives `z+r*(z-Cx)/R-xD = +.685813493mm`. Its pressure contact x differs from the analytical root by +.659778707mm. If combined with printed suction x=29.133 its midpoint is24.830mm rather than24.5mm. Conversely the analytical solution reproduces the printed thickness centre and Q to <.00023mm. The code does not copy the wrong root or loosen its tolerance.

The exact mean parabola gives at x=24.5 a y of91.905091976 and tangent−19.835306372°. Printed values are91.904 and−19.841° (cubic interpolation of the sampled construction). Differences .001092mm and .005694° are documented numerical-approximation effects; those separate comparisons use .003mm and .01° tolerances. No spline is substituted to force agreement.

The imposed thickness is a **local common-normal diameter**. The source calls it maximum; neither the book nor this implementation proves it is the global maximum over arbitrary user inputs. Metadata states `global_maximum_thickness_verified=False`; engineering validation must independently assess the contour and must not equate this input with a certified maximum.

### Reference figure4.1, p.74

The reconstruction data include both edge circles, pressure circle, thickness circle, tangent control points, Q, inlet and outlet suction joins, chord and source throat point. They reproduce the schematic's inlet circle around(3.78,95.495), thickness circle around(24.5,88.110), pressure circle arc from x≈1.98 to98.185, and suction exit(99.974,−17.839). The figure is a construction diagram, not an extra numerical baseline. Tests anchor its labelled coordinates rather than inventing pixel-level precision.

## Table5.1, pp.75–76

Inputs are table4.1's inputs above. The suction side comprises exactly two quadratics, inlet `l→Q` and outlet `Q→f`, with equal tangent direction at Q. No arbitrary smoothing, tangency repair, or duplicated middle points are introduced. Pressure remains one circular arc, and the finite leading/trailing circles close the contour.

Item2 prints the inlet endpoint `(1.980,98.819,.496rad)` and subsequent points `(2.728,99.206)`, `(3.483,99.560)`, `(5.012,100.171)`, `(8.148,100.997)`, `(14.729,101.075)`. Tests compare the analytic parabola at each printed x, with .002mm allowance for coordinate rounding/graphical subdivision.

**Second source inconsistency:** the different matrix fragment at item3 begins `(2.392,99.011,.376rad)`. It cannot be the same curve's inlet endpoint as item2. Its first x differs by approximately+.412mm, y by+.192mm and tangent by−.120rad. The test records this contradiction explicitly rather than claiming that both matrices match. The specified table4.1 control triangle and table5.1 item2 determine the implemented curve.

### Reference figures, table5.1 items7–8, p.76

Item7 is the complete two-parabola/single-circle profile and a copy at +pitch. Item8 compares it with the earlier single-parabola profile: the prescribed-thickness profile is less full. Reconstruct item7 from `profile.contour()` and its +`(0,pitch)` translation; table5's coordinate tests, common-normal diameter, analytic quadratic degree and closure/tangent tests constrain its numerical geometry. Item8's qualitative statement is not promoted to a universal thickness inequality for arbitrary specs, nor are graphical coordinates manufactured as exact values.

## Table6.1, pp.77–86

Inputs: b=100; VA=100; stagger=78°; inlet metal=40°; outlet metal=20°; pressure radius/chord=15.25/25.61; leading radius/chord=.53/25.61; trailing radius/chord=.2/25.61; Kt=.58, so pitch=59.295754502; trailing wedge=7°; m=1.05. The source does not print a consistent gamma input (see below). The ratio .595 printed on p.78 is rounded; using it instead of15.25/25.61 changes the physical radius and must not be silently done.

The missing Mathcad pressure-circle subroutine is replaced by the exact intersection of circles of radii `(R+rLE)` and `(R+rTE)` about the two edge centres, taking the lower centre. Pressure and edges are externally tangent. The source constant-width limit is a concentric circle of radius `R-a_th`, where `a_th=pitch*sin(outlet)/m`. This is a **geometric** constant-normal-width construction on the two circular portions, not proof that the full finite cascade has no diffuser, no choking or correct flow angles. Inscribed channel circle radius is a_th/2, its centre lies halfway between concentric walls.

The source exit probe is x(source h)−0.1mm. Its normal locates the concentric exit join. Both omitted circle/point blending routines are solved algebraically. Given tangent point P, unit normal n, target centre T and radius r, the internal-tangency blending circle is `B=P-Rb*n`, with

`Rb=(r^2-|T-P|^2)/(2*((T-P)·n+r))`.

The target circle must be inside the blending circle. This reproduces the two printed blend matrices without their unavailable Mathcad files.

| Quantity | Printed | Analytical reconstruction |
|---|---:|---:|
| chord |102.234|102.234059487|
| pressure radius |60.877|60.877368495|
| pressure centre |(43.626,52.169)|(43.626448484,52.168935316)|
| actual pressure inlet contact |(3.510,97.959)|(3.509950760,97.958897156)|
| actual pressure outlet contact |(98.482,78.568)|(98.482188738,78.567832190)|
| throat |19.315|19.314611860|
| exit normal angle |25.942°|25.942465950°|
| concentric exit join |(81.00108,70.35133)|(81.001083541,70.351327718)|
| exit blend radius |288.761|288.760993595|
| exit blend centre |(-178.663,-55.972)|(-178.662558337,-55.972480443)|
| exit blend contact |(99.972,19.828)|(99.972001548,19.827872192)|
| inlet blend radius |76.988|76.987580986|
| inlet blend centre |(76.988,40.255)|(76.987580986,40.254531492)|
| inlet blend contact |(4.485,66.148)|(4.484952185,66.147705167)|

### Source ambiguities and failed inlet: no fake pass

1. Item18's printed theta uses `alpha1−wedge/2`, but its numerical theta=3.552rad and h=(98.469,78.596) use `alpha1+wedge/2`. The implementation exposes this printed numerical `source_h` separately from the true pressure-circle contact. Their distance is **.030637758mm**. The source h controls its exit probe only; it must not create a non-tangent pressure-edge join.
2. Item16 gives beta=17.077° from `alpha1−wedge/2+gamma` but does not supply gamma consistently. This requires gamma=.577°, whereas an explicit gamma=7° gives23.5°, a **6.423°** discrepancy. Gamma does not determine the implemented concentric/blend geometry; `source_beta_deg` reports the value for the caller's explicit input.
3. Item44 explicitly says the direct inlet joining circle exits the cascade and joins the lower rather than upper semicircle. Its direct concentric contact is(16.238,83.431). Item46 replaces the inlet condition by a vertical tangent through `(0,W(0)−pitch)`; this is not an inlet metal-angle-preserving correction. We implement this exact alternative because it is the one used by the final figure, but label it invalid for the prescribed source inlet angle.
4. The final **source-variant** candidate's pressure inlet tangent is41.221539304° from +x and suction tangent at the leading extreme is90°. Their bisector is65.610769652° from +x, equivalent to inlet metal **24.389230348°** rather than40°. The signed axial-angle residual is **+15.610769652°**. `construction_valid=False`, an explicit failure reason, `actual_inlet_metal_deg`, and the signed error accompany the source candidate. A smooth closed contour is not treated as evidence of source correctness.
5. Item50's last interval has a reversed inequality (`... ≤ u ≥ b`); the finite contour uses the intended right endpoint b, not an unbounded branch.

### Reference figures inspected

- Figure6.1, p.77: envelope of throat-radius circles on the neighbouring pressure wall; it motivates the constant-width limit, not a substitute for a real cascade channel solver.
- Table6.1 item7 p.78, item15 p.79: chord, edge circles and full pressure circle.
- Items22–23 pp.80–81: actual finite-edge tangencies. Exact circle intersection determines these, not the contradictory h approximation.
- Item33 p.82: pressure, concentric circle, channel circles and normals (axes are swapped in the printed plot). The reconstruction provides centres/radii and normals for direct plotting without interpreting swapped axes as a new coordinate convention.
- Items41/43 pp.83–84: exit joining circle and concentric exit join. The analytic blend matches the printed radius288.761 and centre.
- Item49 p.85: inlet joining circle; the implausible excursion is preserved and diagnosed, not cosmetically redrawn.
- Item51 p.86: two final periodic active profiles. Plot returned contour and +pitch copy. Closure and tangency tests are independent of the explicitly failing inlet metal-angle condition.

Pages87–88 contain references and the 2010 colophon, no further construction methods. Page87 reference11 identifies the 2008 two-conjugate-parabola article; reference4 is the gas-turbine text cited for figure6.1. No formula was inferred from those unprovided works.

## Acceptance and verification handoff

`tests/blading/test_advanced_reference.py` covers printed table4/5/6 dimensions, quantified source contradictions, exact local thickness and midpoint, analytic quadratic degree, directed edge/side closure and unit tangents, dimensional scaling, failure metadata and quantified invalid domains. Standard printed three-decimal coordinates use ±.0006mm (half a last digit plus a small rounding allowance); composed/rounded construction values use at most .003mm, specifically labelled. Five-decimal joins use .00002mm. Geometry identities use SI tolerances1e−12m and tangent tolerance1e−10. No expected source error is hidden as an xfail or patched reference.

No builds, tests, formatters or verification commands were run in this slice. The parent should run `pytest tests/blading/test_advanced_reference.py` with the integrated geometry package, then the requested repository verification. These are new experimental tests, not edits to immutable reference results. Parent integration must respect `construction_valid=False` and must not present the active source example as accepted engineering geometry.

### Plot-ready reference-figure APIs

`prescribed_reference_figure_data(spec, count=401)` returns named `(N,2)` SI arrays for figure4.1's circles/control triangles and table5.1 items7–8's periodic contours and single-/double-parabola comparison. The single-parabola comparison delegates to the actual table3.1 builder; it is not redrawn from an invented fit.

`active_reference_figure_data(spec, options=None, count=401)` returns named SI arrays for figure6.1's constant-width limit, table6.1 item33's channel circles, item43's exit blend, item49's failed inlet blend and item51's periodic contours. Source pitch shifts and plot-axis conventions are explicit. These data can be rendered with any plotting backend, without adding a mandatory matplotlib dependency. Their tests constrain circle radii and periodic translation.

## Constrained active-profile variant: boundary-preserving numerical design

`build_active_profile(spec)` now means `variant='constrained'`. The exact source control `active_profile_construction` is unchanged. `variant='source'` is required by source-fixture tests, and `active_reference_figure_data` always requests it explicitly.

The circular pressure-side construction and central constant-normal-width architecture are retained, but their radius is **solved from the actual outlet boundary constraints**, not fixed to the source example's incompatible input radius. Specifically table4.1's exact `_pressure_from_exit` relation requires the pressure exit tangent to be `outlet_metal-90°+trailing_wedge/2`. For the source example dimensions this gives pressure radius **59.171873713mm**, ratio **.578788263035**, centre approximately(44.205272205,55.000959728), pressure tangent−66.5° and suction tangent−73.5°. Their bisector gives20° outlet metal and their actual difference gives7° wedge simultaneously. The source radius60.877368495mm is1.705494782mm larger and cannot be retained while satisfying these outlet conditions and the exact edge tangencies.

At either blade end, if the pressure tangent angle from +x is `ap` and the required metal angle in that coordinate system is `am`, the suction tangent is fixed as `as=2*am-ap`. Here `am=90°-inlet_metal` at inlet and `am=outlet_metal-90°` at exit. The suction contact with the corresponding edge circle is exactly `Cedge+redge*(cos(as+90°),sin(as+90°))`. Thus the prescribed metal direction is the bisector of the **actual boundary derivatives**, not a metadata label or substituted flow angle. With the solved radius the attempted analytic clockwise internal inlet circle requires a negative radius, approximately−24.254569785mm (target radius39.857261853mm), and is rejected with quantified details. It is not made positive with an absolute value or silently placed on another branch.

The explicit numerical-design alternative is an **equal-tangent-distance G1 biarc** from the actual leading suction contact to the book's direct concentric inlet contact (item44), now translated into the same profile frame. For unit endpoint tangents t0,t1 and endpoint separation v, its positive tangent distance solves

`2*(1-t0·t1)*d² + 2*v·(t0+t1)*d - |v|² = 0`;

the join is `(P0+P1+d*(t0-t1))/2`. Each member is an exact point/tangent circular arc, with an exact straight-line limit when appropriate. The endpoint and interior tangent directions are checked algebraically. This is a deterministic interpolation design convention, **not** an equation claimed from Vinogradov or an empirical correlation. The source inlet construction is not smoothed or angle-adjusted.

The exit similarly uses an equal-distance G1 biarc between the existing concentric exit join and the trailing-edge contact determined by the specified outlet metal angle. This also removes the small exit-angle inconsistency that the original source circle/point blending procedure can introduce. No outlet tangent is inferred from a source plot.

Both join families reject axial backtracking using analytic extrema of the arc x-derivative, not coarse sampled points. Impossible source-contact ordering, reversed/overlapping edge branches, degenerate joins, G1 residuals and actual endpoint metal-angle residuals raise `ConstructionError` with numerical details. `construction_valid=True` in the constrained variant means this explicit geometric construction and boundary enforcement succeeded; it is **not** full cascade clearance, channel convergence or aerodynamic certification. `full_cascade_clearance_verified=False` and `nondiffusing_channel_verified=False` remain explicit. Parent engineering validation must still reject any conflicting finite cascade.

Additional tests inspect actual inlet/outlet derivatives, all G1 and edge joins, similarity of the default constrained construction, explicit source selection and a quantified impossible forward-biarc case. No tests were run in this worker slice.

### Explicit radius constraints and actual wedge verification

In `variant='constrained'`, omitting `options` solves the radius as above. Supplying `ActiveProfileOptions(...)` explicitly constrains its `pressure_radius_chord_ratio`, even if it equals that class's source default; incompatible radii raise `ConstructionError` with `requested_pressure_radius_m`, `required_pressure_radius_m`, `pressure_radius_residual_m`, outlet metal and wedge. They are never silently overridden. To customize only the probe while constraining this radius, supply the compatible solved ratio explicitly. `variant='source'` and `active_profile_construction` retain their original explicit radius inputs and source numerical outputs.

The constrained builder measures both actual trailing-side derivatives after constructing its curves and rejects a wedge error above1e−7 degrees independently of the metal-bisector residual. Metadata includes `actual_trailing_wedge_deg`, `trailing_wedge_error_deg`, `solved_pressure_radius_m` and the radius constraint mode. The retained source-style inlet-angle residual for the solved-radius control geometry is named `preconstraint_inlet_metal_angle_error_deg`; it is not confused with the original source-radius example's15.61077° failure.

Tests independently measure the angle between the actual pressure and suction exit unit tangents, assert both tangent directions, and exercise rejection of the incompatible original source radius as an explicit constrained input. No checks were run in this slice.

### Source-blend domains are not constrained-design prerequisites

The shared `_active_core_geometry` computes only pressure/edge tangencies, the concentric normal-width circle, source h, exit probe, direct inlet contact and the source-style angle diagnostics. It does **not** solve either source-only circle/point blend. `active_profile_construction` explicitly adds those blends and retains their original source values and branch errors. The constrained builder calls the core directly, then independently builds its boundary-preserving inlet join and exit G1 biarc. An impossible source exit blend therefore cannot incorrectly reject an otherwise admissible constrained biarc; conversely an impossible or looping constrained join still raises its own quantified error. `source_blends_evaluated` distinguishes these paths.

The integration-reported RK3 midpoint failure was a negative-radius source blend (`blend_radius_m=-.2768704519846517`, `target_radius_m=.000411792803`) occurring before the constrained joins were reached. That observation is retained as a source-branch domain failure, not treated as proof that the constrained geometry is impossible. No reported failure was rerun in this worker slice. Tests independently compare all shared-core quantities against the source construction and require the default constrained path not to evaluate source blends.

A separate deterministic domain regression deliberately uses the same source edge/angle dimensions with40mm pitch and the compatible constrained pressure-radius ratio. Its positive concentric radius is admissible, but the source inlet circle/point helper returns a positive radius smaller than its target circle and correctly rejects that internal-tangency branch. The constrained inlet/exit G1 design is required independently to close, preserve both metal directions and measure7° trailing wedge. This new case is explicitly **not** the unpersisted old RK3 midpoint; no old fixture was invented or failed integration rerun.

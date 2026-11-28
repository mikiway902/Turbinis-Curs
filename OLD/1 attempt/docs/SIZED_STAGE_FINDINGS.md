# Actual sized-stage findings — 2026-11-12

## Scope and readiness

This is a verified **reduced-model investigation**, not final project delivery.
The uncooled/no-bleed source cycle and existing meanline are coupled through actual
stage work and GG/FPT total states. Compressor maps/inventory, cooling/mixing, metal
temperature, material allowables/life and radial/3D validation are not supplied.
No fully feasible engine, globally best architecture or production Pareto front is claimed.

## Method and source decisions

- Original cycle book printed p.22 §3.2: prescribed shaft power determines design air
  flow through specific work. At 110 kW **before the reducer**, downstream gearbox
  efficiency is not applied. The lead visually checked the original PDF.
- Each explicit air-flow trial rebuilds actual stages and geometry. Actual compressor
  load is `Lc*G_air`; fuel and gas flows follow the retained source fuel/air ratio.
  This is not multiplication of previously saved stage powers.
- C closes GG against actual compressor work, transfers actual GG total T/p/flow to
  FPT, and closes FPT total outlet pressure. Compressor power is not subtracted twice.
- Original turbine book printed p.87 items 55/58 and p.90 item 99 permit explicit
  heat-drop/angle choices and require recalculation when exit angles are unsuitable.
  Stage weights, angles, speed and hub are recorded design variables; source loss
  correlations, efficiencies and acceptance limits were not tuned to force a pass.
- Pressure-ratio variants recompute the source cycle, fuel/alpha and GG reference
  bracket. A uses two series compressor units with log-ratio splits, not two combustors.

## Calculations and retained failures

The actual sized coarse sweep covers pi=3..13 at every integer for three fixed
families: A/B MAIN3 at 78000 rpm / 0.070 m hub, and C GG2/FPT1 at independent speeds.
Its 33 cases contain **597 inner trials**, all reproduced. Thirty roots close 110 kW;
at pi=3 the declared outer air-flow brackets contain no root for all three schemes.
Those three NUMERICAL_FAILURE cases have successful inner evaluations. They are not
nonphysical states or proof that pi=3 cannot work with another explicit design/bracket.

Before launch: at most 1056 trials / about 16.9 MB estimated. Actual journal: 8666417
bytes; calculation phase 565.24 s, excluding setup and reproduction. No full-engine
GPU acceleration or production throughput is inferred.

Eight completed sizing datasets used by the final cross-run analysis contain 1064
RAW trials / 60 outer cases. Fifty-four successful root memberships deduplicate to
46 unique candidates; six failed case memberships remain visible. Different recorded
design families are not silently treated as identical physical inputs.

## Four-profile screening results

The four checks are distinct: user RHO065 upper ceiling; user stage pi=1.8–2.4;
textbook p.38 intermediate/final alpha2 ranges; **project-assumed** sigma<=290 MPa.
The actual reaction-table l/D domain is checked separately, never extrapolated.

At pi=11 the following examples numerically close 110 kW and pass these checks:

| Scheme | Turbine controls | Air kg/s | Fuel kg/s | Shaft W |
|---|---|---:|---:|---:|
| A | MAIN3, 78000 rpm, hub 0.070 m | 0.709140617 | 0.012353085 | 110000.000520 |
| B | MAIN3, 78000 rpm, hub 0.070 m | 0.704352712 | 0.012280032 | 109999.950304 |
| C | GG2 76000 rpm / 0.073 m; FPT1 61000 rpm / 0.080 m | 0.694056384 | 0.012100521 | 109999.938854 |

Each family also has a pi=12 point passing the same screening. These are six scoped
candidates, not six fully validated designs. Stage counts and geometry differ between
families, so numerical fuel differences do not isolate an architecture-only advantage.
They are not optimized compressor/turbine designs.

The six points were recalculated under `require_final_source_domain` with explicit
±2% flow verification endpoints and the known central root: 18/18 records reproduce,
with actual mass/work/pressure closure. This is local reference verification, not a
new optimum search. A separate seven-point check preserves six valid roots and an
outside-domain A/pi4 MODEL_ERROR with all calculated stages (7/7 reproduce).

## Physical interpretation and unresolved limitations

- At low pi, less useful net work per unit air requires higher flow. In these fixed
  families, larger annuli/heights increase the available centrifugal stress estimate
  and may exceed the source reaction-table domain. All such trials are retained.
- Lowering n alone reduced stress but spoiled alpha2. Increasing hub too far changed
  angles and could put first-stage l/D below the source table. Joint n/hub refinement
  exposed this trade-off without changing the engineering bounds.
- C needs its own GG/FPT speed and stage-work distribution. Copying the A/B reference
  controls produces unsuitable triangles; equal GG pressure weights and independent
  shaft inputs improved the recorded screening, not a compressor-map match.
- Within these families, lower observed air flow near pi≈9–10 does not imply an
  acceptable design: stage pi/angles can fail there. At the upper edge, A/B pi=13
  violates the prompt stage-pi profile. Acceptance must not be replaced by a flow minimum.
- Some positive angle margins are small (e.g. C/pi11 FPT alpha2≈80.068 degrees).
  Source/model uncertainty and unverified material properties preclude robust-design
  claims. Sigma=290 MPa is an assumption, not validated temperature-dependent strength.
- Strict final-source-domain mode rejects unsupported final states as MODEL_ERROR;
  intermediate clipped numerical guesses are not validated physical outputs. The
  historical coarse snapshots retain their explicit legacy-clipped provenance.

## Evidence and reproduction

All paths below are under `reference/architecture_stage_smoke/`:

- `2026-11-11-pi3-13-sized-coarse/`: 597 records, SHA-256
  `761e89d138c07f5e4bacfcad2ee26801ea21b464d504ec8afd26687f32139c92`.
- `2026-11-11-c-independent-refinement/`: 58 records, SHA-256
  `d2574a6e2d3aeb9f1fcb7478422f1f621bee7d6ca858c05f6499de4f240c26aa`.
- `2026-11-12-strict-finalist-sizing/`: 18 records, SHA-256
  `cd89a446f435992c2b486009213a2e8fe0e018adb9c5a9c3ffa8a1ea8a3e62e7`.
- `2026-11-12-final-source-domain-check/`: source-domain success/failure evidence.
- `*-stage-audit-v3.json`: source compressor arithmetic, actual mass, work, pressure,
  GG/FPT continuity, exact RAW-ID accounting and separate outer/inner statuses.
- `2026-11-11-sized-analysis-final/`: analysis JSON and two scatter plots. No curves,
  interpolated heatmaps, full-feasibility front or production readiness are asserted.

Use PROJECT_GUIDE for commands. Computational digest changes intentionally reject
resume of older runs; read-only numerical reproduction remains available. Never
rewrite archived manifests to force resume. Original independent stage snapshots
remain unchanged and are not retroactively called coupled/sized engines.

Final parent checks: 536 engine tests OK (1 GPU skip), 245 scripts tests OK;
static verifier 6 PASS / 7 WARN / 0 FAIL. Independent reviews PASS after fixes.
These are software/arithmetic/source-domain checks, not physical validation.

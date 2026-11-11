# Project State

Last updated: 2026-11-12
Git checkpoint: `48be4ee` (branch `main`); the continuation below is uncommitted working-tree work.

## Current route — two-textbook scope correction

### Latest verified checkpoint — 2026-11-12

Status: NOT DONE. Numerical actual-engine sizing now works for A/B/C; this is not
production or physical validation. Full delivery gates and missing engineering
evidence remain open. Lead model is GPT-6.1 Sol per the user's explicit selection.
Counted checklist: 88 DONE / 31 IN PROGRESS / 13 TODO / 2 SKIPPED = 134 tasks;
44 remain unclosed. This is not an effort estimate or a physical-readiness percentage.

Additional implementation: explicit stage design arrays, independent n/hub controls,
source-cycle compressor-ratio variants, failure-aware v3 audit and read-only cross-run
analysis/plots. Changing pressure ratios recomputes compressor/fuel/alpha/GG reference
states; other source parameters remain locked. Final-source-domain checking is opt-in:
`stage_controls.reaction_table_policy="require_final_source_domain"`. It rejects an
unsupported final reaction-table state as MODEL_ERROR with full calculated partials,
not as physical impossibility. Legacy snapshots remain numerically reproducible.
Intermediate table clipping is numerical continuation only, not validated physics.

Completed bounded actual sized coarse run:
`reference/architecture_stage_smoke/2026-11-11-pi3-13-sized-coarse/`:
33 explicit cases (A/B MAIN3 and C GG2/FPT1 at each integer pi=3..13), 597 inner RAW
trials, 597/597 reproduce. Outer results: 30 roots + 3 NUMERICAL_FAILURE, all at pi=3
with no root in the declared air-flow bracket. All 597 inner trials are SUCCESS;
outer bracket failure must not be counted as an inner nonphysical state. Estimated
upper bound before launch was 1056 records / 16.9 MB; actual journal is 8666417 bytes,
SHA-256 `761e89d138c07f5e4bacfcad2ee26801ea21b464d504ec8afd26687f32139c92`.
Recorded calculation time excluding reproduction/setup: 565.24 s on this host.

C's independent refinement (`2026-11-11-c-independent-refinement/`) retains 58 trials,
58/58 reproduce. GG2 uses 76000 rpm / 0.073 m, equal explicit pressure weights;
FPT1 uses 61000 rpm / 0.080 m. At pi=11 and 12 it passes the four audited profiles
within the source reaction-table domain; pi=10 angle failures are preserved.

Final cross-run report/plots: `2026-11-11-sized-analysis-final/` under the same
reference directory. Eight datasets, 1064 inner trials, 60 outer cases, 54 root
memberships deduplicated to 46 unique root candidates; 6 outer failures retained.
Six unique candidates pass ONLY the four audited profiles at pi=11/12 (A/B/C).
No global best, fully feasible design, Pareto front or production readiness is claimed.
See `docs/SIZED_STAGE_FINDINGS.md` for physical interpretation, exact scope and paths.

Reference recalculation with strict final-source-domain policy:
`2026-11-12-strict-finalist-sizing/` has 18 trials: each of the six screened roots
was recomputed with explicit ±2% air-flow endpoints and the central root. All six
power roots and 18/18 records reproduce. These local endpoints are verification of
known roots, not independent optimization. `2026-11-12-final-source-domain-check/`
separately preserves six valid roots and one known out-of-domain A/pi4 MODEL_ERROR;
all seven reproduce, including the failed record's full computed stages.

Parent independently observed final suites: 536 engine tests OK (1 GPU skip),
245 scripts tests OK; static verifier 6 PASS / 7 WARN / 0 FAIL; diff check clean.
Independent reviewers PASS after fixes for every substantial addition, including
source-domain handling, source-ratio variants, failure accounting and role-correct
plots/deduplication. No commit or push after checkpoint `48be4ee`.

Next concrete P0/P1 action: use strict-source-domain policy in subsequent accepted
flow-sizing runs; inspect the remaining available source/engineering checks (root
reaction, meridional geometry/clearance/speed assumptions) before treating four-profile
screening as broader feasibility. Keep compressor inventory/maps, cooling/mixing,
material/metal-temperature/creep and radial validation explicitly missing. Finish
the remaining checklist gates without inferring those data or restarting thermodynamics.
Context-handoff checkpoint: all commands launched in this continuation have finished;
there are no background calculations to recover. Start from the latest section and
the 44 open checklist tasks, not the older fixed-flow deficit. No user confirmation
is needed for ordinary continuation. Production physical-validation claims are blocked
by the missing engineering evidence; the overall project is still IN PROGRESS, not
declared complete. Preserve the uncommitted working tree and all historical journals.
Historical snapshots below are superseded only where this latest section says so.

### Current continuation — explicit design-flow sizing and independent operating points

Goal: close the actual meanline engine numerically at 110 kW, then assess stage
constraints without treating numerical closure as engineering feasibility.

Source decision: lead visually checked the original cycle textbook printed p.22,
§3.2 (`N_specific=N/G_air`, shaft power from turbine gas work minus compressor air
work). At the user-defined pre-reducer plane the downstream gearbox efficiency is
not applied. Therefore the preliminary 0D-sized flow must not be frozen when actual
meanline work differs. `air_mass_flow_kg_s` is now an explicit optional design unknown;
each trial rebuilds all stages/geometry and actual GG→FPT linkage. No saved stage
power is simply multiplied to produce 110 kW. Fuel/air ratio, gas basis and source
cycle pressure/temperature assumptions stay fixed; actual compressor work is Lc*Gair.

`architecture_flow_sizing_smoke.py` sizes three fixed control cases by explicit
bracketed bisection (0.9–1.4 times preliminary air flow), saving every trial in RAW.
`stage_controls.operating_points` optionally supplies independent per-role speed/hub;
the source 84000 rpm / 0.065 m remains unchanged when this option is absent. Independent
GG/FPT speed compatibility with an actual compressor/map is NOT_CHECKED.

Saved experiments (all successful numerical roots, not validated designs):

| Directory under reference/architecture_stage_smoke | Roots / RAW trials | Result |
|---|---:|---|
| 2026-11-11-flow-sized | 3 / 57 | A/B MAIN3 and C GG3/FPT1 close 110 kW; original stage controls violate known engineering checks |
| 2026-11-11-operating-points | 4 / 77 | Lower-speed A/B and independent C GG2/FPT1 controls; stress/angles trade-offs retained |
| 2026-11-11-hub-sensitivity | 2 / 40 | A/B 72000 rpm, hub 0.080 m; angle and reaction-table-domain violations retained |
| 2026-11-11-speed-hub-boundary | 2 / 39 | A/B 79000 rpm, hub 0.070 m; first-stage exit-angle violations retained |
| 2026-11-11-speed-hub-refinement | 2 / 37 | A/B 78000 rpm, hub 0.070 m; the four audited profiles pass and source l/D stays in domain |

Total: 250 retained sizing trials / 13 roots, not a global sweep. Each run reproduced
all its records. The operating-point run resumed with zero appends and reproduced
77/77. Original independent 12/12 RAW, 12/12 clipped, fixed-flow coupled 15/15 and
default sized 57/57 journals were also rechecked read-only with the current evaluator.
Computational source changes intentionally prevent resuming older source-locked runs;
never rewrite their manifests. Use a new destination for a changed implementation.

Default sized roots: A air 0.7067594631 kg/s → 110000.068091 W; B air
0.7019687615 → 110000.048204 W; C air 0.7111105240 → 109999.997604 W.
Refined A/B at 78000 rpm / 0.070 m: air 0.7091406169 / 0.7043527123 kg/s,
power 110000.000520 / 109999.950304 W. This is screening evidence for four checks
only: RHO065, prompt stage pi=1.8–2.4, textbook p.38 alpha2, assumed sigma<=290 MPa.
It is not full feasibility, an optimum, a material-life assessment or an architecture ranking.

`audit_sized_stage_roots.py` independently recomputes source compressor arithmetic,
actual stage powers, sized compressor targets, mass balance, pressure residuals and
actual GG→FPT T/p/flow continuity. It never reruns meanline or mutates RAW. Final
reports are `*-stage-audit-v2.json`; earlier `*-stage-audit.json` reports are retained
as preliminary v1 artifacts. Overall feasibility remains null/NOT_CHECKED. Original
default A/B stresses exceed 290 MPa; C also violates angles/pi and its FPT l/D table
domain. Existing table clipping is explicitly reported, not silently validated.

Independent reviews PASS for flow sizing, per-role operating points/case provenance,
and the final v2 audit after corrections. Parent's last completed suites before the
latest operating-point/audit additions: 536 engine tests OK (1 GPU skip), 161 scripts
tests OK. Final full suites are running; do not infer their result from agent reports.
No commit/push performed for this continuation.

Next P0/P1: extend source-backed stage work distribution and bounded pi/split coverage
on the actual sized model; keep C's remaining violations explicit. Engineering maps,
cooling, material/thermal/resource and radial validation are still missing and must not
be invented. No production/global optimizer or validated-engine claim is authorized.

### Historical fixed-flow continuation (superseded only where stated above)

### Latest bounded actual-boundary RAW continuation

`scripts/architecture_coupled_stage_smoke.py` now evaluates 15 explicit alternatives:
A/B MAIN with 1/2/3 stages, and C GG 1/2/3 × FPT 1/2/3. It uses existing pressure/work
closures and fresh namespaces, with exact actual GG→FPT total-state/flow continuity.
There is no flow, geometry, speed or bracket fitting. All attempts and complete
successful closure stages are persisted, including partial GG results on typed FPT
failure. This is boundary coupling, not achieved engine power closure.

Parent saved `reference/architecture_stage_smoke/2026-11-11-actual-coupled/`:
13 SUCCESS + 2 NUMERICAL_FAILURE (A/B MAIN/1, boundary_not_bracketed within the
explicit 120000–150000 Pa static bracket). These are not physical impossibility
claims. Reproduce 15/15; resume zero appends. Journal: 185924 bytes, SHA-256
`2277f4e544954c68a259f46f94be30bddddc11ac81d1fe4b0793acb3ca1e8fa2`.

No successful alternative achieves 110 kW at the supplied flow/controls. A MAIN/3
net output is 104894.03 W, B MAIN/3 is 104880.83 W. C GG2/FPT1 yields 108141.98 W;
C GG3/FPT1 remains 106095.69 W. These are fixed illustrative controls, not a ranking
of valid designs. Pi-profile violations remain annotations; overall feasibility is
NOT_CHECKED. The earlier independent 12-point journals are untouched.

Parent checks: 536 engine tests OK (1 GPU skip), 146 scripts tests OK;
static verifier 6 PASS / 7 WARN / 0 FAIL; diff check clean. Independent reviewer PASS
after replacing Python-3.11-only file hashing with streamed hashing (3.10 runtime
itself not exercised; tests ran on 3.12.10). New files: coupled smoke and its tests.
No commit/push. Checklist remains 54 unclosed items: bounded evidence does not close
stage selection, engineering checks or 110 kW engine closure.

Next P0: investigate explicit physical static brackets for A/B MAIN/1 and the
remaining power mismatch using source-supported work distribution/control alternatives;
retain all attempts, avoid silent mass-flow resizing and do not start a global sweep.

### Previous single-link diagnostic

Latest actual-boundary continuation: `ConstHubOrchestration` now closes actual
outlet total pressure or prescribed mechanical work using an explicit static-pressure
bracket, without changing flow, geometry or speed. Both require `raw_unbounded`;
the original namespace pressure is restored even on failure/interrupt.

Lead independently checked A MAIN/3: target 150000 Pa, actual 150000.036502 Pa
(relative residual 2.43347e-7). C GG/3 closes compressor power 200200.924926 W
at 200201.071151 W (7.30389e-7). Its actual outlet, 1009.505001 K / 281298.344088 Pa,
is now explicitly passed into a fresh FPT/1 meanline regression. FPT outlet closes
at 149999.898810 Pa (6.74598e-7), but mechanical output is 106095.685758 W:
**3904.314242 W below 110 kW**. This is a verified linkage, not a closed 110 kW engine.
Full successful closure stage records and inputs are saved separately in
`reference/architecture_stage_smoke/2026-11-11-actual-link-diagnostic.json`.
The older 12-point journals remain independent assembly snapshots and are unchanged.

Latest parent checks after the linkage regression: 536 engine tests OK (1 GPU skip),
131 scripts tests OK; static verifier 6 PASS / 7 WARN / 0 FAIL; `git diff --check`
clean. Independent reviewer PASS for the linkage regression, diagnostic and docs.
No commit or push was performed for this continuation.

The paragraph below describes the preceding snapshot; its outstanding coupling
statement is superseded only for this single fixed C GG/3 → FPT/1 regression.

Latest accepted step-3/M3 continuation: shared `baseline.estimate_stage_count`
implements original part-1 p.87 item 54 without rounding or selecting an integer.
The existing const-hub meanline now has an explicit `raw_unbounded` secant policy;
it does not read the engineering pressure-ratio cap. Legacy remains clipped by default.
Pi-profile margins/pass are separate from unknown overall engineering feasibility.

Saved assembly checks: `reference/architecture_stage_smoke/2026-11-11-raw/` and
`2026-11-11-legacy-clipped/`. Four assemblies × stage counts 1/2/3 = 12 attempts
per run. RAW: 12 SUCCESS, 12/12 reproduce, resume zero appends. Reference: 7 SUCCESS
+ 5 NUMERICAL_FAILURE, 12/12 reproduce. These five stalls were caused by the
engineering pressure-ratio clip, not proof that the corresponding stages cannot
be calculated. High-pi successes are retained and fail the separate pi profile.

Latest parent checks: 523 active engine tests OK (1 optional GPU skip), 131 scripts
tests OK; static verifier 6 PASS/7 WARN/0 FAIL; diff check clean. Independent reviews
PASS for the count/meanline runner and RAW policy after digest/acceptance fixes.

No stage count or best architecture is selected. These assembly runs are labelled
NOT_COUPLED_ARCHITECTURE_STAGE: prescribed static pressure was derived from the 0D
total outlet using an explicit fixture Mach, not the solved meanline velocity.
Actual total-pressure/power mismatches are saved; coupling those actual assembly
outlets/work back into the cycle is the remaining P0 item. See PROJECT_GUIDE and
the new section in ENERGY_METHOD_AUDIT for measured mismatches.

### Prior aggregate/RAW-v2 checkpoint (historical results below)

Latest accepted continuation: textbook process means are now wired through the
existing compressor/turbine components and A/B/C architecture calculation. Component
records separate process cp/k from local endpoint properties; GG outlet is the exact
FPT inlet, with independent GG/FPT process means. Source method requires fuel-based
alpha and the retained mean-cp combustor relation. External composition is not inferred.

RAW input/output v2 now preserves `gas_property_basis` and complete process metadata;
v1 payloads/output dictionaries still describe the unchanged legacy model. New saved
grid: `reference/architecture_smoke/2026-11-11-textbook-v2/`, 13 attempts: 9 SUCCESS,
3 MODEL_ERROR (deliberately incompatible external-state fixtures), 1 NON_PHYSICAL_STATE
(nonpositive-output fixture). All 13 reproduce; resume appended zero duplicates.
Journal SHA-256: `febe2ab92a3dfaf868a280a5c6d684ad7cf2749815d276a787739845a5681a87`.
Maximum target-power residual 8.73115e-11 W; mass residual 0 kg/s. NOT_PHYSICALLY_VALIDATED.

Latest parent suites: 510 active engine tests OK (1 optional GPU skip), 116 scripts
tests OK; static verifier 6 PASS / 7 WARN / 0 FAIL; diff check clean. Independent
reviews PASS for components (after residual/oracle fixes), A/B/C wiring and RAW v2.
Mandatory continuation is source-backed stage inventory/work distribution and
connection to existing meanline/geometry/strength calculations, not another gas model.

### Prior single-cycle checkpoint (historical results below)

Latest accepted P0 continuation: `cycle.calculate_cycle_point(...,
gas_property_basis="textbook_true_cp")` implements the opt-in process-mean textbook
variant in the existing calculation. The p.20 fuel formula and configured cooling
fractions are retained; all gas properties use fuel-based alpha; turbine/nozzle
temperatures are refined and local exit k is consistent with both Mach and pressure
conversion. Invalid mass bookkeeping is diagnosed, not normalized or accepted.

`optimize_extended.build_ns(..., gas_property_basis="textbook_true_cp",
cycle_gas_property_basis="textbook_true_cp")` connects it to existing stage equations.
Default anchors remain unchanged. New cycle/source-state diagnostics and actual basis
metadata propagate to successful and failed stage records. Independent reviews PASS
for cycle closure (after exhaustion-residual fix) and combined wiring.

Latest parent checks: 460 active engine tests OK (1 GPU skip), 116 scripts tests OK;
static verifier 6 PASS / 7 WARN / 0 FAIL; diff check clean. Fixed B87_relaxed point:
cycle G=0.532339684442 kg/s versus legacy 0.577155550361; first stator height
11.031705 mm versus 11.295345 mm; both remain infeasible. No search/sweep was run.

Earlier remaining step-2 scope (subsequently implemented above): per-aggregate process properties in A/B/C architecture
components (including independent GG/FPT process means) and source-backed stage
inventory. The previously saved 13/55-point datasets still describe their labelled
legacy reduced model; they were not overwritten or promoted to textbook results.

### Earlier scope-correction checkpoint (historical results below)

User instruction: follow the cycle textbook and turbine textbook part 1; do not
develop an unnecessary alternative integral cycle. Mandatory route is now the
ordered P0/P1 sequence in `AUTONOMOUS_EXECUTION.md`. The original prompt is preserved.

Audit outcome: keep RAW, reference anchors, diagnostics, shaft/architecture contracts
and bounded experiments. Freeze integral `caloric*` work in `experiments/caloric/`.
It is no longer imported by the active calculation. Its optional 45-test suite has
one known unaccepted-combustor one-ULP assertion failure; this is disclosed, not fixed
by loosening a tolerance and not counted as an active-suite success.

Completed necessary P0 continuation: published cycle-book p.57 true-cp correlation
in existing `gas.py`; explicit `gas_property_basis="textbook_true_cp"` in existing
const-hub `build_ns`. Stage gas, stage cp/k iteration and pressure closure use the
same selected basis. Construction-bound provenance rejects a relabelled namespace.
Legacy defaults/anchors are unchanged; cycle still uses legacy mean cp, reported
separately. No new combustor, integral-cycle integration, search or large sweep.

Latest parent verification: 443 active engine tests OK (1 optional GPU skip),
107 scripts tests OK; static verifier 6 PASS/7 WARN/0 FAIL; diff check clean.
Independent gas-property and stage-wiring reviews PASS after provenance fix.
Fixed B87_relaxed control point remains infeasible in both methods: blade height
11.295345 mm legacy versus 11.153120 mm textbook stage option. This is a method
comparison, not a validated engine or a new optimum.

Remaining P0 work is the existing cycle's process-property/fuel/cooling reconciliation
and source-backed stage-count/meanline integration, not an integral thermochemistry
replacement. Absent maps/material life remain affected-check limitations rather than
reasons to halt every supported calculation.

## Historical continuation evidence — superseded route notes below

Subsequent P0/P1 evidence: `ENERGY_METHOD_AUDIT.md`, saved independent energy audit,
and `current_host_profile_2026-11-11.json`. Original cycle-book pp.20/54/56/57 visually
checked. Typed architecture failure propagation corrected; reviewer PASS. Experimental
`caloric.py` source true-cp units passed the subsequent user-requested correction:
eta*R underflow and component-power overflow now have typed diagnostics; independent
reviewer PASS. Full architecture integration and physical validation remain open.
Latest parent suites: 466 engine tests OK (1 GPU skip), 94 scripts tests OK;
static verifier 6 PASS/7 WARN/0 FAIL; git diff --check clean.
Reconciled checklist: 134 task bullets, 79 done, 23 in progress, 31 TODO,
1 justified skip. 54 remain open; this is not an effort/physical-readiness percentage.
Independent completed work: silent gas fixed-point exhaustion now raises coherent
typed diagnostics (success parity unchanged; reviewer PASS). Saved bounded coarse
experiment `reference/architecture_coarse/2026-11-11-pi3-13/`: 55 attempts, all
SUCCESS, 55/55 reproduced, resume zero appends; reviewer PASS. It is illustrative,
not production: stage rho/strength NOT_CHECKED. See `COARSE_0D_EXPERIMENT.md`.

- `architecture_raw.py` now connects both explicit combustor closures and all three
  0D architectures to immutable RAW, with strict versioned JSON and typed failures.
- Saved grid: `reference/architecture_smoke/2026-11-11-grid/` — 13 attempts,
  12 SUCCESS and 1 NON_PHYSICAL_STATE; all retained. Reproduction matched 13/13;
  resume appended zero duplicates and preserved the journal hash.
- Independent checks: 427 engine tests OK (1 GPU skip), 43 scripts tests OK;
  static verifier 6 PASS / 7 WARN / 0 FAIL. Independent reviewer: PASS after fixes.
- Interrupted runs remain incomplete; actual interruption/resume tests compare
  journal, manifest and checkpoint with uninterrupted execution byte-for-byte.
- This closes bounded RAW integration, not M2 physical validation or M6 production.
  Next P0: independent energy/enthalpy audit of retained fuel, alpha and cp/k
  assumptions, then stage-level physical closure. See `ARCHITECTURE_RAW_SMOKE.md`.
- Newly identified SOURCE_DISCREPANCY: FPT inherits GG outlet cp/k rather than
  refreshing them at its lower inlet temperature, despite module prose implying
  per-turbine inlet evaluation. Confirmed in the saved RAW states; no silent fix.

## Earlier checkpoint context

M2 records the user-confirmed 110 kW
**pre-reducer (gearbox-input)** power plane and applies it consistently to the active
cycle, stage baseline, and performance aggregation. The standalone 0D turbine-expansion
unit (`turbine_engine/turbine.py`, now with a prescribed-work inverse), scalar
combustor heat-addition wrapper (`turbine_engine/combustor.py`), standalone
architecture shaft power balance (`turbine_engine/shaft.py`), and opt-in
architecture reference cycles (`turbine_engine/architecture_cycle.py`) remain in
place with focused tests. Prior dirty paths:
`.gitignore`, `Optimizer/README.md`, `Optimizer/optimize_extended.py`,
`Optimizer/optimizer_core.py`, `README.md`; prior untracked: `Super-Promt.md`,
`Super-Promt-Autonomous.md`, `pyproject.toml`, `scripts/`, `turbine_engine/`.
Current milestone: M2 — Unified physical core + architectures
Current task: the confirmed power-plane correction remains implemented in
`Turbinis-Cycle/Turbinis-Cycle.ipynb`, `turbine_engine/cycle.py`,
`turbine_engine/baseline.py`, `turbine_engine/performance.py`, and the active optimizer
adapters. A standalone explicit architecture shaft power balance helper is in
`turbine_engine/shaft.py`, and `turbine_engine/architecture_cycle.py` now provides
opt-in executable 0D reference cycles for all three selected topologies. The explicit
variant is `uncooled_no_bleed_0d_ideal_mixing`; it closes A/B MAIN shafts and the C
GG/FPT shafts, passes the actual GG outlet state to the FPT, sizes flow for the
pre-reducer 110 kW target, and reports `MISSING_COMPRESSOR_PHYSICS` plus
`NOT_PHYSICALLY_VALIDATED`. An external combustor-state closure is available when
fuel/state inference is not wanted. The legacy cycle, baseline, and optimizer are not
rewired. Documentation records the three selected architectures, the conceptual
shaft/gearbox/propeller contract, an active-path A/B/C/D early-exit audit, and a
source-backed equation/power register. All cached outputs/execution counts in the two
active source notebooks (`Turbinis-Cycle`, `Turbinis-Calc`) were cleared because downstream
stage/geometry numbers were stale after the closure change; no notebook sweep was run.
Explicit gearbox ratio/propeller load models and compressor/turbine stage inventories
still do not exist.
Current backend: reference Python / legacy notebook path (`stage_core=False`); extracted
`stage_core=True` path is the default for current optimizer evaluation; JAX 0.12.0.1
float64 is installed in the Windows project venv for CPU kernel validation, while the
validated CUDA path is the separate user-local WSL2 venv described in `docs/BACKENDS.md`
Current environment: Windows; AMD Ryzen 5 9600X (6C/12T); 31.1 GB RAM; NVIDIA RTX 4060
8188 MiB (driver 610.88); Python 3.12.10 in `.venv`; NumPy 2.5.4, SciPy 1.18.1,
pandas 3.1.1, pyarrow 26.0.1, JAX/JAXlib 0.12.0.1; WSL2 Ubuntu with a separate
`~/.venvs/turbinis-gpu` CUDA venv

## Current status

M0 bootstrap is complete. M1 source audit is recorded with evidence, but it is not fully
closed: the Sergey calculation is now located (the previous "absent" claim was wrong); its
latest-author-version status and method reconciliation remain open, and compressor stage
inventory, full repository-wide early-rejection classification, current-host profile, and
JAX compatibility remain `[~]`. The active model is a three-stage single-shaft turbine with
`π_k = 11`, `T_g* = 1250 K`, and a one-dimensional book meanline.

The user-confirmed `N = 110 kW` plane is now **gearbox input, before the reducer**. The
active cycle uses `N_ud = L_t*g_sum*eta_mech - L_k` (no `eta_reducer`), `G = N/N_ud`, and
exposes downstream `N_reducer_out_ud = N_ud*eta_reducer`. `C_e`/`eta_e` are explicitly
referred to the pre-reducer target; `N_e_ud` is the equivalent shaft-plus-nozzle metric
built on the downstream gearbox output. The legacy post-reducer numbers are retained only
as a labelled differential regression. This is a formula change requested by the user, not
physics validation.

The first M2 contract layer is present in `turbine_engine/architecture.py`: a frozen
validated `StationState` and explicit topology descriptors for
`SINGLE_SHAFT_SINGLE_CC`, `SINGLE_SHAFT_DUAL_CC` and `FREE_POWER_TURBINE`. The
descriptor documentation now states the conceptual `output shaft -> reduction gearbox ->
propeller` relation for each architecture, while still encoding no gear ratio, gearbox
efficiency, shaft speed, propeller load, pressure split, or aerodynamic stage inventory.
 `turbine_engine/compressor.py` now adds the
first reusable component: a standalone 0D/mean lumped variable-k polytropic compressor
unit/train consuming `StationState`, with explicit finite-positive ratios, preserved mass
flow, and per-unit work/power diagnostics. It reuses the existing `calc_compressor`
equation through one shared helper; the cycle anchor is unchanged and
`calculate_cycle_point` is not rewired to it. It is not a compressor map, bladed-stage,
corrected-flow, speed/diameter, or pressure-split model. `turbine_engine/turbine.py` adds a
standalone 0D/mean lumped total-to-total turbine-expansion unit consuming `StationState`:
it reuses the existing `calc_eta_expansion` and cycle work/drop expression, preserves mass
flow/`cp`/`k`, exposes work, power, efficiency and an unfiltered pressure residual, and
rejects nonphysical outlet states. It is not a bladed stage, stage-inventory, cooling, map,
or architecture cycle; `calculate_cycle_point` is not rewired, and the cycle's coupled
nozzle iteration still owns `pi_t`. `turbine_engine/combustor.py` adds a standalone scalar
heat-addition wrapper: it calls exactly the existing `calc_alpha(T_in, T_gas)` and
`calc_g_m(T_in, T_gas, config)` and records `g_m`, `alpha`, and the inlet/target
temperatures in an immutable `CombustorResult`. It adds no mass-flow conversion, fuel or
composition update, `cp`/`k` state, pressure loss, or separate combustion-efficiency model,
and it does not produce a `StationState`; `calculate_cycle_point` is not rewired. There is
still no free-power cycle, dual-CC cycle, explicit gearbox/propeller component,
3D/radial-equilibrium implementation, RAW dataset, GPU batch engine, or CLI in this
checkout.

`turbine_engine/shaft.py` now provides a standalone explicit architecture power-balance
helper, not a cycle. `evaluate_shaft_power_balance` accepts mass-flow-weighted compressor
and turbine gas powers in W, explicit per-turbine mechanical efficiencies, an explicit
required gearbox-input power, and an explicit gearbox efficiency. It enforces the exact
per-architecture counts and order: `SINGLE_SHAFT_DUAL_CC` sums two series compressor loads
against one MAIN turbine; `SINGLE_SHAFT_SINGLE_CC` has one compressor and one turbine;
`FREE_POWER_TURBINE` closes a gas-generator shaft (`eta_mech_GG*P_GG - P_compressor_GG`)
separately from the free-power-turbine output shaft (`eta_mech_FPT*P_FPT`). Mechanical
efficiency is applied once to turbine gas power; the downstream gearbox output is
`actual_input*eta_gearbox`, not `required_input*eta_gearbox`. Immutable derived properties
return signed, unfiltered MAIN/GG/FPT residuals, so insufficient turbine power is retained
as a negative balance rather than rejected. There are no efficiency defaults and no
accessory/shaft-loss additions. The helper is not wired into `calculate_cycle_point`, the
baseline, or the optimizer.

`turbine_engine/architecture_cycle.py` now supplies the missing executable layer for all
three selected topologies, as an **opt-in reference cycle** rather than a replacement for
the legacy path. The explicit variant is `uncooled_no_bleed_0d_ideal_mixing`: no
cooling/leak/bleed/return flow, one combustor, all compressor air entering it, ideal
mixing to the requested total temperature, total-to-total turbines, and constant turbine
`cp`/`k` at the turbine inlet. `PolynomialGasModel` requires explicit `R`, `l0`,
`q_lhv_kj`, `eta_combustion`, `T_ref`, and an explicit turbine-gas `alpha` basis
(fuel-based `1/(f*l0)` or the empirical `calc_alpha` polynomial). Fuel/air ratio comes
from the retained `calc_g_m`; `ExternalCombustorClosure` accepts an explicit outlet state
and fuel/air ratio instead. A/B close one MAIN shaft with the turbine driving all
compressor units plus 110 kW; C prescribes GG turbine work from compressor power,
inverts the turbine relation for the real GG outlet state, passes that state to the FPT,
expands the FPT to the caller's **total** outlet pressure, and sizes `G_air` separately.
The target 110 kW is pre-reducer; `eta_gearbox=0.98` gives 107.8 kW downstream. Results
carry unfiltered component/shaft/mass residuals, `target_power_status`, and the
limitations register. Tests prove formula parity, equal-ratio dual-compressor pressure
and temperature propagation, GG->FPT state transfer, independent GG/FPT residuals,
parameter recomputation, invalid-domain errors, and the preserved legacy anchor. This is
software/equation closure, not physical validation; compressor maps/blading, stage
inventories, cooling semantics, gear ratio, and propeller load remain `MISSING_*`.

The user-approved `rho <= 0.65` is an **acceptance-only** ceiling. It does not change
physics, does not filter calculated RAW candidates, and does not extend the digitized
Fig. 18/19 graph functions. Fig. 18's dotted edge is graph-frame clipping near
`rho ~= 0.65`; Fig. 19 is limited to `rho <= 0.5`. Do not extrapolate either.

The project-tracking documents and scalar combustor wrapper were independently reviewed
(PASS). M2 topology review
initially found and then verified fixed the ability to construct enum/shaft mismatches;
second review was PASS. Compressor-result review found a consistency gap; thermal-work
checks were added and follow-up review was PASS. Turbine review likewise found a coordinated
temperature/work inconsistency; source-equation validation was added and follow-up review
was PASS. Typed contracts/components are not wired into cycle architecture paths, and
software tests/reviews are not physical validation.

`docs/README.md` and `README.md` describe `docs/engineering/*` as normative, but that
directory is absent. This is an unresolved source-of-truth mismatch. Do not cite absent
authority documents as if they existed. M2 must not silently invent their content.

## Source notebooks and bounded provenance

All four candidate notebooks are present. Full SHA-256, cell counts, git history, overlap
and scope separation are in `docs/comparisons/Источники_и_методики.md`. Summary:

| Case | File | Cells | Git evidence | Bounded role |
|---|---|---|---|---|
| Sergey | `TVD_pi6_design_steps_1_3.ipynb` | 48 | added in `1263ac2` | current working candidate for cycle + design steps 1–3 |
| Sergey | `archive/comparisons/legacy_workbench/TVD_pi6.ipynb` | 83 | `c6223b9` R100 from `_compare/TVD_pi6.ipynb`; originally added in `1263ac2` | historical/comparison variant with expanded turbine-design sections; 38 shared nonempty cells |
| Anton | `reference/published_cases/1Anton-Mathcad_ST_cooling_Jupyter (3).ipynb` | 575 | added in `c6223b9` | preferred detailed/citable case |
| Anton | `Mathcad_ST_cooling_Jupyter.ipynb` | 60 | added in `05d6031` | compact computational companion/reconstruction |

The four working-copy mtimes are all 2026-11-10, and commit dates reflect repository
organization; neither proves author chronology. No external author timestamps or original
provenance records are present, so this state does **not** claim a "latest author version".
The root Sergey notebook is the working candidate for its cycle + design-steps-1–3 path;
the 83-cell archive is a legacy/comparison variant, not a straightforward newer revision.
For Anton, the 575-cell reference is preferred for detailed citation and the 60-cell root
is a compact companion; only 11 identical nonempty source cells are shared. No notebook is
a validated regression baseline.

## Last verified results

- `git rev-parse --short HEAD` → `85e6102`; branch `main`.
- `python scripts/verify.py` (venv) → 6 PASS, 7 WARN, 0 FAIL, exit 0. WARNs: absent
  `docs/engineering/*` (5) and absent `reference/baselines/`; physics NOT verified.
- `python -m unittest discover -s scripts -p "test_*.py" -v` (venv) → 43 tests OK.
- `.venv\Scripts\python.exe -m unittest discover -s turbine_engine/tests -p "test_*.py" -v`
  → 399 tests OK, 1 skipped (Windows GPU-only skip). These are
  contract/software-drift/parity tests, not physical validation.
  The focused `test_baseline_core.py` run is 19 tests OK, including the last-stage boundary
  ratios `1.000001`, `1.01`, `1.02`, `1.020001` (reached the stage stub) and exact/below
  `p_t` rejection.
- The post-guard-change anchor remains unchanged: `scripts.test_optimizer_regression`
  (`B87_relaxed`) and `scripts.test_cycle_point_regression` both pass without recalibration.
- The cycle anchor at `pi=11`, `T_g=1250 K`, `N=110000 W` now gives
  `N_ud=190589.8677 W/(kg/s)`, `N_reducer_out_ud=186778.0704`,
  `G=0.5771555504 kg/s`, `C_e=0.3180573704 kg/(kW*h)`; `N_e_ud` remains
  `201275.2313`. See `scripts/test_cycle_point_regression.py`.
- Single deterministic `eval_candidate(WARM, "B87_relaxed", q_n_limit=0.995)` anchor was
  re-evaluated once after the power-plane change: `status=ok`, `feasible=False`,
  `N=110000 W`, `l_c1=0.011295345088530603 m` (11.295 mm). This is an
  engineering-infeasible point, not a calculation failure and not an optimizer run.
- No new performance timing was taken in this task; prior single-sample timings predate
  the pre-reducer formula change and must not be compared directly.
- Source checks: user calculation, Sergey root/archive notebooks, and Anton root/reference
  notebooks present; book part 1/2 PDFs and scan/Markdown present;
  `reference/published_cases/README.md` says no published case is approved for automated
  validation. Provenance and the new equation/exit audit are in
  `docs/comparisons/Источники_и_методики.md`.

## Current blockers

- `[~]` Sergey free-power reference: the calculation is present as
  `TVD_pi6_design_steps_1_3.ipynb` (48 cells; cycle + design steps 1–3) and as
  `archive/comparisons/legacy_workbench/TVD_pi6.ipynb` (83-cell historical/comparison
  variant). It is a separate two-shaft 150 hp = 110.325 kW, `pi_k = 6`, `T_g = 1250 K`
  case with one compressor-turbine stage and one free-power stage. It is not reproduced,
  not wired, and not a validated baseline. The old `[B]` "source absent" is closed; the
  remaining task is to decide whether/how to reproduce it as an external comparison.
- `[x]` `N = 110 kW` boundary is **confirmed** as mechanical power at the gearbox input
  (pre-reducer). Active cycle, stage baseline, and performance aggregation use the same
  plane; reducer loss is downstream (`N_reducer_out_ud = N_ud*eta_reducer`). Legacy
  post-reducer anchors are retained only as a labelled differential. See
  `docs/comparisons/Источники_и_методики.md` §6.
- `[~]` Compressor stage count/split is not represented; the cycle has only one `π_k`.
- `[x]` Active-path A/B/C/D early-exit classification is now recorded in
  `docs/comparisons/Источники_и_методики.md` §8. It covers `_evaluate` (both stage paths),
  `eval_candidate`, cycle, stage, baseline, performance, orchestration, expansion, gas and
  reaction. The legacy 2 % last-stage pressure margin `p0 <= 1.02*p_t` was D pending
  justification; the checked textbook equation (printed p. 92, items 138–140) supports only
  `p0 > p_t`, so the margin was removed from the extracted module and active notebook
  cell 16. That specific D finding is closed; other D items (`(None, None)`/`ст_b is None`
  returns and silent non-convergence) remain open. The earlier categorical “no harmful C
  gate” claim is retracted. It is not a repository-wide audit; archive/scan/exploratory
  paths are excluded with rationale. M3 still needs structured diagnostic records for
  A/B/D failures.
- `[~]` No current-host multi-candidate performance profile; the existing
  `Optimizer/results/PERF_AUDIT.md` is on a Ryzen 7 PRO 5850U and is not comparable.
- `[x]` JAX CPU and WSL2 RTX 4060 GPU numerical self-tests now pass for the bounded
  gas-property kernel at float64 (`docs/backend_validation.json`). This is not an engine
  speedup claim; the tested GPU kernel was slower than NumPy for the small batch.
- `docs/engineering/*` normative files are absent (source-of-truth mismatch).
- `[x]` RAW infrastructure now exists (`raw.py`, `dataset.py`, `sampling.py`,
  `turbine_engine/constraints.py`) with JSONL journal, manifest, checkpoint/resume,
  Parquet export, and reproduce tests. No production RAW dataset has been generated yet;
  prior `Optimizer/results/` CSV/JSON remain historical optimizer output.

## Important unresolved engineering questions

- How many compressor stages/units does the active model actually represent, and where is
  the compressor split for a two-CC architecture?
- What are the real shaft/gearbox/propeller layout, gear ratio, gearbox efficiency beyond
  `eta_reducer`, and propeller load? Only the topology statement and `eta_reducer` exist
  today.
- Is the Anton two-shaft/free-power case a valid reference? The Baikal-like 750 kW-after-
  reduction, two-stage free-turbine case is present; the 575-cell
  `reference/published_cases/... (3).ipynb` is the preferred detailed/citable copy and the
  root 60-cell notebook is a compact companion. It is still not quality-approved; its
  `rho` inputs are mean values `.350`/`.415` with local reaction calculated through the
  span.
- Which Fig. 18 dotted-curve points are inside the visible frame, and how should an
  acceptance ceiling of `rho <= 0.65` be annotated without changing graph interpolation?
- Material, metal temperature, creep/resource, thermal stress, and the `σ <= 300 MPa`
  limit remain `NOT_CHECKED`/preliminary (`Turbinis-Calc/PENDING.md`).
- Do not treat the existing 0-feasible-result optimizer runs as proof that no feasible
  region exists; they are not exhaustive.

## Next executable task

Continue M2/M3 integration: connect `architecture_cycle.py` results to the generic RAW
record/evaluator API through an explicit adapter, then run a small labelled smoke grid
(not a production sweep) to verify persisted inputs, mass/energy/shaft residuals, failure
classification, manifest and resume behavior. Priority remains the compressor stage
inventory and `cp`/`alpha`/cooling assumptions that the reference cycles deliberately
leave as `MISSING_*`; do not let the zero-dimensional closure be presented as a validated
stage design. Keep the legacy `calculate_cycle_point` path and its anchors unchanged. Do
not invent stage counts, shaft speeds, gear ratios, gearbox efficiencies, propeller loads
or flow splits. Decide whether the Sergey 110.325 kW two-shaft case is reproduced only as
an external comparison; do not treat either Sergey file or either Anton file as an
approved regression baseline. Do not run a global optimizer, DE, or production sweep
until the method gates are recorded as resolved.

## Latest datasets/runs

- run_id: none (no RAW-schema run exists)
- path: `Optimizer/results/` contains historical CSVs/JSON/logs from prior optimizer work;
  they predate the RAW schema and are not a production design-space dataset
- config: not applicable to the new checklist/state tracking
- status: historical only; do not treat as M3/M5 output

## Latest regression status

- `scripts`: 43/43 OK (verified in the current session; includes the concurrent
  RAW-pipeline regression tests).
- `turbine_engine/tests`: 399/399 OK (one skipped), including the separate
  RAW/analysis/backend pipeline tests from the concurrent workstream plus architecture contracts,
  compressor, combustor, turbine (forward and prescribed-work inverse), the standalone
  shaft power balance, the three architecture reference cycles, cycle power-plane,
  baseline (including the last-stage pressure boundary), performance and wiring tests.
- Focused architecture cycles: `test_architecture_cycle.py` passes 27 tests; A/B close
  at `110000 W` pre-reducer and `107800 W` with `eta_gearbox=0.98`; C passes the actual GG
  outlet state to the FPT and closes the GG/FPT residuals independently.
- Focused shaft balance: `test_shaft.py` → 25 tests OK. The current-cycle
  `SINGLE_SHAFT_SINGLE_CC` anchor closes at `110000 W` actual gearbox input and
  `107800 W` downstream output; the helper itself is not wired into the cycle.
- Static verifier: 6 PASS, 7 WARN, 0 FAIL (warnings are the absent `docs/engineering/*`
  authority files and absent `reference/baselines/`).
- Reference path vs extracted path parity is covered by the test suite; this is a
  software-drift check, not an independent physics validation.
- The cycle and optimizer anchors were intentionally re-baselined for the pre-reducer
  power plane. The old post-reducer cycle values remain in a labelled legacy differential
  test; the old optimizer candidate anchor was re-evaluated once, not re-optimized. The
  last-stage guard change did not alter either anchor.

## Latest performance status

- No new timing was taken after the pre-reducer formula change. Prior single samples
  (`_evaluate` and `eval_candidate`) predate it and are not current performance claims.
- The re-evaluated optimizer anchor is correctness data only: `status=ok`,
  `feasible=False`, `l_c1=11.295 mm`; no wall-time claim.
- Prior `PERF_AUDIT.md`: cProfile on Ryzen 7 PRO 5850U, three candidates; hot functions
  `cp_k_ступени`, `газ`, `ступень_const_hub`; not comparable to the current host.
- Bounded gas-property kernel: NumPy median 0.660 ms, JAX CPU median 0.320 ms,
  JAX GPU median 1.252 ms at a 65,536-element batch; compile/warm and steady samples are
  separated in `docs/backend_validation.json`. This is one kernel, not an engine
  benchmark, and the tested GPU path was slower than NumPy. No production backend choice
  or throughput claim is made.

## Milestone gate (prompt §40)

Full project milestone gate has not been run. Latest component/regression checks are:

```text
PHYSICS:            not reviewed here; power-plane change is user-requested, not validation
NUMERICS:           pre-reducer cycle/stage/performance target is consistent in tests
REGRESSION:         software parity passes (399 + 43 tests)
DATA INTEGRITY:     RAW infrastructure tested synthetically; no production dataset
PERFORMANCE:        bounded gas-property kernel benchmarked; no engine benchmark
DOCUMENTATION:      checklist/state/guide + source/method/equation/power-plane/exit audit
NEXT MILESTONE SAFE: YES for RAW-to-reference-cycle integration and a small labelled smoke
grid; NO for physical architecture comparisons or broad sweeps until the MISSING
compressor/turbine/cooling physics and remaining P0 source/method definitions are explicit.
```

Do not mark M1 `[x]` until the remaining `[~]` items are explicitly accepted or resolved.

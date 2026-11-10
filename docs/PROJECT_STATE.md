# Project State

Last updated: 2026-11-11
Git commit: `85e6102` — `docs(exports): add turbine calculation workbooks` (branch `main`)
Working tree: dirty from prior work. M2 now also records the user-confirmed 110 kW
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

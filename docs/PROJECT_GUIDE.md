# How Turbinis-Calc Works

Last updated: 2026-11-11
Describes: the actual checkout at `85e6102` plus the pre-existing dirty working tree.
Does not describe: the target architecture in `Super-Promt-Autonomous.md` as if it existed.

## 1. Purpose

The checkout calculates and searches a **three-stage, single-shaft, constant-hub turbine**
for a TPE331-class engine study:

- design point: `N = 110 kW` at the gearbox input (pre-reducer), `T_g* = 1250 K`,
  `pi_k = 11`;
- one-dimensional book meanline (`book_approx`), not a 3D or validated radial-equilibrium
  model;
- current optimizer work explores `n`, hub diameter, stage work/angle distribution and
  strength margins around that point.

The master prompt `Super-Promt-Autonomous.md` sets a larger target: three architectures
(single-shaft one-CC, single-shaft two-CC, free-power turbine), a RAW design-space dataset,
JAX/RTX 4060 acceleration, and analysis/Pareto/finalists. Most of that target is **not
implemented**. This guide separates what runs today from what is only planned.

## 2. Actual engine architecture

| Item | Actual checkout | Evidence / limitation |
|---|---|---|
| Shafts | One shaft | `z = 3`; common `n`; no second shaft |
| Turbine stages | 3 | `Turbinis-Calc/Ограничения_проекта.md` §2; `optimizer_core.validate` checks `len(st) == 3` |
| Compressor | Active cycle uses one pressure ratio `pi_k = 11`; a standalone 0D lumped unit/train component exists but is not wired into the cycle. No stage list, map, or compressor split. Topology descriptors distinguish 1 vs 2 conceptual centrifugal units. | `turbine_engine/cycle.py`; `turbine_engine/compressor.py` (standalone 0D); `turbine_engine/architecture.py` (topology only) |
| Combustor | Lumped cycle model with `sigma_combustion`, `eta_combustion`, fuel flow. A standalone scalar heat-addition wrapper (`combustor_heat_addition`) reuses `calc_alpha`/`calc_g_m`; it adds no mass-flow, composition, pressure-loss, or separate efficiency model and is not wired into the cycle. | `turbine_engine/cycle.py`; `turbine_engine/combustor.py` |
| Turbine | Stage meanline + const-hub 3-stage orchestration; a standalone 0D lumped total-to-total expansion unit consumes `StationState` but is not wired into the cycle | `turbine_engine/stage.py`, `orchestration.py`, `baseline.py`, `turbine.py` |
| Gearbox | `eta_reducer = 0.98` is a downstream loss only; the 110 kW target is at the gearbox input (pre-reducer). No ratio, gearbox component, or gearbox-efficiency model beyond `eta_reducer` | `turbine_engine/cycle.py` (`N_ud`, `N_reducer_out_ud`), `turbine_engine/baseline.py`, `turbine_engine/performance.py` |
| Propeller | No load model; `eta_propeller` is used only in the auxiliary `N_e_ud` thrust branch on the downstream gearbox-output plane | `turbine_engine/cycle.py` |
| Free power turbine | Opt-in 0D reference cycle, not validated | `turbine_engine/architecture_cycle.py`: GG work prescribed by compressor power, actual GG outlet state passed to the FPT, FPT expanded to a prescribed total outlet pressure, flow sized for 110 kW; no stage/map/cooling model |
| Dual-CC single shaft | Opt-in 0D reference cycle, not validated | `turbine_engine/architecture_cycle.py`: two explicit series ratios with actual per-unit inlet states, one combustor, one MAIN turbine driving both compressors plus 110 kW; no stage split |
| 3D / radial equilibrium | Not implemented in this checkout | `README.md` “Model status”; `turbine_engine/README.md` limitations |

The active notebook describes this as a three-stage TPE331-class turbine. “TPE331-class”
is a size/architecture analogy, not a validated model of a specific engine.

## 3. Repository and source map

| Path | Actual role |
|---|---|
| `Super-Promt-Autonomous.md` | User's latest master prompt and checklist skeleton |
| `Super-Promt.md` | Previous prompt; superseded, retained |
| `Turbinis-Cycle/Turbinis-Cycle.ipynb` | Cycle boundary conditions and cycle outputs |
| `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb` | Legacy `book_approx` meanline notebook (source of the extracted stage/baseline cells) |
| `Turbinis-Calc/cycle_link.py` | Loads the cycle namespace for the optimizer/model |
| `Turbinis-Calc/constraints.py` | Single source of book constraints and digitized Fig. 18/19; `BOOK_CONSTRAINTS`, `DESIGN_ASSUMPTIONS`, `stage_checks` |
| `Optimizer/optimizer_core.py` | Model loader, cycle wrapper, candidate evaluation, validation, search helpers |
| `Optimizer/optimize_extended.py` | Const-hub extended search, scenarios, WARM point, `build_ns` |
| `Optimizer/scan_*.py`, `stage6*.py` | Historical scans/diagnostics; not part of the new RAW pipeline |
| `turbine_engine/architecture.py` | `StationState` (total `T`, total `p`, mass flow, `cp`, `k`) and explicit `EngineArchitecture` topology descriptors/registry for single-shaft single-CC, single-shaft dual-CC and free-power turbine; topology contracts only |
| `turbine_engine/architecture_cycle.py` | Opt-in explicit 0D reference cycles for all three topologies. `uncooled_no_bleed_0d_ideal_mixing` closes A/B MAIN shafts and C GG/FPT shafts, sizes flow for the pre-reducer 110 kW target, passes the actual GG state to the FPT, and exposes fuel/mass/shaft residuals plus `MISSING_COMPRESSOR_PHYSICS`/`NOT_PHYSICALLY_VALIDATED`. External combustor-state closure available. Not a physical validation, map, stage inventory, cooling, gearbox ratio, or propeller model; legacy cycle not rewired |
| `turbine_engine/compressor.py` | Standalone 0D/mean lumped variable-k polytropic compressor unit/train consuming `StationState`; explicit finite-positive ratios, preserved mass flow, per-unit/total work and power, ratio product and unfiltered residual. Not a compressor map, bladed-stage, corrected-flow, speed/diameter, or pressure-split model; not wired into the active cycle |
| `turbine_engine/combustor.py` | Standalone scalar heat-addition wrapper over `cycle.calc_alpha` and `cycle.calc_g_m`; immutable `CombustorResult` with scalar inlet/target temperatures, `g_m`, and `alpha`. No mass-flow conversion, fuel/composition update, `cp`/`k` state, pressure loss, or separate combustion-efficiency model; not a geometry/map/state model and not wired into the active cycle |
| `turbine_engine/turbine.py` | Standalone 0D/mean lumped total-to-total turbine-expansion unit consuming `StationState`; explicit expansion ratio `p_in_total/p_out_total`, preserved mass flow/`cp`/`k`, specific work, power, efficiency and unfiltered pressure residual. Reuses `cycle.calc_eta_expansion` and the cycle work/drop expression; rejects nonphysical outlet states. Not a bladed stage, stage-inventory, cooling, map, or architecture cycle; not wired into the active cycle |
| `turbine_engine/shaft.py` | Standalone explicit architecture shaft power balance (not a cycle). Takes mass-flow-weighted compressor/turbine gas powers in W, explicit per-turbine mechanical efficiencies, required gearbox-input power and gearbox efficiency; enforces exact per-architecture counts/order and returns immutable derived MAIN/GG/FPT net powers, raw signed residuals, actual gearbox input, and downstream output. No map, stage split, speed, gear ratio, propeller load, efficiency defaults, or cycle wiring |
| `turbine_engine/cycle.py` | Stateless cycle core from `Turbinis-Cycle` |
| `turbine_engine/gas.py` | `gas(T)`, `cp_k_stage(T0,H0)`, `H0_по_p` / `calc_h0_for_pressure` |
| `turbine_engine/stage.py` | Scalar stage meanline core from notebook cell 12 |
| `turbine_engine/reaction.py` | Root-reaction relation transcript |
| `turbine_engine/orchestration.py` | Const-hub stage/turbine/π-distribution functions |
| `turbine_engine/baseline.py` | Patched notebook cell-16 multi-stage orchestration; last-stage guard is `p0 <= p_t` (no positive drop), with the former 2 % margin removed |
| `turbine_engine/expansion.py` | Turbine expansion iteration and preliminary sizing |
| `turbine_engine/performance.py` | Cell-20 aggregate performance |
| `turbine_engine/geometry.py` | Cell-22 meridional flowpath/widths |
| `scripts/verify.py` | Static structure/notebook verifier; does not execute physics |
| `scripts/test_*.py` | Regression/parity anchors |
| `TVD_pi6_design_steps_1_3.ipynb` | Sergey two-shaft 150 hp = 110.325 kW, `pi_k = 6`, `T_g = 1250 K` case; 48-cell working candidate for cycle + design steps 1–3; external example, not a baseline |
| `archive/comparisons/legacy_workbench/TVD_pi6.ipynb` | 83-cell historical/comparison variant of the Sergey case; expanded turbine-design sections; 38 shared nonempty cells with the root; not a proven newer author revision |
| `Mathcad_ST_cooling_Jupyter.ipynb` | Anton 60-cell compact computational companion for the 750 kW-after-reduction, two-stage free-turbine case; not the preferred citation copy |
| `reference/published_cases/` | Anton retained external cases: 575-cell `1Anton-Mathcad_ST_cooling_Jupyter (3).ipynb` preferred for detailed citation, 60-cell root companion; no approved automated validation case |
| `Turbinis-Calc/*.pdf`, `book-scan.md`, `images/`, `photos/` | Book part 1/2 PDFs, scan text/images |
| `turbine_prompt_assets/` | Page scans, digitized Fig. 18/19 CSVs, overlays |
| `Optimizer/results/` | Historical optimizer CSVs/JSON/logs/plots, not a RAW dataset |
| `exports/`, `archive/` | Exported workbooks/PDFs and archived diagnostics; historical |

## 4. Calculation flow and call graph

The current execution path is:

```text
Turbinis-Cycle/Turbinis-Cycle.ipynb
  → Turbinis-Calc/cycle_link.load_cycle()
  → Optimizer/optimizer_core.cycle_point()
  → turbine_engine.cycle.calculate_cycle_point()
       compressor → combustor/fuel → turbine expansion → G (pre-reducer target),
       N_ud, N_reducer_out_ud, p_t, T_t
  → Optimizer/optimizer_core._evaluate()
       executes Main_Turbinis_Calc_Book.ipynb parameter/core cells
       installs extracted adapters:
         cell 7  gas       → turbine_engine.gas
         cell 9  expansion → turbine_engine.expansion
         cell 12 stage     → turbine_engine.stage
         cell 14 reaction  → turbine_engine.reaction
         cell 16 baseline  → turbine_engine.baseline
         cell 20 performance → turbine_engine.performance
         cell 22 geometry (full=True) → turbine_engine.geometry
  → candidate result: lc1, a2, pi, sigma, u, n_t, D_t, N_val, eta, residuals
  → Optimizer/optimizer_core.validate()
       own project/book checks + Turbinis-Calc/constraints.stage_checks()
       (PASS / WARNING / FAIL / NOT_CHECKED; hard/project/recommendation/target/info)
```

`Optimizer/optimize_extended.eval_candidate()` takes the same namespace, installs the
const-hub functions from `turbine_engine.orchestration`, scales `G` to the 110 kW
pre-reducer target, and returns engineering violations without throwing away the candidate.
It is a scalar evaluation, not a batch/GPU kernel.

Legacy/independent paths:

- `Optimizer/optimizer_core._evaluate(..., stage_core=False)` executes the patched notebook
  cells in full; this is the independent reference path.
- `Optimizer/optimize_extended.build_ns(..., orchestration_core=False)` keeps the notebook
  const-hub functions as the independent comparison path.
- `scripts/verify.py` only checks structure/notebook JSON; it does not run the model.

## 5. Source of truth — and the current mismatch

There is no single physical-formula module yet. The current de facto sources are:

1. `turbine_engine/cycle.py` for the cycle, sourced from `Turbinis-Cycle.ipynb`;
2. `turbine_engine/stage.py`, `orchestration.py`, `baseline.py`, `expansion.py`,
   `performance.py`, `geometry.py` for the meanline path, sourced from the book notebook;
3. `Turbinis-Calc/constraints.py` for limits and digitized graphs;
4. `Optimizer/optimizer_core.py` for loading/executing and validation, not an independent
   physics copy.

`docs/README.md` and `README.md` say `docs/engineering/CONSTRAINTS.md` and
`docs/engineering/EQUATIONS.md` are normative. **That directory is absent from this
checkout.** `scripts/verify.py` reports five WARNs for it. Do not cite those files as
authority and do not recreate their content from memory. The absence is a source-of-truth
mismatch, not a green light to invent limits.

`README.md` retains stale exact filenames (`Sergey-TVD_pi6_sequential.ipynb`,
`Anton-Mathcad_ST_cooling_Jupyter (2).ipynb`) that are not in the checkout. The Sergey
calculation is present as `TVD_pi6_design_steps_1_3.ipynb` and
`archive/comparisons/legacy_workbench/TVD_pi6.ipynb`; the Anton case as
`Mathcad_ST_cooling_Jupyter.ipynb` and the 575-cell reference copy. Do not infer author
chronology from file mtimes or git commits; see
`docs/comparisons/Источники_и_методики.md`.

The source/method boundary is recorded in `docs/comparisons/Источники_и_методики.md`.
That note includes a selective direct check of the cycle book §§2.2–2.3, 3.1–3.3 and
turbine book §§4, 6, 9, plus evidence from the prior comparison docs. It now also contains
the exact pre-reducer power-plane register (§6), the selected architecture/shaft contract
(§7), and the active-path A/B/C/D early-exit audit (§8). It is not a
complete cell-by-cell validation of the external notebooks; formula/property and
architecture discrepancies remain open. Cases are examples, not approved regression
baselines.

## 6. Physical models actually present

- **Cycle:** variable-`k` compressor temperature iteration, fuel/air ratio polynomial,
  turbine expansion iteration with `sigma_in`, `sigma_combustion`, `pi_nozzle`,
  polytropic efficiencies, lean/return-flow bookkeeping, nozzle thrust branch.
- **Stage:** scalar meanline velocity triangles, losses, `eta_u`, reaction iteration,
  stress estimate, geometry; source-preserving extraction, not a new model.
- **Const-hub orchestration:** 60/30 iteration limits, `1e-7`/`1e-5`/`1e-12` tolerances,
  half-relaxation, `rho = 0.3` start, `D0 = D_hub + 0.02`, interstage pressure correction,
  clipped secant for stage `pi`, `НетРешения` failure handling.
- **Root reaction:** transcription of the notebook relation; `alpha1=const` and
  `Gamma=const` branches. It is not an accepted radial-equilibrium implementation.
- **Geometry:** meridional contour/width bisection (cell 22); only on `full=True`.
- **Constraints:** `BOOK_CONSTRAINTS` and `DESIGN_ASSUMPTIONS`; statuses
  `book_requirement`, `book_recommendation`, `project_assumption`, `derived_check`,
  `not_checked`. Material, metal temperature, creep, resource and thermal stress remain
  `NOT_CHECKED` (`Turbinis-Calc/PENDING.md`).

The book's part 2 pp. 44-45 gives `rho_sr = 0.3...0.35`, `y0 = 0.55...0.6` as indicative
for multistage turbines, with `y0` around `0.55...0.6` for `rho = 0.2...0.45`. The current
project treats these as recommendations/checks, not as a new hard law.

## 7. State, units and flows

**Units (SI unless stated):** `n` rpm; angles degrees; diameters and blade lengths metres;
mass flows kg/s; pressures Pa; temperatures K; work/enthalpy J/kg; power W; stress MPa;
`rho`, `u/c0`, `l/D`, `pi` dimensionless.

`turbine_engine/architecture.py` now defines a frozen, validated `StationState` that stores
only total `T`, total `p`, mass flow, `cp`, `k` and an optional identifier. The first
consumer is `turbine_engine/compressor.py`, a standalone 0D/mean lumped variable-k
polytropic unit/train with explicit pressure ratios and preserved mass flow; the active
cycle still uses `cycle.calc_compressor`, which shares the same private temperature helper,
and `calculate_cycle_point` is not rewired. The compressor train is not a map, bladed-stage,
corrected-flow, speed/diameter, or pressure-split model. `turbine_engine/turbine.py` adds a
standalone 0D/mean lumped total-to-total turbine-expansion unit consuming `StationState`; it
reuses `cycle.calc_eta_expansion` and the cycle work/drop expression, preserves mass
flow/`cp`/`k`, and exposes specific work, power, efficiency and an unfiltered pressure
residual. Its `pressure_ratio` is the expansion ratio `p_in_total/p_out_total`; it is not a
bladed stage, stage-inventory, cooling, map, or architecture cycle, and the active cycle
still owns its turbine pressure ratio through the coupled nozzle iteration.
`turbine_engine/combustor.py` adds a scalar heat-addition wrapper: it calls exactly
`calc_alpha(T_in, T_gas)` and `calc_g_m(T_in, T_gas, config)` and records the scalar
temperatures, `g_m`, and `alpha`. It performs no mass-flow conversion, fuel/composition
update, `cp`/`k` state, pressure-loss calculation, or separate combustion-efficiency model;
`StationState` still has no fuel or composition fields, and `calculate_cycle_point` is not
rewired. `turbine_engine/shaft.py` adds a standalone explicit architecture power-balance
helper: it consumes already mass-flow-weighted W values plus explicit per-turbine
mechanical efficiencies, required gearbox-input power, and gearbox efficiency. It enforces
the exact topology counts/order, applies mechanical efficiency once to turbine gas power,
and returns signed raw MAIN/GG/FPT residuals and downstream gearbox output from the actual
input. `turbine_engine/architecture_cycle.py` consumes this helper for an opt-in
per-architecture reference cycle: A/B close one MAIN shaft, C closes GG and FPT separately
and passes the actual GG outlet state to the FPT. Flow is sized for the pre-reducer 110 kW
target; `eta_gearbox=0.98` gives 107.8 kW downstream. The legacy `calculate_cycle_point`
path still uses the notebook's named variables and stage dictionaries and is not rewired.
The architecture registry describes counts, shaft grouping and the conceptual
output-shaft -> gearbox -> propeller relation only; no speed, gearbox ratio, gearbox
efficiency beyond the explicit input, propeller load, pressure split, stage inventory, or
physical validation is defined. The reference cycles are not a full architecture closure
and carry `MISSING_COMPRESSOR_PHYSICS`.

Cycle definitions in use:

- `pi_k` — compressor total pressure ratio (fixed 11 in the active cycle);
- `pi_t` — turbine total expansion ratio, including inlet/combustor/nozzle recoveries;
- `T_g*` — turbine inlet total temperature (1250 K);
- `N` — requested mechanical shaft power at the **gearbox input, before the reducer**
  (110 kW; user-confirmed 2026-11-10);
- `N_ud` — specific shaft power on that pre-reducer plane;
- `N_reducer_out_ud` — downstream gearbox-output specific power,
  `N_ud * eta_reducer`;
- `G` — cycle air mass flow, computed as `N / N_ud` on the pre-reducer plane;
- `g_m` — fuel/air ratio; `g_t = (1 + g_m)(1 - g_leak - g_cooling)`;
- `g_ohl` / `g_cooling` — cooling bleed; `g_voz` / `g_return` — return flow
  (`0.6 * g_cooling`); `G_c1 = G * g_t`; `G_l_ost = G * (g_t + g_voz)`;
- `eta_mech_turbine = 0.99`; `eta_reducer = 0.98`; `eta_compressor_poly = 0.85`;
  `eta_turbine_poly = 0.90`; `sigma_in = 0.98`; `sigma_combustion = 0.96`;
  `eta_combustion = 0.98`; `g_leak = 0.0075`; `q_lhv = 43150 kJ/kg`; `l0 = 14.61`;
  `M_t = 0.3379`; `pi_nozzle = 1.05`.

At exactly `T_g = 1250 K`, `cooled = T_g > 1250` is **false**, so the cycle uses the
non-cooled bleed branch (`g_cooling = 0.015`), not the 5 % cooled branch. This is a
source-code detail that matters when comparing “cooling on/off” cases.

## 8. Power and mass balances

The cycle computes:

```text
N_ud             = L_t * (g_t + g_return) * eta_mech_turbine - L_k
G                = N / N_ud
N_reducer_out_ud = N_ud * eta_reducer
N_e_ud           = N_reducer_out_ud + thrust-term
C_e              = fuel-flow-factor * 3600 / (N_ud / 1e3)
```

The meanline/optimizer candidate computes:

```text
N_val = sum(G_c * L_t) * eta_mech_turbine - G_air * L_k
```

and validates `N_val` against the requested `N` on the same pre-reducer plane. The reducer
efficiency is **not** inside the 110 kW number; it is applied only to
`N_reducer_out_ud`/`N_shaft_reducer_out` for downstream gearbox output. `C_e`/`eta_e` are
referred to the pre-reducer target. `N_e_ud` is the equivalent shaft-plus-nozzle metric
built on the downstream gearbox output. The checkout still does **not** contain a gearbox
ratio, a gearbox component, a propeller load, or a separate “power at gearbox input”
measurement beyond this model plane. See
`docs/comparisons/Источники_и_методики.md` §6 for the exact formula register and the
legacy-differential note.

Continuity and energy residuals are checked in `optimizer_core.validate`; they are not a
separate mass-balance module. Cooling/bleed/fuel/return flows are lumped cycle scalars,
not solved flow networks.

## 9. Backends and environment

| Backend | Status |
|---|---|
| Reference Python / legacy notebook path (`stage_core=False`) | Present; used as independent comparison |
| Extracted Python (`stage_core=True`, `turbine_engine`) | Present; default current evaluation path |
| NumPy/SciPy CPU | Present; `numpy 2.5.4`, `scipy 1.18.1` |
| JAX CPU | Not installed |
| JAX GPU / CUDA / WSL2 | Not configured; no numerical GPU self-test |
| Batch/GPU engine, auto batch size, OOM recovery | Not implemented |
| `turbinis` CLI | Not implemented; no console entry point in `pyproject.toml` |

Hardware seen by the OS: Ryzen 5 9600X (6C/12T), 31.1 GB RAM, RTX 4060 8188 MiB
(driver 610.88). Presence of the GPU is not evidence of a working JAX backend. There is
no claim of GPU acceleration.

Use the repo-local venv. The global `python` on this host does not have NumPy installed;
the README commands assume the venv is active.

## 10. Search space and constraints

Actual current variables:

- `optimize_extended`: `n`, `D_hub`, `w1`, `w2`, `alpha1_1..3`, `kappa1..3`;
  `SCENARIOS` set `alpha2` and `l_t/D_t` variants; `WARM` is a fixed start point, not a result.
- `optimizer_core`: `D1`, `D2`, `lD`, `a11..a13`, `w1`, `w2`, optionally `k1..k3`,
  `sigma`; mode B/C can add `pi`/`N`; mode C can add `Tg`; base mode is fixed.
- `pi_k` is fixed at 11 in the active cycle; the prompt's `pi_k = 3...13` sweep is not
  implemented.
- `pi` stages are distributed by `pi_веса`; `pi_t` is a cycle output, not an independent
  search variable in the current const-hub path.
- `D` and `l` are computed from continuity/geometry; they are not all independent inputs.

Constraints are split by status in `constraints.py`. `hard`/`project` checks can make a
candidate infeasible; they do not stop storage in the extended evaluator. `recommendation`
and `info` checks are diagnostic. `NOT_CHECKED` means exactly that, not “pass”.

Active-path early exits are classified in
`docs/comparisons/Источники_и_методики.md` §8: A (mathematically impossible) and B
(nonphysical state) stop the calculation; C (engineering constraint) normally annotates and
stores the point; D covers legacy/unclear solver quirks. The legacy 2 % last-stage pressure
margin `p0 <= 1.02*p_t` was D pending justification; the checked textbook equation (printed
p. 92, items 138–140) supports only `p0 > p_t`, so the margin was removed from the extracted
module and active notebook cell 16. The guard now rejects only `p0 <= p_t`; a positive-drop
candidate is passed to the stage calculations, which remain the physical authority. That
specific D finding is closed; other D items (`(None, None)`/`ст_b is None` returns and
silent non-convergence) remain open. The earlier categorical “no harmful C gate” claim is
retracted. No other guard was changed. This is an active-path audit, not a
repository-wide one.

`rho <= 0.65` is the user-approved **acceptance-only** ceiling. It is not added to
`validate`, `eval_candidate`, or any graph function. Store the calculated row, then
annotate the acceptance result. Fig. 18 dotted values remain `frame_clipped`; Fig. 19
`eta_u` remains limited to `rho <= 0.5`; no extrapolation.

## 11. Dataset pipeline — target vs current

Target (M3/M5/M6): RAW schema → annotated dataset → feasibility/sensitivity/Pareto →
finalists, with manifest/checkpoint/reproduce and streaming Parquet.

Current: `turbine_engine/raw.py`, `dataset.py`, `sampling.py` and
`turbine_engine/constraints.py` provide the generic RAW record, journal, manifest,
checkpoint/resume, Parquet export, annotation and reproduce contracts. They are tested
with synthetic candidates and are not yet connected to a production architecture
evaluator. There is no production RAW dataset or `results/raw/architecture=.../run=...`
directory yet. `Optimizer/results/` contains historical CSVs/JSON/logs/plots from prior
optimizer work; they are useful diagnostics but do not have the M3 identity/diagnostic/
causal-chain fields.

## 12. Commands that are safe today

From the repository root, with the venv active (or use `.\.venv\Scripts\python.exe`):

```powershell
# static structure/notebook check only
python scripts/verify.py

# regression/parity anchors
python -m unittest discover -s scripts -p "test_*.py" -v
python -m unittest discover -s turbine_engine/tests -p "test_*.py" -v

# whitespace/conflict check after documentation edits
git diff --check
```

Observed on 2026-11-11 at the checkpoint:

- `scripts/verify.py`: 6 PASS, 7 WARN, 0 FAIL; warnings are absent
  `docs/engineering/*` and absent `reference/baselines/`; physics NOT verified.
- `scripts` suite: 43 tests OK.
- `turbine_engine/tests`: 399 tests OK, 1 skipped (Windows GPU-only test).
- WSL2 JAX CUDA gas-property kernel validation: PASS; see `docs/backend_validation.json`.

Do **not** use these commands as a physics validation. Do not run
`Optimizer/optimize_extended.py`, DE, scan/sweep scripts, or a global optimizer until the
M2/M3 method uncertainties are resolved. Safe single-point process samples are
documented in `docs/PROJECT_STATE.md`; they are not throughput benchmarks.

## 13. Results layout

- `Optimizer/results/` — historical CSV/JSON/logs/plots, `PERF_AUDIT.md`, reports.
- `exports/` — exported workbooks/PDFs; not RAW data.
- `reference/published_cases/` — retained Anton reference copy (575 cells) and root
  compact companion (60 cells); no approved validation case.
- `reference/baselines/` — absent in this checkout.
- `docs/comparisons/`, `docs/history/` — historical analysis snapshots, not current limits.
- No `results/raw/`, no manifest, no checkpoint directory.

## 14. How to resume work

1. Read `docs/PROJECT_CHECKLIST.md`.
2. Read this state file: `docs/PROJECT_STATE.md`.
3. Read this guide if the task touches the model, tests, or data.
4. Inspect `git status --short`; do not overwrite the pre-existing dirty/untracked work.
5. Continue the first `[~]`, then the highest-priority `[ ]` item. Do not restart the source
   audit; it is recorded. Do not ask “where did we stop?” when the checklist/state answer it.

## 15. Known limitations

- No **validated** full architecture closure. `architecture_cycle.py` provides opt-in
  explicit 0D reference cycles for free-power and dual-CC/single-CC, but they are not
  physical validation. No explicit gearbox ratio or propeller load exists; the output-
  shaft -> gearbox -> propeller relation is documented but not componentized.
- No compressor map/blading/corrected-flow/speed/diameter model, no turbine stage
  inventory, and no cooling/bleed/composition model in the reference cycles;
  `MISSING_COMPRESSOR_PHYSICS` is reported explicitly.
- No validated 3D/radial-equilibrium implementation; the accepted relation's
  `turbine_engine/radial/equilibrium.py` is absent.
- No compressor stage/map model or compressor split; `compressor.py` is a 0D lumped
  unit/train only and is not wired into the active cycle.
- No turbine stage inventory or assembly model in the new component path; `turbine.py` is a
  0D lumped total-to-total expansion unit only and is not wired into the active cycle.
- No combustor flow/composition/performance model; `combustor.py` is a scalar wrapper over
  the existing `calc_alpha`/`calc_g_m` helpers only. It does not produce an outlet
  `StationState`, mass balance, pressure loss, or separate combustion-efficiency model and
  is not wired into the active cycle.
- JAX CPU/GPU validation exists only for the bounded gas-property kernel. There is no
  GPU batch engine, full-cycle JAX port, `turbinis` CLI, or production sweep runner.
  The RAW/annotation/manifest/checkpoint/Parquet/reproduce infrastructure exists in
  `turbine_engine/`, but has not been integrated with a production evaluator.
- No production RAW dataset; historical optimizer results are not a substitute.
- `docs/engineering/*` normative files are absent; README authority claims are unmet.
- Sergey source is present in two variants (48-cell working candidate, 83-cell archive
  comparison); its latest author version cannot be proven and it is not a validated
  baseline. Do not invent missing details.
- Anton case is retained in two variants (60-cell root, 575-cell reference) but is not
  quality-approved as a regression baseline.
- Fig. 18 dotted curve is clipped by the graph frame; Fig. 19 is `rho <= 0.5`; neither may
  be extrapolated, and `rho <= 0.65` is acceptance-only.
- Material/creep/resource/thermal stress and the `sigma <= 300 MPa` limit remain
  preliminary/`NOT_CHECKED`.
- `PERF_AUDIT.md` is from a different host; current-host performance is known only from
  single samples.
- Existing tests establish software parity, not physical correctness.

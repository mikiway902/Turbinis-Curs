# Turbinis-Calc Project Checklist

Last updated: 2026-11-11
Master prompt: `Super-Promt-Autonomous.md` (§37.1 checklist structure; §35 autonomous loop; §36 priority rule)
Progress source of truth: this file. `docs/PROJECT_STATE.md` is the short operative state and
`docs/PROJECT_GUIDE.md` describes the actual checkout.

Status legend:

- `[ ]` TODO
- `[~]` IN PROGRESS
- `[x]` DONE
- `[B]` BLOCKED by external information/access
- `[S]` SKIPPED — reason required

## Accepted acceptance profile — `rho <= 0.65`

The user approved a global acceptance ceiling of `rho <= 0.65` on 2026-11-10. It is an
**acceptance-only** profile:

- It does not change physics, equations, solver tolerances, stage design, or the cycle.
- It is not a search bound and not a filter on calculated RAW candidates. In the M3 RAW
  pipeline every calculated point is stored; `rho <= 0.65` is annotated afterwards as
  pass/fail with its actual value and margins.
- It does not extend, replace, or reinterpret the digitized graph functions:
  `Turbinis-Calc/constraints.py` remains source-limited. `uc1_opt_from_rho` keeps
  `fig18_uc1 = "frame_clipped"` (dotted edge meets the graph frame near `rho ~= 0.65`;
  `(0.7, 1.2)` is a frame intersection, not a curve value). `eta_u_from_uc0_rho` keeps
  its `rho = 0...0.5` domain. No extrapolation is permitted.
- `Turbinis-Calc/PENDING.md` still lists Fig. 18 and material/creep/thermal-stress items
  as unresolved. `0.65` does not resolve them.
- Do not silently turn this acceptance profile into `BOOK`, `RELAXED`, `COMPROMISE`, or
  `CONST_HUB` semantics. Existing profiles keep their own definitions.

Recommended profile id in future annotations: `RHO065` (acceptance ceiling only).

## Phase map — prompt §§35-36

| Milestone | Prompt section | Status | Meaning |
|---|---|---|---|
| M0 | §37.1 | `[x]` | Bootstrap state and environment recorded |
| M1 | §37.1 | `[~]` | Source audit recorded; source notebooks located; several method items remain `[~]` |
| M2 | §37.1 | `[~]` | In progress — contracts, components, shaft balance and opt-in 0D architecture reference cycles landed; stage inventory, cooling/map semantics and physical validation remain open |
| M3 | §37.1 | `[ ]` | Calculation semantics and RAW data |
| M4 | §37.1 | `[ ]` | Performance path |
| M5 | §37.1 | `[ ]` | Sweep engine |
| M6 | §37.1 | `[ ]` | Production design-space calculation |
| M7 | §37.1 | `[ ]` | Engineering analysis |
| M8 | §37.1 | `[ ]` | Final audit and delivery |

Working rule (§35): `IF THE NEXT STEP FOLLOWS FROM THIS SPEC AND DOES NOT REQUIRE NEW USER INFORMATION → DO IT NOW`.
Do not stop between milestones for confirmation. Stop only for a genuine §35.1 blocker; even
then do every independent item first.

Priority rule (§36): P0 correctness/reproducibility/final result → P1 next enabling step →
P2 only if necessary → P3 skip. Do not run a global optimizer, DE run, or design-space sweep
until the M2/M3 method uncertainties below are resolved. Use the §36.2 short cycle:
small smoke test → focused regression → representative batch → large run.

Current transition: **M2 in progress**. The contract layer (`StationState` and topology
descriptors), standalone 0D compressor-unit/train and turbine-expansion components (with
prescribed-work inverse), scalar combustor heat-addition wrapper, standalone architecture
shaft balance, and opt-in executable 0D reference cycles for all three selected topologies
are implemented. The cycles carry an explicit `uncooled_no_bleed_0d_ideal_mixing` variant,
fuel/`alpha`/gas-property basis, GG->FPT state transfer, and
`MISSING_COMPRESSOR_PHYSICS`/`NOT_PHYSICALLY_VALIDATED` limitations. The
user-confirmed `N=110 kW` pre-reducer (gearbox-input) plane is implemented consistently,
and the active-path A/B/C/D early-exit classification plus equation register are recorded.
Compressor-stage inventory, cooling/map semantics, full physical validation, and
repository-wide (non-active) exit classification remain `[~]`; they are M2/M3 inputs, not
reasons to restart the audit.

## M0 — Bootstrap / project state

- [x] Найти существующие roadmap/status/docs и не создавать дубли.
  — Evidence: `docs/README.md` is the only documentation index; no `docs/PROJECT_CHECKLIST.md`,
  `docs/PROJECT_STATE.md`, `docs/PROJECT_GUIDE.md`, or equivalent roadmap/status file existed
  before this task. No duplicate roadmap was created.
- [x] Создать/обновить PROJECT_CHECKLIST.
  — Evidence: this file.
- [x] Создать/обновить PROJECT_STATE.
  — Evidence: `docs/PROJECT_STATE.md`.
- [x] Создать/обновить PROJECT_GUIDE.
  — Evidence: `docs/PROJECT_GUIDE.md`.
- [x] Зафиксировать текущий git commit и рабочее состояние.
  — Evidence: branch `main`; HEAD `85e6102` (`docs(exports): add turbine calculation workbooks`).
  Pre-existing dirty paths: `.gitignore`, `Optimizer/README.md`, `Optimizer/optimize_extended.py`,
  `Optimizer/optimizer_core.py`, `README.md`; pre-existing untracked paths: `Super-Promt.md`,
  `Super-Promt-Autonomous.md`, `pyproject.toml`, `scripts/`, `turbine_engine/`. These were
  not modified by this documentation task.
- [x] Проверить shell/Python/tests для агентов.
  — Evidence: Python 3.12.10; repository-local `.venv\Scripts\python.exe` 3.12.10.
  `python -m unittest discover -s scripts -p "test_*.py" -v` → 43 tests OK.
  `python -m unittest discover -s turbine_engine/tests -p "test_*.py" -v` → 399 tests OK.
  `python scripts/verify.py` → 6 PASS, 7 WARN, 0 FAIL, exit 0 (warnings are the absent
  authority docs and absent `reference/baselines/`). The global `python` lacks NumPy;
  use the venv (see `docs/PROJECT_GUIDE.md`).
- [x] Зафиксировать environment и доступные CPU/GPU backends.
  — Evidence: Windows; AMD Ryzen 5 9600X (6C/12T); 31.1 GB RAM; NVIDIA RTX 4060 8188 MiB,
  driver 610.88. `.venv`: NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.1, pyarrow 26.0.1,
  JAX/JAXlib 0.12.0.1 (CPU). A user-local WSL2 Ubuntu venv
  (`~/.venvs/turbinis-gpu`) was validated with `jax[cuda12]==0.12.0.1` and the RTX 4060;
  see `docs/BACKENDS.md` and `docs/backend_validation.json`. The GPU evidence is for a
  bounded float64 kernel, not full-engine throughput.

## M1 — Source audit + trustworthy reference

Status: `[~]`. No `[B]` items remain; three M1 items are `[~]` (compressor stage
inventory, current-host profile, JAX compatibility) and the rest are recorded with checkable
evidence. The N boundary and active-path A/B/C/D classification are closed.

- [x] Построить repository map.
  — Evidence: `docs/PROJECT_GUIDE.md` §§3-5 maps the notebooks, `Optimizer/`, `turbine_engine/`,
  `Turbinis-Calc/constraints.py`, tests, references, book scans, and result folders.
- [x] Найти основной notebook и computational call graph.
  — Evidence: `Turbinis-Cycle/Turbinis-Cycle.ipynb` (cycle) → `Turbinis-Calc/cycle_link.py`
  → `Optimizer/optimizer_core.cycle_point` → `turbine_engine/cycle.py`. The meanline path is
  `Optimizer/optimizer_core.load_model` / `_evaluate`, which executes the relevant cells of
  `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb` and installs the extracted adapters; see
  `Optimizer/optimizer_core.py` (`load_model`, `_evaluate`) and `turbine_engine/README.md`.
- [x] Найти расчёты пользователя, Сергея и Антона.
  — Evidence: user calculation is present (`Turbinis-Cycle/Turbinis-Cycle.ipynb`,
  `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb`). Sergey calculation is present:
  `TVD_pi6_design_steps_1_3.ipynb` (48 cells, SHA-256
  `fe8da90402d05cebefefcb40ef6367b14fbf884baa2fb4ef1991294821fce4c7`) and
  `archive/comparisons/legacy_workbench/TVD_pi6.ipynb` (83 cells, SHA-256
  `13a0165e5ec35927e05263de63c645f2c8c2a0877126f79dcd5ed78d9c8ff652`; 38 identical
  nonempty source cells = 79.2 % of the root's 48 unique nonempty cells). Anton case is
  present at `reference/published_cases/1Anton-Mathcad_ST_cooling_Jupyter (3).ipynb`
  (575 cells, SHA-256
  `5510d3e44c263b9a77a2895d094e3e0d0799c9408ea7ce504a65dbcf3d406ec3`) and
  `Mathcad_ST_cooling_Jupyter.ipynb` (60 cells, SHA-256
  `d525e7a289be01f82174c52515726ae191684299c6094fe483e1baedf7ce5166`). Root Sergey is
  the working candidate for cycle + design steps 1–3; the archive is a
  historical/comparison variant, not a straightforward newer revision. The 575-cell Anton
  reference is the preferred detailed/citable copy; the 60-cell root is a compact
  companion. Git revisions: root Sergey `1263ac2`; archive `c6223b9` R100 from
  `_compare/TVD_pi6.ipynb` (originally added `1263ac2`); root Anton `05d6031`; Anton
  reference `c6223b9` (added, not renamed). Working-copy mtimes and commit dates are not
  author chronology; no external author timestamps/original provenance exist, so no
  "latest author version" is claimed. No Sergey/Anton file is a validated baseline. Full
  table: `docs/comparisons/Источники_и_методики.md`.
- [x] Найти учебник, scan/Markdown и связанные источники.
  — Evidence: book part 1 and part 2 PDFs in `Turbinis-Calc/`; `Turbinis-Calc/book-scan.md`,
  `book-scan-review.md`, `Turbinis-Calc/images/`, `Turbinis-Calc/photos/`, and
  `turbine_prompt_assets/` (pages 38, 53, 54, 56, 57, 58, 87, 90; part 2 pp. 42-45).
  Page interpretations are recorded in `Turbinis-Calc/Проверка_формул_2026-11-08.md`.
- [x] Восстановить текущий gas path.
  — Evidence: at the cycle/meanline level the active path is
  `INLET → compressor (single π_k) → combustor → 3-stage turbine → exhaust`, single shaft.
  The cycle is `Turbinis-Cycle/Turbinis-Cycle.ipynb` with `π_k = 11`, `T_g* = 1250 K`;
  the meanline path is `Main_Turbinis_Calc_Book.ipynb` cells 7/9/12/14/16/20. This is a
  one-dimensional book approximation, not a 3D flowpath. **The compressor stage count and
  compressor split are not encoded**; only one pressure ratio is.
- [x] Восстановить shaft layout.
  — Evidence: one shaft (`z = 3` turbine stages); `n` is a common rpm. `η_р = 0.98` is a
  downstream reduction-gear loss; the 110 kW target is at the gearbox input (pre-reducer).
  There is still no explicit gearbox ratio, gearbox component, propeller-load component, or
  free-power-turbine shaft implementation in the active cycle. See
  `turbine_engine/cycle.py` (`N_ud`, `N_reducer_out_ud`), `turbine_engine/baseline.py`,
  `turbine_engine/performance.py`, and the topology-only
  `turbine_engine/architecture.py`.
- [x] Однозначно определить смысл N=110 kW.
  — Evidence: user confirmed on 2026-11-10 that 110 kW is mechanical power at the gearbox
  input, **before** the reducer. Active code now uses
  `N_в_уд = L_t·g_sum·η_м_тур − L_k` (no `η_р`), `G_в = N_зад/N_в_уд`, and exposes
  `N_ред_уд = N_в_уд·η_р`. `C_e`/`eta_e` are explicitly pre-reducer; `N_e_уд` uses the
  downstream gearbox output plus nozzle thrust. Legacy post-reducer numbers are retained
  only as a labelled differential test. See
  `docs/comparisons/Источники_и_методики.md` §6.
- [~] Определить фактическое число compressor/turbine stages.
  — Evidence: turbine is 3 stages (`z = 3`; `Turbinis-Calc/Ограничения_проекта.md` §2;
  `Optimizer/optimize_extended.py`). Compressor stage count is not a model variable; the
  code uses one compressor pressure ratio `π_k = 11` and no compressor-stage list.
- [x] Найти hardcoded N, Tг, πк, stage counts, weights, constraints.
  — Evidence: `Turbinis-Cycle/Turbinis-Cycle.ipynb` (`N_зад`, `T*г = 1250`, cycle
  efficiencies); `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb` (parameters cell, `z`, arrays);
  `Optimizer/optimizer_core.py` (`БАЗА`, `КНИГА`, `границы`, `q_n`, sigma/limits);
  `Optimizer/optimize_extended.py` (`WARM`, `BOUNDS`, `SCENARIOS`, `K_N`, `SIGMA_LIM`);
  `Turbinis-Calc/constraints.py` (`BOOK_CONSTRAINTS`, `DESIGN_ASSUMPTIONS`).
- [x] Найти early rejection и классифицировать A/B/C/D (active path).
  — Evidence: `docs/comparisons/Источники_и_методики.md` §8 classifies the active
  `optimizer_core._evaluate` (both stage paths), `optimize_extended.eval_candidate`, cycle,
  stage, baseline, performance, orchestration, expansion, gas, and reaction paths. A =
  mathematically impossible; B = nonphysical state; C = engineering-only annotation with
  the point stored; D = legacy/unclear solver quirks. The legacy 2 % last-stage pressure
  margin `p0 <= 1.02*p_t` was D pending source justification; the checked textbook
  equation (printed p. 92, items 138–140) supports only `p0 > p_t`, so the margin was
  removed from the extracted module and active notebook cell 16. That specific D finding
  is closed; other D items (the `(None, None)`/`ст_b is None` returns and silent
  non-convergence) remain open. This is not a repository-wide audit; archive/scan/
  exploratory paths are excluded with rationale. M3 still needs structured diagnostic
  records for A/B/D failures.
- [x] Найти legacy/duplicate calculation paths.
  — Evidence: legacy notebook path `stage_core=False` in
  `Optimizer/optimizer_core._evaluate`; independent const-hub notebook functions via
  `Optimizer/optimize_extended.build_ns(..., orchestration_core=False)`; extracted modules
  in `turbine_engine/`. `turbine_engine/README.md` and `Optimizer/README.md` document both.
- [x] Найти source discrepancies.
  — Evidence:
  1. `docs/README.md` and `README.md` call `docs/engineering/*` normative, but that directory
     is absent; `scripts/verify.py` reports five WARNs for it.
  2. Fig. 18 dotted curve is `frame_clipped` near `rho ~= 0.646`; `(0.7, 1.2)` is the frame
     intersection, not a curve value (`Turbinis-Calc/PENDING.md`; `constraints.py`
     `DIGITIZATION`; `Проверка_формул_2026-11-08.md` §7.1).
  3. `l_т/D_т` differs between book p. 58 and p. 87; A87 strict/relaxed and A58 remain
     separate scenarios (`Проверка_формул_2026-11-08.md` §7.3; `Optimizer/README.md`).
  4. `README.md` retains stale exact filenames (`Sergey-TVD_pi6_sequential.ipynb`,
     `Anton-Mathcad_ST_cooling_Jupyter (2).ipynb`) that are not in the checkout. The
     Sergey calculation is present under the root/archive names above; the Anton case is
     present as the root 60-cell and `reference/published_cases/... (3).ipynb` 575-cell
     copies. Do not infer author chronology from mtimes/git; see
     `docs/comparisons/Источники_и_методики.md`.
  5. Anton's retained case is a Baikal-like 750 kW-after-reduction, two-shaft/two-stage
     free turbine with cooling; the active project model is 110 kW single-shaft 3-stage
     (`pi_k = 11`). Sergey is a separate two-shaft 150 hp = 110.325 kW, `pi_k = 6`,
     `T_g = 1250 K` case with one compressor-turbine stage + one free-power stage. Neither
     is an approved regression baseline (`reference/published_cases/README.md`); 110.325
     kW must not be conflated with the project's 110 kW pre-reducer gearbox-input target.
   6. Textbook-vs-case method scope is not closed. A selective direct PDF check is now
      recorded for cycle book §§2.2–2.3, 3.1–3.3 and turbine book §§4, 6, 9, alongside
      existing comparison evidence. It is not a complete cell-by-cell validation of the
      external notebooks. Mean-vs-true `c_p`, two `alpha` definitions, bleed/`C_e`
      bookkeeping, split-vs-single turbine, empirical losses, and stress/material limits
      remain unresolved. See `docs/comparisons/Источники_и_методики.md`.
  No discrepancy was silently resolved.
- [x] Выбрать reference backend.
  — Evidence: the independent reference is the legacy notebook path
  `Optimizer/optimizer_core._evaluate(..., stage_core=False)`; the default extracted path is
  `stage_core=True`. Parity tests assert equality of the wired and legacy paths (see M1
  regression item). This is a software-reference choice, not a physics validation.
- [x] Зафиксировать минимум один контрольный расчёт текущего двигателя.
  — Evidence: `scripts/test_optimizer_regression.py` fixes one current-model candidate
  (`B87_relaxed`, `q_n = 0.995`) with `status == "ok"`, `N = 110e3`, zero continuity/energy/
  exit-pressure residuals, and a deliberately infeasible engineering result. A single
  direct `eval_candidate(WARM, "A87_strict", 0.995)` on 2026-11-10 returned `status=ok`,
  `feasible=False`, `l_c1 = 0.011482379226410723 m` (11.4824 mm) in 0.1298 s.
  `scripts/test_cycle_point_regression.py` fixes cycle anchors.
- [x] Добавить regression tests ключевых промежуточных величин.
  — Evidence: `python -m unittest discover -s turbine_engine/tests -p "test_*.py" -v` → 399
  tests OK, including stage/cycle/gas/performance/geometry parity, the pre-reducer
  power-plane relationship, and optimizer wiring. `python -m unittest discover -s scripts -p
  "test_*.py" -v` → 43 tests OK, including the candidate anchor and cycle-point anchors.
  These are software-drift tests, not physical validation. The old post-reducer cycle
  values remain only in a labelled legacy differential.
- [~] Выполнить короткий baseline profile.
  — Evidence: `Optimizer/results/PERF_AUDIT.md` is a prior cProfile on a Ryzen 7 PRO 5850U
  (3 candidates); it is **not comparable** to the current Ryzen 5 9600X. Current-host
  single samples: `_evaluate(oc.warm_start(CONFIG), full=False)` 0.6385 s and
  `eval_candidate` 0.1298 s. No current-host multi-candidate profile yet.
- [x] Отметить hot functions и JAX compatibility.
  — Evidence: prior audit identifies `cp_k_ступени`, `газ`, and the `ступень_const_hub`
  fixed-point loop as hot (`PERF_AUDIT.md` §1). A bounded gas-property kernel now has
  NumPy/JAX CPU/GPU float64 validation (`docs/BACKENDS.md`); the scalar cycling/stage
  kernels remain unported. Do not claim GPU acceleration: the tested GPU kernel was slower
  than NumPy for the small batch.
- [x] Не продолжать аудит ради аудита, если данных уже достаточно для M2.
  — Evidence: repository map, call graph, reference path, cycle/candidate anchors, and
  regression suite are sufficient to start M2's common core. The remaining audit gaps
  (compressor-stage inventory and repository-wide/non-active exit classification) are
  carried into M2/M3 as P0/P1 tasks, not used to restart source reading. The N boundary and
  active-path A/B/C/D classification are now closed.

## M2 — Unified physical core + architectures

- [ ] Выделить единый source of truth физических формул.
- [x] Создать/нормализовать StationState.
  — Evidence: `turbine_engine/architecture.py` adds a frozen, validated `StationState`
  storing only `T_total`, `p_total`, `mass_flow`, `cp`, `k` and an optional identifier.
  It rejects non-finite/non-positive values and `k <= 1`; focused state tests pass.
  It is not wired into the cycle; the standalone `compressor.py` component now consumes it.
- [~] Реализовать переиспользуемые component models.
  — In progress: `turbine_engine/compressor.py` now provides a standalone 0D/mean
  variable-k polytropic compressor unit/train consuming `StationState`. Pressure ratios are
  explicit finite-positive inputs (not normalised or range-constrained); total pressure is
  multiplied, mass flow is preserved, and per-unit work, total specific work, power, ratio
  product and an unfiltered residual are exposed. It reuses the existing
  `cycle.calc_compressor` equation through one shared helper; the cycle anchor is unchanged
  and `calculate_cycle_point` is not rewired. This is not a compressor map, bladed-stage,
  corrected-flow, speed/diameter, or pressure-split model.
  `turbine_engine/turbine.py` now provides a standalone 0D/mean lumped total-to-total
  turbine-expansion unit consuming `StationState`. It reuses the existing
  `cycle.calc_eta_expansion` and cycle work/drop expression, preserves mass flow/`cp`/`k`,
  and exposes specific work, power, efficiency and an unfiltered pressure residual.
  Pressure ratio is the explicit expansion ratio `p_in_total/p_out_total`; finite-positive
  values are accepted without an operating-map range, and nonphysical outlet states are
  rejected. It is not a bladed stage, stage-inventory, cooling, map, or architecture cycle;
  `calculate_cycle_point` is not rewired and its coupled nozzle iteration still owns `pi_t`.
  `turbine_engine/combustor.py` now provides a standalone scalar heat-addition wrapper. It
  calls exactly the existing `cycle.calc_alpha(T_in, T_gas)` and
  `cycle.calc_g_m(T_in, T_gas, config)` and records `g_m`, `alpha`, and the scalar
  inlet/target temperatures in an immutable `CombustorResult`. It adds no mass-flow
  conversion, fuel/product composition update, `cp`/`k` state, pressure loss, or separate
  combustion-efficiency model, does not produce a `StationState`, and is not wired into
  `calculate_cycle_point`. Architecture cycles are implemented separately in
  `architecture_cycle.py`; this combustor wrapper remains a scalar equation-preserving
  record, not a flow/performance model.
  — Evidence: `.venv\Scripts\python.exe -m unittest turbine_engine.tests.test_compressor -v`
   → 31 tests OK; `test_turbine.py` → 42 tests OK; `test_combustor.py` → 16 tests OK;
   `test_cycle_core.py` → 20 tests OK; full `turbine_engine/tests` → 399 OK. Compressor and
   turbine component changes received independent review; the combustor wrapper is verified
   by the focused parity/validation tests above, not by a new physics review.
- [x] Реализовать EngineArchitecture.
  — Evidence: `turbine_engine/architecture.py` defines `EngineArchitecture` (`Architecture`
  is an alias),
  `ArchitectureDescriptor`, `ShaftGroup` and the read-only `ARCHITECTURES` registry for
  `SINGLE_SHAFT_SINGLE_CC` (1 compressor unit, 1 combustor, 1 turbine assembly),
  `SINGLE_SHAFT_DUAL_CC` (2 compressor units, 1 combustor, 1 turbine assembly) and
  `FREE_POWER_TURBINE` (1 compressor, 1 combustor, separate gas-generator and free-power
  turbine shafts). The descriptor validates exact per-architecture component and shaft
  membership; contradictory topologies are rejected. Documentation now states the
  conceptual output-shaft -> reduction gearbox -> propeller relation for each architecture,
  including that the free-power-turbine shaft is the output group and the gas-generator
  shaft is not. This module is topology only; the executable cycles live in
  `architecture_cycle.py`. No pressure split, shaft speed, gearbox ratio, propeller load,
  or stage inventory is defined here. The focused contract tests pass.
- [x] Зафиксировать и реализовать плоскость N=110 kW (pre-reducer).
  — Evidence: `N_ud = L_t·g_sum·η_м_тур − L_k`, `G = N/N_ud`,
  `N_reducer_out_ud = N_ud·η_р` in `turbine_engine/cycle.py`; the same plane is used in
  `turbine_engine/baseline.py` (`N_на_валу`), `turbine_engine/performance.py`
  (`N_shaft`), `Optimizer/optimizer_core.py`, and `Optimizer/optimize_extended.py`.
  `C_e`/`eta_e` are pre-reducer; `N_e_ud` uses the downstream gearbox output. Tests:
  `test_cycle_core.test_power_plane_is_pre_reducer_and_gearbox_output_is_explicit`,
  `test_baseline_core.test_shaft_power_target_is_independent_of_reducer_efficiency`,
  `test_cycle_point_regression.test_legacy_post_reducer_anchor_is_a_labelled_differential`.
- [x] Добавить regression tests на topology/state contracts.
  — Evidence: `.venv\Scripts\python.exe -m unittest turbine_engine.tests.test_architecture -v`
  → 25 tests OK; `test_cycle_core.py` → 20 tests OK; full turbine-engine suite → 399 tests OK.
- [x] Реализовать architecture-specific reference cycles (opt-in, explicit variant).
  — Evidence: `turbine_engine/architecture_cycle.py` implements
  `uncooled_no_bleed_0d_ideal_mixing` for all three topologies. Basis:
  `docs/comparisons/Источники_и_методики.md` §9.1; printed p. 84 §9 and pp. 92–93
  items 142–145 separate prescribed outlet pressure from prescribed work. A/B close one
  MAIN shaft (`eta_mech*G_gas*L_t - sum(G_air*L_c) = 110 kW`); C closes the GG shaft by
  required compressor work, inverts the turbine relation for the real GG outlet state,
  passes that `StationState` to the FPT, expands to the caller's **total** outlet
  pressure, and sizes `G_air` for the output. `PolynomialGasModel` makes the gas
  constants and `alpha` basis explicit; `ExternalCombustorClosure` is the
  no-inference fallback. No stage count, rpm, gear ratio, propeller load, cooling/
  composition, or compressor map is invented; the result carries
  `MISSING_COMPRESSOR_PHYSICS` and `NOT_PHYSICALLY_VALIDATED`.
  - Tests: `test_architecture_cycle.py` (27 tests) proves deterministic closure for all
  three, independent formula parity, equal-ratio dual-compressor pressure/temperature
  propagation, GG->FPT state transfer, independent GG/FPT residuals, parameter
  recomputation, and clean domain errors. Full `turbine_engine/tests` → 399 OK.
- [x] Воспроизвести текущую одновальную схему.
  — Evidence: the legacy `calculate_cycle_point` path is preserved and its anchor is
  still `N_ud=190589.8677 W/(kg/s)`, `G=0.5771555504 kg/s` in
  `scripts/test_cycle_point_regression.py`; the new architecture cycles are opt-in and
  do not rewire it.
- [x] Реализовать single-shaft / single-CC.
  — `SINGLE_SHAFT_SINGLE_CC` closes the turbine against one compressor plus 110 kW.
- [x] Реализовать single-shaft / dual-CC.
  — `SINGLE_SHAFT_DUAL_CC` uses two explicit series ratios, actual per-unit inlet
  temperature/pressure, preserved ratio product, one combustor, one MAIN turbine.
- [x] Реализовать free-power-turbine architecture.
  — `FREE_POWER_TURBINE` closes GG and FPT independently and does not subtract the
  compressor again from FPT output.
- [x] Для каждой архитектуры обеспечить собственный cycle.
  — One `calculate_architecture_cycle` dispatch with per-architecture counts, closures,
  and result ordering; no shared “one cycle split three ways”.
- [x] Реализовать отдельные shaft power balances.
  — In progress: `turbine_engine/shaft.py` provides a standalone explicit
  `ShaftPowerBalance`/`evaluate_shaft_power_balance` helper for all three selected
  topologies. It enforces exact per-architecture compressor/turbine counts and ordering
  (`SINGLE_SHAFT_DUAL_CC`: two series compressor loads on MAIN;
  `SINGLE_SHAFT_SINGLE_CC`: one compressor and one turbine on MAIN;
  `FREE_POWER_TURBINE`: GG compressor on the gas-generator shaft, turbines ordered GG then
  FPT). Mechanical efficiency is supplied per turbine and applied once; the required
  gearbox-input power is explicit; the downstream gearbox output is
  `actual_input*eta_gearbox`. Raw MAIN/GG/FPT residuals are signed and unfiltered. The helper
  accepts mass-flow-weighted W inputs, adds no defaults, and is not wired into
  `calculate_cycle_point`, the baseline, or the optimizer.
  — Evidence: `.venv\Scripts\python.exe -m unittest turbine_engine.tests.test_shaft -v`
  → 25 tests OK, including independent synthetic balances, retained imbalance, gearbox
  isolation, mass-flow boundary, validation/immutability, and the current-cycle common-shaft
  anchor (`110000 W` actual, `107800 W` gearbox output). The helper is now consumed by
  `architecture_cycle.py`; it is still not wired into the legacy cycle.
- [x] Проверить mass balance.
  — `G_gas = G_air*(1+f)` with no bleed/leak/cooling in the reference variant; each
  result exposes `mass_balance_residual` and validates it to zero. A/B/C tests assert it.
- [~] Проверить energy balance.
  — Component arithmetic is consistent (`w=cp*ΔT`, turbine drop/temperature, shaft
  powers). Fuel/air comes from the retained `calc_g_m` source relation; no independent
  absolute-enthalpy/exergy audit or calorimetric validation was performed, and the
  compressor/turbine `cp` models are explicitly not claimed equivalent.
- [x] Реализовать полный пересчёт cycle при изменении πк.
  — Every call rebuilds compressor states, fuel flow, turbine states, and `G`; tests
  change a ratio/outlet pressure and assert changed states and `G`, with the target still
  closed.
- [x] Проверить compressor pressure-ratio distribution.
  — Dual-CC equal ratios are tested against `p_in*r1`, then `p_in*r1*r2`; the product is
  preserved and the second unit uses the first unit's actual temperature.
- [x] Проверить turbine pressure-ratio distribution.
  — A/B use the explicit total-pressure ratio from combustor exit to the prescribed total
  outlet; C solves the GG ratio from prescribed work and uses the resulting state for the
  independent FPT ratio. No stage-count split is assumed.
- [ ] Подтвердить/исправить stage-count assumptions.
  — Deliberately not inferred. The reference cycles are conceptual assembly closures,
  not stage inventories; this remains a P0 M2/M3 input.
- [x] Reproduce reference points после рефакторинга.
  — The legacy cycle/candidate anchors are unchanged and still tested; the new
  architecture cycles are not reference-validated and do not rebaseline the legacy path.

## M3 — Full calculation semantics + RAW data

- [~] Убрать engineering early rejection из calculation path.
  — The 2 % last-stage pressure margin `p0 <= 1.02*p_t` has been removed: the checked
  textbook equation (printed p. 92, items 138–140) supports only `p0 > p_t`, and the
  guard now rejects only `p0 <= p_t`. This item stays open because the §8 audit cannot
  claim a categorical absence of other harmful C gates; legacy cell-14 engineering
  repairs (`ρ` floor/cap, Γ-law inlet-angle floor) still modify the conventional
  candidate and are labelled in optimizer outputs, not silently removed.
- [~] Оставить остановку только для mathematical/non-physical failure.
  — Solver exhaustion now raises typed `NumericalConvergenceError`; namespace
  `НетРешения` remains for invalid evaluated states. Legacy stage construction still
  contains engineering repair, so the full active path is not yet pure calculation.
- [~] Разделить calculation_status и constraint_status.
  — `raw.py` and `analysis.py` implement distinct calculation/constraint statuses and
  conservative feasibility levels; they are not yet wired into the legacy optimizer.
- [x] Сохранять diagnostic record для failures.
  — `diagnostics.py` provides JSON-safe stage/reason/status/last-valid-state records;
  `_score_x` retains failed attempt inputs and phase; RAW records retain failed inputs.
- [x] Создать RAW schema.
  — `turbine_engine/raw.py` defines `RAW_SCHEMA_VERSION`, strict `CalculationStatus`,
  immutable `RawCandidate`/`EvaluationOutcome`, canonical hashes and JSON-safe payloads.
- [x] Добавить run_id/candidate_id/architecture/sampling_method.
  — Deterministic IDs and run configuration are enforced by `raw.py`; tests cover
  determinism and configuration-mismatch rejection.
- [~] Сохранять ключевые causal-chain intermediates.
  — Generic payload/writer supports them, and `architecture_cycle.py` exposes component
  and shaft residuals; a production evaluator adapter has not yet been written.
- [x] Реализовать constraint profiles поверх RAW.
  — `turbine_engine/constraints.py`: explicit `ConstraintProfile`/rules; `rho <= 0.65`
  is `rho_065_profile()` opt-in only, never a search bound or filter.
- [x] Реализовать margins/normalized violations/pass-fail.
  — `analysis.py` requires explicit bounds/scales; `NOT_CHECKED` never contributes zero;
  `screening_feasible` and `fully_verified` are separate.
- [x] Сделать RAW неизменяемым по смыслу pipeline.
  — Annotated records are separate; `test_dataset` verifies acceptance changes do not
  mutate RAW or rerun the evaluator.
- [x] Реализовать manifest.
  — `dataset.py` writes count/hash manifest and rejects truncation/tamper/config mismatch.
- [x] Реализовать reproduce(candidate).
  — `reproduce_candidate` and `compare_candidates` use explicit `rtol`/`atol` and ignore
  diagnostics; tested with successful and failed records.
- [x] Проверить round-trip одного candidate.
  — Dataset tests cover JSONL round-trip, resume, duplicate idempotence, and
  JSONL/Parquet consistency (with `pyarrow` installed).

## M4 — Performance path

- [~] Измерить hot path.
  — Prior `Optimizer/results/PERF_AUDIT.md` is on another host and predates the power-plane
  change; no current-host full-cycle profile exists. The bounded backend benchmark is not a
  substitute.
- [ ] Удалить очевидные Python bottlenecks без изменения физики.
- [~] Векторизовать только горячие kernels.
  — `turbine_engine/backends.py` vectorizes only the gas-property kernel; the scalar stage
  and cycle loops are unchanged.
- [ ] Реализовать optimized CPU path там, где это выгодно.
- [~] Реализовать JAX CPU для подходящих kernels.
  — Lazy JAX float64 implementation of the gas-property kernel only.
- [x] Сравнить reference vs JAX CPU.
  — `docs/backend_validation.json`: max relative discrepancy `5.71e-14` for cp and
  `1.80e-14` for k at the tested inputs, `rtol=atol=1e-12`; not a full-cycle comparison.
- [x] Настроить JAX GPU/WSL2 RTX 4060.
  — User-local WSL2 venv; `jax.devices("gpu")` returns `CudaDevice(id=0)` on the RTX 4060.
- [x] Выполнить реальный numerical GPU self-test.
  — Float64 GPU kernel executed; max absolute parity error `0.0` for the tested batch.
- [~] Реализовать batch/chunk execution.
  — Backend helper supports batched kernel calls and OOM-only chunk halving; no streaming
  production sweep exists.
- [~] Реализовать auto batch size / OOM recovery.
  — Chunk halving is implemented and tested for the kernel; auto sizing has not been
  benchmarked across shapes.
- [x] Сравнить reference vs JAX GPU.
  — Bounded gas-property kernel parity passed; the tested GPU time was `1.896×` NumPy
  time, so no acceleration is claimed.
- [x] Проверить float64.
  — All validated paths explicitly enable/require float64.
- [S] Проверять float32/mixed только если это реально ускоряет.
  — Skipped: no demonstrated speed benefit; mixed precision is not admitted.
- [x] Выполнить минимально достаточный benchmark.
  — One kernel, one representative batch, compile/warm/steady separated. This is not an
  engine benchmark.
- [ ] Выбрать production backend по измерениям.
  — Blocked on current-host vectorized hot-path implementation and representative
  full-cycle measurements.
- [x] Не оптимизировать оставшиеся cold paths.
  — No cold-path rewrite was done.

## M5 — Sweep engine

- [x] Реализовать deterministic grid для ключевых outer variables.
  — `turbine_engine/sampling.py` provides deterministic Cartesian/targeted grids but no
  physics-specific preset.
- [x] Реализовать constraint-preserving parameterization.
  — `split_pressure_ratio_log` preserves the total ratio for later explicit use.
- [x] Реализовать batch generation.
  — Generic evaluator/batch APIs exist in `dataset.py`; no production architecture preset.
- [x] Реализовать streaming Parquet write.
  — `export_parquet` streams batches; requires optional `pyarrow` and reports a clear
  dependency error otherwise. JSONL is the durable journal.
- [x] Реализовать checkpoint/resume.
  — Append-only journal, manifest/cursors, single-writer lock, config/version checks.
- [x] Проверить resume после искусственного прерывания.
  — Tests cover journal-before-manifest interruption with identical result versus an
  uninterrupted run.
- [ ] Реализовать quick preset.
- [ ] Реализовать normal preset.
- [ ] Реализовать deep preset только если оправдан runtime/storage.
- [ ] Оценивать N_candidates/storage перед крупным run.
- [ ] Проверить все архитектуры на small sweep.
- [ ] Проверить πк=3...13 на coarse sweep.
- [ ] Проверить compressor split.
- [ ] Проверить turbine split.
- [ ] Проверить основные geometry/search variables.
- [x] Не запускать бессмысленный Cartesian explosion.
  — No production sweep was launched; only synthetic infrastructure tests and a
  three-point labelled smoke check were run.

## M6 — Production design-space calculation

- [ ] Запустить достаточный coarse RAW sweep всех архитектур.
- [ ] Проверить dataset integrity.
- [ ] Проверить numerical failure statistics.
- [ ] Проверить energy/mass residual distributions.
- [ ] Найти boundaries/high gradients/extrema.
- [ ] Выполнить adaptive refinement только там, где он добавляет информацию.
- [ ] Выполнить local perturbations вокруг важных candidates.
- [ ] При необходимости расширить sampling.
- [ ] Не считать дополнительные миллионы точек без ожидаемого информационного выигрыша.

## M7 — Engineering analysis

- [~] Создать annotated dataset.
  — Post-processing annotation contracts and profiles exist; no production dataset has
  been annotated yet.
- [ ] Построить feasibility map.
- [~] Построить near-feasibility analysis.
  — `analysis.py` returns full raw/normalized violation vectors and conservative ranking;
  tested only on synthetic rows.
- [ ] Определить active constraints.
- [~] Выполнить controlled sensitivity analysis.
  — `paired_sensitivity` supports exact paired perturbations; no real paired dataset yet.
- [ ] Исследовать влияние πк на G/πт/geometry/blade heights.
- [ ] Исследовать влияние n и D на blade heights/stress/angles.
- [ ] Исследовать compressor split.
- [ ] Исследовать turbine split.
- [ ] Сравнить архитектуры на одинаковой базе.
- [ ] Построить только информативные plots/heatmaps.
- [~] Построить Pareto.
  — Pareto utility with explicit objective directions is implemented and synthetically
  tested; no real front exists yet.
- [ ] Выбрать feasible/near-feasible/boundary finalists.
- [ ] Пересчитать finalists reference backend.
- [ ] Объяснить результаты физически, не только статистически.

## M8 — Final audit / delivery

- [~] Reference tests PASS либо расхождения документированы.
  — Current software suites pass (399 engine + 43 scripts), but this is not an independent
  physical validation; external cases remain unapproved.
- [~] Energy balances проверены.
  — Component arithmetic and shaft residuals are tested for the explicit 0D variant; no
  independent absolute-enthalpy/exergy audit is complete.
- [x] Mass balances проверены.
  — `gas = air + fuel` is enforced and tested for the no-bleed reference variant.
- [x] CPU fallback работает.
  — Reference Python and NumPy paths pass; JAX CPU is optional.
- [x] GPU path реально работает на RTX 4060 либо blocker документирован.
  — WSL2 JAX CUDA float64 kernel validation passed on the RTX 4060.
- [x] Checkpoint/resume проверен.
  — Synthetic interruption/resume tests pass with matching manifest and records.
- [x] Reproduce candidate проверен.
  — Synthetic successful and failed candidate reproduction tests pass.
- [ ] RAW dataset сохранён.
- [ ] Analysis outputs сохранены.
- [x] PROJECT_GUIDE соответствует фактическому проекту.
- [x] PROJECT_STATE соответствует фактическому состоянию.
- [ ] Нет критических TODO/BLOCKED, делающих выводы недостоверными.
  — Compressor/turbine stage inventories, cooling/maps, material life and production
  sweep remain open; conclusions are bounded accordingly.
- [ ] Сформирован final engineering report.
- [ ] Даны ответы на основные инженерные вопросы ТЗ.
- [ ] Выполнен финальный GPT-6 Luna review.
  — Independent review is pending; one reviewer attempt was blocked by the external
  subscription limit, so this gate is not claimed passed.

## Gate before starting M2

M1 is recorded, not fully closed. Before or at the start of M2:

1. `[x]` Fix the `N = 110 kW` boundary (P0): it is the gearbox input (pre-reducer), with
   one definition used by the active cycle, stage baseline, and performance aggregation.
2. Inventory the compressor: actual stage count/split, or mark `MISSING_COMPRESSOR_PHYSICS`
   rather than inventing one.
3. `[x]` Complete the A/B/C/D classification of the active candidate path and separate
   calculation failure from engineering infeasibility (see
   `docs/comparisons/Источники_и_методики.md` §8). Repository-wide/exploratory paths remain
   out of scope; M3 diagnostic records are still open.
4. Keep `rho <= 0.65` as acceptance-only; do not add it to graph interpolation or to the
   calculation path.
5. Start M2 with `StationState` and reusable components while reproducing the current
   single-shaft 3-stage anchor before adding dual-CC or free-power architectures.

The Sergey source is now located (root working candidate + archive comparison variant);
this unblocks the free-power reference branch, but neither file is a validated baseline.
Continue the independent M2 common core and do not invent Sergey details. Close the
source/method boundary in `docs/comparisons/Источники_и_методики.md` before architecture
comparisons.

# DeepSeek agent conclusions — 2026-11-11 checkpoint

This file preserves structured summaries of the implementation-agent reports at the
checkpoint, not a verbatim transcript. They are work reports, not independent physical validation and not
final project approval. Cross-checked verification is summarized in
`docs/STAGE_REPORT_2026-11-11.md` and `docs/PROJECT_STATE.md`.

## 1. Numerical diagnostic-iterate coherence fix

- Fixed the P2 diagnostic-iterate coherence issue.
- `baseline.решить` exhaustion record now captures the final evaluated state:
  - `A_t`, `n_t`, `D_t`, `u_t`, `A_fact` all refer to the same pre-relaxation
    evaluation.
  - `relative_residual` is the residual actually tested.
  - Half-relaxed proposal is separate as `A_t_next`.
- `baseline.calculate` exhaustion record now also includes the evaluated `N_b`
  residual and relative residual against `N_target`.
- `orchestration.ступень_const_hub` exhaustion record now captures:
  - evaluated `D_c`, `D_l`, `rho`;
  - original unhalved deltas `D_c_delta`, `D_l_delta`, `rho_delta`;
  - targets separately;
  - half-relaxed proposals separately as `D_c_next`, `D_l_next`, `rho_next`.
- Convergence tests still use exactly the same quantities, tolerances, equations,
  and order as before. Successful paths are unchanged.
- Added coherence assertions using the actual formulas:
  - `n_t`, `D_t` recomputed from recorded `A_t`;
  - recorded `relative_residual == abs(A_fact - A_t) / A_t`;
  - `A_t_next == 0.5*(A_t + A_fact)`;
  - orchestration deltas equal `target - evaluated` and are not halved;
  - proposed next iterate equals half-relaxation from evaluated to target.
- Updated `docs/NUMERICAL_FAILURES.md` to describe coherent record semantics.
- No other agents' files touched.

### Changed files

- `turbine_engine/baseline.py`
- `turbine_engine/orchestration.py`
- `turbine_engine/tests/test_baseline_core.py`
- `turbine_engine/tests/test_orchestration.py`
- `docs/NUMERICAL_FAILURES.md`

No changes to `expansion.py`, diagnostics module, `__init__.py`, notebooks,
optimizer, raw/dataset, backends, or project docs.

### Tests

Focused diagnostic tests now assert state/residual coherence, not just exception
type.

```text
python -m py_compile turbine_engine\baseline.py turbine_engine\orchestration.py turbine_engine\tests\test_baseline_core.py turbine_engine\tests\test_orchestration.py
=> EXIT=0

python -m unittest turbine_engine.tests.test_baseline_core.ExplicitDiagnosticsTests turbine_engine.tests.test_orchestration.ExplicitDiagnosticsTests -v
=> Ran 10 tests in 0.001s
=> OK

python -m unittest discover -s turbine_engine\tests -p "test_*.py" -t .
=> Ran 398 tests in 31.091s
=> OK (skipped=1)
```

The skip is the optional JAX GPU backend unavailable in that environment.

```text
python scripts/test_verify.py                 => Ran 24 tests, OK
python scripts/test_optimizer_regression.py   => Ran 1 test, OK
python scripts/test_cycle_point_regression.py => Ran 5 tests, OK
```

All reference success-parity tests passed.

### Unknowns and risks

- No end-to-end optimizer/sweep run was performed; typed errors still rely on the
  separately owned adapter/raw-record integration.
- Additional diagnostic fields (`A_t_next`, `D_c_target`, `D_c_next`, `residual`,
  etc.) are additive; existing readers of `last_valid_state` are unaffected.
- No physical validation is claimed.
- Failure-path records expose pre-relaxation state, while `A_t` no longer reflects
  the post-relaxation value. The proposal is explicitly available as `*_next`.
- Successful-path arithmetic, order, tolerances, and returned values are unchanged.

## 2. Conservative analysis fixes

Applied the reviewer findings in a bounded second cycle; no new feature areas were
added.

### Fixes

- **P1 — profile-aware group feasibility counts**
  - `summarize_design_groups(..., required_constraints=...)` now computes
    feasibility per row using the declared profile.
  - Missing required constraints produce `UNVERIFIED`, while calculation failures
    remain `CALCULATION_FAILED` and known constraint failures remain
    `ENGINEERING_INFEASIBLE`.
  - Tests assert the feasibility counts, not just constraint absence.
- **P2 — strict boolean policy flags**
  - `pareto_front` rejects non-bool `include_calculation_failures`,
    `include_infeasible`, and `include_unchecked`.
  - `rank_near_feasible` rejects non-bool `include_calculation_failures`.
  - Strings such as `"False"` and integers such as `1` are rejected.
- **P2 — safe input freezing**
  - `inputs` now accepts only JSON-style values: `None`, bool, int, finite float,
    str, sequences, and string-keyed mappings.
  - `bytearray`, `bytes`, sets, `array.array`, NumPy arrays, and arbitrary objects
    are rejected instead of being stored as mutable aliases.
  - Added tests for `bytearray`, `array.array`, and `numpy.ndarray`.
- **P2 — explicit constraint arithmetic overflow/underflow**
  - Non-finite derived margins are rejected explicitly at constraint construction.
  - A normalized violation that overflows or underflows to zero returns `None`.
  - Near-feasibility leaves such rows unranked with a reason rather than silently
    ranking a zero.
- **P2 — exact numeric conversion**
  - Numeric fields accept real numbers only; numeric strings and booleans are
    rejected.
  - Integers that cannot be represented exactly as `float` are rejected, e.g.
    `9007199254740993`.
  - The same rule applies to constraint values, bounds, scales, metrics,
    `pi_compressor`, and sensitivity parameters.
  - Added tests for lossy large integers and numeric strings.

### Changed files

- `turbine_engine/analysis.py`
- `turbine_engine/tests/test_analysis.py`

No edits to physical sources, notebooks, optimizer, docs, `__init__.py`, README,
or pyproject.

### Tests

Focused analysis suite: 36 tests passed.

```text
.\.venv\Scripts\python.exe -m unittest turbine_engine.tests.test_analysis -v
Ran 36 tests in 0.005s
OK

.\.venv\Scripts\python.exe -m py_compile turbine_engine\analysis.py turbine_engine\tests\test_analysis.py
exit code 0, py_compile OK
```

The full engine suite was not rerun in that cycle because other agents were editing
the tree; the parent verified the combined suite afterward.

### Unknowns and risks

- Strict JSON-style input freezing rejects NumPy scalar/array inputs; callers must
  convert them to Python scalars/sequences first.
- Raw constraint margin overflow is now a hard `AnalysisError`; normalized
  overflow/underflow yields `None` and an unranked near-feasibility entry.
- The `required_constraints` profile remains caller-supplied; rows cannot infer
  globally applicable criteria by themselves.
- No real dataset integration, CLI, plots, or physical validation was added.

## 3. RAW/M5 infrastructure

Implemented generic M3/M5 infrastructure with no turbine physics and no rewiring of
legacy paths:

- Immutable, versioned RAW candidate records with strict calculation statuses and
  canonical JSON hashes.
- Separate post-calculation constraint profiles and annotations; RAW records are
  never mutated and evaluators are never rerun for acceptance changes.
- Deterministic sampling helpers with no physical assumptions.
- Append-only canonical JSONL dataset with manifest/checkpoint integrity, crash
  recovery, resume, duplicate handling, reproduction comparison, and optional
  streaming Parquet export.
- Synthetic tests cover successful/infeasible retention, all failure statuses,
  serialization errors, interrupts, tamper/config rejection,
  interrupted-vs-uninterrupted resume, reproduction tolerance, and JSONL/Parquet
  consistency.

### Changed files

- `turbine_engine/raw.py`
- `turbine_engine/constraints.py`
- `turbine_engine/sampling.py`
- `turbine_engine/dataset.py`
- `turbine_engine/tests/test_raw.py`
- `turbine_engine/tests/test_constraints.py`
- `turbine_engine/tests/test_sampling.py`
- `turbine_engine/tests/test_dataset.py`

### Key API

`turbine_engine.raw`:

- `RAW_SCHEMA_VERSION`, `CalculationStatus`, `ConstraintStatus`
- `RawCandidate`, `EvaluationOutcome`
- `NumericalFailure`, `NonPhysicalStateError`, `ModelError`,
  `EvaluationSerializationError`
- `evaluate_candidate(...)`
- `make_candidate_id`, `make_run_id`, `normalize_run_config`, `canonical_json`,
  `canonical_hash`, `copy_json_value`

`turbine_engine.constraints`:

- `ConstraintRule`, `ConstraintProfile`, `ConstraintCheck`, `AnnotatedCandidate`
- `annotate_candidate`, `annotate_records`
- `rho_065_profile()` / `RHO_LE_0_65_PROFILE_NAME` — opt-in only, never a search
  bound

`turbine_engine.sampling`:

- `iter_grid`, `deterministic_grid`
- `targeted_values`
- `split_pressure_ratio_log`

`turbine_engine.dataset`:

- `DatasetWriter.create(...)`, `DatasetWriter.open(...)`
- `append`, `append_many`, `evaluate`, `evaluate_and_append`, `finish`, `close`
- `read_records`, `iter_records`, `load_manifest`
- `reproduce_candidate`, `compare_candidates`
- `export_parquet`

### Tests

```text
.\.venv\Scripts\python.exe -m unittest turbine_engine.tests.test_raw turbine_engine.tests.test_constraints turbine_engine.tests.test_sampling turbine_engine.tests.test_dataset
Ran 39 tests in 0.603s
OK

.\.venv\Scripts\python.exe -m unittest discover -s turbine_engine\tests -t .
Ran 396 tests in 31.040s
OK (skipped=1)
```

Concurrent `scripts/test_candidate_provenance.py` failures observed during the
workstream were superseded by the final combined run (43 scripts tests OK).

### Unknowns and integration

- `turbine_engine/__init__.py` was not edited by this workstream; RAW package-level
  exports were not subsequently added. Use direct module imports.
- `pyproject.toml` was not edited by this workstream; the parent added optional
  `parquet`/`jax` groups.
- No engineering profile numbers were invented; profiles are explicit inputs.
- Parquet export stores nested payload/output/intermediate structures as canonical
  JSON strings in a stable flat schema; it is not physics-specific flattening.
- `read_records` and resume verify the journal by reading it; this is a scalability
  tradeoff, not a streaming append limitation.

### Risks

- An incomplete final JSONL line is truncated on resume and recorded in
  `repair_notes`; it is not silently dropped, but the partial line is not recovered.
- Manifest/checkpoint hashes detect ordinary truncation/tampering but are not
  cryptographically signed.
- Unexpected evaluator exceptions are classified as `MODEL_ERROR` with
  `evaluator_exception`; explicit typed failures are preferred.
- Parquet requires optional `pyarrow`; without it, `export_parquet` raises an explicit
  dependency error.

## 4. Optimizer candidate-provenance fixes

Implemented bounded candidate-retention and provenance fixes in existing optimizer
adapters, without changing notebook formulas, physics policy, search settings, or
running optimization/sweeps/DE.

- `_score_x` now records failed attempts with copied `x`/`cfg`, `full`, failure phase
  (`calculate`/`validate`/`penalty`), any computed `res`/`checks`, and structured
  exception metadata. Returns remain backward-compatible (`1e9`; failed rows
  explicitly infeasible). Interrupts are still not converted to records.
- `Archive.add` accepts optional extra fields while preserving the old three-argument
  call. Empty check lists no longer make a result feasible.
- Optional figure-18 `uc0_opt` diagnostics no longer erase known physical candidates.
  Out-of-range `rho` produces `None` + `NOT_CHECKED` + reason, is excluded from
  critical finite checks, and is conservatively penalized in `--soft` scoring.
- `evaluate_sizing` now reports optional `u_c0_opt` state/reason instead of raising.
- Extended successful results now include full `stages`, `a1_requested` vs
  `a1_actual`, actual `rho`, diagnostic statuses, and model-variant/adjustment
  metadata.
- Conventional `optimizer_core.evaluate` explicitly labels legacy cell-14 engineering
  repairs (`rho` floor/cap and `Gamma`-law `alpha1` floor), while keeping
  `res["p"]["a1"]` as the requested angle.
- Added `docs/LEGACY_RAW_LIMITATIONS.md` documenting limitations, fix scope,
  integration checklist, and remaining risks.

### Changed files

- `Optimizer/optimizer_core.py`
- `Optimizer/optimize_extended.py`
- `scripts/test_attempt_diagnostics.py`
- `scripts/test_candidate_provenance.py`
- `docs/LEGACY_RAW_LIMITATIONS.md`

### Tests

```text
.\.venv\Scripts\python.exe scripts\test_attempt_diagnostics.py  => Ran 7 tests, OK
.\.venv\Scripts\python.exe scripts\test_candidate_provenance.py => Ran 6 tests, OK
.\.venv\Scripts\python.exe scripts\test_optimizer_regression.py => Ran 1 test, OK
.\.venv\Scripts\python.exe scripts\test_cycle_point_regression.py => Ran 5 tests, OK
.\.venv\Scripts\python.exe scripts\test_verify.py => Ran 24 tests, OK
.\.venv\Scripts\python.exe -m unittest discover -s turbine_engine\tests -t .
=> Ran 399 tests in 30.567s, OK (skipped=1)
```

No optimizer, sweep, or DE command was run.

### Unknowns and risks

- No end-to-end `optimize_extended.main()`/CSV write was run.
- The legacy `rho_max = 0.6` cap may still prevent satisfying the root reaction floor
  for extreme geometries; this patch only labels provenance and does not validate or
  change that behavior.
- `copy.deepcopy(cfg)` per `_score_x` attempt adds minor overhead and assumes
  JSON-style configs; opaque config objects could fail copying.
- `is_feasible`/`is_strict` now require at least one applicable check, so info-only
  check lists are infeasible.
- Full `stages` and provenance increase archive/CSV size.
- The soft missing-diagnostic penalty (`1e3` per missing stage) is a conservative
  ranking policy, not a physical result.
- Legacy engineering repairs remain active and labelled, not RAW.

## 5. Architecture-specific 0D reference cycles

Implemented opt-in, executable 0D architecture-specific cycles for all three selected
topologies, without rewiring the legacy cycle:

- **A `SINGLE_SHAFT_SINGLE_CC`**: one compressor, one combustor, one MAIN turbine
  driving compressor + 110 000 W pre-reducer output.
- **B `SINGLE_SHAFT_DUAL_CC`**: two series compressor units with explicit per-unit
  ratios and actual inlet-temperature propagation, one combustor, one MAIN turbine
  driving both compressors + 110 000 W.
- **C `FREE_POWER_TURBINE`**: GG compressor/turbine balanced separately by prescribed
  work; actual GG outlet `StationState` passed to the FPT; FPT expanded to the
  caller's total outlet pressure; `G_air` sized for 110 000 W output; no compressor
  re-subtraction from FPT.

The explicit reference variant is `uncooled_no_bleed_0d_ideal_mixing`. An
external-combustor-state/fuel-flow closure is also supported.

> Note: the archive's enum names are `SINGLE_SHAFT_SINGLE_CC` (B) and
> `SINGLE_SHAFT_DUAL_CC` (A); the prompt letter map is A = dual-CC, B = single-CC.

### Sources read and equations chosen

Read:

- `Turbinis-Cycle/Михальцев_В_Е_Расчет_параметров_цикла_при_проектировании_газотурбинных.pdf`
  — pp. 6–8 (efficiency/loss ranges, mechanical/reducer efficiencies) and pp. 15–22
  (`N_к = N_т·η_м`, `G_т = G_в + G_m − G_ут − G_охл`, relative fuel-flow `g_m`,
  effective power).
- `Turbinis-Calc/book-scan.md` (turbine book part 1) — §1.1 pp. 4–7 (multishaft
  compressor turbine prescribed by compressor power), §9 pp. 84–93, especially items
  138–145 (prescribed outlet pressure vs prescribed work) and 150–153 (`L_т`, `η_т`,
  `η_м`, `N = L_тΣ·G_c·η_м`).
- `docs/comparisons/Источники_и_методики.md` §9.1.
- Existing source-preserving code: `cycle.py`, `compressor.py`, `turbine.py`,
  `shaft.py`, `architecture.py`, `gas.py`.

Equations:

```text
Compressor (existing): T_out = T_in · π^((k−1)/(k·η_c)), k at mean T
Combustor fuel:        f = calc_g_m(T_comp, T_g); G_gas = G_air·(1+f)
Alpha variant:         FUEL_BASED = 1/(f·l0)  |  EMPIRICAL = calc_alpha(T_comp,T_g)
Turbine:               η_t = calc_eta_expansion(π_t,k,η_poly)
                       w = cp·T_in·(1 − π_t^(−(k−1)·η_poly/k))
                       T_out = T_in − w/cp; p_out = p_in/π_t
A/B shaft:             η_mech·G_gas·w_t − Σ(G_air·w_c,i) = 110 000 W
C GG shaft:            w_GG = w_c/(η_mech_GG·(1+f))
C FPT shaft:           η_mech_FPT·G_gas·w_FPT = 110 000 W
Gearbox:               N_out = N_actual·η_gearbox
```

C GG pressure ratio is obtained by analytical inversion:

```text
π_GG = (1 − w_GG/(cp·T_g))^(−k/((k−1)·η_poly_GG))
```

### Important assumptions / limits

- All states are total states; turbine outlet pressure is explicitly total.
- No cooling, leakage, bleed, or return flows; all compressor air enters the single
  combustor.
- Compressor uses mean-temperature variable `k`; each turbine holds constant `cp/k`
  at its inlet total temperature. These are explicitly not claimed equivalent.
- Fuel/air comes from retained `calc_g_m`; alpha basis is explicit in
  `PolynomialGasModel`.
- No stage count, maps, blading, corrected flow, speed, diameter, gear ratio,
  propeller load, or control law.
- Results carry `NOT_PHYSICALLY_VALIDATED`, `MISSING_COMPRESSOR_PHYSICS`,
  cooling/thermochemistry/gearbox limitations.
- Physical inability to supply positive output raises
  `ArchitectureCyclePhysicalError`, not an acceptance filter.

### Code paths

- New: `turbine_engine/architecture_cycle.py`
- Modified: `turbine_engine/turbine.py` — added
  `turbine_pressure_ratio_for_work`
- Modified: `turbine_engine/__init__.py` — exports
- New tests: `turbine_engine/tests/test_architecture_cycle.py`
- Extended tests: `turbine_engine/tests/test_turbine.py`
- Docs: `README.md`, `turbine_engine/README.md`, `docs/PROJECT_STATE.md`,
  `docs/PROJECT_CHECKLIST.md`, `docs/PROJECT_GUIDE.md`,
  `docs/comparisons/Источники_и_методики.md`

### Tests

```text
.venv\Scripts\python.exe -m unittest turbine_engine.tests.test_architecture_cycle
→ 27 tests OK

.venv\Scripts\python.exe -m unittest turbine_engine.tests.test_turbine
→ 42 tests OK

.venv\Scripts\python.exe -m unittest discover -s turbine_engine/tests -p "test_*.py"
→ 399 tests OK (1 skipped), exit 0

.venv\Scripts\python.exe -m unittest discover -s scripts -p "test_*.py"
→ 43 tests OK, exit 0

.venv\Scripts\python.exe scripts/verify.py
→ 6 PASS, 7 WARN, 0 FAIL, exit 0
```

### Unresolved issues / risks

- These are software/equation closures, not physical validation.
- No compressor map/stage inventory, turbine stage physics, cooling/bleed semantics,
  gear ratio, or propeller model.
- `calc_g_m`, `alpha`, `cp/k`, and thermochemistry remain retained project
  assumptions; no independent enthalpy/exergy audit.
- `target_power_status` is a numerical reporting status, not a feasibility/acceptance
  model.

## 6. CPU/JAX/GPU backend validation

Implemented bounded optional JAX CPU/GPU validation and benchmarking for the existing
gas-property kernel `turbine_engine.gas.calc_gas_cp_k` /
`cycle.calc_cp_kerosene`. The port preserves the exact source equation, 700 K branch,
1 K finite difference, 273.15 K offset, and 4187 factor.

- Added `turbine_engine/backends.py`: reference delegate, vectorized NumPy float64,
  lazy JAX float64, explicit `jax.devices("cpu")` / `jax.devices("gpu")` selection,
  `block_until_ready` timing, compile/warmup separation, and OOM-only chunk halving
  with dtype preservation.
- Added `scripts/validate_backends.py`: reproducible CPU/GPU CLI, strict parity
  checks, JSON report, explicit GPU selection, and recorded CPU fallback when GPU is
  unavailable.
- Added a user-local WSL setup script and created `~/.venvs/turbinis-gpu`
  successfully with Python 3.14.4, NumPy 2.5.4, JAX/JAXlib 0.12.0.1 plus
  `jax[cuda12]`. No sudo, apt, or system Python changes.
- GPU validated successfully on WSL: `CudaDevice(id=0)`,
  `NVIDIA GeForce RTX 4060`, float64, 40-point max absolute error `0.0`.
- Final report: `docs/backend_validation.json`, status `PASS`.

### Changed files

- `turbine_engine/backends.py`
- `turbine_engine/tests/test_backends.py`
- `scripts/validate_backends.py`
- `scripts/setup_wsl_jax_cuda.sh`
- `docs/BACKENDS.md`
- `docs/backend_validation.json`

### Tests and benchmark

```text
Windows full turbine_engine/tests: 399 tests OK, 1 skipped
Windows focused test_backends: 10 tests OK, 1 skipped
WSL focused test_backends: 12 tests OK, including explicit GPU device/parity
Windows scripts suite: 43 tests OK
scripts/verify.py: 6 PASS, 7 WARN, 0 FAIL
py_compile and bash -n: exit 0
```

WSL final validation:

```text
NumPy vs reference: PASS (max abs cp=0, k=0)
JAX CPU vs reference: PASS (device=cpu, dtype=float64)
JAX GPU vs reference: PASS (device=NVIDIA GeForce RTX 4060, dtype=float64)
NumPy kernel median: 0.660 ms (99.27 M elements/s)
JAX CPU kernel median: 0.320 ms (first call 39.1 ms, includes compilation)
JAX GPU kernel median: 1.252 ms (first call 64.4 ms, includes compilation)
Result: PASS
```

Final JSON values: JAX CPU max relative error `5.71e-14` for `cp` and `1.80e-14`
for `k`; GPU max absolute error `0.0`; tolerance remains `rtol=1e-12`,
`atol=1e-12`. Steady ratios versus NumPy: JAX CPU `0.485`, JAX GPU `1.896`
(GPU slower for this small 65,536-element batch; no speedup claim).

### Unknowns and risks

- Benchmark covers one bounded kernel and one batch size; not full-cycle, stage, or
  engine throughput, and not a larger-batch GPU crossover.
- Windows JAX remains CPU-only. The validated GPU path exists only inside the
  dedicated WSL venv.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set process-locally by the validation CLI
  only when the caller did not set it.
- The port was compared against the existing Python reference, not the notebook as a
  separate physical validation.
- GPU speed was not faster for this small kernel; no acceleration claim is made.
- The WSL setup consumes about 5.5 GB in the dedicated venv; no system packages or
  existing venvs were changed.
- The report `docs/backend_validation.json` was generated in WSL and records Linux
  host metadata, while the repository is on Windows.

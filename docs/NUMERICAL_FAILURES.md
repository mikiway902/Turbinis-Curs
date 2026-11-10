# Numerical failure diagnostics (M3 P0)

Status: module-specific implementation note. This file is not an engineering
acceptance rule, a physical validation, or a change to the source notebooks.

## Purpose

The extracted baseline, const-hub orchestration and expansion solvers keep the
notebook equations, evaluation order, iteration limits and tolerances. Their
failure branches, however, used to leak ambiguous Python outcomes:

- `baseline.распределить_π` could return `(y_b, None)` after a failed final
  trial; `baseline.решить` then indexed `None` and raised `TypeError`.
- `orchestration.распределить_π_const_hub` could have `f_a is None` and then
  subtract it from `f_b`; it could also return `(y_b, None)`.
- Both pressure-ratio secants and the const-hub stage reaction fixed point
  could exhaust their limits and return a stale iterate as if it had closed.
- `expansion.calculate_turbine_expansion` returned its last temperature iterate
  after 50 iterations without marking non-convergence.

The original notebooks remain the independent reference and are not modified.
The extracted modules deliberately diverge only on these failure outcomes, not
on successful numerical results.

## Diagnostic contract

`turbine_engine.diagnostics` defines:

- `SolverDiagnosticsError`: typed exception with `stage`, `reason`, `status`,
  `last_valid_state`, and `as_record()`.
- `NumericalConvergenceError`: the typed failure used when a solve does not
  close to its existing tolerance.
- `NUMERICAL_FAILURE`: iteration/stall/no-valid-evaluation status.
- `NON_PHYSICAL_STATE`: separate status available for an invalid evaluated
  state. The extracted solvers keep the namespace's own `НетРешения` for an
  invalid individual stage evaluation; they do not turn it into an engineering
  acceptance filter.

`last_valid_state` and `as_record()` are JSON friendly. Mappings become plain
dictionaries, sequences become lists, and non-finite numbers become `null`
(`None`), so a record never contains `NaN` or `Infinity`. The record includes
the existing limit/tolerance and the last finite iterate or input available.

Exhaustion records describe one coherent evaluated iteration, not a mixture of
the evaluated state and the next relaxed state:

- `baseline.решить` records `A_t`, `n_t`, `D_t`, `u_t` and `A_fact` from the
  same evaluation, plus `relative_residual` actually tested. The half-relaxed
  proposal is separate as `A_t_next`.
- `orchestration.ступень_const_hub` records the evaluated `D_c`, `D_l`, `rho`,
  the original deltas `D_c_delta`, `D_l_delta`, `rho_delta`, and the targets.
  The half-relaxed proposal is separate as `D_c_next`, `D_l_next`, `rho_next`.
  Deltas are never halved by relaxation before being recorded.
- `baseline.calculate` records the evaluated `N_b` and its `residual` /
  `relative_residual` against `N_target`.

## Sites now explicit

| Stage | Reason | Raised when |
| --- | --- | --- |
| `baseline.распределить_π` | `no_valid_final_evaluation` | final trial still returns the namespace `НетРешения` |
| | `secant_denominator_stalled` | zero/non-finite secant denominator |
| | `pi_secant_iteration_exhausted` | 30 iterations without `abs(residual) < 1e-5` |
| | `non_finite_residual` | evaluated residual is not finite |
| `baseline.решить` | `stage_solution_unavailable` | no stage list from the pressure solve |
| | `area_iteration_exhausted` | 40 iterations without the existing `1e-4` relative area test |
| `baseline.calculate` | `outer_flow_iteration_exhausted` | 20 outer flow iterations without the existing `1e-5` relative shaft-power target |
| `orchestration.распределить_π_const_hub` | `no_valid_final_evaluation` | final trial still returns the namespace `НетРешения` |
| | `secant_denominator_stalled` | `abs(f_b - f_a) < 1e-12` while the residual is still above `1e-5` |
| | `pi_secant_iteration_exhausted` | 30 iterations without `abs(residual) < 1e-5` |
| | `non_finite_residual`, `non_finite_secant_denominator` | non-finite evaluated arithmetic |
| `orchestration.ступень_const_hub` | `reaction_fixed_point_exhausted` | 60 iterations without the existing `1e-7` reaction/geometry test |
| `expansion.calculate_turbine_expansion` | `temperature_iteration_exhausted` | 50 iterations without `abs(T_т_н - T_т) < 1e-6` |

An invalid initial pressure-ratio evaluation still raises the namespace's
`НетРешения` in both baseline and const-hub orchestration, preserving the
residual wrapper's existing catch.

A tiny secant denominator is not treated as convergence. Convergence still
requires the residual itself to be below the existing tolerance; the
denominator check runs only after that residual test.

## Verification

Focused tests:

- `turbine_engine.tests.test_diagnostics`
- `turbine_engine.tests.test_baseline_core.ExplicitDiagnosticsTests`
- `turbine_engine.tests.test_orchestration.ExplicitDiagnosticsTests`
- `turbine_engine.tests.test_expansion_core.FormulaTests.test_exact_iteration_limit_raises_typed_nonconvergence`

The existing numerical-success parity tests in the same modules are unchanged.
They compare successful extracted results with the executed notebook reference
at the existing tolerances.

## Remaining sites (not changed here)

- `Optimizer/optimizer_core._score_x` broad `except Exception` → score `1e9`
  and raw diagnostic records are a separate owner's task.
- `Optimizer/optimize_extended.eval_candidate` non-finite/`model_error` record
  handling is unchanged.
- `raw.py`, dataset, analysis and architecture/backends code are outside this
  change.
- `turbine_engine/__init__.py`, README and project documentation are not edited;
  the parent session must integrate this module note and export decision.
- No optimizer, sweep, differential-evolution or global-environment behavior is
  changed, and no commit is created by this task.

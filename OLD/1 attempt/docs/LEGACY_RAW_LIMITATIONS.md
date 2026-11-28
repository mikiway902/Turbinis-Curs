# Legacy paths, engineering repair, and RAW limitations

Status: bounded implementation note for the existing optimizer adapters.
Date: 2026-11-11.

This note records what was fixed for candidate-data retention and what was
deliberately **not** changed. It is not a physical validation report and it does
not claim that any legacy result is a complete/RAW closure.

## Terms used here

* **RAW** — a result produced without hidden stage-construction repairs or
  filters, with all requested and actual inputs distinguishable.
* **Engineering repair** — source-model construction that adjusts a parameter
  to satisfy a design rule (for example raising reaction or inlet angle). It is
  not a mathematical guard and not evidence of physical validity.
* **Attempt record** — a diagnostic archive entry. It is not durable RAW data.

## Findings and fixes

### 1. Exceptions in `_score_x` lost the candidate

`Optimizer/optimizer_core.py::_score_x` previously wrapped calculation,
validation, and penalty in one `except Exception` and appended only
`res=None, checks=[], score=1e9`. The input `x`, configuration, already-computed
result, checks, and the failure location were lost.

Now the archive row retains:

* a copied `x` and a deep-copied JSON-style `cfg`;
* `full` and the failure `phase`: `calculate`, `validate`, or `penalty`;
* any already-computed `res` and `checks`;
* `error.type`, `error.message`, and structured `status`/`stage`/`reason`/
  `last_valid_state` when the exception exposes them.

The returned score is still `1e9` for failed attempts. `feasible`/`strict` are
explicitly false on failed attempts. Empty check lists are no longer treated as
feasible (`all([])` trap). `KeyboardInterrupt`/`SystemExit` are not converted to
records. Attempt rows are provenance for debugging, not durable RAW claims.

### 2. Optional figure-18 diagnostics could erase a known physical candidate

`Optimizer/optimize_extended.py::eval_candidate` previously computed
`cs.uc0_opt_from_rho` for every stage inside the main `try`. A `ValueError` for
ρ outside the digitized figure-18 range reached the general handler and replaced
the whole candidate with `status="model_error"` and no stages.

Now each `uc0_opt` entry is either a finite value or `None` with
`uc0_opt_status="NOT_CHECKED"` and the no-extrapolation reason. The optional
diagnostic is **excluded** from the critical finite-value list, so it cannot
convert a computed physical point into `status="numerical"`. `uc_dev` is
`None` where the diagnostic is unavailable. Physical fields, violations,
`feasible`, and `score1` are unchanged when only this diagnostic is missing.

Soft scoring (`--soft`) treats missing diagnostics conservatively: it is marked
`scoring_status="unavailable_diagnostic"`, adds
`SOFT_MISSING_DIAGNOSTIC_PENALTY` per missing stage, and does not let such a
candidate win a soft comparison by omission. With `soft=False`, `score2`
matches the previous finite-anchor values.

`Optimizer/optimizer_core.py::evaluate_sizing` uses the same
`_optional_graph_diagnostic` rule: `u_c0_opt` may contain `None`, with
`u_c0_opt_status` and `u_c0_opt_reason`. Other sizing/geometry fields remain
computed.

### 3. Legacy cell-14 engineering repair is preserved, not hidden

The conventional notebook path (`optimizer_core.evaluate` via the cell-16
baseline) still uses the active cell-14 functions:

* `ступень_с_подбором_ρ` — selects mean reaction from the `l/D` table and, when
  root reaction is below `ρ_корня_мин`, bisects upward (bounded by
  `ρ_макс = 0.6`);
* `ступень_авто` — for the `Γ = const` law only, raises the mean `α1` until the
  root angle reaches `α1_мин_корня` (11°) plus a 0.15° margin, up to 8 passes.

These are engineering model-construction rules, not mathematical guards. They
were not removed and their formulas were not changed.

The conventional result now carries:

* `model_variant = "legacy_cell16+cell14_engineering_repair"`;
* `a1_requested` (same values as `res["p"]["a1"]`) and `a1_actual` from the
  stages;
* `rho_requested = None` (ρ is not an input) and `rho_actual`;
* `engineering_adjustment` with `active`, `legacy_repair`, mechanism names,
  `ρ_корня_мин`, `ρ_макс`, `α1_мин_корня`, requested/actual values, and which
  `α1` entries changed.

Thus `res["p"]["a1"]` remains the requested angle and must not be read as the
actual stage angle. The legacy path is not RAW.

### 4. Extended const-hub path provenance

`optimize_extended.eval_candidate` now returns the full `stages` records plus
requested vs actual `α1` and actual `ρ`. The const-hub branch does not call the
cell-14 `ступень_авто` repair (`legacy_cell14_repair=False`), but it does use an
internal const-hub reaction fixed point. That is still engineering model
construction, not a claim of full RAW closure. Unexpected post-calculation
errors retain any computed stages in the `model_error` record.

## Fix scope

Changed:

* `Optimizer/optimizer_core.py` — attempt provenance, optional graph diagnostic
  helper, sizing metadata, conventional requested/actual provenance.
* `Optimizer/optimize_extended.py` — optional graph diagnostic handling,
  conservative soft ranking, full stages and provenance fields, additive CSV
  columns.
* `scripts/test_attempt_diagnostics.py` — failure-phase/archive tests and
  sizing graph-out-of-range metadata.
* `scripts/test_candidate_provenance.py` — extended graph coverage, soft policy,
  full-stages/provenance, conventional legacy-repair labels.

Not changed:

* notebook sources (`Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb`,
  `Turbinis-Cycle/Turbinis-Cycle.ipynb`);
* cell-14/cell-16 formulas, tolerances, iteration limits, or `ρ_макс = 0.6`;
* optimization/search policy, DE settings, or scenario bounds;
* raw/dataset/backends/analysis/new-architecture modules owned elsewhere.

## Parent integration checklist

1. Preserve the additive fields when merging with other agents' changes:
   `_score_x` attempt rows, `evaluate_sizing` `u_c0_opt_status`/reason,
   `eval_candidate` `stages`/`a1_requested`/`a1_actual`/`rho_actual`/
   `uc0_opt_status`/`scoring_status`, and conventional
   `engineering_adjustment`.
2. Do not interpret attempt rows, `NOT_CHECKED` graph diagnostics, or
   `model_error` partial stages as RAW results.
3. If a future RAW closure removes or replaces cell-14 repairs, do it in a
   separate explicit change with new anchors; do not silently delete them while
   merging this patch.
4. Any later CSV consumer must tolerate `None` entries in `uc0_opt` and the new
   optional columns.
5. Re-run focused tests, then the existing script suite and engine tests. Do not
   run optimizer sweeps/DE as part of this verification.

## Remaining unknowns / risks

* The legacy `ρ_макс = 0.6` cap can leave the root-reaction floor unsatisfied
  for extreme geometries; this note does not prove otherwise.
* `engineering_adjustment` reports the active mechanism and actual values; it
  does not prove every stage was changed, nor that the repair is physically
  desirable.
* The const-hub path's reaction fixed point is engineering construction even
  though it is not the cell-14 repair; its outputs are not RAW.
* The new architecture/raw work owned by other agents is out of scope here and
  is not validated by these adapter fixes.

# Turbinis-Curs engineering rules

This repository contains safety-relevant turbine calculations. A successful
run is not evidence that the physics is correct.

## Authority

1. `docs/engineering/EQUATIONS.md` records equation status.
2. `docs/engineering/CONSTRAINTS.md` records hard limits.
3. `reference/baselines/` is immutable unless the user explicitly approves a
   baseline change after the cause of the numerical difference is explained.
4. The notebook and `Optimizer/` remain the reproducible `book_approx` legacy
   implementation until each equation is migrated and independently checked.

Only the user may promote an equation from `EXPERIMENTAL` to `ACCEPTED`.
Never change a formula, its test, and its reference result in one task.

## Required workflow

Before changing calculation code:

1. identify the equation, source, assumptions, units, limiting cases, and
   acceptance tolerance;
2. classify it as accepted physics, experimental model, empirical
   correlation, numerical approximation, or software assumption;
3. define nominal, boundary, analytical-limit, dimensional, and regression
   checks;
4. implement the smallest change without tuning constants to reach a target;
5. run `python scripts/verify.py` and report every skipped or failed check.

Do not average competing physical closures. Do not update a baseline merely
because the implementation changed. Optimizers may call the physics core but
must not own private copies of physical equations.

## Current project status

- The corrected static-pressure radial equilibrium equation is accepted.
- Coupled rotor-exit closure is experimental: stages 1–2 have numerical
  evidence; stage 3 and stator-exit closure remain open.
- Existing reaction bands and optimization points are diagnostic
  `book_approx`, not validated uncertainty bands of a closed radial model.
- Temperature strength, creep/resource, thermal stress, cooling, final
  third-stage root reaction, and real blade profiles remain `NOT_CHECKED`.

Never push or commit unless the user explicitly requests it.

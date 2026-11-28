# Caloric experiments (frozen)

This directory preserves the opt-in true-`cp` caloric and caloric-combustor
experiment exactly as it was left. It is **not accepted** engine code and is
**not** part of `turbine_engine`.

The files here are a software transcription of the source textbook method
(true-`cp` fit and its integrated component relations). They are frozen so that
experimental results and provenance are not lost; they are deliberately excluded
from main `turbine_engine` test discovery. The active book-method implementation
in `turbine_engine` does not import anything from `experiments`.

All source coefficients, test assertions, and tolerances are preserved. Only
import paths and filesystem locations changed in the move.

## Known failure (NOT fixed)

`test_caloric_combustor.py` contains a known interrupted-combustor test failure
of one unit in the last place (1 ULP) in
`test_nonzero_positive_and_negative_fuel_sensible_enthalpy`
(`775085.7128656327 != 775085.7128656328`). It is recorded here as an
unresolved experimental result. The assertions and tolerances were **not**
weakened. The experiment suite currently reports 45 tests with 1 failure; do
not claim these experimental tests pass.

## Run (from the repository root)

```text
.venv python -m unittest discover -s experiments/caloric/tests -p test_*.py -t .
```

On Windows this is equivalent to:

```text
.venv\Scripts\python.exe -m unittest discover -s experiments/caloric/tests -p "test_*.py" -t .
```

The empty `__init__.py` files in `experiments/`, `experiments/caloric/`, and
`experiments/caloric/tests/` exist only so the command above can import the
experiment package unambiguously.

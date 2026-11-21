# Turbinis-Curs

Engineering study of a three-stage TPE331-class turbine: `N = 110 kW`,
`T_g* = 1250 K`, `pi_k* = 11`, one shaft, constant hub diameter.

A green test run is not evidence that the physics is correct. Read the
authority documents before trusting any number produced here.

## Start here

1. `docs/engineering/README.md` — what equation status means.
2. `docs/engineering/METHOD.md` — calculation layers and the change gate.
3. `docs/engineering/CONSTRAINTS.md` — hard project limits.
4. `docs/engineering/EQUATIONS.md` — which relations are accepted.
5. `docs/engineering/VALIDATION_CASES.md` — required solver evidence.

## Authority hierarchy

1. `docs/engineering/CONSTRAINTS.md` and `docs/engineering/EQUATIONS.md`.
2. `reference/baselines/` — immutable regression baselines; changing one needs
   a separate, user-approved task that explains the numerical difference.
3. Code and historical notes in `Turbinis-Calc/`, `Turbinis-Cycle/`,
   `Optimizer/`, and the notebooks.

Later sources never override earlier ones. Conflicting closures are not
averaged. Only the user promotes an equation from `EXPERIMENTAL` to `ACCEPTED`.

## Canonical calculation paths

| Path | Role |
|---|---|
| `turbine_engine/` | new extracted calculation package: stateless single-point cycle (`cycle.py`), gas properties (`gas.py`), scalar meanline stage (`stage.py`), root reaction relation (`reaction.py`), constant-hub three-stage orchestration (`orchestration.py`), turbine expansion iteration and preliminary sizing (`expansion.py`), meridional flowpath geometry (`geometry.py`), turbine aggregate performance (`performance.py`), the conventional multi-stage patched cell-16 baseline (`baseline.py`), and opt-in explicit 0D reference cycles for the three selected architectures (`architecture_cycle.py`); the architecture cycles are not physical validation and the package is not a 3D model |
| `turbine/` | retained experimental radial-equilibrium package with unit, physics-invariant, regression, and optimization software checks; distinct from `turbine_engine/` |
| `1-Turbinis-Curs-Last/` | separate two-stage course calculations: `TVD_1CBK_2st_v8.ipynb` checks the previous layout at 2000 h; `TVD_1CBK_2st_2000h.ipynb` and `TVD_1CBK_2st_10000h.ipynb` reproduce redesigned candidates with preliminary Larson–Miller normal-stress checks; not qualified engine-life assessments |
| `turbine_engine/README.md` | package scope, provenance, and current limitations |
| `Turbinis-Cycle/Turbinis-Cycle.ipynb` | cycle boundary conditions: flow, pressure ratio, temperatures, power |
| `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb` | legacy `book_approx` meanline, reproducible until each equation is migrated |
| `Optimizer/` | search and diagnostics that must call the shared core; owns no private copies of physical equations |
| `reference/baselines/` | frozen JSON baselines (`BOOK-APPROX-001`, `RADIAL-ANALYTIC-001`) |

The extracted `turbine_engine` package includes opt-in explicit 0D reference
cycles, but not accepted static-pressure radial equilibrium at
`turbine_engine/radial/equilibrium.py`, full architecture closure, or 3D validation.
The engineering authority documents and the separate `turbine/` implementation
are retained in this merged checkout; their presence does not validate either model.

`Optimizer/` and the notebooks remain `book_approx`; they are not closed
three-dimensional physics.

## Tests and verifier

```bash
python scripts/verify.py
python scripts/verify.py --executable-checks
python -m unittest discover -s scripts -p "test_*.py" -v
python -m unittest discover -s turbine_engine/tests -p "test_*.py" -v
```

The `scripts` suite covers the verifier and existing regression anchors; the
`turbine_engine` suite covers the extracted cycle, gas, stage, reaction,
const-hub orchestration, turbine expansion/sizing, meridional flowpath
geometry, turbine aggregate-performance modules, and the conventional
multi-stage cell-16 baseline. These tests check repository structure and
software-drift anchors only; they do not run an optimizer search or establish
calculation correctness.

The default verifier is read-only and static. `--executable-checks` additionally
selects the retained `turbine/` syntax/import checks, test groups, and baseline
reproduction runner; it is not physical validation of either calculation family.

`turbine_engine/baseline.py` copies the *currently executed patched* notebook
cell 16, including its stable-flow reset wrapper `_N_of(G)`, iteration limits,
tolerances, clamps, failure handling, stage sequence and iteration log. It
changes no equations or physical assumptions. `_evaluate(stage_core=True)`
skips the notebook cell and installs the module outputs; the cell's pandas
presentation tables and `display` calls are omitted on that path.
`_evaluate(stage_core=False)` still executes the patched notebook cell in full
as the independent comparison path.

In the project container:

```bash
podman exec -i -w /home/jovyan/work/Turbinis-Curs jupyter \
  python scripts/verify.py --executable-checks --allow-baseline-changes
```

`--allow-baseline-changes` exists for setup/review only; it is never an
approval of changed physics.

## Python environment

`pyproject.toml` declares dependencies and installs the extracted `turbine`
and `turbine_engine` packages. Legacy hyphenated calculation directories retain
their path-based imports; no CLI entry point is declared.

Runtime dependencies for the calculation and optimizer core:

- Python 3.10 or newer (developed and checked with 3.12).
- `numpy` (`Optimizer/`, `Turbinis-Calc/constraints.py`).
- `scipy` (`Optimizer/optimizer_core.py`, `Optimizer/optimize_extended.py`,
  `Optimizer/rotor_exit_solve.py`, `Turbinis-Calc/constraints.py`).
- `pandas` (`Optimizer/optimizer_core.py` and the JSON/CSV helper scripts).

Optional groups:

| Group | Packages | Needed for |
|---|---|---|
| `plot` | `matplotlib` | `Optimizer/scan2d.py`, `Optimizer/stage6.py`, and notebook plots |
| `test` | `pytest>=8` | executable package test groups |
| `notebook` | `jupyterlab`, `nbconvert`, `ipykernel`, `openpyxl`, `jinja2`, `matplotlib` | executing the notebooks; `openpyxl` is used by the workbook export and `jinja2` by pandas `Styler` |
| `digitize` | `Pillow` | `Turbinis-Calc/plot_user_digitization.py` |
| `parquet` | `pyarrow>=14` | streaming RAW Parquet export; JSONL remains available without it |
| `jax` | `jax>=0.12` | optional float64 kernel validation; see `docs/BACKENDS.md` |

The validated CUDA path uses the user-local WSL2 environment described in
`docs/BACKENDS.md`, not the Windows CPU-only JAX install.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[plot,notebook]"
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[plot,notebook]"
```

If you prefer not to install the project, install the libraries directly:

```bash
python -m pip install numpy scipy pandas matplotlib
python -m pip install jupyterlab nbconvert ipykernel openpyxl jinja2
```

The documented dependency set was installed in the repository-local `.venv` on
Windows with Python 3.12.10. The calculation dependencies resolved to NumPy
2.5.4, SciPy 1.18.1, and pandas 3.1.1. No lock file is checked in, so fresh
installs may resolve different versions; notebook execution has not yet been
validated end to end.

## Model status

- Architecture-specific 0D cycles: `REFERENCE` only. `turbine_engine/architecture_cycle.py`
  closes the explicit uncooled/no-bleed variant for single-shaft single-CC,
  single-shaft dual-CC and free-power-turbine topologies, including the
  GG -> FPT total-state transfer and pre-reducer power sizing. It is not
  physically validated and does not supply compressor maps, blading, stage
  inventory, cooling, gear ratio, or propeller load.
- Static-pressure radial equilibrium: `ACCEPTED` as a governing relation, but
  its implementation (`turbine_engine/radial/equilibrium.py`) is absent from
  this checkout.
- Equivalent thermodynamic and general radial residuals: `ACCEPTED` as
  independent implementations (they are not extra momentum terms).
- Rotor energy relation and coupled rotor-exit system B: `EXPERIMENTAL`;
  stage residuals are roughly 0.4–0.7 % (stage 1), 1.8–2.0 % (stage 2), and
  12–15 % (stage 3) under the polynomial `c2u` closure. This is diagnostic
  evidence, not accepted closure.
- Entropy-augmented pressure residual: `REJECTED` (double-counts entropy).
- `BOOK-APPROX-001` output is diagnostic: it detects software drift, it does
  not prove momentum, energy, loss, and integral-flow closure.
- Temperature-dependent strength, creep/resource, thermal stress, cooling,
  real blade profiles, and third-stage root reaction: `NOT_CHECKED`.

## Index

- `docs/README.md` — documentation index (engineering, comparisons, history).
- `docs/comparisons/` — comparisons against the external Mathcad and
  `TVD_pi6` models.
- `archive/README.md` — archived diagnostics and the legacy comparison
  workbench.
- `Optimizer/results/README.md` — what the generated result families are.
- `reference/README.md` — baseline and published-case policy.
- `exports/pdf/variant_length_only_book_approx.pdf`
- `exports/pdf/variant_longest_rotor_with_violations.pdf`

## Retained root source notebooks

`Mathcad_ST_cooling_Jupyter.ipynb`, `Anton-Mathcad_ST_cooling_Jupyter (2).ipynb`,
and `Sergey-TVD_pi6_sequential.ipynb` are external comparison notebooks kept at
the repository root. They are path-sensitive source files: they are not part
of the project authority and must not be rebuilt or relocated without an
explicit instruction.

`reference/published_cases/1Anton-Mathcad_ST_cooling_Jupyter (3).ipynb` is a
retained external case notebook, byte-identical to the file that used to sit at
the root.

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
| `turbine/` | new calculation core; currently `radial/` (equilibrium, entropy budget, rotor-exit) with package placeholders elsewhere |
| `turbine/radial/equilibrium.py` | accepted static-pressure radial equilibrium |
| `Turbinis-Cycle/Turbinis-Cycle.ipynb` | cycle boundary conditions: flow, pressure ratio, temperatures, power |
| `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb` | legacy `book_approx` meanline, reproducible until each equation is migrated |
| `Optimizer/` | search and diagnostics that must call the shared core; owns no private copies of physical equations |
| `reference/baselines/` | frozen JSON baselines (`BOOK-APPROX-001`, `RADIAL-ANALYTIC-001`) |

`Optimizer/` and the notebooks remain `book_approx`; they are not closed
three-dimensional physics.

## Tests and verifier

```bash
python scripts/verify.py
```

In the project container:

```bash
podman exec -i -w /home/jovyan/work/Turbinis-Curs jupyter \
  python scripts/verify.py --allow-baseline-changes
```

`--allow-baseline-changes` exists for setup/review only; it is never an
approval of changed physics.

## Model status

- Static-pressure radial equilibrium: `ACCEPTED`.
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

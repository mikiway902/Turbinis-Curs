# Assumptions and model boundaries

## Active assumptions

- steady, axisymmetric flow;
- negligible radial velocity in radial-equilibrium equations;
- ideal-gas state within each local residual evaluation;
- constant hub diameter and three stages;
- `p_2z = p_t` and existing cycle thermodynamics are invariant;
- `book_approx` is preserved as a reproducible baseline, not treated as closed
  three-dimensional physics;
- M0, M1, and M2 are separate model closures and are never averaged.

## Numerical approximations

- radial derivatives use a finite grid and must report grid dependence;
- current system B parameterizes `c2u(r)` with a polynomial, which is a
  numerical approximation rather than a blade law;
- equal-span and equal-mass streamline mappings were close for the current
  geometry, but that observation is not universal.

## Not checked

Temperature-dependent strength, creep/resource, thermal stress, cooling,
third-stage final root reaction, `gamma/lambda/l/b_a/delta`, radial loss data
for M3, and real blade profiles are `NOT_CHECKED`.

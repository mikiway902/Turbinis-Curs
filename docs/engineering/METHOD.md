# Engineering method

## Scope

The project develops a three-stage TPE331-class turbine at 110 kW,
`T_g* = 1250 K`, and `pi_k* = 11`, with constant hub diameter. The immediate
task is physical closure of radial fields before further geometry search.

## Calculation layers

1. **Cycle** supplies gas flow, turbine pressure ratio, temperatures, and
   power demand.
2. **Meanline** calculates stage work and velocity triangles.
3. **Radial model** must satisfy momentum, state, energy, entropy/loss closure,
   boundary conditions, and integral mass flow simultaneously.
4. **Constraints** evaluate geometry, angles, reaction, strength, efficiency,
   speed, power, and invariants.
5. **Optimizer** searches only through the shared calculation core.

The current notebook and `Optimizer/` are the reproducible legacy
`book_approx`. New equations move to `turbine/` one at a time, with independent
tests, before callers are switched. The notebook remains an interface only
after that migration is complete.

## Change gate

Every new physical relation requires: nominal, boundary, analytical-limit,
dimensional, and regression checks. Important solvers also require an
independent implementation or residual evaluation, mesh dependence,
multiple initial guesses, positivity checks, and convergence-domain evidence.

No result is accepted when a required check is skipped. No baseline is updated
until the numerical change is explained and approved independently of the code
change.

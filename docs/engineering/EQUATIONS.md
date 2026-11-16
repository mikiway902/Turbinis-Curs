# Equation register

## Static-pressure radial equilibrium

**Status:** ACCEPTED  
**Class:** accepted physics

\[
\frac{1}{\rho}\frac{dp}{dr}=\frac{c_u^2}{r}
\]

Variables use SI units: `p [Pa]`, `rho [kg/m3]`, `c_u [m/s]`, `r [m]`.
Assumptions: steady axisymmetric flow, negligible radial velocity, no radial
body force. Implemented in `turbine/radial/equilibrium.py`. Verified by
`tests/physics/test_radial_equilibrium.py`.

## Equivalent thermodynamic residual

**Status:** ACCEPTED  
**Class:** accepted physics / independent implementation

\[
R_T=c_p\frac{dT}{dr}-T\frac{ds}{dr}-\frac{c_u^2}{r}
\]

For a thermodynamically consistent state it equals the pressure residual.
The entropy term is not added to the pressure momentum equation.

## General radial-equilibrium residual

**Status:** ACCEPTED  
**Class:** accepted physics / independent implementation

\[
R_{GRE}=\frac{dh_0}{dr}-T\frac{ds}{dr}
-c_a\frac{dc_a}{dr}-\frac{c_u}{r}\frac{d(rc_u)}{dr}.
\]

For consistent fields, `R_GRE = R_T = R_p` within the stated numerical
tolerance.

## Ideal-gas state and entropy difference

**Status:** ACCEPTED  
**Class:** accepted physics

\[
p=\rho R T,
\qquad
s-s_{ref}=c_p\ln(T/T_{ref})-R\ln(p/p_{ref}).
\]

Current scope assumes constant `R` and `c_p` inside one residual evaluation.

## Rotor energy relation used by system B

**Status:** EXPERIMENTAL  
**Class:** inherited `book_approx` closure

\[
T_2=T_1+\frac{w_1^2-w_2^2+u_2^2-u_1^2}{2c_p}.
\]

The relation is currently implemented in `Optimizer/rotor_exit_solve.py`.
Its coupling to streamline mapping and loss closure is not accepted for final
design evidence.

## Rotor loss entropy increment

**Status:** EXPERIMENTAL  
**Class:** empirical/book approximation

The current implementation derives `Delta s_l` from `w2` and `psi = 0.96`.
Its radial applicability and source-page evidence are incomplete. It must not
be modified merely to close stage 3.

## Coupled rotor-exit system B

**Status:** EXPERIMENTAL

Unknowns are `p2(r)`, `T2(r)`, `c2u(r)`, and one axial-flow scale (with state
density eliminated). Required equations are radial momentum, the existing
rotor energy relation, entropy/state matched to the loss model, integral mass
flow, and one pressure boundary condition. Gibbs is a consequence of state
entropy and is not an extra equation.

Current evidence: stage 1 residual `0.44–0.69%`, stage 2 `1.8–2.0%`, stage 3
`12–15%` under polynomial `c2u` parameterization. This is diagnostic evidence,
not accepted closure.

## Entropy-augmented pressure residual

**Status:** REJECTED

\[
\frac{1}{\rho}\frac{dp}{dr}-\frac{c_u^2}{r}+T\frac{ds}{dr}=0
\]

This double-counts entropy when interpreted as the radial momentum equation.
Previous `13.4%` alpha1 and `0.00–0.01%` Gamma closure claims derived from it
must not be reused.

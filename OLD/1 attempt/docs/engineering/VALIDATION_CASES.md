# Validation cases

## RADIAL-ANALYTIC-001 — isothermal solid-body swirl

For `c_u = omega r`, constant `T`, and ideal gas:

\[
p(r)=p_0\exp\left[\frac{\omega^2(r^2-r_0^2)}{2RT}\right].
\]

Checks the pressure residual against an analytical solution. Acceptance:
interior normalized residual below `2e-4` on the stored grid.

## RADIAL-ZERO-002 — zero swirl

For constant pressure and `c_u = 0`, radial residual is zero. This is the
boundary/limiting case.

## RADIAL-GIBBS-003 — independent residual equivalence

Construct `s(T,p)` from the accepted ideal-gas entropy difference and require
`R_p`, `R_T`, and `R_GRE` to agree within finite-difference tolerance.

## BOOK-APPROX-001 — legacy control point

`n=90000 rpm`, `D_hub=62 mm`, `w2=0.8125`, `w1=0.70`,
`kappa2=1.04375`, `kappa3=1.25`, `alpha1,1=13.10 deg`.
Stored results are diagnostic and must carry `model_status=book_approx`.
They may detect software drift but cannot validate a closed radial model.

## Required solver evidence

Coupled solvers additionally require at least three physically distinct initial
guesses, positivity of `p,T,rho,c_a`, `w2>c2a`, energy and integral-flow
closure, mesh refinement, and a separately evaluated radial residual. The
target accepted radial residual is at most `1%`.

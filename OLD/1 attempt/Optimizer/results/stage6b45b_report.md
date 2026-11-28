# Stage 6B-4.5-B — coupled rotor-exit closure

Engineering model: **EXPERIMENTAL**. A numerical pass does not promote the loss closure or the equal-span mapping to accepted physics.

## Control point

n = 90000 rpm; D_hub = 62 mm; w1 = 0.7; w2 = 0.8125; kappa2 = 1.04375; kappa3 = 1.25; alpha1_stage1 = 13.10 deg.

Laws: stage 1: Γ, stage 2: α1, stage 3: Γ.

## Provenance

Python 3.11.10; NumPy 2.0.2; SciPy 1.14.1; platform Linux-7.2.13-200.fc44.x86_64-x86_64-with-glibc2.39.

## Equation count

Unknowns: N axial log velocities `y_i = ln(c2a_i/c2a_ref)`.

The square branch solve uses N-1 finite-volume radial-momentum intervals + 1 integral mass equation = N equations. The `p2` construction enforces N-1 relative entropy/state equations plus the prescribed pressure boundary condition. The complete required set is therefore N-1 momentum + 1 mass + 1 absolute cross-rotor entropy condition = N+1 conditions on N unknowns. The absolute entropy condition is a compatibility residual of the prescribed control point; there is no physically allowed extra global unknown. `residual` and its analytic Jacobian are the square momentum+mass subsystem only, so their conditioning does not describe the N+1 compatibility set. Loss/state and the pressure BC are not claimed to be independently satisfied.

## Acceptance thresholds

epsilon_RRE <= 0.01; \|mass\| <= 0.005; work <= 1e-06; R-form difference <= 0.0001 (numerical finite-difference consistency cap, not a physics tolerance); smoothness <= 0.5; p,T,rho positive finite; w2>c2a; alpha2 70..90 deg stages 1-2 and 80..90 deg stage 3; entropy anchor max\|E_s\| <= 1.0e-06 J/(kg K) (strict algebraic numerical closure, not model uncertainty).

Solved branch member: valid and `max\|core.residual\| <= 1e-05` (dimensionless). Solver success alone is neither necessary nor sufficient; unfinished iterates are not branch confirmations or contradictions.

Classification rules: **YES** requires every finest-mesh criterion (including the absolute entropy anchor) to pass, at least two genuinely independent cold starts on the same numerically closed branch on some mesh, a branch continuously refined to the finest mesh with every best numerically closed, conditioning evidence, and no branch inconsistency. **NO** additionally requires the same failure on every mesh using numerically closed bests, robustness to the alternative pressure BC with a numerically closed alternative best, a clear margin and no continuing improvement with mesh. Warm/continuation starts are never counted as independent. A failed solver is never evidence of non-existence. Anything else is **INCONCLUSIVE**. Starts within 1% relative c2a of the best are treated as geometrically close for reporting; branch membership still requires numerical closure.

## Stage 1 (Γ)

**Classification: INCONCLUSIVE**
 — alpha2 robustness alternative-BC best is not numerically closed


| criterion | value | threshold | pass |
|---|---|---|---|
| epsilon_RRE | 0.0001% | <= 1.00% | yes |
| \|mass\| | 0.0005% | <= 0.50% | yes |
| energy residual | 2.107e-16 | algebraic | yes |
| work residual | 0.000e+00 | <= 1e-06 | yes |
| max\|R_T-R_p\|/scale | 1.018e-08 | <= 1.0e-04 | yes |
| max\|R_GRE-R_p\|/scale | 8.538e-07 | <= 1.0e-04 | yes |
| positive finite | yes | yes | yes |
| smoothness | 0.00154 | <= 0.5 | yes |
| min(w2-c2a) | 234.689 m/s | > 0 | yes |
| alpha2 range | 69.16..72.04 deg | 70..90 / 80..90 | no |
| entropy anchor max\|E_s\| | 1.534e-01 J/(kg K) | <= 1.0e-06 | no |
| equation residual max\|f\| | 4.584e-06 | <= 1e-05 | yes |

R-form numerical consistency: achieved max\|R_T-R_p\|/scale = 1.018e-08 and max\|R_GRE-R_p\|/scale = 8.538e-07, versus the numerical-consistency cap 1.0e-04. These are finite-difference discretization differences, not percent-level physical discrepancies.

Ranges: p 541.80..542.92 kPa; T 1076.1..1079.0 K; rho 1.7508..1.7521 kg/m3; c2u 35.85..50.63 m/s; c2a 110.6..133.0 m/s; alpha2 69.16..72.04 deg. lambda_a = 1.0113; dp/dr sign changes = 0.

Independent scales: max\|(1/rho)dp/dr\| = 8.270e+04 m/s2; max\|R_p\| = 6.548e-02; max\|R_T\| = 6.611e-02; max\|R_GRE\| = 8.233e-02 m/s2.

### Absolute cross-rotor entropy

| quantity | value [J/(kg K)] |
|---|---|
| min E_s | -1.533984e-01 |
| max E_s | -1.533984e-01 |
| mean E_s | -1.533984e-01 |
| max\|E_s\| | 1.533984e-01 |
| span (max-min) | 4.100720e-12 |
| anchor E_s | -1.533984e-01 |

`E_s = cp ln(T2/T1) - R ln(p2/p1) - Delta_s_l` uses absolute inlet and exit states. The relative `p2` construction makes `E_s` nearly constant across the span, so the derivative equivalence `R_p = R_T = R_GRE` cannot detect a constant `E_s` offset. Only the anchor/max absolute value tests that offset.

### Independent cold-start evidence

Independent cold-start confirmation: **yes**; confirming meshes: 41.

| N | cold starts on best branch | names |
|---|---|---|
| 41 | 2 | uniform, hub-biased |
| 81 | 0 | — |
| 161 | 0 | — |
| 401 | 0 | — |
| 801 | 0 | — |

| N coarse | N fine | max relative c2a difference | same branch |
|---|---|---|---|
| 41 | 81 | 7.583e-06 | yes |
| 81 | 161 | 1.904e-06 | yes |
| 161 | 401 | 7.029e-07 | yes |
| 401 | 801 | 7.551e-08 | yes |

First-to-finest relative c2a difference: 1.02e-05 (limit 1%).

No branch inconsistency among cold starts on the branch.

### Mesh and start evidence

| N | start | kind | success | nfev | cost | optimality | epsilon_RRE | mass | lambda | delta c2a [m/s] | relative | same | eq res max\|f\| | solver closed | entropy max\|E_s\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 41 | uniform | cold | yes | 15 | 1.764e-19 | 1.149e-07 | 0.0302% | 0.0000% | 1.0113 | 0 | 0.000e+00 | yes | 5.13e-10 | yes | 0.154 |
| 41 | hub-biased | cold | yes | 34 | 2.033e-19 | 2.499e-07 | 0.0302% | 0.0000% | 1.0113 | 1.352e-07 | 1.016e-09 | yes | 4.95e-10 | yes | 0.154 |
| 41 | tip-biased | cold | no | 400 | 1.212e-03 | 3.897e-02 | 0.0289% | 4.9224% | 1.0592 | 6.563 | 4.934e-02 | no | 0.0492 | no | 0.91 |
| 81 | uniform | cold | no | 400 | 6.566e-07 | 1.669e-04 | 0.0077% | 0.1146% | 1.0102 | 0.1542 | 1.159e-03 | yes | 0.00115 | no | 0.136 |
| 81 | hub-biased | cold | no | 400 | 3.046e-02 | 4.117e-02 | 0.0429% | 24.6823% | 0.7714 | 34.47 | 2.592e-01 | no | 0.247 | no | 3.1 |
| 81 | tip-biased | cold | no | 400 | 3.839e-02 | 2.611e-02 | 0.0180% | 27.7105% | 1.2809 | 36.14 | 2.717e-01 | no | 0.277 | no | 4.87 |
| 81 | warm | continuation | no | 400 | 6.153e-12 | 6.239e-07 | 0.0077% | 0.0004% | 1.0113 | 0 | 0.000e+00 | yes | 3.51e-06 | yes | 0.153 |
| 161 | uniform | cold | no | 400 | 2.504e-06 | 2.543e-04 | 0.0020% | 0.2238% | 1.0091 | 0.3008 | 2.262e-03 | yes | 0.00224 | no | 0.12 |
| 161 | hub-biased | cold | no | 400 | 3.092e-02 | 5.489e-03 | 0.0041% | 24.8671% | 0.7696 | 34.75 | 2.612e-01 | no | 0.249 | no | 3.12 |
| 161 | tip-biased | cold | no | 400 | 3.875e-02 | 1.145e-02 | 0.0069% | 27.8396% | 1.2821 | 36.31 | 2.730e-01 | no | 0.278 | no | 4.9 |
| 161 | warm | continuation | no | 400 | 9.574e-12 | 1.387e-05 | 0.0020% | 0.0004% | 1.0113 | 0 | 0.000e+00 | yes | 4.38e-06 | yes | 0.153 |
| 401 | uniform | cold | no | 400 | 2.916e-06 | 5.912e-05 | 0.0003% | 0.2415% | 1.0089 | 0.3246 | 2.440e-03 | yes | 0.00241 | no | 0.117 |
| 401 | hub-biased | cold | no | 400 | 3.177e-02 | 2.864e-03 | 0.0004% | 25.2070% | 0.7663 | 35.25 | 2.650e-01 | no | 0.252 | no | 3.16 |
| 401 | tip-biased | cold | no | 400 | 3.905e-02 | 1.721e-02 | 0.0005% | 27.9465% | 1.2832 | 36.44 | 2.740e-01 | no | 0.279 | no | 4.92 |
| 401 | warm | continuation | yes | 4 | 1.034e-11 | 2.487e-05 | 0.0003% | 0.0005% | 1.0113 | 0 | 0.000e+00 | yes | 4.55e-06 | yes | 0.153 |
| 801 | uniform | cold | yes | 163 | 3.186e-06 | 1.084e-04 | 0.0001% | 0.2524% | 1.0088 | 0.3392 | 2.550e-03 | yes | 0.00252 | no | 0.115 |
| 801 | hub-biased | cold | no | 400 | 3.181e-02 | 2.488e-03 | 0.0016% | 25.2217% | 0.7662 | 35.27 | 2.652e-01 | no | 0.252 | no | 3.16 |
| 801 | tip-biased | cold | no | 400 | 3.910e-02 | 4.222e-02 | 0.0021% | 27.9641% | 1.2833 | 36.47 | 2.741e-01 | no | 0.28 | no | 4.92 |
| 801 | warm | continuation | yes | 4 | 1.051e-11 | 1.018e-04 | 0.0001% | 0.0005% | 1.0113 | 0 | 0.000e+00 | yes | 4.58e-06 | yes | 0.153 |

Best finest branch spread: max c2a difference = 0 m/s (0 relative) over 1 numerically closed branch runs. Warm/continuation starts are excluded from independent confirmation.

Geometrically distant endpoints at the finest mesh: hub-biased (cold): 35.27 m/s, 2.652e-01 relative, numerically closed: no; tip-biased (cold): 36.47 m/s, 2.741e-01 relative, numerically closed: no. Distance alone does not establish distinct solved branches. Unclosed endpoints are unfinished, not evidence of non-existence.

### Geometrically near but numerically unfinished runs

| N | start | kind | eq res max\|f\| | entropy max\|E_s\| |
|---|---|---|---|---|
| 81 | uniform | cold | 0.00115 | 0.136 |
| 161 | uniform | cold | 0.00224 | 0.12 |
| 401 | uniform | cold | 0.00241 | 0.117 |
| 801 | uniform | cold | 0.00252 | 0.115 |

These iterates are geometrically close to the best branch but do not satisfy the solver-closure cap, so they are not confirmations, branch spread, continuity, inconsistency, or non-existence evidence.

### Conditioning evidence

Method: deterministic `onenormest(J) * onenormest(J^-1)` with fixed seed; `J^-1` applies a SuperLU `splu` factorization and its transpose solve. `J` is the N x N square momentum+mass subsystem Jacobian, not the N+1 full compatibility set (which adds the absolute entropy residual), so the estimate describes the branch solve only. Rank proxy is `min|U_ii| / max|U_ii|`. No pass threshold is invented; factorization/finiteness failure would make the evidence incomplete.

| N | factorization | cond_1 estimate | pivot ratio | \|J\|_1 est. | \|J^-1\|_1 est. | dense cond_1 | available |
|---|---|---|---|---|---|---|---|
| 41 | yes | 5.83e+04 | 0.00113 | 1444 | 40.37 | 5.861e+04 | yes |
| 81 | yes | 2.314e+05 | 0.0005652 | 2903 | 79.72 | — | yes |
| 161 | yes | 9.222e+05 | 0.0003556 | 5821 | 158.4 | — | yes |
| 401 | yes | 5.75e+06 | 0.0001423 | 1.458e+04 | 394.5 | — | yes |
| 801 | yes | 2.298e+07 | 7.113e-05 | 2.916e+04 | 788 | — | yes |

Finest-mesh conditioning available: **yes**.

### Tolerance sensitivity (N=801, initialized from the found branch)

| case | success | optimality | epsilon_RRE | mass | R-form max | lambda |
|---|---|---|---|---|---|---|
| standard | yes | 9.197e-05 | 0.0001% | 0.0005% | 8.557e-07 | 1.0113 |
| loose | yes | 9.197e-05 | 0.0001% | 0.0005% | 8.557e-07 | 1.0113 |

Standard vs loose field difference: 0 m/s (0 relative).

### BC sensitivity (finest mesh, hub anchor vs primary mean anchor)

- primary mean: epsilon_RRE=0.0001%, |mass|=0.0005%, p_hub=541.80 kPa, rho_hub=1.7519, alpha2 69.16/72.04 deg, lambda=1.0113
- alternative hub: epsilon_RRE=0.0001%, |mass|=0.1254%, p_hub=542.51 kPa, rho_hub=1.7541, alpha2 69.12/71.99 deg, lambda=1.0088
- hub-anchor sensitivity: 0.130% on p_hub. The hub anchor is a sensitivity case only; it is not selected by reaction/pass.

## Stage 2 (α1)

**Classification: INCONCLUSIVE**
 — entropy_anchor robustness alternative-BC best is not numerically closed


| criterion | value | threshold | pass |
|---|---|---|---|
| epsilon_RRE | 0.0003% | <= 1.00% | yes |
| \|mass\| | 0.0002% | <= 0.50% | yes |
| energy residual | 1.241e-16 | algebraic | yes |
| work residual | 0.000e+00 | <= 1e-06 | yes |
| max\|R_T-R_p\|/scale | 8.121e-08 | <= 1.0e-04 | yes |
| max\|R_GRE-R_p\|/scale | 1.379e-05 | <= 1.0e-04 | yes |
| positive finite | yes | yes | yes |
| smoothness | 0.00319 | <= 0.5 | yes |
| min(w2-c2a) | 208.327 m/s | > 0 | yes |
| alpha2 range | 74.39..84.92 deg | 70..90 / 80..90 | yes |
| entropy anchor max\|E_s\| | 3.060e-01 J/(kg K) | <= 1.0e-06 | no |
| equation residual max\|f\| | 1.518e-06 | <= 1e-05 | yes |

R-form numerical consistency: achieved max\|R_T-R_p\|/scale = 8.121e-08 and max\|R_GRE-R_p\|/scale = 1.379e-05, versus the numerical-consistency cap 1.0e-04. These are finite-difference discretization differences, not percent-level physical discrepancies.

Ranges: p 257.79..258.19 kPa; T 914.2..915.8 K; rho 0.9794..0.9827 kg/m3; c2u 16.93..48.91 m/s; c2a 175.0..190.7 m/s; alpha2 74.39..84.92 deg. lambda_a = 1.0081; dp/dr sign changes = 0.

Independent scales: max\|(1/rho)dp/dr\| = 7.717e+04 m/s2; max\|R_p\| = 2.091e-01; max\|R_T\| = 2.058e-01; max\|R_GRE\| = 8.551e-01 m/s2.

### Absolute cross-rotor entropy

| quantity | value [J/(kg K)] |
|---|---|
| min E_s | -3.059576e-01 |
| max E_s | -3.059576e-01 |
| mean E_s | -3.059576e-01 |
| max\|E_s\| | 3.059576e-01 |
| span (max-min) | 2.538414e-12 |
| anchor E_s | -3.059576e-01 |

`E_s = cp ln(T2/T1) - R ln(p2/p1) - Delta_s_l` uses absolute inlet and exit states. The relative `p2` construction makes `E_s` nearly constant across the span, so the derivative equivalence `R_p = R_T = R_GRE` cannot detect a constant `E_s` offset. Only the anchor/max absolute value tests that offset.

### Independent cold-start evidence

Independent cold-start confirmation: **yes**; confirming meshes: 41.

| N | cold starts on best branch | names |
|---|---|---|
| 41 | 3 | uniform, hub-biased, tip-biased |
| 81 | 1 | uniform |
| 161 | 0 | — |
| 401 | 0 | — |
| 801 | 0 | — |

| N coarse | N fine | max relative c2a difference | same branch |
|---|---|---|---|
| 41 | 81 | 3.563e-05 | yes |
| 81 | 161 | 8.267e-06 | yes |
| 161 | 401 | 1.738e-06 | yes |
| 401 | 801 | 3.356e-07 | yes |

First-to-finest relative c2a difference: 3.09e-05 (limit 1%).

No branch inconsistency among cold starts on the branch.

### Mesh and start evidence

| N | start | kind | success | nfev | cost | optimality | epsilon_RRE | mass | lambda | delta c2a [m/s] | relative | same | eq res max\|f\| | solver closed | entropy max\|E_s\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 41 | uniform | cold | yes | 13 | 1.011e-20 | 1.104e-07 | 0.1010% | 0.0000% | 1.0081 | 0 | 0.000e+00 | yes | 4.51e-11 | yes | 0.306 |
| 41 | hub-biased | cold | yes | 29 | 1.647e-20 | 1.495e-07 | 0.1010% | 0.0000% | 1.0081 | 3.864e-09 | 2.026e-11 | yes | 6.32e-11 | yes | 0.306 |
| 41 | tip-biased | cold | yes | 31 | 1.473e-18 | 1.732e-07 | 0.1010% | 0.0000% | 1.0081 | 3.295e-07 | 1.728e-09 | yes | 1.71e-09 | yes | 0.306 |
| 81 | uniform | cold | no | 400 | 2.203e-11 | 5.975e-06 | 0.0262% | 0.0007% | 1.0081 | 0.001342 | 7.035e-06 | yes | 6.64e-06 | yes | 0.306 |
| 81 | hub-biased | cold | no | 400 | 2.489e-02 | 2.313e-02 | 0.0265% | 22.3136% | 0.7925 | 44.73 | 2.346e-01 | no | 0.223 | no | 7.8 |
| 81 | tip-biased | cold | no | 400 | 4.050e-02 | 2.058e-02 | 0.0262% | 28.4599% | 1.2776 | 53.68 | 2.815e-01 | no | 0.285 | no | 13.3 |
| 81 | warm | continuation | yes | 31 | 3.521e-14 | 9.774e-07 | 0.0262% | 0.0000% | 1.0081 | 0 | 0.000e+00 | yes | 2.65e-07 | yes | 0.306 |
| 161 | uniform | cold | no | 400 | 2.496e-05 | 3.157e-04 | 0.0067% | 0.7065% | 1.0148 | 1.372 | 7.196e-03 | yes | 0.00707 | no | 0.593 |
| 161 | hub-biased | cold | no | 400 | 2.534e-02 | 1.499e-02 | 0.0074% | 22.5105% | 0.7905 | 45.14 | 2.367e-01 | no | 0.225 | no | 7.86 |
| 161 | tip-biased | cold | no | 400 | 4.085e-02 | 5.762e-02 | 0.0068% | 28.5847% | 1.2788 | 53.91 | 2.827e-01 | no | 0.286 | no | 13.3 |
| 161 | warm | continuation | yes | 4 | 6.272e-13 | 3.055e-06 | 0.0067% | 0.0001% | 1.0081 | 0 | 0.000e+00 | yes | 1.12e-06 | yes | 0.306 |
| 401 | uniform | cold | yes | 148 | 6.373e-06 | 6.891e-05 | 0.0011% | 0.3570% | 1.0115 | 0.6939 | 3.639e-03 | yes | 0.00357 | no | 0.451 |
| 401 | hub-biased | cold | no | 400 | 2.565e-02 | 2.900e-03 | 0.0011% | 22.6501% | 0.7892 | 45.43 | 2.382e-01 | no | 0.227 | no | 7.91 |
| 401 | tip-biased | cold | no | 400 | 4.092e-02 | 7.908e-03 | 0.0011% | 28.6089% | 1.2790 | 53.95 | 2.829e-01 | no | 0.286 | no | 13.4 |
| 401 | warm | continuation | yes | 4 | 1.100e-12 | 2.078e-05 | 0.0011% | 0.0001% | 1.0081 | 0 | 0.000e+00 | yes | 1.48e-06 | yes | 0.306 |
| 801 | uniform | cold | yes | 205 | 6.626e-06 | 1.346e-04 | 0.0003% | 0.3640% | 1.0116 | 0.7075 | 3.710e-03 | yes | 0.00364 | no | 0.454 |
| 801 | hub-biased | cold | no | 400 | 2.575e-02 | 3.465e-02 | 0.0015% | 22.6927% | 0.7888 | 45.52 | 2.387e-01 | no | 0.227 | no | 7.92 |
| 801 | tip-biased | cold | no | 400 | 4.095e-02 | 8.114e-02 | 0.0021% | 28.6179% | 1.2791 | 53.97 | 2.830e-01 | no | 0.286 | no | 13.4 |
| 801 | warm | continuation | yes | 4 | 1.152e-12 | 9.331e-05 | 0.0003% | 0.0002% | 1.0081 | 0 | 0.000e+00 | yes | 1.52e-06 | yes | 0.306 |

Best finest branch spread: max c2a difference = 0 m/s (0 relative) over 1 numerically closed branch runs. Warm/continuation starts are excluded from independent confirmation.

Geometrically distant endpoints at the finest mesh: hub-biased (cold): 45.52 m/s, 2.387e-01 relative, numerically closed: no; tip-biased (cold): 53.97 m/s, 2.830e-01 relative, numerically closed: no. Distance alone does not establish distinct solved branches. Unclosed endpoints are unfinished, not evidence of non-existence.

### Geometrically near but numerically unfinished runs

| N | start | kind | eq res max\|f\| | entropy max\|E_s\| |
|---|---|---|---|---|
| 161 | uniform | cold | 0.00707 | 0.593 |
| 401 | uniform | cold | 0.00357 | 0.451 |
| 801 | uniform | cold | 0.00364 | 0.454 |

These iterates are geometrically close to the best branch but do not satisfy the solver-closure cap, so they are not confirmations, branch spread, continuity, inconsistency, or non-existence evidence.

### Conditioning evidence

Method: deterministic `onenormest(J) * onenormest(J^-1)` with fixed seed; `J^-1` applies a SuperLU `splu` factorization and its transpose solve. `J` is the N x N square momentum+mass subsystem Jacobian, not the N+1 full compatibility set (which adds the absolute entropy residual), so the estimate describes the branch solve only. Rank proxy is `min|U_ii| / max|U_ii|`. No pass threshold is invented; factorization/finiteness failure would make the evidence incomplete.

| N | factorization | cond_1 estimate | pivot ratio | \|J\|_1 est. | \|J^-1\|_1 est. | dense cond_1 | available |
|---|---|---|---|---|---|---|---|
| 41 | yes | 1.122e+05 | 0.0007626 | 2808 | 39.97 | 1.122e+05 | yes |
| 81 | yes | 4.431e+05 | 0.0003813 | 5616 | 78.91 | — | yes |
| 161 | yes | 1.761e+06 | 0.0001871 | 1.123e+04 | 156.8 | — | yes |
| 401 | yes | 1.096e+07 | 7.485e-05 | 2.808e+04 | 390.5 | — | yes |
| 801 | yes | 4.38e+07 | 3.743e-05 | 5.616e+04 | 779.9 | — | yes |

Finest-mesh conditioning available: **yes**.

### Tolerance sensitivity (N=801, initialized from the found branch)

| case | success | optimality | epsilon_RRE | mass | R-form max | lambda |
|---|---|---|---|---|---|---|
| standard | yes | 9.366e-05 | 0.0003% | 0.0002% | 1.379e-05 | 1.0081 |
| loose | yes | 9.366e-05 | 0.0003% | 0.0002% | 1.379e-05 | 1.0081 |

Standard vs loose field difference: 0 m/s (0 relative).

### BC sensitivity (finest mesh, hub anchor vs primary mean anchor)

- primary mean: epsilon_RRE=0.0003%, |mass|=0.0002%, p_hub=257.79 kPa, rho_hub=0.9794, alpha2 74.39/84.92 deg, lambda=1.0081
- alternative hub: epsilon_RRE=0.0003%, |mass|=0.4914%, p_hub=258.10 kPa, rho_hub=0.9807, alpha2 74.45/84.93 deg, lambda=1.0116
- hub-anchor sensitivity: 0.123% on p_hub. The hub anchor is a sensitivity case only; it is not selected by reaction/pass.

## Stage 3 (Γ)

**Classification: INCONCLUSIVE**
 — finest-mesh best is not numerically closed (equation_residual_inf > 1e-05)


| criterion | value | threshold | pass |
|---|---|---|---|
| epsilon_RRE | 0.0002% | <= 1.00% | yes |
| \|mass\| | 0.2736% | <= 0.50% | yes |
| energy residual | 1.515e-16 | algebraic | yes |
| work residual | 0.000e+00 | <= 1e-06 | yes |
| max\|R_T-R_p\|/scale | 1.562e-05 | <= 1.0e-04 | yes |
| max\|R_GRE-R_p\|/scale | 3.129e-04 | <= 1.0e-04 | no |
| positive finite | yes | yes | yes |
| smoothness | 0.00249 | <= 0.5 | yes |
| min(w2-c2a) | 134.373 m/s | > 0 | yes |
| alpha2 range | 89.54..89.70 deg | 70..90 / 80..90 | yes |
| entropy anchor max\|E_s\| | 2.934e-01 J/(kg K) | <= 1.0e-06 | no |
| equation residual max\|f\| | 2.736e-03 | <= 1e-05 | no |

R-form numerical consistency: achieved max\|R_T-R_p\|/scale = 1.562e-05 and max\|R_GRE-R_p\|/scale = 3.129e-04, versus the numerical-consistency cap 1.0e-04. These are finite-difference discretization differences, not percent-level physical discrepancies.

Ranges: p 106.36..106.37 kPa; T 746.1..750.6 K; rho 0.4930..0.4960 kg/m3; c2u 1.23..2.04 m/s; c2a 234.5..254.9 m/s; alpha2 89.54..89.70 deg. lambda_a = 1.0040; dp/dr sign changes = 0.

Independent scales: max\|(1/rho)dp/dr\| = 1.348e+02 m/s2; max\|R_p\| = 2.874e-04; max\|R_T\| = 2.117e-03; max\|R_GRE\| = 4.219e-02 m/s2.

### Absolute cross-rotor entropy

| quantity | value [J/(kg K)] |
|---|---|
| min E_s | -2.933590e-01 |
| max E_s | -2.933590e-01 |
| mean E_s | -2.933590e-01 |
| max\|E_s\| | 2.933590e-01 |
| span (max-min) | 2.460254e-12 |
| anchor E_s | -2.933590e-01 |

`E_s = cp ln(T2/T1) - R ln(p2/p1) - Delta_s_l` uses absolute inlet and exit states. The relative `p2` construction makes `E_s` nearly constant across the span, so the derivative equivalence `R_p = R_T = R_GRE` cannot detect a constant `E_s` offset. Only the anchor/max absolute value tests that offset.

### Independent cold-start evidence

Independent cold-start confirmation: **no**; confirming meshes: none.

| N | cold starts on best branch | names |
|---|---|---|

| N coarse | N fine | max relative c2a difference | same branch |
|---|---|---|---|

First-to-finest relative c2a difference: — (limit 1%).

No branch inconsistency among cold starts on the branch.

### Mesh and start evidence

| N | start | kind | success | nfev | cost | optimality | epsilon_RRE | mass | lambda | delta c2a [m/s] | relative | same | eq res max\|f\| | solver closed | entropy max\|E_s\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 41 | uniform | cold | yes | 12 | 3.755e-06 | 3.882e-02 | 0.0765% | 0.2740% | 1.0041 | 0 | 0.000e+00 | yes | 0.00274 | no | 0.294 |
| 41 | hub-biased | cold | yes | 6 | 2.986e-02 | 1.563e-01 | 0.0765% | 24.4386% | 0.7758 | 60.18 | 2.361e-01 | no | 0.244 | no | 17.6 |
| 41 | tip-biased | cold | yes | 7 | 4.347e-02 | 6.476e-02 | 0.0765% | 29.4854% | 1.2714 | 68.55 | 2.689e-01 | no | 0.295 | no | 27.6 |
| 81 | uniform | cold | yes | 13 | 4.675e-06 | 1.258e-01 | 0.0199% | 0.3058% | 1.0038 | 0.07691 | 3.017e-04 | yes | 0.00306 | no | 0.267 |
| 81 | hub-biased | cold | yes | 7 | 3.288e-02 | 1.960e+00 | 0.0201% | 25.6445% | 0.7642 | 63.31 | 2.484e-01 | no | 0.256 | no | 18.4 |
| 81 | tip-biased | cold | yes | 8 | 4.153e-02 | 1.143e+00 | 0.0199% | 28.8202% | 1.2656 | 67.08 | 2.632e-01 | no | 0.288 | no | 27 |
| 81 | warm | continuation | yes | 4 | 3.746e-06 | 2.128e-01 | 0.0199% | 0.2737% | 1.0041 | 0 | 0.000e+00 | yes | 0.00274 | no | 0.293 |
| 161 | uniform | cold | yes | 15 | 4.931e-06 | 7.259e-01 | 0.0051% | 0.3141% | 1.0037 | 0.097 | 3.805e-04 | yes | 0.00314 | no | 0.26 |
| 161 | hub-biased | cold | yes | 9 | 3.304e-02 | 2.023e+00 | 0.0051% | 25.7047% | 0.7637 | 63.47 | 2.490e-01 | no | 0.257 | no | 18.4 |
| 161 | tip-biased | cold | yes | 10 | 4.174e-02 | 4.067e+00 | 0.0051% | 28.8919% | 1.2662 | 67.24 | 2.638e-01 | no | 0.289 | no | 27 |
| 161 | warm | continuation | yes | 4 | 3.744e-06 | 7.366e-01 | 0.0051% | 0.2737% | 1.0041 | 0 | 0.000e+00 | yes | 0.00274 | no | 0.293 |
| 401 | uniform | cold | yes | 16 | 4.942e-06 | 5.744e+00 | 0.0008% | 0.3144% | 1.0037 | 0.09782 | 3.837e-04 | yes | 0.00314 | no | 0.26 |
| 401 | hub-biased | cold | yes | 10 | 3.319e-02 | 6.363e+00 | 0.0023% | 25.7648% | 0.7631 | 63.63 | 2.496e-01 | no | 0.258 | no | 18.4 |
| 401 | tip-biased | cold | yes | 10 | 4.188e-02 | 2.388e+01 | 0.0027% | 28.9415% | 1.2667 | 67.35 | 2.642e-01 | no | 0.289 | no | 27.1 |
| 401 | warm | continuation | yes | 4 | 3.744e-06 | 5.232e+00 | 0.0008% | 0.2736% | 1.0041 | 0 | 0.000e+00 | yes | 0.00274 | no | 0.293 |
| 801 | uniform | cold | yes | 18 | 4.997e-06 | 2.565e+01 | 0.0002% | 0.3161% | 1.0037 | 0.102 | 4.002e-04 | yes | 0.00316 | no | 0.258 |
| 801 | hub-biased | cold | yes | 13 | 3.324e-02 | 1.820e+01 | 0.0048% | 25.7849% | 0.7629 | 63.68 | 2.498e-01 | no | 0.258 | no | 18.5 |
| 801 | tip-biased | cold | yes | 13 | 4.193e-02 | 5.764e+01 | 0.0052% | 28.9574% | 1.2668 | 67.38 | 2.643e-01 | no | 0.29 | no | 27.1 |
| 801 | warm | continuation | yes | 4 | 3.744e-06 | 2.165e+01 | 0.0002% | 0.2736% | 1.0040 | 0 | 0.000e+00 | yes | 0.00274 | no | 0.293 |

Best finest branch spread: max c2a difference = — m/s (— relative) over 0 numerically closed branch runs. Warm/continuation starts are excluded from independent confirmation.

Geometrically distant endpoints at the finest mesh: hub-biased (cold): 63.68 m/s, 2.498e-01 relative, numerically closed: no; tip-biased (cold): 67.38 m/s, 2.643e-01 relative, numerically closed: no. Distance alone does not establish distinct solved branches. Unclosed endpoints are unfinished, not evidence of non-existence.

### Geometrically near but numerically unfinished runs

| N | start | kind | eq res max\|f\| | entropy max\|E_s\| |
|---|---|---|---|---|
| 41 | uniform | cold | 0.00274 | 0.294 |
| 81 | uniform | cold | 0.00306 | 0.267 |
| 81 | warm | continuation | 0.00274 | 0.293 |
| 161 | uniform | cold | 0.00314 | 0.26 |
| 161 | warm | continuation | 0.00274 | 0.293 |
| 401 | uniform | cold | 0.00314 | 0.26 |
| 401 | warm | continuation | 0.00274 | 0.293 |
| 801 | uniform | cold | 0.00316 | 0.258 |
| 801 | warm | continuation | 0.00274 | 0.293 |

These iterates are geometrically close to the best branch but do not satisfy the solver-closure cap, so they are not confirmations, branch spread, continuity, inconsistency, or non-existence evidence.

### Conditioning evidence

Method: deterministic `onenormest(J) * onenormest(J^-1)` with fixed seed; `J^-1` applies a SuperLU `splu` factorization and its transpose solve. `J` is the N x N square momentum+mass subsystem Jacobian, not the N+1 full compatibility set (which adds the absolute entropy residual), so the estimate describes the branch solve only. Rank proxy is `min|U_ii| / max|U_ii|`. No pass threshold is invented; factorization/finiteness failure would make the evidence incomplete.

| N | factorization | cond_1 estimate | pivot ratio | \|J\|_1 est. | \|J^-1\|_1 est. | dense cond_1 | available |
|---|---|---|---|---|---|---|---|
| 41 | yes | 7.785e+07 | 9.532e-07 | 2.042e+06 | 38.13 | 7.801e+07 | yes |
| 81 | yes | 3.082e+08 | 4.766e-07 | 4.092e+06 | 75.31 | — | yes |
| 161 | yes | 1.226e+09 | 2.646e-07 | 8.192e+06 | 149.7 | — | yes |
| 401 | yes | 7.639e+09 | 1.058e-07 | 2.049e+07 | 372.8 | — | yes |
| 801 | yes | 3.052e+10 | 5.291e-08 | 4.099e+07 | 744.6 | — | yes |

Finest-mesh conditioning available: **yes**.

### Tolerance sensitivity (N=801, initialized from the found branch)

| case | success | optimality | epsilon_RRE | mass | R-form max | lambda |
|---|---|---|---|---|---|---|
| standard | yes | 2.071e+01 | 0.0002% | 0.2736% | 3.121e-04 | 1.0040 |
| loose | yes | 2.071e+01 | 0.0002% | 0.2736% | 3.121e-04 | 1.0040 |

Standard vs loose field difference: 0 m/s (0 relative).

### BC sensitivity (finest mesh, hub anchor vs primary mean anchor)

- primary mean: epsilon_RRE=0.0002%, |mass|=0.2736%, p_hub=106.36 kPa, rho_hub=0.4960, alpha2 89.54/89.70 deg, lambda=1.0040
- alternative hub: epsilon_RRE=0.0003%, |mass|=0.3188%, p_hub=106.37 kPa, rho_hub=0.4960, alpha2 89.54/89.70 deg, lambda=1.0036
- hub-anchor sensitivity: 0.000% on p_hub. The hub anchor is a sensitivity case only; it is not selected by reaction/pass.

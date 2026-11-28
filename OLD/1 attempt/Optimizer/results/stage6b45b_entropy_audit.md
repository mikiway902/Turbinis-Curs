# Stage 6B-4.5-B — absolute entropy-offset attribution

Engineering status: **EXPERIMENTAL**. This is a read-only diagnostic of saved fields. It is not a closure, a new solver, or a new existence decision. The original all-INCONCLUSIVE classification is unchanged.

## Scope and source

The ideal-gas state entropy `s - s_ref = cp ln(T/T_ref) - R ln(p/p_ref)` is **ACCEPTED** in `docs/engineering/EQUATIONS.md`. The inherited rotor energy relation and the `psi = 0.96` loss entropy are **EXPERIMENTAL/`book_approx`**. The meanline pressure is an inherited book approximation. No primary source-page applicability data (Mach, Reynolds, geometry, validity range) is available; this is a disclosed limitation.

Book source: `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb` cell 12, points 90–93 and 107; `Turbinis-Calc/mathcad/make_mathcad.py` lines 296–325. The relations are evaluated as inherited inputs, not re-derived or tuned.

Equation count: unchanged. The saved coupled system still has `N` unknowns and `N+1` compatibility conditions (`N-1` momentum + 1 mass + 1 absolute cross-rotor entropy). This audit adds no unknown and frees no boundary condition.

## Units, domains, tolerances

- SI units: `T [K]`, `p [Pa]`, `cp, R [J/(kg K)]`, `w2 [m/s]`, entropy terms `J/(kg K)`.

- Valid domain: finite positive `T`, `p`; `cp > 0`, `R > 0`; `0 < psi <= 1`; `h_l >= 0`; conditional log `0 <= x < 1`, `x = h_l/(cp T2)`.

- Budget reconstruction: `|sum of terms - core absolute_entropy_residual(anchor)| <= 1e-09 J/(kg K)` (algebraic identity, not model uncertainty).

- Book identity checks: energy identity 1e-09 K absolute / 1e-10 relative; `cp/R` identity 1e-10 relative. These are floating-point compatibility caps, not physical acceptance bands.

## Provenance and input integrity

Python 3.11.10; NumPy 2.0.2; SciPy 1.14.1; platform Linux-7.2.15-300.fc45.x86_64-x86_64-with-glibc2.39.

Saved results SHA-256: `0430003a53023174d80f3eff260df9eb6a0981a4f51863c9d854371e9e72beb1`.

Setup regeneration check: **yes**; 18 unique setups and 24 saved solves compared at rtol/atol 1e-12/1e-12; max relative difference `0.000e+00`. No mixed-history fields were used.

## Control point

n = 90000 rpm; D_hub = 62 mm; w1 = 0.7; w2 = 0.8125; kappa2 = 1.04375; kappa3 = 1.25; alpha1_stage1 = 13.10 deg; R = 287.4 J/(kg K); phi = 0.97.

Laws: stage 1: Γ, stage 2: α1, stage 3: Γ.

## Signed entropy budget at the meanline anchor

Definitions (all `J/(kg K)`):

`E_s = cp ln(T2/T1) - R ln(p2/p1) - ds_l`.

At anchor `a`:

`E_s = E_book + inlet_state_shift + exit_temperature_shift + exit_pressure_shift + loss_shift`,

`E_book = cp ln(T2*/T1*) - R ln(p2*/p1*) - h_l*/T2*`,

`inlet_state_shift = -cp ln(T1_a/T1*) + R ln(p1_a/p1*)`, `exit_temperature_shift = cp ln(T2_a/T2*)`, `exit_pressure_shift = -R ln(p2_a/p2*)`, `loss_shift = -(ds_l_a - h_l*/T2*)`.

| stage | E_book | inlet | exit T | exit p | loss | sum | E_s | mismatch | span | max\|E_s\| |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | +1.934922e-02 | +0.000000e+00 | -1.583677e-01 | -0.000000e+00 | -1.437988e-02 | -1.533984e-01 | -1.533984e-01 | -7.721601e-14 | +4.100720e-12 | 0.15339838 |
| 2 | +3.349168e-02 | +0.000000e+00 | -3.106461e-01 | -0.000000e+00 | -2.880314e-02 | -3.059576e-01 | -3.059576e-01 | -1.237899e-13 | +2.538414e-12 | 0.30595757 |
| 3 | +6.684005e-02 | +0.000000e+00 | -3.286411e-01 | -0.000000e+00 | -3.155790e-02 | -2.933590e-01 | -2.933590e-01 | -8.171241e-14 | +2.460254e-12 | 0.29335898 |

## Finest saved endpoint and input ratios

| stage | N | start | kind | valid | closed | eq res max\|f\| |
|---|---|---|---|---|---|---|
| 1 | 801 | warm | continuation | yes | yes | 4.58e-06 |
| 2 | 801 | warm | continuation | yes | yes | 1.52e-06 |
| 3 | 801 | warm | continuation | yes | no | 0.00274 |

| stage | T1_a/T1* | p1_a/p1* | T2_a/T2* | p2_a/p2* | ds_l_a/(h_l*/T2*) | c2a_a/c2a* | w2_a/w2* |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 0.999865 | 1 | 1.00214 | 1.01127 | 1.001 |
| 2 | 1 | 1 | 0.999728 | 1 | 1.0033 | 1.00807 | 1.00151 |
| 3 | 1 | 1 | 0.999704 | 1 | 1.0026 | 1.00405 | 1.00115 |

Signed term interpretation is algebraic. `exit_pressure_shift` is zero at the primary mean anchor because the saved boundary condition pins `p2(mean) = p2*`; this is a software assumption, not physical evidence. Inlet ratios identical to one at the mean anchor are likewise a construction observation.

## Inherited meanline identities

`energy_resid = (T2 - T2_ad) - h_l/cp` [K]. `conditional` uses `x = h_l/(cp T2)` and is marked NOT_APPLICABLE when its assumptions fail; no invalid log is evaluated.

| stage | h_l [J/kg] | h_l key diff | energy_resid [K] | cp/R diff | p2 recon rel. | ds_state [J/(kg K)] | h_l/T2 [J/(kg K)] | conditional | diff = -cp ln(1-x)-h_l/T2 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7256.021 | 0 | -4.44e-15 | 0 | 0 | 6.753062343 | 6.733713127 | APPLICABLE | 0.019349216 |
| 2 | 7979.4912 | 0 | 3.82e-14 | 0 | 0 | 8.758493203 | 8.72500152 | APPLICABLE | 0.033491683 |
| 3 | 9073.5635 | 0 | 7.11e-15 | 1.15e-16 | 0 | 12.19507285 | 12.1282328 | APPLICABLE | 0.06684005 |

`h_l key diff` is `h_l(from psi, w2) - h_l(stored key)` when `h_л`/`h_l` is present. For the saved control point it is floating-point negligible.

## All saved endpoints (72)

Audited: 72 endpoints; evaluated: 72; not evaluated: 0. Max absolute budget reconstruction error: `+2.122746e-13` J/(kg K) (atol 1e-09).

| stage | total | valid | closed | unfinished | invalid | evaluated |
|---|---|---|---|---|---|---|
| 1 | 24 | 24 | 8 | 16 | 0 | 24 |
| 2 | 24 | 24 | 10 | 14 | 0 | 24 |
| 3 | 24 | 24 | 0 | 24 | 0 | 24 |

| case type | total | valid | closed | unfinished | invalid | evaluated |
|---|---|---|---|---|---|---|
| bc_hub | 9 | 9 | 0 | 9 | 0 | 9 |
| mesh | 57 | 57 | 14 | 43 | 0 | 57 |
| tolerance | 6 | 6 | 4 | 2 | 0 | 6 |

`closed` means `valid` and `equation_residual_inf <= 1e-05`; `unfinished` means valid but not closed; `invalid` means the saved run was not valid. Unfinished and invalid runs are not used as existence evidence.

### Stage 1 (Γ) mesh/start/tolerance/BC table

| case | N | bc | start | kind | valid | closed | unfinished | eq res | E_anchor | E_book | inlet | exit T | exit p | loss | mismatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mesh N=41 | 41 | mean | uniform | cold | yes | yes | no | 5.13e-10 | -1.535470e-01 | +1.934922e-02 | +0.000000e+00 | -1.585040e-01 | -0.000000e+00 | -1.439226e-02 | -1.158240e-13 |
| mesh N=41 | 41 | mean | hub-biased | cold | yes | yes | no | 4.95e-10 | -1.535470e-01 | +1.934922e-02 | +0.000000e+00 | -1.585040e-01 | -0.000000e+00 | -1.439226e-02 | -1.469658e-13 |
| mesh N=41 | 41 | mean | tip-biased | cold | yes | no | yes | 0.0492 | -9.104701e-01 | +1.934922e-02 | +0.000000e+00 | -8.523982e-01 | -0.000000e+00 | -7.742109e-02 | +5.395684e-14 |
| mesh N=81 | 81 | mean | uniform | cold | yes | no | yes | 0.00115 | -1.361843e-01 | +1.934922e-02 | +0.000000e+00 | -1.425866e-01 | -0.000000e+00 | -1.294686e-02 | +6.111778e-14 |
| mesh N=81 | 81 | mean | hub-biased | cold | yes | no | yes | 0.247 | +3.100541e+00 | +1.934922e-02 | +0.000000e+00 | +2.825005e+00 | -0.000000e+00 | +2.561870e-01 | -8.526513e-14 |
| mesh N=81 | 81 | mean | tip-biased | cold | yes | no | yes | 0.277 | -4.871126e+00 | +1.934922e-02 | +0.000000e+00 | -4.482695e+00 | -0.000000e+00 | -4.077805e-01 | +4.440892e-15 |
| mesh N=81 | 81 | mean | warm | continuation | yes | yes | no | 3.51e-06 | -1.534362e-01 | +1.934922e-02 | +0.000000e+00 | -1.584024e-01 | -0.000000e+00 | -1.438303e-02 | -7.116530e-14 |
| mesh N=161 | 161 | mean | uniform | cold | yes | no | yes | 0.00224 | -1.197706e-01 | +1.934922e-02 | +0.000000e+00 | -1.275393e-01 | -0.000000e+00 | -1.158049e-02 | -5.676015e-14 |
| mesh N=161 | 161 | mean | hub-biased | cold | yes | no | yes | 0.249 | +3.121519e+00 | +1.934922e-02 | +0.000000e+00 | +2.844241e+00 | -0.000000e+00 | +2.579293e-01 | +2.886580e-14 |
| mesh N=161 | 161 | mean | tip-biased | cold | yes | no | yes | 0.278 | -4.895638e+00 | +1.934922e-02 | +0.000000e+00 | -4.505159e+00 | -0.000000e+00 | -4.098279e-01 | -6.661338e-14 |
| mesh N=161 | 161 | mean | warm | continuation | yes | yes | no | 4.38e-06 | -1.534083e-01 | +1.934922e-02 | +0.000000e+00 | -1.583768e-01 | -0.000000e+00 | -1.438071e-02 | -2.259304e-14 |
| mesh N=401 | 401 | mean | uniform | cold | yes | no | yes | 0.00241 | -1.171094e-01 | +1.934922e-02 | +0.000000e+00 | -1.250996e-01 | -0.000000e+00 | -1.135896e-02 | -1.026401e-13 |
| mesh N=401 | 401 | mean | hub-biased | cold | yes | no | yes | 0.252 | +3.159837e+00 | +1.934922e-02 | +0.000000e+00 | +2.879376e+00 | -0.000000e+00 | +2.611117e-01 | +6.128431e-14 |
| mesh N=401 | 401 | mean | tip-biased | cold | yes | no | yes | 0.279 | -4.915991e+00 | +1.934922e-02 | +0.000000e+00 | -4.523812e+00 | -0.000000e+00 | -4.115280e-01 | -1.776357e-14 |
| mesh N=401 | 401 | mean | warm | continuation | yes | yes | no | 4.55e-06 | -1.533995e-01 | +1.934922e-02 | +0.000000e+00 | -1.583687e-01 | -0.000000e+00 | -1.437997e-02 | -4.188316e-14 |
| mesh N=801 | 801 | mean | uniform | cold | yes | no | yes | 0.00252 | -1.154715e-01 | +1.934922e-02 | +0.000000e+00 | -1.235981e-01 | -0.000000e+00 | -1.122261e-02 | -9.582612e-14 |
| mesh N=801 | 801 | mean | hub-biased | cold | yes | no | yes | 0.252 | +3.161489e+00 | +1.934922e-02 | +0.000000e+00 | +2.880891e+00 | -0.000000e+00 | +2.612489e-01 | -3.286260e-14 |
| mesh N=801 | 801 | mean | tip-biased | cold | yes | no | yes | 0.28 | -4.919332e+00 | +1.934922e-02 | +0.000000e+00 | -4.526874e+00 | -0.000000e+00 | -4.118071e-01 | +8.881784e-16 |
| mesh N=801 | 801 | mean | warm | continuation | yes | yes | no | 4.58e-06 | -1.533984e-01 | +1.934922e-02 | +0.000000e+00 | -1.583677e-01 | -0.000000e+00 | -1.437988e-02 | -7.721601e-14 |
| tolerance/standard | 801 | mean | branch | continuation | yes | yes | no | 4.58e-06 | -1.533984e-01 | +1.934922e-02 | +0.000000e+00 | -1.583677e-01 | -0.000000e+00 | -1.437988e-02 | +1.360023e-14 |
| tolerance/loose | 801 | mean | branch | continuation | yes | yes | no | 4.58e-06 | -1.533984e-01 | +1.934922e-02 | +0.000000e+00 | -1.583677e-01 | -0.000000e+00 | -1.437988e-02 | +1.360023e-14 |
| bc_hub | 801 | hub | uniform | cold | yes | no | yes | 0.00125 | -4.873643e-01 | +1.934922e-02 | -3.357051e-01 | -1.563610e+00 | -0.000000e+00 | +1.392601e+00 | -7.394085e-14 |
| bc_hub | 801 | hub | hub-biased | cold | yes | no | yes | 0.251 | +2.789671e+00 | +1.934922e-02 | -3.357051e-01 | +1.444236e+00 | -0.000000e+00 | +1.661791e+00 | +6.483702e-14 |
| bc_hub | 801 | hub | tip-biased | cold | yes | no | yes | 0.282 | -5.303580e+00 | +1.934922e-02 | -3.357051e-01 | -5.983052e+00 | -0.000000e+00 | +9.958279e-01 | -7.194245e-14 |

### Stage 2 (α1) mesh/start/tolerance/BC table

| case | N | bc | start | kind | valid | closed | unfinished | eq res | E_anchor | E_book | inlet | exit T | exit p | loss | mismatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mesh N=41 | 41 | mean | uniform | cold | yes | yes | no | 4.51e-11 | -3.064559e-01 | +3.349168e-02 | +0.000000e+00 | -3.111021e-01 | -0.000000e+00 | -2.884543e-02 | -2.853273e-14 |
| mesh N=41 | 41 | mean | hub-biased | cold | yes | yes | no | 6.32e-11 | -3.064559e-01 | +3.349168e-02 | +0.000000e+00 | -3.111021e-01 | -0.000000e+00 | -2.884543e-02 | -8.476553e-14 |
| mesh N=41 | 41 | mean | tip-biased | cold | yes | yes | no | 1.71e-09 | -3.064560e-01 | +3.349168e-02 | +0.000000e+00 | -3.111022e-01 | -0.000000e+00 | -2.884544e-02 | -7.827072e-14 |
| mesh N=81 | 81 | mean | uniform | cold | yes | yes | no | 6.64e-06 | -3.063964e-01 | +3.349168e-02 | +0.000000e+00 | -3.110477e-01 | -0.000000e+00 | -2.884038e-02 | -9.103829e-15 |
| mesh N=81 | 81 | mean | hub-biased | cold | yes | no | yes | 0.223 | +7.799183e+00 | +3.349168e-02 | +0.000000e+00 | +7.108707e+00 | -0.000000e+00 | +6.569841e-01 | +6.394885e-14 |
| mesh N=81 | 81 | mean | tip-biased | cold | yes | no | yes | 0.285 | -1.328593e+01 | +3.349168e-02 | +0.000000e+00 | -1.218384e+01 | -0.000000e+00 | -1.135579e+00 | -2.664535e-14 |
| mesh N=81 | 81 | mean | warm | continuation | yes | yes | no | 2.65e-07 | -3.061167e-01 | +3.349168e-02 | +0.000000e+00 | -3.107918e-01 | -0.000000e+00 | -2.881665e-02 | +3.058664e-14 |
| mesh N=161 | 161 | mean | uniform | cold | yes | no | yes | 0.00707 | -5.932534e-01 | +3.349168e-02 | +0.000000e+00 | -5.735585e-01 | -0.000000e+00 | -5.318653e-02 | +5.084821e-14 |
| mesh N=161 | 161 | mean | hub-biased | cold | yes | no | yes | 0.225 | +7.862281e+00 | +3.349168e-02 | +0.000000e+00 | +7.166482e+00 | -0.000000e+00 | +6.623069e-01 | -1.012523e-13 |
| mesh N=161 | 161 | mean | tip-biased | cold | yes | no | yes | 0.286 | -1.334892e+01 | +3.349168e-02 | +0.000000e+00 | -1.224144e+01 | -0.000000e+00 | -1.140976e+00 | +1.598721e-14 |
| mesh N=161 | 161 | mean | warm | continuation | yes | yes | no | 1.12e-06 | -3.060000e-01 | +3.349168e-02 | +0.000000e+00 | -3.106849e-01 | -0.000000e+00 | -2.880674e-02 | +3.058664e-14 |
| mesh N=401 | 401 | mean | uniform | cold | yes | no | yes | 0.00357 | -4.509233e-01 | +3.349168e-02 | +0.000000e+00 | -4.433089e-01 | -0.000000e+00 | -4.110604e-02 | -7.061018e-14 |
| mesh N=401 | 401 | mean | hub-biased | cold | yes | no | yes | 0.227 | +7.906888e+00 | +3.349168e-02 | +0.000000e+00 | +7.207327e+00 | -0.000000e+00 | +6.660698e-01 | -6.306067e-14 |
| mesh N=401 | 401 | mean | tip-biased | cold | yes | no | yes | 0.286 | -1.336114e+01 | +3.349168e-02 | +0.000000e+00 | -1.225261e+01 | -0.000000e+00 | -1.142023e+00 | +8.881784e-15 |
| mesh N=401 | 401 | mean | warm | continuation | yes | yes | no | 1.48e-06 | -3.059623e-01 | +3.349168e-02 | +0.000000e+00 | -3.106504e-01 | -0.000000e+00 | -2.880354e-02 | +2.386980e-15 |
| mesh N=801 | 801 | mean | uniform | cold | yes | no | yes | 0.00364 | -4.537698e-01 | +3.349168e-02 | +0.000000e+00 | -4.459139e-01 | -0.000000e+00 | -4.134763e-02 | -7.799317e-14 |
| mesh N=801 | 801 | mean | hub-biased | cold | yes | no | yes | 0.227 | +7.920472e+00 | +3.349168e-02 | +0.000000e+00 | +7.219764e+00 | -0.000000e+00 | +6.672156e-01 | -2.122746e-13 |
| mesh N=801 | 801 | mean | tip-biased | cold | yes | no | yes | 0.286 | -1.336569e+01 | +3.349168e-02 | +0.000000e+00 | -1.225677e+01 | -0.000000e+00 | -1.142412e+00 | -4.440892e-14 |
| mesh N=801 | 801 | mean | warm | continuation | yes | yes | no | 1.52e-06 | -3.059576e-01 | +3.349168e-02 | +0.000000e+00 | -3.106461e-01 | -0.000000e+00 | -2.880314e-02 | -1.237899e-13 |
| tolerance/standard | 801 | mean | branch | continuation | yes | yes | no | 1.52e-06 | -3.059576e-01 | +3.349168e-02 | +0.000000e+00 | -3.106461e-01 | -0.000000e+00 | -2.880314e-02 | -1.074696e-13 |
| tolerance/loose | 801 | mean | branch | continuation | yes | yes | no | 1.52e-06 | -3.059576e-01 | +3.349168e-02 | +0.000000e+00 | -3.106461e-01 | -0.000000e+00 | -2.880314e-02 | -1.074696e-13 |
| bc_hub | 801 | hub | uniform | cold | yes | no | yes | 0.00491 | -8.078665e-01 | +3.349168e-02 | -4.161112e+00 | +1.433554e+00 | -0.000000e+00 | +1.886200e+00 | -2.025047e-13 |
| bc_hub | 801 | hub | hub-biased | cold | yes | no | yes | 0.225 | +7.547255e+00 | +3.349168e-02 | -4.161112e+00 | +9.093520e+00 | -0.000000e+00 | +2.581354e+00 | -9.592327e-14 |
| bc_hub | 801 | hub | tip-biased | cold | yes | no | yes | 0.288 | -1.374844e+01 | +3.349168e-02 | -4.161112e+00 | -1.042188e+01 | -0.000000e+00 | +8.010664e-01 | -8.881784e-15 |

### Stage 3 (Γ) mesh/start/tolerance/BC table

| case | N | bc | start | kind | valid | closed | unfinished | eq res | E_anchor | E_book | inlet | exit T | exit p | loss | mismatch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mesh N=41 | 41 | mean | uniform | cold | yes | no | yes | 0.00274 | -2.937719e-01 | +6.684005e-02 | +0.000000e+00 | -3.290179e-01 | -0.000000e+00 | -3.159409e-02 | -5.828671e-14 |
| mesh N=41 | 41 | mean | hub-biased | cold | yes | no | yes | 0.244 | +1.759217e+01 | +6.684005e-02 | +0.000000e+00 | +1.600017e+01 | -0.000000e+00 | +1.525161e+00 | -1.314504e-13 |
| mesh N=41 | 41 | mean | tip-biased | cold | yes | no | yes | 0.295 | -2.762912e+01 | +6.684005e-02 | +0.000000e+00 | -2.524441e+01 | -0.000000e+00 | -2.451556e+00 | -1.136868e-13 |
| mesh N=81 | 81 | mean | uniform | cold | yes | no | yes | 0.00306 | -2.670339e-01 | +6.684005e-02 | +0.000000e+00 | -3.046228e-01 | -0.000000e+00 | -2.925121e-02 | -6.550316e-15 |
| mesh N=81 | 81 | mean | hub-biased | cold | yes | no | yes | 0.256 | +1.837257e+01 | +6.684005e-02 | +0.000000e+00 | +1.671312e+01 | -0.000000e+00 | +1.592609e+00 | -6.750156e-14 |
| mesh N=81 | 81 | mean | tip-biased | cold | yes | no | yes | 0.288 | -2.696069e+01 | +6.684005e-02 | +0.000000e+00 | -2.463574e+01 | -0.000000e+00 | -2.391788e+00 | -7.815970e-14 |
| mesh N=81 | 81 | mean | warm | continuation | yes | no | yes | 0.00274 | -2.934639e-01 | +6.684005e-02 | +0.000000e+00 | -3.287369e-01 | -0.000000e+00 | -3.156710e-02 | -5.007106e-14 |
| mesh N=161 | 161 | mean | uniform | cold | yes | no | yes | 0.00314 | -2.600570e-01 | +6.684005e-02 | +0.000000e+00 | -2.982572e-01 | -0.000000e+00 | -2.863988e-02 | -1.076916e-14 |
| mesh N=161 | 161 | mean | hub-biased | cold | yes | no | yes | 0.257 | +1.841138e+01 | +6.684005e-02 | +0.000000e+00 | +1.674858e+01 | -0.000000e+00 | +1.595963e+00 | -7.105427e-15 |
| mesh N=161 | 161 | mean | tip-biased | cold | yes | no | yes | 0.289 | -2.703243e+01 | +6.684005e-02 | +0.000000e+00 | -2.470107e+01 | -0.000000e+00 | -2.398201e+00 | -9.592327e-14 |
| mesh N=161 | 161 | mean | warm | continuation | yes | no | yes | 0.00274 | -2.933867e-01 | +6.684005e-02 | +0.000000e+00 | -3.286665e-01 | -0.000000e+00 | -3.156034e-02 | -5.367928e-14 |
| mesh N=401 | 401 | mean | uniform | cold | yes | no | yes | 0.00314 | -2.597510e-01 | +6.684005e-02 | +0.000000e+00 | -2.979780e-01 | -0.000000e+00 | -2.861307e-02 | -8.715251e-14 |
| mesh N=401 | 401 | mean | hub-biased | cold | yes | no | yes | 0.258 | +1.844994e+01 | +6.684005e-02 | +0.000000e+00 | +1.678381e+01 | -0.000000e+00 | +1.599295e+00 | +5.684342e-14 |
| mesh N=401 | 401 | mean | tip-biased | cold | yes | no | yes | 0.289 | -2.708215e+01 | +6.684005e-02 | +0.000000e+00 | -2.474634e+01 | -0.000000e+00 | -2.402646e+00 | -9.592327e-14 |
| mesh N=401 | 401 | mean | warm | continuation | yes | no | yes | 0.00274 | -2.933621e-01 | +6.684005e-02 | +0.000000e+00 | -3.286439e-01 | -0.000000e+00 | -3.155818e-02 | +6.017409e-14 |
| mesh N=801 | 801 | mean | uniform | cold | yes | no | yes | 0.00316 | -2.583068e-01 | +6.684005e-02 | +0.000000e+00 | -2.966603e-01 | -0.000000e+00 | -2.848652e-02 | +6.117329e-14 |
| mesh N=801 | 801 | mean | hub-biased | cold | yes | no | yes | 0.258 | +1.846287e+01 | +6.684005e-02 | +0.000000e+00 | +1.679562e+01 | -0.000000e+00 | +1.600411e+00 | -4.618528e-14 |
| mesh N=801 | 801 | mean | tip-biased | cold | yes | no | yes | 0.29 | -2.709805e+01 | +6.684005e-02 | +0.000000e+00 | -2.476082e+01 | -0.000000e+00 | -2.404068e+00 | -4.263256e-14 |
| mesh N=801 | 801 | mean | warm | continuation | yes | no | yes | 0.00274 | -2.933590e-01 | +6.684005e-02 | +0.000000e+00 | -3.286411e-01 | -0.000000e+00 | -3.155790e-02 | -8.171241e-14 |
| tolerance/standard | 801 | mean | branch | continuation | yes | no | yes | 0.00274 | -2.933590e-01 | +6.684005e-02 | +0.000000e+00 | -3.286411e-01 | -0.000000e+00 | -3.155790e-02 | -4.907186e-14 |
| tolerance/loose | 801 | mean | branch | continuation | yes | no | yes | 0.00274 | -2.933590e-01 | +6.684005e-02 | +0.000000e+00 | -3.286411e-01 | -0.000000e+00 | -3.155790e-02 | -4.907186e-14 |
| bc_hub | 801 | hub | uniform | cold | yes | no | yes | 0.00319 | -2.569326e-01 | +6.684005e-02 | -9.051880e-01 | -2.911503e+00 | -0.000000e+00 | +3.492918e+00 | +4.218847e-14 |
| bc_hub | 801 | hub | hub-biased | cold | yes | no | yes | 0.257 | +1.843813e+01 | +6.684005e-02 | -9.051880e-01 | +1.420622e+01 | -0.000000e+00 | +5.070259e+00 | +8.881784e-14 |
| bc_hub | 801 | hub | tip-biased | cold | yes | no | yes | 0.292 | -2.734254e+01 | +6.684005e-02 | -9.051880e-01 | -2.767198e+01 | -0.000000e+00 | +1.167787e+00 | +4.973799e-14 |

## Interpretation

The decomposition is diagnostic only. It does not assign a true physical cause to the residual and it does not average alternative closures. On the finest saved primary endpoints:

- Stage 1: `exit_temperature_shift` has the largest absolute term (-1.583677e-01 J/(kg K)); `E_book` is +1.934922e-02 J/(kg K) while `E_s` is -1.533984e-01 J/(kg K). The sum of terms reproduces `E_s` to -7.721601e-14 J/(kg K).
- Stage 2: `exit_temperature_shift` has the largest absolute term (-3.106461e-01 J/(kg K)); `E_book` is +3.349168e-02 J/(kg K) while `E_s` is -3.059576e-01 J/(kg K). The sum of terms reproduces `E_s` to -1.237899e-13 J/(kg K).
- Stage 3: `exit_temperature_shift` has the largest absolute term (-3.286411e-01 J/(kg K)); `E_book` is +6.684005e-02 J/(kg K) while `E_s` is -2.933590e-01 J/(kg K). The sum of terms reproduces `E_s` to -8.171241e-14 J/(kg K).

Across the three primary stages `book_offset` is positive and small compared with the negative `E_s`; `exit_temperature_shift` is negative and dominant; `exit_pressure_shift` is zero at the mean anchor because of the prescribed pressure boundary condition; `inlet_state_shift` is zero at the mean anchor; and `loss_shift` is a negative secondary term. Therefore the hypothesis that the book loss/energy approximation alone explains the offset is **not sufficient** on these saved fields. The dominant term is the difference between the radial-solution exit temperature at the anchor and the book meanline `T2`, which follows from the solved axial scale and radial field, not from a new closure. This statement is an observation, not a license to modify the meanline or radial values.

Hub-anchor sensitivity endpoints show different offsets and are reported in the JSON endpoint rows. They are sensitivity cases, not selected design evidence and not an alternative closure.

## Tests and limitations

- `tests/optimization/test_entropy_budget.py` checks exact synthetic budget reconstruction, each perturbed term (including pressure), zero loss at `psi=1`, small-`x` logarithmic comparison, domain rejection (`x>=1`, non-positive `p/T`), conditional meanline applicability, and the saved three-stage identity when artifacts are present.

- No new nonlinear sweep, no mesh refinement, and no new initial guesses were run. Saved solver closure and labels are as recorded; no run is upgraded to existence evidence.

- The book meanline pressure and loss are inherited `book_approx`; their radial applicability and primary source page are not established. The conditional log identity is reported only when `T2 - T2_ad = h_l/cp` holds numerically.

- Not checked: primary source-page applicability and validity range; independent radial closure; stator-exit closure; temperature strength, creep/resource, thermal stress, cooling, stage-3 root reaction, and real blade profiles. No mass/angle pass or existence claim is made.

- The `book_offset` positive values are consistent with the conditional identity `-cp ln(1-x) - h_l/T2 = cp x^2/2 + ...`, which is positive; they are not evidence that the experimental loss model is accepted.

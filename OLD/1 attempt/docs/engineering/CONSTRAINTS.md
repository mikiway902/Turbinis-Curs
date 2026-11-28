# Project constraints

These limits may not be relaxed to obtain feasibility.

| Quantity | Requirement |
|---|---|
| stages | exactly 3 |
| shaft power | `N = 110 kW` |
| turbine inlet total temperature | `T_g* = 1250 K` |
| compressor pressure ratio | `pi_k* = 11` |
| stage 1–2 `alpha2` | `70..90 deg` |
| stage 3 `alpha2` | `80..90 deg` |
| root reaction | `rho_root >= 0.05` |
| root `alpha1` | `>= 11 deg` |
| total turbine efficiency | `eta_t* >= 0.90` |
| blade stress | `sigma <= 300 MPa` |
| speed margin | `q_n <= 0.995` |
| blade speed | `u <= 400 m/s` |
| rotor heights | `16 <= l_rk1 < l_rk2 < l_rk3`, preferred `l_rk1=20 mm` |

`q_n = sqrt(sigma_max / 300)` and stress are mathematically linked and do not
constitute independent physical evidence. Category E requires rotor geometry,
angles/reaction, strength, efficiency, power, and invariant checks.

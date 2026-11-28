# Clearance variant delta_rel = 0.003, delta_min = 0.15 mm

**Status: EXPERIMENTAL diagnostic. Not accepted design evidence.**

## Requested case vs legacy book_approx

| quantity | legacy book_approx | requested diagnostic |
|---|---|---|
| delta_rel | 0.005 | 0.003 |
| delta_min | 0.2 mm | 0.15 mm |

- Applied relation: `max(delta_rel*l_RK, delta_min)` (software assumption, user-prescribed parameters, not a verified manufacturing limit).
- Relative-branch crossover: 50 mm (legacy 40 mm); the current rotor heights are below both, so both cases sit on their floors.
- This is a new explicit diagnostic entry point, **not** a global default
  migration: no existing default, notebook, core module or baseline was edited.
- No new unknowns, equations or solver. One fixed meanline control point is
  reused with the existing flow-to-power iteration.

## Classification

- `applied_gap`: SOFTWARE ASSUMPTION: user-prescribed 0.003 / 0.15 mm; not a verified manufacturing limit
- `hz_correlation`: EMPIRICAL CORRELATION / EXPERIMENTAL: Main_Turbinis_Calc_Book points 113-115, formula unchanged; applicability and uncertainty not verified
- `friction_loss_and_energy_coupling`: EXPERIMENTAL legacy book_approx
- `ideal_gas_state`: ACCEPTED
- `new_unknowns`: 0
- `new_equations`: 0
- `new_solver`: False

## Software checks (diagnostic, fail-closed)

Software acceptance: **PASS** (problems: none).
Scope: diagnostic project checks only; not a full Category E acceptance.

## Engineering acceptance: FAIL (EXPERIMENTAL, no verified claim)

Engineering acceptance: **FAIL**; feasible = False; diagnostic project checks passed = True.
- Category E is not satisfied at this control point: l_RK1 < 16 mm and root reaction 1 < 0.05
- the 0.003 / 0.15 mm gap is a user software assumption, not a thermal or mechanical approval
- radial closure, stage-3 root reaction and stator-exit closure are not validated
- standard verifier baseline gate: FAIL in this working tree (reference/ untracked); verify.py not rerun here
- no overall 'verified' claim is made.

## Root reaction estimate: approximate geometry exponent

- method: `approximate_geometry_exponent_estimate` (not a radial profile, not the baseline law)
- source: Optimizer/optimize_extended.py:194-199 -> rho_root = 1 - (1-rho) * (D_c/(D_c-l_c))**(2*cos(alpha1)**2)
- assumption: alpha1 = const on stages 1-2; D_c and l_c from meanline geometry; geometric exponent estimate, not a radial field
- stages 1-2, old case: 0.047859, 0.067304; new case: 0.048358, 0.067334
- `BOOK-APPROX-001` stores [0.0609, 0.0673]: Stage-1 estimate differs from the law-specific baseline diagnostic 0.0609 (stage 2 agrees to 0.0673). Values are reported as-is, never averaged or substituted; the physical model and the baseline are unchanged.

## Source hashes (before == after)

| file | sha256 |
|---|---|
| `Mathcad_ST_cooling_Jupyter.ipynb` | `095c31129b88d0f21b1e4b058ca7213416fd979074607a47a367815bd3b6161f` |
| `Anton-Mathcad_ST_cooling_Jupyter (2).ipynb` | `6804c290b2f243e6b0f47458145c3cbef3ccbd324442df82f4e658805bc0c651` |
| `Sergey-TVD_pi6_sequential.ipynb` | `f24d18fa4928c30fda67057ae2ef65dc9f57f94633c72115e2e55d172c295352` |
| `Turbinis-Cycle/Turbinis-Cycle.ipynb` | `e0fa542d7d2b5f02c01a926a24ffe9b0eb96e93ff7e85cb4823fd03a65f7f34a` |
| `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb` | `494c97d4acf8fafa14e196aeb874d0cd7eab362f934b93290845f2aea2375728` |
| `Optimizer/optimize_extended.py` | `85677873bb2f9344a83a58abf70ffd245ebbe9f3c0b4bac1766aa89a9f2cce53` |
| `Optimizer/optimizer_core.py` | `b5d8226a18dd011d2544c4968491b19b6a74046062463901ada4d71e1c9ac7b4` |
| `Optimizer/rotor_exit_coupled.py` | `75879e2b4aa7baca0df34badff44e85775d4cb2d1431225538c912c001498df6` |
| `turbine/radial/__init__.py` | `c447dd0d95c8a8474a43c551f83d1946777647b35c17e579057df630063ca8f7` |
| `turbine/radial/entropy_budget.py` | `dd99114a864aae7179f16a20e4e0c5364ac56ef737a1bd68b4f6adf5a651191c` |
| `turbine/radial/equilibrium.py` | `381bfa2d9b13cf565eae7f7c90bf3770aef96e04604d467fc35aaae899d16eaf` |
| `turbine/radial/rotor_exit.py` | `aed63962802750313d43f3930c4a97723539955ecf45ef515887008810564c64` |
| `reference/baselines/book_approx_control_001.json` | `0db0e5e0275e08d806cc03839111337a4be80647d8284a51a31f8314a1359af9` |
| `reference/baselines/radial_analytic_001.json` | `70c8dadb051aa1e623256cb6525fd1c8f20c2ff0c37ae7595d3d3f04cb03a7c4` |
| `Optimizer/results/stage6b45b_entropy_audit.json` | `3426ce0e92e68380961a64261070e29119a6aa552b2da51b69fe06e0883b1ab9` |
| `Optimizer/results/stage6b45b_entropy_audit.md` | `46ff5a472d084f7a6d4a5b2383c7e834c16e50870cad424bd99737f99a6e5624` |
| `Optimizer/results/stage6b45b_report.md` | `b57575f96e28c71320e6ad09673b554835a039cf053b767905f38580534e2c35` |
| `Optimizer/results/stage6b45b_results.json` | `0430003a53023174d80f3eff260df9eb6a0981a4f51863c9d854371e9e72beb1` |
| `Optimizer/results/stage6b45b_runs.csv` | `4d3ade1c6715f327ec4f1e6ee722e6b66db914510c58902c9fc016cd3b3b39db` |
| `Optimizer/results/stage6b45b_stage_summary.csv` | `189acfbf19accd88cd8a236e8662ad4435df3c3ae954db966324d7373bb6d1a2` |

## A. Frozen-field local loss comparison (first order, fixed field)

Identical stage inputs are passed to the existing stage calculator in both
namespaces; only delta changes. This is not a new coupled solution.

| stage | l_RK mm | delta old mm | delta new mm | hz old J/kg | hz new J/kg | dhz J/kg | hz new/old | local fields |
|---|---|---|---|---|---|---|---|---|
| 1 | 12.782696224 | 0.200000 | 0.150000 | 7272.952185 | 5454.714139 | -1818.238046 | 0.750000000000 | unchanged |
| 2 | 14.699351529 | 0.200000 | 0.150000 | 6264.454030 | 4698.340523 | -1566.113508 | 0.750000000000 | unchanged |
| 3 | 20.610157717 | 0.200000 | 0.150000 | 4858.051701 | 3643.538776 | -1214.512925 | 0.750000000000 | unchanged |

Max |hz ratio - 0.75| = 1.110e-16; tolerance 1e-12.

## B. Recomputed whole meanline, fixed controls (not a fixed solution)

The flow is refitted to 110 kW for each clearance case, so geometry and
velocities may move. Fixed controls, not a fixed solution.

| property | old | new |
|---|---|---|
| eta_t (global total turbine efficiency) | 0.904349633113 | 0.910562280943 |
| N | 110.000000000 kW | 110.000000000 kW |
| N relative residual | 3.817e-13 | -5.482e-13 |
| G_air | 0.649085224 kg/s | 0.636620023 kg/s |
| G_stator stages 1-3 | 0.645410412, 0.651252179, 0.651252179 | 0.633015783, 0.638745363, 0.638745363 |
| G_rotor stages 1-3 | 0.645410412, 0.651252179, 0.651252179 | 0.633015783, 0.638745363, 0.638745363 |
| rotor heights | 12.782696, 14.699352, 20.610158 mm | 12.548678, 14.421768, 20.207283 mm |
| gap delta | 0.200000, 0.200000, 0.200000 mm | 0.150000, 0.150000, 0.150000 mm |
| hz | 7272.952185, 6264.454030, 4858.051701 J/kg | 5524.568405, 4750.940986, 3674.720471 J/kg |
| L_u | 194408.144774, 184196.111206, 178131.961224 J/kg | 194387.196566, 183884.866930, 177446.760035 J/kg |
| L_t | 186400.327252, 177462.116134, 172903.216925 J/kg | 188124.842247, 178662.994960, 173402.089774 J/kg |
| T0 | 1250.000000, 1091.521890, 936.167659 K | 1250.000000, 1090.055217, 933.609808 K |
| p0 | 1048333.440000, 554457.775949, 272587.580583 Pa | 1048333.440000, 554463.014439, 272640.832960 Pa |
| T1 | 1128.488981, 971.097292, 821.619254 K | 1127.919171, 969.109290, 818.468394 K |
| T2 | 1077.566096, 914.554709, 748.135667 K | 1077.545292, 913.260466, 745.983874 K |
| alpha1 | 13.100000, 20.000000, 23.000000 deg | 13.100000, 20.000000, 23.000000 deg |
| alpha2 | 71.110406, 81.158946, 89.642822 deg | 71.056933, 81.265468, 89.869273 deg |
| sigma max | 294.617516 MPa | 287.449802 MPa |
| q_n | 0.990988590 | 0.978859544 |
| u max | 389.291190 m/s | 387.392688 m/s |
| u mean | 363.857173 m/s | 362.552274 m/s |
| mass residual (unheated) | 2.888e-16 | 1.476e-16 |
| energy residual (unheated) | 8.982e-16 | 6.561e-16 |
| p2z residual | 4.818e-14 | 3.264e-14 |
| p2 static heated exit | 542505.8, 258103.5, 106365.0 Pa | 542431.6, 258065.0, 106365.0 Pa |
| T_ст static heated exit | 1084.374, 920.450, 752.853 K | 1082.870, 917.833, 749.634 K |
| rho = p2/(R*T_ст) | 1.740758, 0.975679, 0.491589 | 1.742939, 0.978315, 0.493699 |
| signed mass error rho*A*c2a/G_л-1 | -0.006279, -0.006405, -0.006266 | -0.004917, -0.004982, -0.004870 |
| T_ст_п total heated exit (reference only, never a density) | 1091.522, 936.168, 780.179 K | 1090.055, 933.610, 777.101 K |
| p_ст_п total heated exit (reference only, never a density) | 557289.9, 276071.4, 122044.5 Pa | 557314.2, 276147.2, 122192.9 Pa |

Unheated residuals are the existing eval_candidate identities and are
unchanged. The heated diagnostic uses the static exit state p2/T_ст, not
the total state; total fields are listed only to keep the two states distinct.

## Static heated exit continuity diagnostic (diagnostic identity only)

`rho = p2/(R*T_ст)` [kg/m3], area `A = pi*D_л*l_л` [m2], velocities [m/s],
flow [kg/s]; the reported signed error `rho*A*c2a/G_л - 1` is dimensionless
(positive = the static heated density would carry more mass than delivered).
With the existing identity `A = G_л*v2/(c2a*pi*D_л)` it equals `T2/T_ст - 1`.

| case | stage | p2 Pa | T_ст K | rho kg/m3 | G_л kg/s | signed error | T2/T_ст-1 |
|---|---|---|---|---|---|---|---|
| old | 1 | 542505.784963790 | 1084.374366312 | 1.740757869 | 0.645410412 | -6.278523693e-03 | -6.278523693e-03 |
| old | 2 | 258103.473258442 | 920.449796469 | 0.975679048 | 0.651252179 | -6.404572097e-03 | -6.404572097e-03 |
| old | 3 | 106365.000000005 | 752.852892491 | 0.491588661 | 0.651252179 | -6.265799531e-03 | -6.265799531e-03 |
| new | 1 | 542431.609601506 | 1082.869580351 | 1.742938535 | 0.633015783 | -4.916832502e-03 | -4.916832502e-03 |
| new | 2 | 258064.950499853 | 917.832973289 | 0.978314757 | 0.638745363 | -4.981851463e-03 | -4.981851463e-03 |
| new | 3 | 106365.000000003 | 749.634492253 | 0.493699195 | 0.638745363 | -4.869864650e-03 | -4.869864650e-03 |

Heated fields finite and positive: **True**.
This is a diagnostic identity at the meanline only; it is not evidence of
radial validity and it adds no equation.

## Efficiency difference

- eta old = 0.904349633113
- eta new = 0.910562280943
- delta absolute = +0.006212647830
- delta percentage points = +0.621264783 pp
- delta relative = +0.686974109 %

The frozen hz ratio is 0.75 here because both cases sit on their gap
floors; that is not a 25 % total efficiency gain and no such claim is made.

## Checks and tolerances

| check | pass | detail | limit |
|---|---|---|---|
| power_old | True | N_rel=3.817e-13; from N_W=3.817e-13 | <=1e-10 |
| power_new | True | N_rel=-5.482e-13; from N_W=-5.482e-13 | <=1e-10 |
| frozen_hz_ratio_expected | True | max|ratio-expected|=1.110e-16; expected 0.75 on the floor branch | <=1e-12 |
| frozen_delta_prescribed | True | old 0.2 mm, new 0.15 mm at current heights; declared floor branch matches |  |
| frozen_local_fields_unchanged | True | T1, T2, L_u, alpha1, alpha2 identical in both namespaces |  |
| frozen_inputs_match_meanline | True | frozen rows reproduce the raw old stage inputs and their local fields |  |
| same_control_point_x | True | one fixed x for both cases |  |
| finite_positive_fields | True | all reported inputs/outputs finite and positive |  |
| protected_hashes_unchanged | True | 20 protected files |  |

## Diagnostic project checks (not Category E)

| constraint | pass | value | limit |
|---|---|---|---|
| old: N=110 kW | True | 110.000000000 kW |  |
| old: alpha2 strict | True | 71.11041, 81.15895, 89.64282 | 70..90,70..90,80..90 |
| old: sigma<=300 MPa | True | 294.617516 MPa |  |
| old: q_n<=0.995 | True | 0.990988590 (linked to sigma, not independent) |  |
| old: u<=400 m/s | True | 389.291190 m/s |  |
| old: eta_t>=0.90 | True | 0.904349633 |  |
| old: l_rk1>=16 mm | False | 12.782696 mm |  |
| old: heights increasing | True | 12.782696, 14.699352, 20.610158 mm |  |
| old: rho_root_1>=0.05 (approximate geometry exponent estimate) | False | 0.047859 |  |
| old: rho_root_2>=0.05 (approximate geometry exponent estimate) | True | 0.067304 |  |
| old: existing eval_candidate feasible | False | l_c1 < 14 мм:0.20025; ρ_корня_1 < 0.05:0.042813 |  |
| new: N=110 kW | True | 110.000000000 kW |  |
| new: alpha2 strict | True | 71.05693, 81.26547, 89.86927 | 70..90,70..90,80..90 |
| new: sigma<=300 MPa | True | 287.449802 MPa |  |
| new: q_n<=0.995 | True | 0.978859544 (linked to sigma, not independent) |  |
| new: u<=400 m/s | True | 387.392688 m/s |  |
| new: eta_t>=0.90 | True | 0.910562281 |  |
| new: l_rk1>=16 mm | False | 12.548678 mm |  |
| new: heights increasing | True | 12.548678, 14.421768, 20.207283 mm |  |
| new: rho_root_1>=0.05 (approximate geometry exponent estimate) | False | 0.048358 |  |
| new: rho_root_2>=0.05 (approximate geometry exponent estimate) | True | 0.067334 |  |
| new: existing eval_candidate feasible | False | l_c1 < 14 мм:0.214033; ρ_корня_1 < 0.05:0.032843 |  |

Feasible: **False**

## Unresolved / NOT_CHECKED

- stage-3 root reaction is NOT_CHECKED (the estimate covers stages 1-2 only and is not a radial profile)
- stator exit closure and stator radial fields are NOT_CHECKED
- radial closure and mesh refinement (801-point grid) were not rerun for this case
- multiple starts of any radial solver were not performed
- material, temperature-dependent strength, creep/resource, thermal stress and cooling are NOT_CHECKED
- thermal and mechanical approval of the 0.15 mm minimum gap, and life/durability, are NOT_CHECKED
- full Category E feasibility cannot pass at this control point
- no optimization, geometry search or real blade profile was performed

## Provenance

- `python`: 3.11.10
- `python_full`: 3.11.10 | packaged by conda-forge | (main, Oct 16 2024, 01:27:36) [GCC 13.3.0]
- `numpy`: 2.0.2
- `scipy`: 1.14.1
- `platform`: Linux-7.2.15-300.fc45.x86_64-x86_64-with-glibc2.39
- `machine`: x86_64

Generated by `Optimizer/clearance_sensitivity.py` (no optimization, no
nonlinear sweep, no new radial solve).

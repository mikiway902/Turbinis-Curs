# `turbine_engine` — extracted calculation package

This is the new, clearly named package for extracted turbine calculations. It
is the canonical location for reusable calculation code; the notebooks remain
independent legacy comparison paths and are not modified by this package.

## Current contents

| Module | Scope |
|---|---|
| `architecture.py` | First M2 contract layer: an immutable, validated `StationState` (total `T`, total `p`, mass flow, `cp`, `k`, optional identifier) and explicit `EngineArchitecture` topology descriptors for `SINGLE_SHAFT_SINGLE_CC`, `SINGLE_SHAFT_DUAL_CC` and `FREE_POWER_TURBINE`. `CC` means centrifugal compressor. The descriptors record only conceptual unit counts and mechanical shaft groups; they contain no cycle, pressure split, speed, gearbox ratio, or stage-count assumption. |
| `compressor.py` | Standalone **0D/mean lumped** variable-k polytropic compressor unit/train consuming `StationState`. `compressor_unit` and `compressor_train` return immutable exit/intermediate states, per-unit and total specific work, and power; total pressure is multiplied by explicit finite-positive ratios and mass flow is preserved. The train re-evaluates mean `k` per unit and exposes the ratio product and an unfiltered pressure-closure residual. It reuses the existing `cycle.calc_compressor` equation through one shared helper. It is not a compressor map, aerodynamic stage model, corrected-flow model, speed/diameter model, or pressure-split rule. |
| `combustor.py` | Standalone **scalar** heat-addition wrapper over the existing `cycle.calc_alpha(T_in, T_gas)` and `cycle.calc_g_m(T_in, T_gas, config)` equations. `combustor_heat_addition` returns an immutable `CombustorResult` recording scalar inlet/target temperatures (K), relative fuel flow `g_m`, and excess-air ratio `alpha`. It performs no mass-flow conversion, fuel/composition update, `cp`/`k` state, pressure loss, or separate combustion-efficiency model; it is not a combustor geometry, performance-map, or thermodynamic state model and is not wired into the cycle. |
| `turbine.py` | Standalone **0D/mean lumped** total-to-total turbine-expansion unit consuming `StationState`. `turbine_expansion` applies the existing `calc_eta_expansion` efficiency and cycle work/drop `cp*T_in*(1 - pi**((1-k)/k))*eta`, returning an immutable outlet state, specific work, power, efficiency, and unfiltered pressure residual. `turbine_pressure_ratio_for_work` is the analytical inverse for a prescribed positive specific work; it solves `pi = (1 - w/(cp*T_in))**(-k/((k-1)*eta_poly))` and returns the same validated result type. `pressure_ratio` is the expansion ratio `p_in_total/p_out_total`; finite-positive values are accepted without an operating range, mass flow/`cp`/`k` are preserved, and nonphysical outlet states are rejected. It is not a bladed stage, stage-inventory model, map, cooling model, or architecture cycle. |
| `architecture_cycle.py` | Opt-in executable 0D reference cycles for the three selected topologies. The explicit `uncooled_no_bleed_0d_ideal_mixing` variant closes, per architecture, the compressor train, combustor fuel/air relation, total-to-total turbines, mass flow, and pre-reducer shaft power. A/B close one MAIN shaft (`eta_mech*G_gas*L_t - sum(G_air*L_c) = 110 kW`); C prescribes GG turbine work from compressor power, passes the actual GG outlet `StationState` to the FPT, expands the FPT to the caller's total outlet pressure, and sizes flow for the output power. An `external_combustor` closure accepts an explicit combustor-exit state and fuel/air ratio instead of inferring them. It is not physical validation, a compressor map/blading model, a stage inventory, a cooling/BLEED model, a gearbox ratio, or a propeller load. |
| `shaft.py` | Standalone explicit **architecture shaft power balance**, not a cycle. `evaluate_shaft_power_balance` takes the architecture, mass-flow-weighted compressor and turbine gas powers (W), explicit per-turbine mechanical efficiencies, the required gearbox-input power (W), and the gearbox efficiency. It enforces exact per-architecture counts/order (two series compressor units for `SINGLE_SHAFT_DUAL_CC`; gas-generator then free-power turbine for `FREE_POWER_TURBINE`) and returns an immutable result whose net powers, independent raw residuals, actual gearbox input, and downstream output are derived properties. Mechanical efficiency is applied once to turbine gas power; the gearbox efficiency is applied only downstream to the actual input. Signed engineering-infeasible balances are retained, not clamped. It adds no map, stage split, speed, gear ratio, propeller load, or cycle wiring. |
| `cycle.py` | Stateless single-point cycle core extracted from `Turbinis-Cycle/Turbinis-Cycle.ipynb`, including the derived fields formerly computed by `Optimizer/optimizer_core.cycle_point`. |
| `gas.py` | Stateless gas-property helpers `газ(T)`, `cp_k_ступени(T0, H0)` and the pressure-based enthalpy-drop helper `H0_по_p(T0, p0, p2)` (exposed as `calc_gas_cp_k`, `calc_stage_cp_k`, `calc_h0_for_pressure`) extracted from the gas cell of `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb`. |
| `stage.py` | Stateless scalar meanline calculation of one turbine stage, extracted from the stage cell of `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb`. |
| `reaction.py` | Transcription of the current root reaction relation `ρ_корня(r, закон)` from the notebook profiling cell. |
| `orchestration.py` | Source-preserving extraction of the three constant-hub orchestration functions `ступень_const_hub`, `турбина_const_hub` and `распределить_π_const_hub` from cell `const-hub-scan` (before the excluded top-level `D_hub` sweep). It resolves stage, gas, table, array and exception dependencies from a supplied live namespace and imports no NumPy/SciPy. |
| `performance.py` | Formula-preserving extraction of the numerical aggregation at the start of code cell 20 (`g_i`, `L_тΣ`, `η_т_турб`, `h_вz`, `η_л_турб`, `p_т_полн`, `cp_т`, `k_т`, `H_т_полн`, `η_т_полн`, `N_т`, `N_вал`). It takes the stage mappings and scalars as explicit arguments plus an injected gas callable, and imports no NumPy/pandas. |
| `expansion.py` | Formula-preserving extraction of the numerical block at the start of code cell 9: the turbine expansion temperature iteration (`π_т`, `T_т`, `H_т`, `cp0`, `k0`) and preliminary single-stage sizing (`v_т`, `A_т0`, `n_max0`, `n_раб0`, `c0_т`, `u_η`, `D_η`, `l_η`, `D_т0`). It takes every namespace value explicitly plus the injected gas callable, and imports no NumPy/pandas. |
| `geometry.py` | Formula-preserving extraction of the numerical meridional flowpath block in code cell 22: the `γ_участка`/`ширина`/`r_нв` helpers, contour sections, `геом` chord widths, per-section opening angles (`γ_max`), axial length (`dx_т`) and overall opening angle (`γ_турб`). It takes the stage mappings plus every cell constant explicitly, keeps the exact 50-iteration width bisection, and imports no NumPy/pandas. |
| `baseline.py` | Source-preserving extraction of the conventional multi-stage turbine orchestration in code cell 16: `турбина`, `распределить_π`, `решить`, `N_на_валу` and the outer flow-secant solve. It keeps the currently executed patched stable-flow reset (`_N_of`), all iteration limits/tolerances/clamps, the stage and iteration-log shapes, and the namespace ``НетРешения`` control flow. Dependencies are read live from the supplied namespace; `np.clip`/`np.mean` are injected. No pandas/DataFrame imports. |
| `tests/` | Unit, parity, and optimizer-wiring software-drift tests for the station/architecture contracts, extracted cores, gas helper, root reaction relation, const-hub orchestration, turbine expansion sizing, meridional geometry, turbine performance aggregation, and conventional multi-stage baseline. |

`architecture.py` is contract-only. `StationState` is a frozen, validated
record of the explicitly available total quantities; it does not imply fuel,
composition, or cooling-flow state. The architecture registry records
conceptual compressor-unit, combustor, and turbine-assembly counts plus
mechanical shaft groups. Every target architecture is declared with one
combustor; unknown speeds, gearbox ratios, pressure-ratio splits, and stage
counts are deliberately absent. `architecture.py` remains contract-only;
`architecture_cycle.py` is the opt-in executable reference layer, while
`calculate_cycle_point` remains the current-engine legacy path.

`compressor.py` is the first reusable component consuming `StationState`. It is
a 0D/mean lumped model only: one unit uses the current variable-k polytropic
temperature equation and mean-gas `cp = R*k/(k-1)` specific work, and a train
applies that same unit equation sequentially with the current
`eta_compressor_poly`. Ratios are explicit inputs, not a preferred operating
range or a split rule; they are not normalised. `cycle.calc_compressor` now
delegates to the same private temperature helper, so its default path and the
single-point cycle anchor are unchanged. `calculate_cycle_point` is **not**
rewired to the component, and no compressor map, corrected flow, speed,
diameter, bleed, or bladed stage inventory is introduced.

`combustor.py` is a scalar equation-preserving wrapper, not a flow/composition
model. `combustor_heat_addition` calls exactly `cycle.calc_alpha(T_in, T_gas)`
and `cycle.calc_g_m(T_in, T_gas, config)` and records the returned `alpha`,
`g_m` and the scalar inlet/target temperatures in an immutable
`CombustorResult`. `g_m` and `alpha` therefore keep the current inherited cycle
assumptions (fuel heating value, combustion efficiency, `l0`, `T_ref`, and the
existing mean-`cp` and excess-air polynomials). No mass-flow conversion, fuel
or product composition, `cp`/`k` state, pressure loss, or separate combustion
efficiency model is added. `StationState` intentionally carries no fuel or
composition fields, so this wrapper does not fabricate an outlet station state
or a mass balance. `calculate_cycle_point` is **not** rewired: it continues to
call the same helpers directly, and its outputs are unchanged.

`turbine.py` is the turbine counterpart of the compressor unit: one lumped
total-to-total expansion relation consuming `StationState`. It reuses
`cycle.calc_eta_expansion` and the exact cycle work expression, takes `cp` and
`k` from the inlet because the state carries no gas-composition information,
and returns the outlet state plus specific work, power, efficiency, and an
unfiltered pressure residual. `pressure_ratio` is the explicit expansion ratio
`p_in_total / p_out_total`; values greater than one expand and extract work,
values below one are kept as the equation's pressure-rising mathematical
branch, and exactly one is rejected by the retained efficiency-helper
denominator check. `calculate_cycle_point` is **not** rewired to this unit: the
cycle's coupled nozzle iteration still owns `pi_t`, and the cycle anchor is
unchanged. This is not a bladed turbine stage, stage inventory, cooling split,
operating map, or architecture cycle; future turbine assembly/stage inventories
remain unknown.

`shaft.py` is a standalone balance helper, not an architecture cycle.  It takes
explicit mass-flow-weighted compressor and turbine gas powers in W, explicit
per-turbine mechanical efficiencies, the required pre-reducer gearbox-input
power, and the gearbox efficiency.  The two single-shaft architectures close one
MAIN shaft as `sum(eta_mech*P_turbine) - sum(P_compressor)`; the free-power
architecture closes a gas-generator shaft as
`eta_mech_GG*P_GG - P_compressor_GG` and a separate free-power-turbine output
shaft as `eta_mech_FPT*P_FPT`.  The raw residuals are independent and
unfiltered; the downstream gearbox output is `actual_input*eta_gearbox`, not
`required_input*eta_gearbox`.  The helper does not equate specific works across
different mass-flow bases, does not add accessory/shaft-loss defaults, and is
**not** wired into `calculate_cycle_point`, the baseline, or the optimizer.

`architecture_cycle.py` is the first executable architecture layer, but it is
still an opt-in reference cycle and **not** a replacement for
`calculate_cycle_point`. The only inferred variant is explicitly named
`uncooled_no_bleed_0d_ideal_mixing`: no cooling, no leak, no bleed, one
combustor, all compressor air entering it, ideal mixing to the requested
combustor-exit total temperature, total-to-total turbines, and constant turbine
`cp`/`k` evaluated at the turbine inlet. `PolynomialGasModel` records the gas
constants and whether turbine-gas `alpha` is fuel-based (`1/(f*l0)`) or the
retained empirical `calc_alpha` polynomial. Fuel/air ratio comes from the
retained `calc_g_m`; the separate `ExternalCombustorClosure` accepts an
explicit outlet total state and fuel/air ratio when inference is not wanted.
The module reports `MISSING_COMPRESSOR_PHYSICS`, `NOT_PHYSICALLY_VALIDATED`,
and the missing stage/gearbox/propeller scope in every result. It preserves
the legacy cycle and is not wired into the baseline or optimizer.

The extraction preserves the source equations and numerical behaviour. Tests
compare the cores with observed legacy outputs as software-drift diagnostics,
not as physical validation.

`orchestration.py` keeps the notebook's operation order, 60/30 iteration
limits, `1e-7`/`1e-5`/`1e-12` tolerances, half-relaxation, `rho = 0.3`,
`D0 = D_hub + 0.02`, interstage pressure correction and failure handling. The
optimizer supplies its fast interpolant and `numpy.clip`, so the default
`build_ns` path is arithmetic-identical to the notebook. The default path also
installs the extracted `газ`/`H0_по_p`, `ступень` and `ρ_корня` adapters
immediately after their source cells (before notebook cells 9, 16 and 20 run),
matching `optimizer_core._evaluate(stage_core=True)`. The explicit
`build_ns(..., orchestration_core=False)` mode retains every notebook function
as the independent comparison path. The extracted orchestration still depends
on the injected stage, gas and constraint helpers; it is not a complete 3D
architecture and does not add an acceptance model.

`performance.py` keeps the cell-20 expression order and assigns every legacy
intermediate (`g_i`, `L_тΣ`, `η_т_турб`, `h_вz`, `η_л_турб`, `p_т_полн`,
`cp_т`, `k_т`, `H_т_полн`, `η_т_полн`, `N_т`, `N_вал`) from one pure call. In
`Optimizer/optimizer_core._evaluate(stage_core=True)` the cell is detected by
its semantic marker, its numerical aggregation is replaced by this module, and
its pandas table/display code is not executed. With `stage_core=False` the
notebook cell 20 is still executed in full as the independent comparison path.
The extraction adds no guards or clamps, so calculable engineering-poor points
keep the source arithmetic. The presentation tables are not produced on the
wired path because the optimizer consumes none of them.

`expansion.py` keeps the cell-9 operation order, the exact 50-iteration limit,
the strict `abs(T_т_н - T_т) < 1e-6` test, the post-loop `T_т = T_т_н`
assignment, and native division/square-root failure behaviour. In
`Optimizer/optimizer_core._evaluate(stage_core=True)` the cell is detected by
its semantic markers (`π_т = p_г_полн / p_т` and `табл_п1_31`), its numerical
block is replaced by this module after the cell-7 gas adapter is installed, and
its pandas table is not built. With `stage_core=False` the notebook cell 9 is
still executed in full as the independent comparison path. The extraction adds
no guards, clamps, or physical changes; it is a source transcription and
software-parity check, not model validation.

`geometry.py` keeps the cell-22 contour order, the `γ_цель` width bisection
with its exact 50-iteration limit and strict comparisons, the `γ_макс`
transition-channel slope limit, and the final per-section/overall opening-angle
aggregation. In `Optimizer/optimizer_core._evaluate(stage_core=True,
full=True)` the cell is detected by its semantic markers (`γ_участка`,
`ширина`, `r_нв`, `контур, геом, x = [], [], 0.0`), its numerical block is
replaced by this module, and its pandas display objects are not built. The
returned `контур`, `геом`, `γ_max`, `dx_т` and `γ_турб` identifiers are
installed in the namespace so cells 24 and 26 see the expected geometry. With
`stage_core=False` the notebook cell 22 is still executed in full as the
independent comparison path, and with `full=False` it is not executed at all.
The extraction adds no guards, filters, or clamps; calculable engineering-poor
geometry keeps the source arithmetic.

`reaction.py` is explicitly a transcription of the current root reaction
relation, not a radial-equilibrium validation or acceptance model. It adds no
constraints, no acceptance filtering, and no new physical interpretation; only
the exact `"Γ"` string selects the source's constant-circulation branch and
every other law value keeps the source's general fallback.

`baseline.py` copies the numerical cell 16 as it is actually executed by
`Optimizer/optimizer_core.load_model` / `_evaluate(stage_core=True)`, including
the already patched stable-flow wrapper `_N_of(G)`. Every trial flow starts
from the original `(G0, Gc10, Gl0, A0)`, then scales those four quantities by
`G / G0`; the raw notebook secant side-effect accumulation is not revived. The
source's `(0.9 G0, 1.1 G0)` bracket, 20 outer iterations, `1e-5` relative
residual, `1e-9` denominator guard, `[0.5 G0, 2.0 G0]` clamp, 40-iteration
area loop, `1e-4` area tolerance, half-relaxation, 30-step clipped pressure
secant, and `НетРешения` failure handling are unchanged. The one intentional
boundary correction is the last-stage pressure guard: the unsourced legacy
`p0 <= 1.02*p_t` margin was replaced by `p0 <= p_t` in both this module and the
active notebook cell 16. Turbine book part 1, printed p. 92, items 138–140
prescribes `p2z = p_t` and a drop that is positive only for `p0 > p_t`; it has no
percentage margin. A passing `p0 > p_t` value is passed to the same stage
calculation, which remains the physical-feasibility authority. No other formula,
boundary, power definition, loss model, cooling model, reaction model, or
acceptance criterion is changed.

In `Optimizer/optimizer_core._evaluate(stage_core=True)` cell 16 is detected by
its semantic markers (`турбина`, `распределить_π`, `решить`, `N_на_валу` and
`_N_of(G)`), its numerical block is replaced by this module after the gas,
expansion, stage and root-reaction adapters are installed, and its pandas
tables/`display` calls are not built. The module also does not mutate the
namespace; `_evaluate` installs the returned `legacy_values()` so cell 20 and
the full cells see the same final flow state, stage sequence and scalar
outputs. With `stage_core=False` the patched notebook cell 16 is still executed
in full, including both presentation tables, as the independent comparison
path. The extraction is a software-parity check, not model validation; apart
from the documented last-stage boundary, it adds no guards, clamps, or filtering
beyond the copied source and keeps calculable engineering-poor points.

## Status and limitations

- The package currently contains the station-state/architecture contracts,
  the 0D variable-k compressor unit/train component, the scalar combustor
  heat-addition wrapper, the 0D lumped turbine-expansion unit (with its
  prescribed-work inverse), the standalone architecture shaft power balance,
  the explicit 0D architecture reference cycles, the
  single-point cycle core, the stage gas-property helper, the
  meanline-stage core, the transcribed root reaction relation, the constant-hub
  three-stage orchestration, the turbine expansion iteration and preliminary
  sizing, the meridional flowpath geometry, the turbine aggregate performance
  calculation, and the conventional multi-stage cell-16 orchestration.
- `architecture_cycle.py` is an executable **reference** cycle for all three
  selected topologies, not a validated engine model. It implements the explicit
  uncooled/no-bleed 0D variant and an external-combustor closure; it does not
  establish compressor stage inventory, maps, blading, cooling, gear ratio,
  propeller load, control law, or the missing physical validation. The legacy
  cycle remains the current-engine authority and is not rewired.
- `compressor.py` is a 0D/mean lumped component. It is not a compressor map,
  stage/aerodynamic model, corrected-flow model, or speed/diameter model, and
  it is not wired into `calculate_cycle_point`. The cycle anchor is preserved
  through the shared temperature helper, not through a new cycle formulation.
- `combustor.py` is a scalar heat-addition wrapper over the existing
  `calc_alpha` and `calc_g_m` helpers. It stores only scalar temperatures,
  `g_m`, and `alpha`; it is not a combustor geometry, performance-map,
  operating-range, composition, mass-flow, or thermodynamic state model, and
  it is not wired into `calculate_cycle_point`. It does not produce a
  `StationState` or a mass balance, and it adds no separate combustion
  efficiency model.
- `turbine.py` is a 0D/mean lumped total-to-total expansion unit. It is not a
  bladed turbine stage, stage-inventory model, cooling split, operating map, or
  architecture cycle, and it is not wired into `calculate_cycle_point`. Future
  turbine assembly/stage inventories remain unknown and are deliberately not
  encoded.
- `shaft.py` is an architecture balance helper only. It does not define a
  compressor split, a turbine expansion, a shaft speed, a gearbox ratio, a
  propeller load, a stage inventory, or an architecture-specific cycle, and it
  is not wired into `calculate_cycle_point`, the baseline, or the optimizer.
  It records explicit raw balances and leaves feasibility to later analysis.
- The legacy notebooks are retained independently as the comparison path.
  This includes the notebook's own const-hub functions, available through
  `Optimizer/optimize_extended.build_ns(..., orchestration_core=False)`, and
  the patched notebook cell 16, available through
  `Optimizer/optimizer_core._evaluate(..., stage_core=False)`.
- A 0D architecture **reference closure** now exists for all three selected
  topologies, but full 3D/meanline architecture closure is **not implemented or
  validated** here. In particular, this package does not contain a complete 3D
  flow model, accepted radial-equilibrium implementation, stage inventory, or
  map-based component validation.
- A green test run only checks software consistency; it does not prove that the
  underlying physics is correct.

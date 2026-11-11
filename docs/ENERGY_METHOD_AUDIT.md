# Energy and gas-property method audit — 2026-11-11

## Bounded actual-boundary RAW alternatives

New coupled-stage smoke reuses the existing meanline closures without changing
physics or tuning inputs. Fixed A/B counts 1–3 and nine C GG/FPT count pairs give
15 attempts: 13 SUCCESS and two NUMERICAL_FAILURE (A/B MAIN/1 total-pressure roots
not bracketed by 120000–150000 Pa static). The latter do not prove physical failure;
there is no automatic bracket widening. All successful points miss 110 kW.

Selected net mechanical results at unchanged source flow and illustrative controls:
A MAIN/2 102198.85 W, /3 104894.03 W; B MAIN/2 102204.12 W, /3 104880.83 W;
C GG2/FPT1 108141.98 W; C GG3/FPT1 106095.69 W. These are not architecture rankings
or validated optimum stage counts. Overall feasibility remains null/NOT_CHECKED.

Parent independently executed both test suites (536 engine tests, one GPU skip;
146 scripts tests), saved the full run and resumed it with zero appended records.
All 15 attempts reproduce, including both bracket failures. Saved journal:
`reference/architecture_stage_smoke/2026-11-11-actual-coupled/`, 185924 bytes,
SHA-256 `2277f4e544954c68a259f46f94be30bddddc11ac81d1fe4b0793acb3ca1e8fa2`.
Configuration locks the fixed plan, borrowed constants and computational source files.
Existing independent assembly snapshots and the single-link diagnostic are unchanged.
Independent reviewer PASS after Python 3.10-compatible streamed digest correction;
Python 3.10 execution remains unverified. No physical validation claim follows.

## Actual-boundary linkage follow-up

The following supersedes the earlier "not yet passed" statement only for the fixed
C GG/3 → FPT/1 diagnostic, not the saved 12-point assembly journals. Lead independently
ran the new pressure/work closures and saved full stages, fixture payloads, brackets
and current source digest in `reference/architecture_stage_smoke/2026-11-11-actual-link-diagnostic.json`.
This is a diagnostic artifact, not a resumable RAW sweep or engineering acceptance.

- A MAIN/3 actual total outlet: 150000.036501989 Pa, target 150000 Pa;
  solved static pressure 140145.263671875 Pa, relative residual 2.4334659e-7.
- C GG/3 mechanical work: 200201.071150633 W versus compressor 200200.924926023 W;
  relative residual 7.3038928e-7, mechanical efficiency 0.99.
- GG actual outlet 1009.505000657 K / 281298.344088335 Pa is passed unchanged to FPT/1;
  gas flow, 84000 rpm and 0.065 m hub remain fixed illustrative inputs.
- FPT/1 actual total outlet: 149999.898810257 Pa versus 150000 Pa (6.7459829e-7).
  Mechanical output: 106095.685758257 W, **−3904.314241743 W** relative to 110 kW.

The regression checks exact inlet-state/flow continuity, actual-stage metrics and
namespace-pressure restoration. Neither closure uses fixture Mach as its final metric.
The remaining power deficit is retained: successful component boundary closure is
not engine power closure, stage-count selection or physical validation.

## Latest step-3/M3 result — stage count and engineering pressure clip

Lead visually checked original turbine part 1 printed p.87 (PDF p.88), items 45–54.
The source formula is `z=(c0^2/v^2-1)/(u_mean^2/(v^2*y0^2)-1)`, with the explicit
recovered mean exit velocity v. It already existed inline in the baseline and is now
shared as `estimate_stage_count`; successful arithmetic is unchanged. No rounding,
automatic stage-count selection or transfer of the old three-stage assumption occurs.

`scripts/architecture_stage_smoke.py` evaluates existing const-hub equations for the
four source aggregate inlets (A dual-CC MAIN, B single-CC MAIN, C GG and C FPT), each
with 1/2/3 explicit stages. Controls are borrowed illustrative fixtures, not designed
engine dimensions. No optimizer or flow resizing is run. Full stages, geometry,
velocity triangles, Mach, pressure, work and available stress metrics are in RAW.

This exposed an engineering-only secant cap: `[1.05, 1.2*pi_stage_max]` stopped five
calculable assemblies. An explicit RAW policy now removes that cap while retaining
the mathematical ratio>1 domain, existing tolerances and genuine stage-domain failures.
Default library policy still reproduces the clipped reference. Initial non-finite
residuals are now diagnosed before any subsequent evaluation can hide them.

Parent independently ran and saved both policies:

| Assembly / stages | Clipped reference | RAW calculation | Prompt pi profile |
|---|---|---|---|
| A MAIN / 1 | numerical stall | SUCCESS, pi=7.692838 | FAIL |
| A MAIN / 2 | numerical stall | SUCCESS | FAIL |
| A MAIN / 3 | SUCCESS | SUCCESS | PASS |
| B MAIN / 1 | numerical stall | SUCCESS, pi=7.692834 | FAIL |
| B MAIN / 2 | numerical stall | SUCCESS | FAIL |
| B MAIN / 3 | SUCCESS | SUCCESS | PASS |
| C GG / 1 | numerical stall | SUCCESS, pi=2.908479 | FAIL |
| C GG / 2–3 | SUCCESS | SUCCESS | FAIL |
| C FPT / 1 | SUCCESS | SUCCESS, pi=2.072209 | PASS |
| C FPT / 2–3 | SUCCESS | SUCCESS | FAIL |

The profile checks meanline p0_total/p2_static against 1.8–2.4 with signed margins;
its pass flag is independent of missing strength/material verification. Overall
engineering feasibility is null/NOT_CHECKED, never inferred from this one profile.

Both journals have 12 attempts and reproduce 12/12; RAW resume appends zero records:

- `reference/architecture_stage_smoke/2026-11-11-raw/`, 138500 journal bytes,
  SHA-256 `9c7ae704ce9cd13238e1327eb7f7a62040150efa53460648b3bda05565308f9b`.
- `reference/architecture_stage_smoke/2026-11-11-legacy-clipped/`, 108364 bytes,
  SHA-256 `3e06b96f4fd71e60a1ef5e5affad69648ed2efe4cb458c0e1f56de3edd6f7618`.

**Not a coupled engine:** static outlet was translated from the 0D total outlet with
fixture Mach=0.3379; actual meanline exit velocity differs. For A MAIN/1, actual outlet
total pressure differs from the aggregate by +41.30%, and mechanical work is lower by
67103.72 W. For A MAIN/3 the differences are -0.7585% and -4035.91 W. For C GG/3 the
mechanical difference is +257.44 W, while C FPT/1 differs by +7504.30 W. These are
saved mismatches, not numerical closure residuals or accepted engines. The unrounded
A MAIN/3 stage estimate is 2.57699 under its stated preliminary assumptions.
Actual meanline GG output has not yet been used as the FPT meanline inlet; current
GG/FPT aggregate-state continuity must not be confused with that remaining coupling.

Independent reviews PASS after fixing incomplete source hashing and acceptance
semantics. Lead observed 523 active engine tests OK (1 GPU skip), 131 scripts tests
OK; static verifier 6 PASS/7 WARN/0 FAIL; git diff check clean. The model digest now
covers 28 computational/source files, including compressor/turbine/shaft/reaction;
tests reject resume after changes in each previously omitted dependency.

## Latest continuation — existing cycle and stage method aligned explicitly

The existing `cycle.calculate_cycle_point` now accepts
`gas_property_basis="textbook_true_cp"`. Default `legacy_mean_cp` keeps its numerical
anchors. The opt-in calculation uses true cp/k at actual process-mean temperatures;
one alpha=1/(gm*l0) for turbine, nozzle and exit; the original p.20 mean-cp fuel
formula; and the same configured leak/cooling/return fractions. No integral model or
new thermochemical closure is imported. Return above cooling bleed or nonpositive
combustor-air fraction is a diagnosed mass-bookkeeping failure, not repaired.

Compressor/turbine/nozzle fixed points retain evaluated and proposed temperatures
separately. Exhaustion reports the actual nonzero residual; it does not overwrite it
after advancing the iterate. Fuel `CycleSolverError` is not relabelled physical.
Turbine exit static/total pressure and velocity use the same local k/temperature;
the process-mean k is separately reported for the work equation. Source-fit process
cp, common R and cooling bookkeeping without temperature mixing remain approximations.

Existing `build_ns` can explicitly select both cycle and stage textbook options.
The full fixed B87_relaxed comparison, independently executed by the lead:

| Quantity | Legacy cycle + stage | Textbook cycle + stage |
|---|---:|---:|
| Cycle-sized air flow, kg/s | 0.577155550361 | 0.532339684442 |
| Compressor specific work, J/kg air | 358583.408021 | 356096.970147 |
| Fuel-based alpha | 3.97341363949 | 3.92591125104 |
| Turbine-exit velocity, m/s | 180.095822191 | 182.370919995 |
| Stage-solved gearbox-input power, W | 109999.999999939 | 110000.000000033 |
| First stator blade height, mm | 11.2953450885 | 11.0317054088 |
| Stage-reported turbine efficiency | 0.9127057739 | 0.9111862295 |
| Reported stress metric, MPa | 309.07753384 | 301.81415760 |
| Engineering feasible / violation count | false / 5 | false / 5 |

Only the same fixed control point was compared; there was no search or optimum claim.
Original 13/55-point architecture datasets remain unchanged legacy-model datasets.
This work does not yet update all A/B/C compressor/turbine aggregates or justify their
stage inventories. Independent cycle and wiring reviewers PASS. Lead checks: 460
active engine tests OK (1 GPU skip), 116 scripts tests OK; static verifier 6/7/0;
git diff check clean. `test_textbook_cycle.py` has 17 tests with independently
transcribed source equations; combined selector/wiring tests have 22 cases.

## Scope correction and accepted continuation

The user explicitly restricted the route to the two textbooks and necessary work.
The integral component/cycle proposal below is **historical optional work**, not a
prerequisite or the selected engine method. Files were preserved under
`experiments/caloric/` with internal imports adjusted, without changing coefficients
or tests. The interrupted integral combustor is unaccepted; its optional suite has
45 tests with one 1-ULP equality failure (775085.7128656327 versus 775085.7128656328).
Active engine discovery does not include these experiments. No assertion was loosened.

Original turbine part 1 PDF was checked at **printed pp.15–16 and 24** (PDF pp.16–17
and 25). Its mean-process approximation does not require analytic h/phi integration.
The cycle book pp.12–13/19 requires consistent process cp/k, and pp.19–20 already
provides the fuel relation. Appendix 6 p.57 supplies separate true/mean fits.
The required correction is to distinguish these definitions in the existing method,
not introduce a replacement fuel/composition/thermochemistry solver.

`gas.calc_textbook_gas_cp_k` now transcribes the published true-cp fit directly,
including its 750 K branch; mean cp and the notebook derivative remain separate.
Existing stage and pressure loops accept an explicit basis. Existing const-hub
`build_ns(..., gas_property_basis="textbook_true_cp")` routes every relevant stage
helper consistently and labels stage/cycle bases separately. Default behavior remains
the original notebook approximation. The existing cycle still uses legacy mean cp;
this option is not a claim of a fully reconciled textbook cycle.

Parent independently recalculated the same fixed B87_relaxed anchor, without search:

| Quantity | Legacy stage derivative | Published true-cp stage option |
|---|---:|---:|
| Gearbox-input power, W | 109999.99999993911 | 110000.00000003492 |
| First stator blade height, mm | 11.2953450885 | 11.1531201067 |
| Reported turbine efficiency | 0.9127057739 | 0.9114534173 |
| Reported stress metric, MPa | 309.07753384 | 306.21838793 |
| Engineering feasible | false | false |
| Violation count | 5 | 5 |

Both are model calculations, not physical validation. The source option is explicit,
not a new baseline or an accepted design. Construction-bound basis metadata prevents
a mutable namespace marker from relabelling unchanged calculations, including failures.
Independent reviewer PASS for gas source/iteration changes and for stage wiring after
that provenance correction. Parent suites: 443 active engine tests OK (1 GPU skip),
107 scripts tests OK; static verifier 6 PASS/7 WARN/0 FAIL; diff check clean.

The mandatory reduced-work sequence is in `AUTONOMOUS_EXECUTION.md`. Do not resume
the historical integral route merely because its component tests once passed.

## Evidence and status

The lead opened the original cycle textbook PDF and visually checked printed/PDF
pp. 20, 54, 56 and 57. The PDF is
`Turbinis-Cycle/Михальцев_В_Е_Расчет_параметров_цикла_при_проектировании_газотурбинных.pdf`.
Temporary rendering tools were installed only under the approved OpenCode temp
directory, not in the project environment. No physical equation was silently replaced.

- p.20: fuel balance uses mean air/pure-product heat capacities at the respective
  temperatures; alpha=1/(gm*l0). Fuel sensible-enthalpy terms vanish only under the
  explicit assumption T_fuel=T_ref. The retained code implements that reduced form.
- p.57: the mean-cp polynomial has its branch at 700 K; a **different true-cp fit**
  has its branch at 750 K. The source gives kcal/(kg K), converted by 4187 to SI.
  Differentiating the fitted mean-cp curve is not the published true-cp formula.
- pp.54/56: the displayed true/mean curves cover alpha=1 through infinity. Rich
  mixtures are not established by those charts. No full-temperature-range validity
  is proven by merely evaluating the polynomial outside the plotted region.
- p.19 describes cp/k averaged over the expansion process. The legacy architecture
  variant instead holds combustor-exit properties across the entire turbine train.

## Independent numerical audit of the saved RAW fixture

`scripts/audit_architecture_energy.py` imports dataset integrity readers, not
production gas/cycle formulas as its numerical oracle. It reconstructs equations
independently, preserves failed records and external unknowns, and labels alternative
enthalpy conventions as diagnostics, not corrected physics.

Saved output: `reference/architecture_smoke/2026-11-11-grid-energy-audit.json`.
Input journal hash:
`e90d96b65c0d97d9db7917fee5a1570f5e905f88ba1262e17006ab9af54f7200`.

- 13 attempts: 9 successful legacy, 3 successful external closures, 1 physical failure.
- Retained fuel identity maximum: 0 J/kg air. Heat-input identity maximum:
  1.1642e-10 W. Compressor/turbine stored constant-cp identities: 0 J/kg.
  These check arithmetic, not physical correctness. Different units are not ranked.
- Fuel inferred using the diagnostic 273.15 K offset differs from retained fuel by
  -2.8621% to -2.7912%. The source's printed p.20 expression uses cp*T directly;
  the offset alternative is not accepted as a source correction.
- Empirical alpha differs from fuel-based alpha by +1.7439% to +1.7465%.
- In three legacy FPT cases, inherited cp exceeds mean cp evaluated at actual FPT
  inlet by 2.65594%. Actual GG outlet T/p/flow is transferred, but cp/k is inherited.
- Illustrative reference-subtracted combustor energy residuals span
  1356.80 to 22691.85 J/kg air across the explicitly labelled conventions/alpha bases.
  This is not the source's stoichiometric bookkeeping and is not a physics FAIL/PASS.
- External alpha and heat input remain unavailable, not zero and not inferred.

Audit reviewer: PASS after fixing turbine enthalpy-drop sign and removing a mixed-unit
global maximum. Parent observed 64 scripts tests and 427 engine tests OK at that point.

## Typed diagnostic propagation

The audit exposed a separate P0 error-classification issue: `_resolve_combustor`
converted a typed `CycleSolverError` into a physical failure via a broad RuntimeError
catch. It now preserves typed solver exceptions and leaves unexpected RuntimeError
visible as MODEL_ERROR/evaluator_exception. Actual catch-boundary tests were added.
The misleading per-turbine refreshed-cp prose was corrected; success arithmetic and
serialized limitation markers remain unchanged. Independent reviewer: PASS.

## Opt-in integrated true-cp components — numerical review corrected

`turbine_engine/caloric.py` and its tests implement a separate opt-in transcription
of p.57 true cp, analytic h=int(cp dT), phi=int(cp/T dT), and variable-cp compression,
expansion and prescribed-work expansion. Legacy components are untouched.

Thermodynamic derivation at fixed composition:

- compressor: phi_out-phi_in = R*ln(pi_c)/eta_poly;
- turbine: phi_in-phi_out = eta_poly*R*ln(pi_t);
- work is the corresponding positive enthalpy change/drop;
- actual outlet cp/k is evaluated at outlet temperature, not inherited unchanged.

Independent quadrature and source-value tests pass. Reviews found and corrected
deep-expansion bracketing, a hidden near-unit tolerance relaxation, and trial-bracket
overshoot near the cp>R domain boundary. Requested tolerance is now honored or an
explicit floating-resolution exception is raised.

**Historical review at the prior three-cycle limit: FAIL.** The remaining P2 was in inverse expansion:
`eta * R` can underflow to zero before division (`caloric.py`, pressure-ratio solve),
producing an unstructured ZeroDivisionError. Reproducer: R=1e-12 J/(kg K), air,
Tin=288 K, prescribed work=100 J/kg, eta=5e-324. This is an extreme numerical input,
not a supported engine design, but its diagnostic contract is still unmet.
That task stopped at its three-cycle limit. The lead independently reproduced the
unstructured ZeroDivisionError before the subsequent explicit user correction request.

### User-requested correction — closed

The denominator `eta_poly*R` is now checked before division; zero underflow produces
`CaloricModelDomainError`, stage `caloric.turbine_inverse.pressure`, reason
`pressure_denominator_underflow`, with the input factors and evaluated state retained.
Nonzero subnormal denominators with overflowing exponents follow the existing typed
pressure-overflow path. Computed component-power overflow/underflow is likewise
reported explicitly, not left as an unstructured result-construction error.
Normal arithmetic, source coefficients, brackets and requested tolerances are unchanged.

Independent reviewer: **PASS**. Parent ran 30 focused caloric tests, 466 full engine
tests (1 optional GPU skip) and 94 scripts tests: OK. The original underflow
reproducer is now a regression test; power overflow and exponent overflow are covered.
The power-underflow branch has no independent end-to-end reproducer because the
temperature-resolution guard normally triggers first. No new physical validation,
architecture wiring, production sweep, or full-engine acceptance is implied.

## Current-host profiling (independent P1 work)

`scripts/profile_engine_paths.py` profiles the fixed, engineering-infeasible
B87_relaxed regression candidate; no optimizer/search runs. Parent saved
`docs/current_host_profile_2026-11-11.json`. Setup, warmup, warm wall measurements
and cProfile measurements are separate. Seven output fields matched exactly between
extracted and notebook-reference paths (rtol=atol=1e-12). The notebook uses its
existing fast/cache setup: this is not a cold notebook comparison.

Parent timings: extracted warm median 0.132879 s, notebook reference 0.064959 s;
reference/extracted wall-time ratio 0.488860. The extracted path was slower, not
accelerated. Profile evidence points to repeated
stage/gas-property calls and legacy-dictionary conversion. Independent reviewer: PASS.
No speculative JAX port or micro-optimization was performed.

## Additional P0 numerical-semantics closure

`gas.calc_stage_cp_k` and `gas.calc_h0_for_pressure` formerly returned a silent
50th iterate. Exhaustion now raises NumericalConvergenceError with coherent evaluated
cp/k, proposed cp_new/k_new, unhalved residual, T_mean and same-iteration H0.
Strict convergence test, arithmetic/order, and the source's pre-update cp/k success
return remain unchanged. Independent reviewer: PASS. Final parent suites: 463 engine
tests OK (1 optional GPU skip), 94 scripts tests OK; static verifier 6/7/0.

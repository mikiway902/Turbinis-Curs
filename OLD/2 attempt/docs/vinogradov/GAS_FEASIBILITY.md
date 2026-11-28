# Fixed-30700-rpm gas recovery after Vinogradov loss feedback

## Finding and scope

Six bounded calls to the **unchanged native** `resource_geometry_refinement` produced no successful proxy optimization. Replacing the legacy cubic/taper screening floor `(1.95, 1.95)` by geometry-measured maximum-normal-stress/Kn ratios, with the source's existing **1.05 guard**, did **not** by itself recover the native solver.

However, terminal geometries from attempts **3 and 4**, independently recomputed by real native `evaluate`, pass **all 14 native `design_checks`, all gas balances and all source full-radial assertions** at 110000 W after the gearbox. They are reproducible **gas-only recoupling seeds**, not successful native proxy solutions and **not accepted actual-blade designs**. Their stricter native search screens still fail. Actual Vinogradov contours, clearances, capacity, axial envelopes, normal stress and LM strength must be recomputed by the coupled calculation before acceptance; HCF remains `NOT_CHECKED`.

This distinguishes a restrictive search screen from a physical acceptance condition. It does not prove that the native screened problem is globally infeasible, or that either gas-only seed can support acceptable actual blades.

Full inputs, all six outcomes, native solver frame diagnostics, Configs, bounds, per-stage gas states, every native design check/balance and 101-point radial diagnostics are in [`native_gas_multistart_10000h.json`](../../exports/vinogradov/trials/native_gas_multistart_10000h.json).

## Authoritative inputs and unchanged physics

- Input: `exports/vinogradov/trials/feedback_10000h/report.json`, SHA-256 `d116d5d26d5019db86d206a4e0c1d21bc54eeb155b86e1858776a1b82b1ccd6a`.
- Native definitions: `TVD_1CBK_2st_10000h.ipynb`, `course_adapter.SOURCE_CELLS` **except 42**. No archived three-stage calculation or repeated original coupled solve was executed.
- Fixed shaft speed **30700 rpm**, two stages, 1250 K, CBK cycle and **110000 W after gearbox**. Source power acceptance is **0.001 W**, not an invented 1e-6 W tolerance.
- Frozen feedback coefficients: phi = `(0.8818921977644429, 0.9002568414422932)`; psi = `(0.9230975697395587, 0.9371526319047682)`.
- Initial reactions `(0.19460382006761426, 0.2323782218109361)`; reactions may vary only inside native bounds.
- Initial geometry `(30700, 0.21628855281243597, 11.2, 17.23138611020823, 1.0121808618089065, 1.0456293114692594, 0.517006670480376)`.
- Initial native air flow **0.8810736856224782 kg/s**; initial power **109999.99999794646 W**. Initial final-design failures: alpha2 stage1 **69.537317569° < 70°** and height-increment spread **0.538209730 mm > 0.25 mm**.
- Material is the original **IN-738C, not IN-738LC**; density **8110 kg/m³**, C_LM **20**, life **10000 h**, required normal-stress LM margin **1.2**. The unchanged bounded curve and its provenance are saved in the JSON. No LC creep/fatigue data, cooling assumption or extrapolated allowable was substituted.

The source `G` is air flow; individual stage `G` values include the source combustion/leak accounting and are separately saved. No gas equations were copied or reimplemented.

### Original optimizer bounds

| Parameter (source order) | Lower | Upper |
|---|---:|---:|
| n, rpm | 30700 | 30700 |
| Dtip, m | 0.16 | 0.50 |
| alpha1 stage1, deg | 11.2 | 19.8 |
| alpha1 stage2, deg | 11.2 | 19.8 |
| Ca ratio stage1 | 0.8 | 1.65 |
| Ca ratio stage2 | 0.8 | 1.65 |
| stage1 work split | 0.4 | 0.6 |
| G air, kg/s | 0.4 | 1.24 |
| reaction stage1 | 0.15 | 0.65 |
| reaction stage2 | 0.15 | 0.65 |

Every actual initial seed, including its `evaluate`-derived flow, and every terminal vector is inside these bounds. The native SLSQP objective, scaling and constraint mathematics were unchanged; all calls used maxiter=220 and the original ftol=1e-9.

## What actually limits the original failed probe

Attempt 1 reproduces the earlier **min g = -0.3252168516430946** and `Positive directional derivative for linesearch` (SLSQP status 8, 28 iterations). The scalar minimum is the **increment-spread guard**, not a generic solver tolerance or proof of impossible material life.

| Quantity at native terminal iterate | Value | Native search condition |
|---|---:|---:|
| SA1 height | 14.496745479 mm | >=14.5 mm |
| Height increments | 1.965975365; 2.247279578; 2.224494986 mm | each >=0.3 mm |
| Increment spread | **0.281304213 mm** | <=0.20 mm |
| Root Mach stage1 | **0.980115884** | <=0.98 |
| Eta stage1 | **0.707324084** | >=0.72 |
| Eta stage2 | **0.716657624** | >=0.72 |
| Stage1 screening temperature | 1140.547701732 K | native separated-endpoint thermal bound |
| Stage1 Kn stress | 73.302176636 MPa | proxy, not actual normal stress |
| Stage1 factor × Kn | 142.939244441 MPa | 1.95 × Kn |
| Stage1 bounded-curve allowable | 149.577754624 MPa | divided by required margin1.2 for constraint |
| Stage1 proxy LM margin | **1.046442880** | >=1.2 |
| Stage1 LM constraint deficit | **18.291115588 MPa** | allowance/1.2 - factor×Kn >=0 |
| Power residual | **-1040.680929 W** | equality not closed |

Stage2 LM is not the binding life screen (its normalized LM residual is +1.270320055). All 25 normalized residuals are saved. The native SA1, root-Mach, both eta and stage1 LM constraints also fail, even though the spread is the most negative normalized component. Alpha2 stage2 approaches the native 89-degree upper guard. Failed native iterates must not be reported as a 110-kW design: separate `evaluate` calls reclose power below.

## Measured proxy comparison and six starts

The exact input report's maximum actual normal stresses include **both** the five-section tension method and the three-section quadratic tension-plus-gas-bending method.

| Rotor | Maximum actual normal stress, MPa | Native Kn, MPa | Actual/Kn | 1.05 × actual/Kn |
|---|---:|---:|---:|---:|
| RK1 | 122.154795551 | 73.375613924 | 1.664787373 | **1.748026742** |
| RK2 | 155.431224311 | 92.422752419 | 1.681741998 | **1.765829098** |

The original actual RK1 LM minimum is **1.061354282 < 1.2**; a less conservative proxy does not erase this physical failure. The measured ratios apply to the original actual geometry only. Applying them at another gas geometry is an explicitly unvalidated sizing screen, not a newly established stress law.

Attempt 1 uses `(1.95,1.95)`. Attempts 2–6 use the guarded measured factors above, with unchanged physical and native search constraints. The six starts are:

1. Exact feedback geometry and reactions, legacy factors.
2. Exact feedback geometry and reactions, measured guarded factors.
3. Attempt1 terminal geometry/reactions, measured guarded factors; its initial flow is recomputed by native `evaluate`, not copied from the failed power iterate.
4. Compact higher-reaction seed: geometry `[30700,.222,12,17,1.04,1.10,.50]`, reactions `(.30,.35)`.
5. Larger-diameter/lower-loading seed: `[30700,.250,12,18.5,1.10,1.15,.55]`, reactions `(.30,.42)`.
6. Balanced-work/lower-reaction seed: `[30700,.232,11.5,18,1.0,1.07,.48]`, reactions `(.22,.28)`.

All six end with status8, not a success flag. The table separates **unclosed native optimization power** from **real `evaluate` flow at closed 110-kW power**.

| Attempt | SLSQP iterations | Native min g | Native power residual, W | Reclosed air flow, kg/s | Reclosed spread, mm | Final native design checks |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 28 | -0.325217 | -1040.681 | 0.885759067 | 0.282895 | 13/14 |
| 2 | 88 | -8.196671 | +166.730 | 0.880169467 | 2.246011 | 13/14 |
| 3 | 48 | -0.118491 | -15.571 | 0.885100509 | **0.199088** | **14/14** |
| 4 | 40 | -0.115662 | -1109.189 | 0.880426114 | **0.199859** | **14/14** |
| 5 | 110 | -1.417852 | -986.354 | 0.878302393 | 0.560400 | 13/14 |
| 6 | 120 | -1.329780 | -1852.465 | 0.877789480 | 0.543399 | 13/14 |

All reclosed powers meet the original 0.001-W tolerance. All six complete both 101-point native radial solves and pass the source cell74 assertions: max Mc1<1; max Mw2<1; each mass residual<1e-8 kg/s; Euler residual<0.002 J/kg; relative radial-equilibrium error<0.005; grid/Gauss scale difference<1e-4. These do not imply the stricter **0.98** optimizer root-Mach screen passes.

## Reproducible gas-only seeds and remaining screens

Attempt **3** is the preferred gas-only seed because its native optimization equality residual was much smaller than attempt4's. Use its full Config in the JSON, not the baseline reactions:

```python
geometry = [30700.0, 0.2392110374937786, 11.2, 19.769221732085402,
            1.0482315451439432, 1.1176339515317917, 0.5814192123921609]
rho_stage1 = 0.222110440136003
rho_stage2 = 0.40129120624577946
# phi/psi and every other Config field remain the exact feedback inputs.
```

Real `evaluate` yields **109999.99999997829 W**, air flow **0.8851005087736833 kg/s**, SA1 **14.501851551 mm**, spread **0.199087619 mm**, eta `(0.708156999, 0.717211903)`, and Gauss root Mach `(0.980721440, 0.748150779)`. Full-grid max Mc1 is `(0.980723199, 0.748151176)` and max Mw2 `(0.657324003, 0.688480601)`. Both stages pass every source radial assertion and every gas balance.

At its **native failed iterate**, stage1 proxy margin remains only **1.158811111**, with a **4.453104081-MPa** deficit in allowance/1.2 minus guarded stress; eta1 is **0.708150935 < 0.72**, eta2 **0.717205573 < 0.72**, root Mach1 **0.980708919 > 0.98**. Lowering the floor improves LM screening but does not remove these simultaneous search restrictions.

Attempt **4** is a second independent gas-only seed:

```python
geometry = [30700.0, 0.23618692662535348, 11.200000000090867,
            19.0393432473353, 1.0291934680674395, 1.1114762121519604,
            0.5729871305858795]
rho_stage1 = 0.21426124097962035
rho_stage2 = 0.37537972841772516
```

It recloses to **109999.99999962946 W**, air flow **0.8804261144604771 kg/s**, SA1 **14.626268432 mm**, spread **0.199859243 mm**, eta `(0.708856998,0.720164633)`. Again this is not an actual-contour strength result.

### Screening conditions are not final physical acceptance

- Native SA1 guard **14.5 mm** versus final native check **14 mm**.
- Native increment spread **0.20 mm** versus final **0.25 mm**; native increments >=0.3 mm versus final strictly positive increments.
- Native eta >=**0.72** versus final >**0.70**.
- Native root Mach <=**0.98** versus final full-radial assertion <**1**.
- Native alpha2 bands **71–89 / 81–89 degrees** versus final **70–90 / 80–90 degrees**; native mean speed <=395 versus final <=400 m/s; native root reaction >=0.055 versus final >=0.05.
- Both native pressure-ratio bounds remain **1.8–2.4**. Original geometry-variable bounds and fixed RPM remain unchanged throughout.
- Native `design_checks` stress entries are **Kn ring screening** at the source project/200-MPa guide, **not** the separate actual-contour **300-MPa maximum-normal-stress cap**. Passing Kn does not prove passing actual stress.
- The native thermal screen is a separated-endpoint conservative relative-total-temperature bound, not a solved metal temperature. Saved full radial temperatures are independently recomputed; substituting them into native optimizer constraints was not attempted.

Neither changing a proxy factor nor passing all gas checks waives **300 MPa actual maximum normal stress, LM1.2 using the original IN-738C curve, 14-mm final height, original power closure, actual contours, area taper, row counts, capacity or clearances**. No actual blades at either changed gas geometry have been accepted in this investigation.

## Reproduction

Runtime used: container `jupyter`, Python **3.11.10**, NumPy **2.0.2**, SciPy **1.14.1**. Notebook and individual executed-cell hashes are saved in JSON. From the course directory, execute the following with `podman exec -i -w /home/jovyan/work/Turbinis-Curs/1-Turbinis-Curs-Last jupyter python -`:

```python
import json, sys, types
from vinogradov.course_adapter import SOURCE_CELLS

book = json.load(open('TVD_1CBK_2st_10000h.ipynb'))
module = types.ModuleType('_native_gas_reproduction')
sys.modules[module.__name__] = module
ns = module.__dict__
for index in SOURCE_CELLS:
    if index != 42:
        exec(compile(''.join(book['cells'][index]['source']),
                     f'TVD_1CBK_2st_10000h.ipynb:cell{index}', 'exec'), ns)

data = json.load(open('exports/vinogradov/trials/native_gas_multistart_10000h.json'))
for attempt in data['attempts']:
    values = dict(attempt['input_config'])
    values['phi'], values['psi'] = tuple(values['phi']), tuple(values['psi'])
    cfg = ns['Config'](**values)
    captured = {}
    def capture(frame, event, arg):
        if (event == 'return' and
                frame.f_code is ns['resource_geometry_refinement'].__code__):
            captured.update(frame.f_locals)
    sys.setprofile(capture)
    try:
        ns['resource_geometry_refinement'](
            attempt['input_geometry'], cfg, ns['STRENGTH_REQUIREMENTS'],
            stress_factors=attempt['stress_factors'], maxiter=220)
    except ns['Bad'] as exc:
        print(attempt['label'], str(exc))
    finally:
        sys.setprofile(None)
    print(captured['solution'], captured['cons'], captured['power'] * 10000)
    # Separate true-power reclosure; do not accept failed optimization flow.
    from dataclasses import replace
    x = captured['x']
    cc = replace(cfg, rho_stage1=float(x[8]), rho_stage2=float(x[9]))
    real = ns['evaluate'](x[:7], cc)
    print(real['power_W'], real['G'], ns['design_checks'](real))
    print(ns['balances'](real))
    for stage in real['stages']:
        print(ns['stator_radial_screen'](stage, cc))
        radial = ns['radial_stage'](stage, x[:7], cc, 'alpha_const', points=101)
        print(radial['mass_in_residual'], radial['mass_out_residual'],
              radial['euler_max_residual'], radial['radial_eq_relative'],
              radial['inlet']['M_c'].max(), radial['outlet']['M_w'].max())
```

The profile hook only copies real frame locals on return/unwind; it does not patch `minimize`, source formulas, constraints, success flags or exceptions. Every scientific computation was inline; no throwaway script was created. No production Python or notebook was edited, and no tests/build/lint were run.

## Interpretation

Observed recovery is **not blocked solely by the 1.95 proxy floor**: measured guarded factors still leave stage1 LM and eta/root-Mach search violations. No start escaped local SLSQP status8. Conversely, actual native gas bounds and final gas checks do **not** exclude all recovered meanline geometries: two saved examples pass them after real power closure. The remaining question is whether recoupled actual blades can satisfy all physical requirements, and whether the unchanged stricter native screens have another feasible basin. These six starts establish neither global impossibility nor coupled design convergence.

## Necessary annulus CF/LM bound for gas-only seeds 3 and 4

The unchanged source `resource_annulus_bound(result, full_radials, requirements)` was additionally called for attempts3/4 after real `evaluate` and complete101-point radial solves. It uses the original density, speed, radii, actual radial relative-total-temperature bounds, IN-738C curve and LM1.2 requirement. It is independent of blade count, absolute blade scale and the1.95 screening factor. Target Aroot/Atip remains **1.6**; the native function retains its existing0.5% taper tolerance, hence ratio_max=**1.608**. This is a necessary lower bound on root tension for monotone area, not a constructive profile or a sufficient strength test.

| Gas seed | Rotor | Full-radial thermal upper bound, K | Necessary root tension lower bound, MPa | LM allowable, MPa | Allowable/1.2, MPa | Headroom over necessary bound, MPa | Native necessary-bound result |
|---|---|---:|---:|---:|---:|---:|---|
| 3 | RK1 | 1138.954271484 | 96.088454011 | 151.681823858 | 126.401519882 | 30.313065870 | Pass |
| 3 | RK2 | 1027.105992085 | 119.680723704 | 370.905574740 | 309.087978950 | 189.407255246 | Pass |
| 4 | RK1 | 1140.064041425 | 96.525893824 | 150.213307434 | 125.177756195 | 28.651862371 | Pass |
| 4 | RK2 | 1025.231766777 | 120.361987453 | 375.817961151 | 313.181634293 | 192.819646839 | Pass |

Thus this existing native bound **does not reject either seed before profile selection**. It does not establish that an actual area distribution can attain the lower tension bound, or that gas bending, actual maximum normal stress, LM margin, capacity and all contour constraints will pass. In particular, passing this necessary bound is compatible with the failed, more restrictive measured-factor screening margins reported above.

The JSON field `native_annulus_necessary_bound_followup` saves both source tables, complete inlet/outlet thermal-bound radius/temperature arrays, Configs, power/flow, function provenance, derived headroom and the source taper tolerance. No additional optimizer start or full profile recalculation was performed.

To reproduce the follow-up using the loader above:

```python
for number in (3, 4):
    saved = data['attempts'][number - 1]['true_power_reevaluation']
    values = dict(saved['config'])
    values['phi'], values['psi'] = tuple(values['phi']), tuple(values['psi'])
    cfg = ns['Config'](**values)
    real = ns['evaluate'](saved['geometry'], cfg)
    radials = [ns['radial_stage'](stage, real['geometry'], cfg,
                                'alpha_const', points=101)
               for stage in real['stages']]
    print(ns['resource_annulus_bound'](real, radials, ns['STRENGTH_REQUIREMENTS']))
    thermal = [ns['candidate_temperature_bound'](rd, stage, cfg)
               for rd, stage in zip(radials, real['stages'])]
    print(thermal)
```

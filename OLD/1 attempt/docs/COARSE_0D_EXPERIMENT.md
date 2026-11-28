# Bounded 0D coarse experiment — 2026-11-11

**ILLUSTRATIVE / NOT PHYSICALLY VALIDATED / NOT A PRODUCTION SWEEP.**

## Reproduce the experiment

```powershell
.\.venv\Scripts\python.exe scripts/architecture_coarse.py --destination reference/architecture_coarse/2026-11-11-pi3-13 --resume
```

Without `--resume`, a nonempty destination is refused. For a fresh experiment choose
a new destination. All assumptions and the ordered payload-plan hash/count are in
run config; incompatible resume is refused. The script never imports experimental
`caloric.py`. Source arithmetic is the reviewed legacy opt-in 0D architecture model.

Plan: pi_k integers 3..13 for B (single CC) and C (free-power turbine); for A (dual CC),
three logarithmic split weights 0.4/0.5/0.6 at each pi_k. The second ratio is computed
as total/first to retain the product. This covers both architectures around 7..8;
it does not assert compressor feasibility anywhere in the range.

Uniform illustrative fixtures: total inlet 288 K/101300 Pa, combustor exit 1250 K,
110000 W gearbox input, gearbox eta=0.98, compressor poly eta=0.85, turbine poly
eta=0.90, mechanical eta=0.99, final turbine total pressure=150000 Pa. No cooling,
bleed, shaft speeds, maps or stage inventory is modelled. These are assumptions,
not measured engine data. Full cycles are recalculated at every pressure/split point.

## Observed results

- 55/55 calculations SUCCESS, all stored, none accepted as an engineering design.
- Reproduction 55/55 matched at rtol=1e-12, atol=1e-9; resume appended zero records.
- Journal 704366 bytes, SHA-256
  `12239e6ed2d1ea0021c55333b2a45e51c57c0c19d63fb93454b0f19f42049e9d`.
- Maximum target power residual 8.73115e-11 W; mass residual 0 kg/s.
  These are algebraic closures, not independent thermochemistry validation.
- Stage rho and strength: NOT_CHECKED for every record. No feasible count is inferred.
- `summary.json` includes every point, status by architecture, post-hoc unknown checks,
  adjacent-pi gas/fuel-flow finite differences and fixed-pi dual-split spreads.
  Failed adjacencies are not bridged; failure retention is also covered synthetically
  because no point in this particular plan failed naturally.
- `controlled_cycle_slices.png` displays gas/fuel flows, including all split curves;
  the lead generated it from the saved summary only, without recalculating physics.

For B, gas flow changes from 1.55926 kg/s at pi=3 to 0.753021 kg/s at pi=13;
fuel flow changes from 0.0353016 to 0.0120448 kg/s. Gas-flow slope changes sign
between the 9..10 and 10..11 intervals. This identifies a reduced-model trend,
not an engineering optimum. For A at pi=13, the three sampled gas flows span
0.755964..0.756171 kg/s. Their small spread is sensitive to the retained compressor
mean-property discretization; it cannot establish an optimum centrifugal split.

**B and C give essentially identical flow/fuel results here.** With constant gas
properties, equal turbine polytropic efficiencies and equal mechanical efficiencies,
cascading expansions retains the same total enthalpy drop, and shaft bookkeeping
alone does not create an advantage. No independent shaft-speed/map/stage/loss model
is present. Equality is a consequence of this reduced model, not proof that the
architectures are equivalent or that a free turbine has no real advantage.

## Checks and limits

New script/tests reviewed independently: PASS. Parent observed 463 engine tests
OK (1 GPU skip), 94 scripts tests OK; static verifier 6 PASS / 7 WARN / 0 FAIL.
The original 13-record smoke journal and this 55-record journal remain separate;
changing constraints never rewrites them. Summary git/timestamp metadata describes
the reporting invocation, not a retroactive claim that the dirty code was committed.

This closes a bounded M5 sampling/persistence experiment and supplies limited M7
controlled slices. It does not close M2 energy-method/stage validation, M6 production,
architecture ranking, geometry/stress sensitivity, Pareto finalists or final delivery.
Pending caloric architecture integration and absent stage/map/cooling evidence are recorded in
`ENERGY_METHOD_AUDIT.md`; no best-engine claim is supported.

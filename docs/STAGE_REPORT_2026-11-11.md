# Stage checkpoint — 2026-11-11

This is a **checkpoint report**, not the final project report and not a physical
validation. It records what was implemented, what was independently checked, and
what remains open at the point where the work paused for the checkpoint commit.

## Power and architecture baseline

- Net target is **110 kW at the gearbox input, before the reducer**.
- The active legacy cycle and stage power closure use
  `N_ud = L_t·g_sum·η_м_тур − L_k`; reducer efficiency is downstream only,
  giving `107.8 kW` for `η_р = 0.98`.
- Selected topologies (one combustor each):
  - **A / `SINGLE_SHAFT_DUAL_CC`** — two centrifugal compressors in series,
    turbine and both compressors on one shaft, gearbox → propeller.
  - **B / `SINGLE_SHAFT_SINGLE_CC`** — one centrifugal compressor, turbine on
    one shaft, gearbox → propeller.
  - **C / `FREE_POWER_TURBINE`** — compressor + gas-generator turbine on one
    shaft; separate free-power turbine → gearbox → propeller.

## Implemented in this stage

| Area | Evidence | Status |
|---|---|---|
| Pre-reducer power plane | `turbine_engine/cycle.py`, `baseline.py`, `performance.py`, active notebooks and optimizer adapters | implemented and anchored |
| Explicit 0D architecture cycles | `turbine_engine/architecture_cycle.py`; variant `uncooled_no_bleed_0d_ideal_mixing` | reference only, not physically validated |
| GG → FPT state transfer | actual `StationState` passed from GG turbine to FPT; work/flow bases kept separate | implemented |
| Shaft balances | `turbine_engine/shaft.py`; signed raw MAIN/GG/FPT residuals | implemented |
| Last-stage pressure boundary | textbook p. 92, items 138–140 supports `p0 > p_t`; unsupported 2 % margin removed from active module and notebook | corrected; not full stage validation |
| Numerical failure diagnostics | `diagnostics.py`; no `None`/stale iterate/accidental `TypeError` on audited solver exhaustion paths | implemented for inspected paths |
| RAW/reproducibility | `raw.py`, `dataset.py`, `sampling.py`, `constraints.py`; immutable JSONL journal + manifest, Parquet export, resume, reproduce | infrastructure tested synthetically; no production dataset |
| Post-processing analysis | `analysis.py`; conservative feasibility, near-feasibility, Pareto, paired sensitivity | utilities tested synthetically; no real analysis outputs |
| CPU/GPU numerical evidence | `backends.py`, `scripts/validate_backends.py`, WSL2 JAX CUDA venv | bounded gas-property kernel validated; no engine speedup claim |
| Optimizer provenance | failed attempts retain inputs/phase; optional graph diagnostics no longer erase computed stage results; legacy engineering repairs labelled | adapter fixes; legacy repairs still active |

## Representative 0D smoke result

Inputs: inlet total `288 K / 101300 Pa`, total compressor ratio `11`, `T_g = 1250 K`,
turbine outlet total pressure `106365 Pa`, compressor polytropic efficiency `0.85`,
turbine polytropic efficiency `0.90`, mechanical efficiency `0.99`, gearbox
efficiency `0.98`, target `110000 W`. The dual-CC case used series ratios
`3.5 × 3.142857`. This is a software/equation smoke test, not a design selection.

| Architecture | Air flow, kg/s | Fuel flow, kg/s | Gearbox input, W | Gearbox output, W | Target status |
|---|---:|---:|---:|---:|---|
| Single-shaft single-CC | 0.499215 | 0.008600 | 110000.0000 | 107800.0000 | met |
| Single-shaft dual-CC | 0.500281 | 0.008612 | 110000.0000 | 107800.0000 | met |
| Free-power turbine | 0.499215 | 0.008600 | 110000.0000 | 107800.0000 | met |

At identical 0D assumptions, all three topologies close the same ideal cycle;
the numerical differences above come from the dual-compressor split and
variable-`k` mean-property path. This does **not** show that the architectures
are equivalent in practice: maps, blading, stage counts, cooling, mechanical
arrangement and off-design behaviour are absent.

## Verification actually run

- `turbine_engine/tests`: **399 tests OK, 1 skipped** (Windows; skip is the
  optional GPU test because Windows JAX is CPU-only).
- `scripts`: **43 tests OK**.
- `scripts/verify.py`: **6 PASS, 7 WARN, 0 FAIL**; warnings are the absent
  `docs/engineering/*` authority files and absent `reference/baselines/`.
  This verifier checks structure only; physics is not verified.
- `git diff --check`: clean.
- WSL2 JAX GPU validation: `CudaDevice(id=0)` on RTX 4060, float64, max absolute
  parity error `0.0` for the bounded gas-property kernel. The kernel benchmark
  was slower than NumPy for the tested 65,536-element batch (`1.896×` NumPy
  time); no speedup is claimed. See `docs/backend_validation.json`.

## Open engineering gaps

- Compressor stage inventory, maps, corrected flow, surge/diffusion limits.
- Turbine stage counts, velocity triangles, cooling/mixing and blade geometry.
- Cooling, leak, bleed and composition semantics outside the no-bleed variant.
- Gear ratio, shaft speeds, gearbox efficiency beyond `eta_reducer`, propeller
  load and control law.
- Material, metal temperature, creep/resource, thermal stress and the
  `σ <= 300 MPa` limit remain `NOT_CHECKED`.
- `cp`/`alpha`/thermochemistry consistency is not independently validated.
- The legacy optimizer's stage construction still includes engineering repair
  steps; the new RAW infrastructure must receive a separate closure rather than
  treating repaired legacy output as RAW.
- No production RAW dataset, full design-space sweep, final Pareto analysis,
  finalists recalculation or final engineering report exists yet.

## Source boundary

The external Sergey and Anton notebooks remain examples, not approved
regression baselines. Their bounded provenance and unresolved authorship
chronology are in `docs/comparisons/Источники_и_методики.md`. The textbook
cross-check is selective and source-scoped, not a complete page-by-page
validation of every external notebook cell.

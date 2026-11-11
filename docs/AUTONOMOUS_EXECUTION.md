# Autonomous execution gates

Updated: 2026-11-11. This is an execution-gate map, not a claim of completion.
Progress source of truth remains `PROJECT_CHECKLIST.md`.

## Revised mandatory route — user scope correction, 2026-11-11

This section supersedes the earlier proposal to integrate a new caloric/enthalpy
cycle. The master prompt's goals remain unchanged; the implementation route is
restricted to the two supplied textbooks and the existing calculation. Do not
rewrite `Super-Promt-Autonomous.md` or treat experimental work as a new requirement.

### Method authority checked in the original PDFs

- Cycle book, printed pp.12–13 and 19: use mutually consistent cp/k averaged over
  each compression/expansion process, with cp=R*k/(k-1).
- Cycle book, pp.19–20: retain the published fuel balance using mean heat capacities
  and its stated fuel-reference assumptions; alpha=1/(gm*l0). Do not replace it by
  a new integral-mixture combustor merely to force a different energy residual to zero.
- Cycle book, appendix 6 p.57: mean cp (700 K split) and true cp (750 K split) are
  different published correlations. Neither may silently substitute for the other.
- Turbine part 1, **printed pp.15–16**: first estimate k at a mean expansion
  temperature, then refine it using the calculated outlet temperature. Printed p.24
  distinguishes endpoint mean heat capacities from the true cp approximation at
  mean process temperature. Original PDF pagination takes precedence over the
  Markdown's shifted page labels.

### Audit of work already done

Keep: extracted legacy reference and anchors; 110 kW pre-reducer definition;
architecture/shaft/state contracts; explicit GG-to-FPT state transfer; RAW journal,
resume/reproduce; post-hoc constraints; typed failure diagnostics; bounded profiling;
13/55-point illustrative datasets and useful flow slices. They implement the prompt,
but do not prove physical correctness or engineering feasibility.

Freeze outside the mandatory route: `experiments/caloric/`. This preserves the
integral property/component work and the interrupted integral combustor without
importing it from `turbine_engine`. The optional experiment suite has a known
one-ULP combustor assertion failure; it is not repaired or presented as passing.
Do not spend further effort on integral cycles, new thermochemistry, additional
storage formats, cosmetic CLI or extreme-number support unrelated to the selected
engineering method. No experiment is deleted and no acceptance tolerance is loosened.

### Ordered P0/P1 route

1. **Properties and method selection:** distinguish published true cp from mean cp
   and the original notebook derivative approximation. Implement only the direct
   source correlation and reuse the existing process/stage iteration. Preserve the
   reference default; compare an explicitly labelled textbook option on existing
   control points. Record the actual selected basis, including failure records.
2. **Cycle by the cycle book:** reconcile process cp/k, alpha, fuel/cooling/mass
   bookkeeping and total/static boundaries in the existing cycle/components. Keep
   the p.20 fuel relation; test its own balance and assumptions, not a replacement
   caloric model. Separate each aggregate's process-mean properties; retain actual
   GG outlet T/p/flow for FPT. No new general-purpose engine framework.
3. **Turbines and stages by part 1:** use established prescribed-pressure versus
   prescribed-work closures, source-backed stage-count/work distribution and existing
   meanline equations. Check velocities, Mach, reaction, angles, geometry and the
   available strength model. Do not automatically copy three stages into all schemes.
4. **Connect that calculation to RAW:** retain every attempted candidate and causal
   stage result; engineering violations annotate rather than terminate calculation.
   Keep source-domain/solver failures distinct from engineering infeasibility.
5. **Bounded sweep, then sufficient coverage:** smoke/regression/resume checks and
   measured time/storage first; pi_k=3..13, meaningful compressor/turbine splits and
   independent geometry/speed variables. Refine only informative boundaries/slices.
6. **Analysis and delivery:** profiles, feasibility/near-feasibility, controlled
   sensitivities, useful plots, Pareto and reference recalculation of finalists.
   Explain architecture differences and missing checks; no best-engine claim from 0D.

Current evidence for steps 1–2: the existing single-point cycle now accepts an
explicit `textbook_true_cp` option, retaining the p.20 mean-cp fuel relation while
using process-mean true cp/k, one fuel-based alpha, refined actual outlet temperatures
and consistent local turbine-exit Mach/pressure conversion. The existing const-hub
path can select this cycle and the textbook stage basis together. Defaults remain
legacy. Tests/reviews pass; no integral model is imported.

Subsequent step-2 evidence: the existing A/B/C aggregate components now also support
the explicit textbook basis, with separate process means and local endpoint cp/k.
GG output is passed unchanged to FPT, which evaluates its own process properties.
Strict RAW v2 persists the method and component property records; v1 remains legacy.
Saved 13-point source smoke reproduces 13/13, including explicit unsupported-input
and physical-failure records. Stage inventory, source-backed meanline coupling and
missing cooling/map/strength checks remain open; aggregate closure is not stage validation.

Step-3 continuation: original turbine part 1 printed p.87 item 54 was visually
checked. Its unrounded stage-count estimate is reused from `baseline.py`; n,
geometry, y0 and recovered exit velocity remain explicit assumptions, not hidden
stage-count defaults. The bounded stage smoke evaluates A MAIN, B MAIN, C GG and
C FPT with 1/2/3 stages using existing const-hub equations.

A necessary M3 correction was found during that check: the reference secant clips
pi by the engineering `pi_stage_max`. `orchestration` now has an explicit
`raw_unbounded` policy enforcing only the mathematical expansion domain, while
default `legacy_clipped` retains reference results. The saved raw 12-point plan
calculates all 12; the clipped reference stalls on 5. Pi acceptance is post-hoc,
independent of overall NOT_CHECKED feasibility. The runs are independent assembly
checks, not a coupled meanline engine: outlet total-pressure and power differences
from the aggregate cycle are retained and must be closed before architecture ranking.

Actual-boundary follow-up: explicit bracketed closures now evaluate actual last-stage
total pressure or sum of actual stage mechanical work, restoring the namespace static
pressure after every call. A fixed GG/3 → FPT/1 regression passes the actual GG total
T/p unchanged into FPT. Compressor work and FPT outlet pressure close within 1e-6
relative, but FPT output is 106095.69 W, not 110000 W. This establishes only the
linkage; engine power closure, justified stage alternatives and full RAW coupling
remain open. No flow or geometry was rescaled to hide the 3904.31 W deficit.

Bounded step-3/4 follow-up now persists 15 actual-boundary alternatives through
`scripts/architecture_coupled_stage_smoke.py`: A/B counts 1–3 and all nine C GG/FPT
count pairs. Saved run `reference/architecture_stage_smoke/2026-11-11-actual-coupled/`
has 13 SUCCESS / 2 NUMERICAL_FAILURE, reproduces 15/15 and resumes without duplicates.
A/B count1 failures mean only that the explicit static-pressure bracket did not
contain a root. All successful points retain nonzero 110 kW deficits, so this does
not authorize a production sweep or a feasible-engine claim. Actual full stages,
GG/FPT linkage, signed power residuals and post-hoc pi profiles remain in RAW.

Missing maps, material life or detailed cooling physics are marked NOT_CHECKED /
MISSING_* at their affected checks. They do not justify inventing those models or
blocking independent source-supported cycle/stage investigations. Validated full
engine claims remain gated by the missing evidence.

Performance work follows the measured bottleneck only. Existing RTX 4060 kernel
evidence is sufficient to retain an optional GPU path; it does not require another
backend rewrite before the textbook calculation is connected end-to-end.

## Authority and scope

`Super-Promt-Autonomous.md` replaces the older prompt. The user's autonomous-mode
instruction removes the older prompt's Phase-0 confirmation stop. The selected
session lead performs the gatekeeper role; historical model names do not require
changing providers/models. Independent review remains mandatory for meaningful changes.
Each implementation task has at most three correction/review cycles; unresolved
findings are recorded, not hidden by looser tolerances.

110 kW is net mechanical power at the **gearbox input**. A is dual-CC single-shaft,
B is single-CC single-shaft, C has separate gas-generator and free-power shafts.
Every architecture has one combustor. No architecture can be represented by simply
splitting a legacy single-shaft turbine's work. Reaction acceptance `rho <= 0.65`
does not restrict calculation or extend digitized source graphs.

## Phase-to-milestone map

The older prompt's phase numbers are retained only as navigation. Their gate
criteria must still be met; a new helper is not equivalent to completing a phase.

| Phases | Current milestone | Required evidence |
|---|---|---|
| 0–1 | M0/M1 | Sources, physical definitions, independent software reference and anchors |
| 2–6 | M2 | Shared equations and executable architecture-specific cycles; mass/energy/shaft closure; explicit assumptions |
| 7–8 | M3 | No engineering-only rejection; structured failures; immutable RAW, manifest and reproduction |
| 9–11 | M4 | Profile and numerical CPU/JAX/GPU parity; actual device and dtype evidence |
| 12 | M5 | Streaming batches, checkpoint/resume and storage-integrity tests |
| 13–18 | M5/M6 | Validated model, smoke run, size/runtime/storage estimates, then justified coverage/refinement |
| 19–24 | M7 | Post-hoc annotation, paired sensitivity, feasibility, Pareto and reference recalculation |
| 25 | M8 | Final evidence audit and engineering report with honest limitations |

## Active independent workstreams

- **M2:** explicit sourced 0D architecture-cycle variant and real GG outlet state
  transfer. Preserve the current legacy cycle; identify reduced variants explicitly.
- **M3:** strict candidate/failure records, post-hoc constraint annotation and
  deterministic durable dataset recovery/reproduction.
- **M3 numerics:** explicit convergence/failure diagnostics instead of returning
  `None`, accidental `TypeError`, or an unmarked last iterate.
- **M4:** optional float64 numerical kernel parity and actual Windows CPU / WSL2
  RTX 4060 validation. A kernel benchmark is not an engine-cycle speedup.
- **M7 infrastructure:** conservative feasibility, near-feasibility, Pareto and
  controlled paired sensitivity, initially tested with synthetic data only.

## Current gate restrictions

No production engineering sweep or best-engine claim until the relevant
architecture and stage models pass their source/mass/energy/state gates. Reduced
0D variants may support explicitly labelled methodological experiments, but cannot
validate compressor blading, turbine geometry, material life or cooled operation.
Unknown checks must stay `NOT_CHECKED`, never become feasible by omission.

Dependency installation is limited to project/user-local environments. No secrets,
privileged system modifications, push or deployment. The working tree includes
pre-existing user changes; no mixed-work checkpoint commit may silently include
them. Durable checkpoints consist of recorded files/tests until an atomic and
reviewed commit boundary can be isolated safely.

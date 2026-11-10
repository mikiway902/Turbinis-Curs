# Autonomous execution gates

Updated: 2026-11-11. This is an execution-gate map, not a claim of completion.
Progress source of truth remains `PROJECT_CHECKLIST.md`.

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

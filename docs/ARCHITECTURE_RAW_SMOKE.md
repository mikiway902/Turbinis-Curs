# Architecture RAW integration — 2026-11-11

## Textbook aggregate continuation — RAW v2

New saved grid: `reference/architecture_smoke/2026-11-11-textbook-v2/`.
Input/output v2 adds the explicit textbook property method and full component
process records; v1 remains the unchanged legacy model. Compressor process alpha
is serialized as `"dry_air"`, never JSON Infinity. Source RAW dataset label:
`architecture_cycle_0d_textbook_raw_v2`.

Run/resume:

```powershell
.\.venv\Scripts\python.exe scripts/architecture_smoke.py --destination reference/architecture_smoke/2026-11-11-textbook-v2 --preset grid --gas-property-basis textbook_true_cp --resume
```

13 attempts retained: 9 SUCCESS, 3 MODEL_ERROR (the intentionally incompatible
external-state fixtures do not specify the composition needed by this method),
1 NON_PHYSICAL_STATE (nonpositive net-output fixture). No missing composition is
inferred. Reproduction 13/13 matched; resume appended zero records. Rho remains
NOT_CHECKED for all attempts. These are labelled methodological results, not
stage feasibility, a production sweep or physical validation.

Journal: 152948 bytes; SHA-256
`febe2ab92a3dfaf868a280a5c6d684ad7cf2749815d276a787739845a5681a87`.
Plan hash: `dec48f7b6decf1c3eeed8bdbeac8b8780cf94913647861881d83f75e3fc4b60d`.
Max target-power residual 8.73115e-11 W; mass residual 0 kg/s.

In the first source free-power record, the actual GG outlet is exactly the FPT
inlet. GG process mean T=1129.918937 K and cp=1203.682616 J/(kg K); FPT process
mean T=941.700032 K and cp=1165.533434 J/(kg K). Thus endpoint state continuity
does not force the two process-property approximations to be the same.

Independent reviews PASS for components (after residual-equation/oracle fixes),
architecture wiring and v2 persistence. Parent observed 510 active engine tests
OK (1 GPU skip), 116 scripts tests OK; static verifier 6 PASS/7 WARN/0 FAIL.
Old v1 smoke and 55-point coarse data were not overwritten; their numerical
reproduction remains covered by tests. The v1 description below is historical.

## Scope and API

New files: `turbine_engine/architecture_raw.py`,
`turbine_engine/tests/test_architecture_raw.py`, `scripts/architecture_smoke.py`.
No legacy physics, notebook formulas or regression anchors were changed.

`input_to_payload` / `payload_to_input` implement `architecture-cycle-input-v1`.
Every physics field and gas assumption is explicit; both legacy and external-state
combustor closures round-trip. Unknown/missing fields, bool/string numbers,
non-finite and inexact integer conversions are rejected. Outputs include component
states, shaft powers, mass/pressure/power residuals and model limitations.

`evaluate_architecture_payload` returns `EvaluationOutcome`; use it with the
generic RAW/dataset evaluator and derive its architecture argument with
`architecture_from_payload`. Generic RAW labels are caller-supplied:
`evaluate_architecture_record` additionally checks existing record label consistency.
The payload evaluator alone cannot validate a separately supplied record label.

Physical closure failures map to NON_PHYSICAL_STATE. Typed solver diagnostics retain
their declared numerical/nonphysical status and available last state. Schema errors
map to MODEL_ERROR/schema; unexpected exceptions stay visible as evaluator_exception.
Existing physical errors expose only inputs/reason, not unavailable partial components.

## Saved bounded experiment

```powershell
.\.venv\Scripts\python.exe scripts/architecture_smoke.py --destination reference/architecture_smoke/2026-11-11-grid --preset grid
.\.venv\Scripts\python.exe scripts/architecture_smoke.py --destination reference/architecture_smoke/2026-11-11-grid --preset grid --resume
```

The saved journal, manifest and checkpoint contain 13 attempts: four fixture
variants per architecture and one deliberately nonphysical single-shaft point.
A = dual-CC, B = single-CC, C = free-power turbine. Assumptions are illustrative
test fixtures, not validated design values. Each successful point requests 110000 W
at gearbox input; gearbox efficiency 0.98 gives 107800 W downstream.

- 12 SUCCESS, 1 NON_PHYSICAL_STATE; no attempts filtered.
- Reproduced 13/13 with rtol=1e-12 and atol=1e-9 (software consistency only).
- Stage rho unavailable: 13/13 separate annotations NOT_CHECKED.
- Journal: 155240 bytes; SHA-256
  `e90d96b65c0d97d9db7917fee5a1570f5e905f88ba1262e17006ab9af54f7200`.
- Initial calculation/persistence loop: 0.076354 s; resume loop: 0.025330 s.
  These are single observations, excluding reproduction, not throughput benchmarks.
- Resume appended zero records, preserving the journal hash. Ordered payload plan
  count/hash is part of run config; changed plans are refused even at the same version.
- Interruption tests exercise both KeyboardInterrupt and ordinary error after three
  appends: manifest stays incomplete, then resumed journal/manifest/checkpoint equal
  uninterrupted execution byte-for-byte.

At this fixture size, arithmetic projection is about 12 kB per attempt (~12 GB per
million), before Parquet or other products. This is a storage caution, not permission
or a validated estimate to launch a million-point sweep. Component state is duplicated
in outputs/intermediates. No large sweep or full-cycle GPU acceleration was performed.

## Independent verification

Parent executed:

- `.venv\Scripts\python.exe -m unittest discover -s turbine_engine/tests -p "test_*.py" -q`
  — 427 tests, OK (1 optional GPU skip).
- Same discovery under `scripts` — 43 tests, OK.
- `.venv\Scripts\python.exe scripts/verify.py` — 6 PASS, 7 WARN, 0 FAIL;
  structural only, not physics verification.
- Independent GPT reviewer: PASS on corrected adapter/CLI/tests. Initial findings
  concerned nested schema error classification and overwritten nonphysical status;
  both fixed and tested. Parent additionally required exact integer handling,
  interruption completion semantics and plan-bound resume.

## Remaining gates

### SOURCE_DISCREPANCY — FPT gas-property evaluation temperature

Read-only tracing identified a pre-existing description/implementation mismatch:
`architecture_cycle.py` describes cp/k evaluated at each turbine inlet, but its
FPT receives the actual GG outlet with inherited cp/k (`turbine.py` preserves them).
Parent verified the saved first free-power record: GG inlet 1250 K and FPT inlet
984.4646426126983 K both carry cp=1093.9694472656695 J/(kg K), k=1.35632393586728.
The shaft residuals are near zero; this does not resolve the property-method choice.
The RAW adapter faithfully stores the actual states; it does not fix or refresh
properties. This discrepancy must be resolved explicitly in the next method audit,
not by silently changing the frozen reference. The inherited-property behaviour is
an additional limitation of this saved dataset and supersedes any broader claim
that every turbine's properties are refreshed at its inlet.

No compressor maps, stage inventory, turbine stage closure, cooling, materials or
independent combustor enthalpy validation is supplied here. M2 and production M6
remain open. Next independent task: audit fuel/alpha/cp/k energy assumptions against
the sources, reporting discrepancies without silently changing the model.

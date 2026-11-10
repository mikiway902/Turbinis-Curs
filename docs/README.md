# Documentation index

| Directory | Contents | Status |
|---|---|---|
| `engineering/` | planned equation register, constraints, method and validation-case authority | absent from this checkout; not usable as authority |
| `comparisons/` | comparisons against the external Mathcad and `TVD_pi6` models | historical analysis |
| `history/` | dated review notes kept as a record | historical |

## Authority vs DRAFT

The historical documentation names `engineering/CONSTRAINTS.md`,
`engineering/EQUATIONS.md` and `engineering/README.md` as authority documents,
but none exists in this checkout. Do not infer their contents or treat their
absence as physical approval. Current implementation/provenance and open issues
are recorded in `PROJECT_GUIDE.md`, `PROJECT_STATE.md`, `PROJECT_CHECKLIST.md`
and `comparisons/Источники_и_методики.md`.

The formerly referenced `engineering/CONSTRAINTS_COMPARISON.md` is also absent.
No proposal or comparison note overrides an implemented model or supplies a
missing material/resource qualification.

`comparisons/` and `history/` documents are analysis snapshots. Their numbers
describe the code and sources at their date and are not current limits.

## Current project records

- `PROJECT_CHECKLIST.md` — live progress status and evidence (source of truth).
- `PROJECT_STATE.md` — short operative state and next executable task.
- `PROJECT_GUIDE.md` — how the actual checkout works.
- `AUTONOMOUS_EXECUTION.md` — execution-gate and phase map.
- `comparisons/Источники_и_методики.md` — source provenance, equation/exit audit,
  and unresolved method conflicts.
- `NUMERICAL_FAILURES.md` — typed solver-failure behavior.
- `LEGACY_RAW_LIMITATIONS.md` — why the legacy optimizer result is not RAW.
- `BACKENDS.md` and `backend_validation.json` — bounded CPU/JAX/GPU kernel evidence.
- `STAGE_REPORT_2026-11-11.md` — checkpoint conclusions; not a final project report.
- `agent_reports/2026-11-11_deepseek_agent_conclusions.md` — preserved implementation-agent
  reports; evidence to verify, not independent approval.

## Related

- `../archive/README.md` — archived diagnostics and the legacy comparison
  workbench.
- `../reference/README.md` — baseline and published-case policy.
- `../Optimizer/results/README.md` — generated result families.

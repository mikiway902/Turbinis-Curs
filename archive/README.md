# Archive

Archived artifacts are retained, not accepted: nothing here is design
evidence, a validated baseline, or a current default. Do not build new work on
archived numbers without promoting the underlying method into `turbine/` and
`docs/engineering/` first.

## Map

| Path | Contents |
|---|---|
| `diagnostics/clearance_003_min015/` | one fixed `book_approx` meanline run with `delta = max(0.003*l, 0.15 mm)`, marked `EXPERIMENTAL`; the report itself records where the variant fails project constraints |
| `comparisons/legacy_workbench/` | the former root `_compare/` workbench: scripts, dumps, the Mathcad export, the `TVD_pi6` notebook copy, and its chart |

## Archival rules

- Byte-identical moves are preferred over rewrites; do not "clean up" archived
  data dumps (`nb_src.txt`, `repo.txt`, `tvd_pi6_src.txt`) or checkpoints.
- Archived notebooks and dumps keep the path text they had when produced.
- Scripts that still run were made runnable relative to their own location
  rather than to the old root path (`comparisons/legacy_workbench/tvd6.py`).
- An archived file may be cited by `docs/comparisons/`; citing it does not
  raise its status.

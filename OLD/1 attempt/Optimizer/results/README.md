# Optimizer results

Generated output of `Optimizer/` scripts. A report here is a record of a run,
not a design verdict: most numbers are `book_approx` diagnostics, and the
coupled radial work is `EXPERIMENTAL`. Nothing in this directory overrides
`docs/engineering/CONSTRAINTS.md` or `docs/engineering/EQUATIONS.md`.

## Families

| Family | Examples | Meaning |
|---|---|---|
| Reports | `REPORT_STAGE6.md`, `REPORT_STAGE6B.md`, `REPORT_STAGE6B5.md`, `REPORT_full10.md`, `REPORT_full10_vs_qn1.md`, `REPORT_qn1.md`, `REPORT_SCAN2D.md` … `REPORT_SCAN5D.md` | narrative output written by the analysis and scan scripts |
| Summaries and scans | `extended_*_q{0.995,1}_{full10,qn1}.csv`, `scan1d.csv`, `scan2d_*.csv`, `scan3d_A.csv`, `scan4d.csv`, `scan5d.csv`, `SUMMARY_*.csv`, `scan_l14_frontier.csv`, `cycle_coeff_scan.csv`, `tg_scan.csv`, `archive.csv` | sweep inputs and top-level tables |
| Logs and batches | `*.log`, `analysis.log`, `BATCH_DONE`, `SCAN3D_DONE`, `SCAN4D_DONE`, `SCAN5D_DONE`, `*_bundle.zip` | run traces and packaging |
| Coupled radial evidence | `stage6b45b_report.md`, `stage6b45b_results.json`, `stage6b45b_runs.csv`, `stage6b45b_stage_summary.csv`, `stage6b45b_entropy_audit.{json,md}` | system B diagnostics; all stages remain `INCONCLUSIVE`, so this is not accepted closure |
| Staged diagnostics | `stage6_*.csv`, `stage6b_*.csv`, `stage6b5_radial.csv`, `stage6a_compare.csv`, `stage6_rk*.png`, `heatmap_D1_lD.png`, `parametric_1d.png`, `pareto.png`, `optimization_tables.xlsx`, `scaling_solution.json`, `initial_conditions_mode*` | per-stage, per-radius, and sensitivity work from the stage 6 campaign |
| Status notes | `HANDOFF.md`, `PILOT_SUMMARY.md`, `CODE_REVIEW_PACKET.md`, `PERF_AUDIT.md` | handoff and review material, not evidence |

Some sweep outputs are gitignored (`scan2d.csv`, `scan3d_A.csv`, `scan3d_B.csv`,
`scan4d.csv`, `scan5d.csv`, `extended_*_full10.csv`, `extended_*_qn1.csv`,
`*_DONE`, `*.zip`). Local presence does not make them tracked or authoritative.

## Path preservation

`Optimizer/` scripts resolve this directory from `Path(__file__)` and read
exact file names, for example `extended_{scenario}_q0.995_full10.csv`,
`scan2d.csv`, `stage6b_radial.csv`, and the `REPORT_*` outputs. Do not rename,
move, or restructure these files without updating the generating scripts in
the same change.

The clearance diagnostic is not kept here; it lives in
`archive/diagnostics/clearance_003_min015/`.

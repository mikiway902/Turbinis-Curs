# Published validation cases

No case in this directory is approved for automated validation or regression. Cases are
examples only; discrepancies remain unresolved. Full provenance, overlap and method-scope
evidence is in `../../docs/comparisons/Источники_и_методики.md`.

## Candidate inventory

| Case | Path | Cells | SHA-256 | Git evidence | Bounded role |
|---|---|---|---|---|---|
| Anton | `reference/published_cases/1Anton-Mathcad_ST_cooling_Jupyter (3).ipynb` | 575 | `5510d3e44c263b9a77a2895d094e3e0d0799c9408ea7ce504a65dbcf3d406ec3` | added in `c6223b9` | preferred detailed/citable case |
| Anton | `Mathcad_ST_cooling_Jupyter.ipynb` | 60 | `d525e7a289be01f82174c52515726ae191684299c6094fe483e1baedf7ce5166` | added in `05d6031` | compact computational companion/reconstruction |
| Sergey | `TVD_pi6_design_steps_1_3.ipynb` | 48 | `fe8da90402d05cebefefcb40ef6367b14fbf884baa2fb4ef1991294821fce4c7` | added in `1263ac2` | current working candidate for cycle + design steps 1–3 |
| Sergey | `archive/comparisons/legacy_workbench/TVD_pi6.ipynb` | 83 | `13a0165e5ec35927e05263de63c645f2c8c2a0877126f79dcd5ed78d9c8ff652` | `c6223b9` R100 from `_compare/TVD_pi6.ipynb`; originally added in `1263ac2` | historical/comparison variant |

## Scope separation

The active project model is 110 kW, single-shaft, three turbine stages, `pi_k = 11`,
`T_g = 1250 K`. Neither external case is that model:

- **Anton**: Baikal-like 750 kW after reduction, two-shaft free turbine, two-stage power
  turbine with cooling, `pi_k = 10`, total temperature 1440 K at the HPT nozzle throat
  after cooling-air return.
- **Sergey**: separate two-shaft 150 hp = 110.325 kW, `pi_k = 6`, `T_g = 1250 K`, one
  compressor-turbine stage and one free-power stage. 110.325 kW must not be conflated with
  the project's 110 kW boundary.

## Anton variants

Both Anton notebooks describe the same two-stage free turbine. The 575-cell `(3)` file is
the preferred canonical detailed/citable case: it expands the cycle and stage calculations
into per-pass cells, carries anchor IDs and stage/profile appendices, and is the better
source for line-level citation. The 60-cell root notebook is a compact computational
companion/reconstruction, not a proven newer or authoritative replacement. The two share only 11
identical nonempty source cells (18.3 % of the root's 60 unique nonempty cells), so the
reference is not a simple copy of the root.

## Sergey variants

Both Sergey notebooks describe the same two-shaft, `pi_k = 6` case. The root 48-cell file
is the current working candidate for the cycle + design-steps-1–3 path. The 83-cell archive
file is a historical/comparison variant with expanded turbine-design sections 15–22; it
originated as `_compare/TVD_pi6.ipynb` and was renamed (R100) in `c6223b9`. The two share
38 identical nonempty source cells (79.2 % of the root's 48 unique nonempty cells). The
archive is not a straightforward newer revision of the root solely because it has more
cells or a later commit timestamp.

## Provenance boundary

Working-copy mtimes are checkout artifacts; git commit dates reflect repository
organization. No external author timestamps or original provenance records are present in
the checkout. Therefore this directory does not claim a "latest author version" for either
case, and does not promote either case to a regression baseline.

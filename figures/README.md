# figures/ — figure-to-paper mapping

This directory is a scratch gallery: `experiments/make_paper_figs.py` and the
per-experiment scripts write every diagnostic figure here (currently 58 files,
one .pdf/.png pair per figure). It is gitignored except for this note; only
`paper/figs/` copies of the figures actually embedded in the paper are tracked.

Used by the paper:

| File | Paper location | Source artifact |
|---|---|---|
| `mar18_rich_sparsification.pdf` | Fig. 2 (results, sparsification/AUSE) | `experiments/out/mar18_rich_sparsification.json` |
| `change_nov18_mar19.pdf` | Fig. 3 (results, change detection) | `experiments/out/change_nov18_mar19_cells.npz` (regenerate via `experiments/change_detection.py`) |
| `stpls3d_khat_vs_complexity.pdf` | Fig. 4 (discussion, K-hat capacity) | `experiments/out/stpls3d_full_grid.json` |
| `geoacc_mar18.pdf` | Geometric-accuracy figure (copied to `paper/figs/`; companion to the accuracy/completeness table) | `experiments/out/geoacc_mar18.json` |

Everything else (per-epoch khat maps, alternate cell sizes, re-flight variants,
smoke tests) is exploratory output retained for auditability; none of it is
referenced by the paper.

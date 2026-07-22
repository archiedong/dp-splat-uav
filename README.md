# DP-Splat-UAV

Tiled Dirichlet-process Gaussian splatting for UAV/aerial mapping with calibrated
uncertainty. Application companion to DP-Splat (arXiv:2607.10912); method code imports
`dp-splat` and never modifies it.

Paper: *Tiled Dirichlet-Process Gaussian Splatting for UAV Mapping with Calibrated
Uncertainty* (submitted to IEEE TGRS). Every number and figure in the paper is
reproducible from the commands below; the aggregate result JSONs that ground each
table are tracked in `experiments/out/` (large binary artifacts are not — see
Data and license terms).

## Layout
```
src/dp_splat_uav/
  tiling.py      # spatial tile splitter (halo overlap), ownership weights, partition-of-unity gating
  weighted.py    # per-point-weighted suff-stats (halo down-weighting) wrapped around dp_splat
  merge.py       # NIW log-normalizer match score, Hungarian+EPPF matching, prior-subtracted merge
  stitch.py      # global predictive: partition-of-unity stitching of per-tile Student-t mixtures
  baselines.py   # seam baselines: mean-in-tile crop, Voronoi ownership
  io_aerial.py   # LAZ/LAS/PLY loaders (H3D, GauU-Scene, STPLS3D), EPSG-aware
tests/           # independent NumPy oracles for every new update equation
experiments/     # pipeline runner, evaluations, and figure generation
```

## Installation

Requires Python >= 3.12 (all paper numbers: Python 3.14.6, CPU JAX on Apple
silicon) and a checkout of the `dp-splat` method package (arXiv:2607.10912) as a
sibling directory, pinned at its arXiv-v1 tag/commit — this repo imports
dp-splat but never modifies it. There is no pip dependency on dp-splat yet:
the root `conftest.py` prepends `../dp-splat/src` (and `./src`) to `sys.path`
and enables float64, and the experiment scripts do the same, so the sibling
checkout is picked up automatically as long as the two clones sit side by side.
Release pinning moves to the package index once dp-splat is published there.

```bash
git clone <dp-splat repo> dp-splat        # checkout the arxiv-v1 tag
git clone <this repo> dp-splat-uav
python3 -m venv ~/.venvs/dp-splat && source ~/.venvs/dp-splat/bin/activate
pip install -r dp-splat-uav/requirements-lock.txt   # exact versions behind the paper's numbers
cd dp-splat-uav && python -m pytest -q   # 112 oracle/unit tests, ~1 min, no data needed
```

`pyproject.toml` carries the same pins for the direct dependencies;
`requirements-lock.txt` is the full closure (including pytest) of the
environment that produced every artifact in `experiments/out/`.

## Data (not in repo)

Expected at `~/dp-splat-data/`:

| Dataset | Access | License |
|---|---|---|
| `h3d/` Hessigheim 3D (3 UAV-LiDAR epochs) | registration (ifp, Uni Stuttgart) | CC BY-NC-SA 4.0 |
| `gauuscene/SMBU/` GauU-Scene V2 | emailed license request | terms per provider |
| `mill19/` Mill 19 Rubble (images + poses) | direct download (Mega-NeRF) | no formal license file; research use |
| `stpls3d/` STPLS3D synthetic | direct download | CC BY-NC-SA 3.0 |

Derived point clouds and fitted models are never redistributed (code, scripts, and
aggregate statistics only), in keeping with the NC terms.

## Reproducing the paper

All commands use the venv python. Each experiment writes a record JSON capturing
its full configuration; per-script docstrings document every flag.

**Tables I–II (seam comparison; calibration + sparsification), per epoch** — fit,
held-out eval, sparsification (March 2018 shown; Nov. 2018 / March 2019 use their
`--input` and `--subsample 116000000` / `107000000`):
```bash
python experiments/run_tiled_fit.py --input ~/dp-splat-data/h3d/Epoch_March2018/LiDAR/Mar18_train.laz \
    --name mar18_rich --subsample 53500000 --alpha 100 --truncation 256 --seed 0
python experiments/eval_heldout.py --record experiments/out/mar18_rich_record.json \
    --model experiments/out/mar18_rich_model.npz \
    --input ~/dp-splat-data/h3d/Epoch_March2018/LiDAR/Mar18_train.laz --holdout 200000 --seed 11
python experiments/sparsification.py --record experiments/out/mar18_rich_record.json \
    --model experiments/out/mar18_rich_model.npz \
    --input ~/dp-splat-data/h3d/Epoch_March2018/LiDAR/Mar18_train.laz
```
Reference arm: `--alpha 1 --truncation 64`. Merge ablation: add `--no-merge`.

**Second-site scale run (SMBU, 152M points):**
```bash
python experiments/run_tiled_fit.py --input ~/dp-splat-data/gauuscene/SMBU/cloud_merged.las \
    --name smbu_rich --subsample 152000000 --alpha 100 --truncation 256 --seed 0
python experiments/eval_heldout.py --record experiments/out/smbu_rich_record.json \
    --model experiments/out/smbu_rich_model.npz \
    --input ~/dp-splat-data/gauuscene/SMBU/cloud_merged.las --holdout 200000 --seed 11
```

**Geometric accuracy vs the H3D reference mesh (accuracy / completeness table):**
```bash
python experiments/geometric_accuracy.py --record experiments/out/mar18_rich_record.json \
    --model experiments/out/mar18_rich_model.npz \
    --input ~/dp-splat-data/h3d/Epoch_March2018/LiDAR/Mar18_train.laz --name geoacc_mar18
```
(exact point-to-mesh distances both directions, NLPD-tercile stratification,
crop/Voronoi baseline arms; expects the per-tile mesh under
`~/dp-splat-data/h3d/Epoch_March2018/Mesh/per-tile/`).

**Table III (change detection):** `python experiments/change_detection.py` on the
Nov. 2018 / March 2019 production records (see its docstring; 0.5 m / 2 m cells via
`--cell`).

**Table IV (re-flight planning):**
```bash
python experiments/reflight.py --holdback-mode block --rank-mode nlpd  --seeds 5 --target-tile-points 250000 --name reflight_rubble_block
python experiments/reflight.py --holdback-mode block --rank-mode yield --seeds 8 --target-tile-points 120000 --name reflight_rubble_yield12
```
(both include targeted / oracle / mass-only / random arms).

**K-hat capacity study (Fig. 4, Sec. VI-D):** `python experiments/stpls3d_khat_grid.py`
(29 scenes x 3 alphas x 2 seeds; `--prior-scale` for the salvage sweep).

**Camera-ready figures (Figs. 2–4):** `python experiments/make_paper_figs.py`
regenerates all three paper figures from `experiments/out/` artifacts. Note:
Fig. 3 reads `change_nov18_mar19_cells.npz`, which is not tracked (binary
hygiene) — re-run `change_detection.py` first if it is absent; Figs. 2 and 4
build from tracked JSONs alone. Figure-to-file mapping: `figures/README.md`.

**Expected runtimes** (from the `seconds_total` fields of the tracked records;
Apple-silicon Mac, 24–36 GB RAM, CPU JAX):

| Step | Points | Runtime |
|---|---|---|
| `run_tiled_fit` Mar 2018 production | 53.5M | ~705 s |
| `run_tiled_fit` Mar 2019 / Nov 2018 production | 107M / 116M | ~1,340 / ~1,880 s |
| `run_tiled_fit` SMBU production | 152M | ~2,350 s |
| `eval_heldout` (200k complement) | — | 30–50 s |
| `sparsification` | — | ~17 s |
| `change_detection` (0.5 m cells) | — | ~21 s |
| `geometric_accuracy` | — | ~54 s |
| `reflight` (per scenario, all arms + seeds) | — | ~2,200–2,350 s |
| `stpls3d_khat_grid` (full 174-fit grid) | — | ~2,790 s |

**Paper:** `cd paper && tectonic main.tex` (IEEEtran).

## Governing documents

`../paper2-uav/BRIEF.md` (adopted 2026-07-15), `../paper2-uav/LIT_REVIEW.md`;
equation/change log in `../dp-splat/QUESTIONS.md`.

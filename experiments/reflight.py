"""Uncertainty-targeted re-flight experiment on the Mill 19 Rubble anchor cloud.

The closing demonstration: does the model's own predictive uncertainty tell a
UAV operator where to re-fly? Protocol (Phase B-3 spec):

  1. SURVEY SPLIT. Images sorted by name (6-digit capture ids, temporal flight
     order). The held-back re-flight pool is every ``--holdback-every``-th image
     of that ordering (sorted_names[k-1::k], 20% at k=5) — spread across the
     whole site by the temporal interleaving, mimicking available re-fly
     capacity rather than a spatially contiguous gap. A 3D point belongs to the
     initial cloud iff >= 2 of its (distinct) track images are survey images;
     points seen by >= 2 held-back images and < 2 survey images form the
     re-flight pool; the ambiguous remainder (< 2 in both) is discarded. All
     three counts are reported.
  2. INITIAL FIT. Tiled production fit (alpha, T from the CLI; production
     alpha=100, T=256) on the initial cloud, reusing run_tiled_fit's per-tile
     machinery (fit_tile_cavi / fit_tile_svi, shared global NIW priors). Per
     tile we record the mean stitched NLPD of the tile's own (core) points and
     an MC estimate of the tile's spatial predictive entropy; both rankings are
     reported.
  3. INTERVENTION. ARM A (targeted): top ``--top-frac`` of tiles by mean
     per-point NLPD receive the re-flight-pool points falling inside their
     halo-dilated rectangles; those tiles are refit and the scene re-stitched.
     ARM B (control): random tile subsets of the same size, ``--seeds``
     replicates. ARM C (oracle): tiles ranked by true held-out error
     contribution — mean stitched NLPD of the *pool* points per tile under the
     initial model (the pool is exactly the data the initial fit never saw).
  4. SCORING on the fixed evaluation set — an ``--eval-frac`` subset of the
     pool reserved BEFORE any granting (so it is identical across arms, no arm
     or random seed ever refits on an eval point, and selected tiles retain
     eval mass; defining the eval set as the complement of the grants instead
     would leave zero eval points inside any arm's own selected tiles and make
     the within-selection delta vacuous), under each arm's model: (i) delta
     mean NLPD overall and
     within the arm's selected tiles; (ii) coverage shift at 68/90/95%
     (eval_heldout's responsibility-weighted central-interval construction);
     (iii) completeness proxy: fraction of eval points whose nearest-neighbor
     distance to a ``--model-samples`` draw from the arm's spatial predictive
     is below ``--nn-thresh`` meters.
  5. OUTPUTS. experiments/out/<name>.json and figures/<name>.{png,pdf}
     (variance map with selected tiles outlined + per-arm gain bars).

Spec resolutions taken here (documented for the paper):
  * AXES. The COLMAP world frame has axis 1 vertical (Mega-NeRF convention).
    All coordinates are reordered to (x, z, y) = (right, forward, down) before
    tiling, so the first two axes — the ones tiling.make_grid splits and
    stitch.gate_weights gates on — are the horizontal ground plane. The
    permutation is applied once at load; every downstream array lives in the
    reordered frame.
  * CROP. The robust bounding box (0.5–99.5 percentile per axis) is computed
    from the initial-survey cloud (the data available at fit time) and the same
    box crops the pool, keeping every arm and the evaluation inside the fitted
    domain. Drop counts are reported.
  * SKIPPED TILES rank as maximally uncertain (NLPD = +inf) in the targeted
    ordering: a tile with no fittable data is exactly a coverage gap. Tiles
    with no pool points contribute no held-out error and rank last (-inf) in
    the oracle ordering.
  * REFITS keep the original grid geometry, per-tile alpha_t, and shared
    priors; a refit tile's data are its original (halo-weighted) points plus
    the granted pool points, the latter weighted by their full-grid ownership
    counts. Non-selected tiles keep their initial states; the arm's model is
    the re-stitched partition-of-unity predictive.
  * The cross-tile merge stage is not run: every score here is computed from
    the stitched per-tile predictives, which the merge does not alter for the
    single-source majority of components, and per-arm models stay directly
    comparable tile by tile.

Smoke run (wiring check):
  ~/.venvs/dp-splat/bin/python experiments/reflight.py \
      --subsample 400000 --seeds 1 --name reflight_rubble_smoke
Full run:
  ~/.venvs/dp-splat/bin/python experiments/reflight.py
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "dp-splat" / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from dp_splat import cavi
from dp_splat import niw as dniw
from dp_splat import priors as dpriors
from dp_splat.predictive import student_logpdf

from dp_splat_uav import io_aerial, merge, stitch, tiling

import eval_heldout as ev
import run_tiled_fit as rtf

LEVELS = (0.68, 0.90, 0.95)


def parse_args():
    p = argparse.ArgumentParser(description="Uncertainty-targeted re-flight experiment")
    p.add_argument("--colmap-model",
                   default=str(Path.home() / "dp-splat-data/mill19/colmap_work/sparse_opencv"),
                   help="COLMAP text model directory (points3D.txt, images.txt)")
    p.add_argument("--name", default="reflight_rubble")
    p.add_argument("--out-dir", default=str(REPO / "experiments" / "out"))
    p.add_argument("--fig-dir", default=str(REPO / "figures"))
    p.add_argument("--holdback-mode", choices=["interleave", "block"],
                   default="interleave",
                   help="interleave: every k-th image (redundant views of covered "
                        "ground); block: the last fraction of the flight (a "
                        "contiguous unfinished strip -- the operational re-flight "
                        "scenario)")
    p.add_argument("--holdback-frac", type=float, default=0.2,
                   help="block mode: trailing fraction of the flight held back")
    p.add_argument("--holdback-every", type=int, default=5,
                   help="every k-th image (temporal order) is held back for re-flight")
    p.add_argument("--min-track", type=int, default=2,
                   help="distinct images required to claim a point for a split")
    p.add_argument("--rank-mode", choices=["nlpd", "yield"], default="nlpd",
                   help="tile acquisition score: raw mean NLPD, or expected yield "
                        "= mean NLPD x grantable pool mass in the tile (the "
                        "planner's expected-utility form)")
    p.add_argument("--top-frac", type=float, default=0.25,
                   help="fraction of tiles selected for re-flight")
    p.add_argument("--seeds", type=int, default=5,
                   help="random-control replicates (ARM B)")
    p.add_argument("--eval-frac", type=float, default=0.25,
                   help="fraction of the pool reserved as the fixed eval set "
                        "before any granting")
    p.add_argument("--subsample", type=int, default=None,
                   help="subsample the INITIAL cloud (smoke runs); the pool is kept whole")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--truncation", type=int, default=256)
    p.add_argument("--target-tile-points", type=float, default=250_000)
    p.add_argument("--halo", type=float, default=5.0)
    p.add_argument("--max-iters", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--n-min", type=float, default=1.0)
    p.add_argument("--min-tile-points", type=int, default=1000)
    p.add_argument("--svi-threshold", type=float, default=2e6)
    p.add_argument("--svi-batch", type=int, default=65536)
    p.add_argument("--svi-epochs", type=float, default=3.0)
    p.add_argument("--svi-eval-every", type=int, default=25)
    p.add_argument("--svi-eval-points", type=int, default=100_000)
    p.add_argument("--chunk", type=int, default=131_072)
    p.add_argument("--entropy-samples", type=int, default=20_000,
                   help="MC draws per tile for the spatial predictive entropy")
    p.add_argument("--model-samples", type=int, default=200_000,
                   help="model draw size for the completeness proxy")
    p.add_argument("--nn-thresh", type=float, default=0.5,
                   help="completeness NN distance threshold, meters")
    return p.parse_args()


# ---------------------------------------------------------------------------
# COLMAP loading and the survey split
# ---------------------------------------------------------------------------


def load_image_names(model_dir):
    """image_id -> name from images.txt (header lines only; POINTS2D lines skipped)."""
    id2name = {}
    with open(model_dir / "images.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10 and parts[-1].endswith(".jpg"):
                id2name[int(parts[0])] = parts[-1]
    return id2name


def split_points(model_dir, id2name, every, min_track, mode="interleave", frac=0.2):
    """One pass over points3D.txt: coordinates, colors, and the survey split.

    Held-back images are sorted_names[every-1::every] (temporal order — names
    are zero-padded capture ids). Track image ids are deduplicated per point
    before counting. Labels: 0 = initial cloud (>= min_track survey images),
    1 = re-flight pool (< min_track survey, >= min_track held-back),
    2 = discarded remainder.
    """
    names = sorted(id2name.values())
    if mode == "block":
        n_hold = max(1, int(round(len(names) * frac)))
        holdback = set(names[-n_hold:])
    else:
        holdback = set(names[every - 1::every])
    is_hold = np.zeros(max(id2name) + 1, dtype=bool)
    for iid, nm in id2name.items():
        is_hold[iid] = nm in holdback

    xyz, rgb, label = [], [], []
    with open(model_dir / "points3D.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            ids = set(map(int, parts[8::2]))
            nh = int(sum(is_hold[i] for i in ids))
            ns = len(ids) - nh
            xyz.append((float(parts[1]), float(parts[2]), float(parts[3])))
            rgb.append((int(parts[4]), int(parts[5]), int(parts[6])))
            label.append(0 if ns >= min_track else (1 if nh >= min_track else 2))
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64) / 255.0
    label = np.asarray(label, dtype=np.int8)
    split_info = dict(
        n_images=len(id2name),
        n_survey_images=len(names) - len(holdback),
        n_holdback_images=len(holdback),
        n_points=int(label.size),
        n_initial=int((label == 0).sum()),
        n_pool=int((label == 1).sum()),
        n_discarded=int((label == 2).sum()),
    )
    return xyz, rgb, label, split_info


# ---------------------------------------------------------------------------
# Predictive construction (mirrors eval_heldout.tile_submixtures on live
# states: per-tile prune + one model-wide global weight vector for stitching)
# ---------------------------------------------------------------------------


def _niw_index(q, idx):
    return dniw.NIW(m=q.m[idx], kappa=q.kappa[idx], Psi=q.Psi[idx], nu=q.nu[idx])


def pruned_tile_state(state, counts, n_min, alpha_t, sp):
    """Tile state restricted to surviving components, weights rebuilt from counts.

    Returns (state, cfg, kept_counts); a skipped/empty tile yields a
    zero-component state (stitch.tile_predictive handles it).

    The within-tile stick weights built here are an intermediate only: every
    stitched-scoring consumer replaces them with the model-wide global weights
    (globalize_weights below) -- a per-tile-normalized sub-mixture would make
    the stitched density integrate to ~#tiles instead of 1.
    """
    kept = np.flatnonzero(np.asarray(counts) > n_min) if state is not None else np.zeros(0, int)
    if kept.size == 0:
        empty = jnp.zeros((0,))
        niw0 = _niw_index(sp, jnp.zeros(0, int))
        st = cavi.State(niw0, niw0, niw0, niw0,
                        dpriors.StickBreakingPosterior(empty, empty), None)
        return st, cavi.Config(weight_prior="dp", T=0, alpha=float(alpha_t)), np.zeros(0)
    a, b, order = merge.rebuild_weights(jnp.asarray(np.asarray(counts)[kept]), float(alpha_t))
    kept = kept[np.asarray(order)]
    spatial = _niw_index(state.spatial, jnp.asarray(kept))
    color = _niw_index(state.color, jnp.asarray(kept))
    st = cavi.State(spatial, color, spatial, color,
                    dpriors.StickBreakingPosterior(a, b), None)
    cfg = cavi.Config(weight_prior="dp", T=int(kept.size), alpha=float(alpha_t))
    return st, cfg, np.asarray(counts)[kept]


def globalize_weights(preds, kept_counts, alpha):
    """Replace each tile predictive's log_pi with its components' GLOBAL weights.

    One weight vector is rebuilt from ALL tiles' pooled kept counts
    (merge.rebuild_weights with alpha_global -- the same construction as
    run_tiled_fit step 6), and each tile's sub-mixture carries its own
    components' global weights un-renormalized within the tile. The stitched
    density sum_t g_t(s) p_t then integrates to 1 (eval_heldout's weight
    conventions; tests/test_eval_mass_oracle.py): no component is shared
    across tiles here (the merge stage is not run), so the global weights
    partition exactly across the tile sub-mixtures.

    Must be re-applied per arm on the arm's own post-refit kept counts: a
    refit changes the tile's counts and therefore every tile's global weight.
    """
    counts = np.concatenate([np.asarray(kc, dtype=np.float64) for kc in kept_counts])
    if counts.size == 0:
        return list(preds)
    a, b, order = merge.rebuild_weights(jnp.asarray(counts), float(alpha))
    w = np.empty(counts.size)
    w[np.asarray(order)] = np.asarray(dpriors.dp_expected_pi(a, b))
    out, off = [], 0
    for p, kc in zip(preds, kept_counts):
        k = np.asarray(kc).size
        out.append(p._replace(
            log_pi=jnp.log(jnp.asarray(w[off:off + k]) + 1e-300)))
        off += k
    return out


def stitched_nlpd(preds, grid, xs, xc, chunk):
    out = []
    for start in range(0, xs.shape[0], chunk):
        sl = slice(start, start + chunk)
        out.append(-np.asarray(stitch.stitched_logpdf(preds, grid, xs[sl], xc[sl])))
    return np.concatenate(out) if out else np.zeros(0)


def core_tile_of(grid, s):
    """Morton-order tile index of the core rectangle containing each point.

    Cores partition the bounding box on linspace edges; out-of-box points clip
    to the nearest edge tile.
    """
    nx, ny = grid.shape
    xe = np.linspace(grid.bbox_lo[0], grid.bbox_hi[0], nx + 1)
    ye = np.linspace(grid.bbox_lo[1], grid.bbox_hi[1], ny + 1)
    ix = np.clip(np.searchsorted(xe, s[:, 0], side="right") - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(ye, s[:, 1], side="right") - 1, 0, ny - 1)
    pos = np.empty((nx, ny), dtype=np.int64)
    pos[grid.index[:, 0], grid.index[:, 1]] = np.arange(grid.index.shape[0])
    return pos[ix, iy]


def _sample_student_bank(rng, loc, chol, dof, mass, m):
    """Draw m points from a mass-weighted mixture of multivariate Student-t."""
    comp = rng.choice(loc.shape[0], size=m, p=mass / mass.sum())
    z = rng.standard_normal((m, loc.shape[1]))
    g = rng.chisquare(dof[comp])
    x = loc[comp] + np.einsum("nij,nj->ni", chol[comp], z) * np.sqrt(dof[comp] / g)[:, None]
    return x


def tile_spatial_entropy(pred, kept_counts, rng, m):
    """MC differential entropy of a tile's spatial predictive mixture.

    Components are drawn by soft-count mass, points from the component
    Student-t, and the entropy is the mean of -log p over the draws (weights
    renormalized over the tile's surviving components).
    """
    if kept_counts.size == 0:
        return float("inf")
    loc = np.asarray(pred.spatial.loc)
    scale = np.asarray(pred.spatial.scale)
    dof = np.asarray(pred.spatial.dof)
    x = _sample_student_bank(rng, loc, np.linalg.cholesky(scale), dof, kept_counts, m)
    log_pi = np.asarray(pred.log_pi)
    log_pi = log_pi - logsumexp(jnp.asarray(log_pi))
    ll = jnp.asarray(log_pi)[None, :] + student_logpdf(pred.spatial, jnp.asarray(x))
    return float(-np.mean(np.asarray(logsumexp(ll, axis=1))))


def sample_model(preds, kept_counts_list, rng, m):
    """Draw m spatial points from the pooled (count-weighted) per-tile predictives."""
    loc = np.concatenate([np.asarray(p.spatial.loc) for p in preds])
    scale = np.concatenate([np.asarray(p.spatial.scale) for p in preds])
    dof = np.concatenate([np.asarray(p.spatial.dof) for p in preds])
    mass = np.concatenate(kept_counts_list)
    return _sample_student_bank(rng, loc, np.linalg.cholesky(scale), dof, mass, m)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def fit_one_tile(t, xs_t, xc_t, w_t, cfg_t, sp, cp, args):
    t0 = time.perf_counter()
    if xs_t.shape[0] > args.svi_threshold:
        state, counts, history, meta = rtf.fit_tile_svi(
            args.seed + t, xs_t, xc_t, w_t, cfg_t, sp, cp, args)
    else:
        state, counts, history, meta = rtf.fit_tile_cavi(
            args.seed + t, xs_t, xc_t, w_t, cfg_t, sp, cp)
    seconds = time.perf_counter() - t0
    khat = int((counts > args.n_min).sum())
    record = dict(tile=t, n_points=int(xs_t.shape[0]), khat=khat,
                  path=meta["path"], seconds=seconds,
                  elbo_tail=[float(v) for v in history[-3:]])
    return state, counts, record


def initial_fit(s, c, grid, asn, cfg, sp, cp, args):
    """Per-tile weighted fits of the initial cloud (run_tiled_fit steps 4-5)."""
    xs_all, xc_all = jnp.asarray(s), jnp.asarray(c)
    tiles = []
    for t in range(grid.core_lo.shape[0]):
        idx = np.asarray(asn.indices[t])
        if idx.size < args.min_tile_points:
            tiles.append(dict(state=None, counts=np.zeros(cfg.T),
                              record=dict(tile=t, n_points=int(idx.size), khat=0,
                                          path="skipped", seconds=0.0)))
            print(f"[tile {t}] n={idx.size} SKIPPED", flush=True)
            continue
        cfg_t = dataclasses.replace(cfg, alpha=float(asn.alpha[t]))
        state, counts, record = fit_one_tile(
            t, xs_all[jnp.asarray(idx)], xc_all[jnp.asarray(idx)],
            jnp.asarray(asn.weight[idx]), cfg_t, sp, cp, args)
        tiles.append(dict(state=state, counts=counts, record=record))
        print(f"[tile {t}] n={idx.size} path={record['path']} K-hat={record['khat']} "
              f"{record['seconds']:.1f}s", flush=True)
    return tiles


def build_predictives(tiles, asn, args, sp):
    preds, cfgs, kept_counts = [], [], []
    for t, tile in enumerate(tiles):
        st, cfg_t, kc = pruned_tile_state(
            tile["state"], tile["counts"], args.n_min, asn.alpha[t], sp)
        preds.append(stitch.tile_predictive(st, cfg_t))
        cfgs.append(cfg_t)
        kept_counts.append(kc)
    return preds, cfgs, kept_counts


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def rank_tiles_desc(values):
    """Tile indices by descending value; ties and infinities resolve by tile index."""
    return np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")


def grant_pool(grid, selected, s_pool, grantable=None):
    """Pool-point indices inside the union of the selected tiles' dilated
    rectangles, restricted to the grantable mask (points not reserved for
    evaluation). ``grantable=None`` means every point may be granted."""
    lo = grid.core_lo[selected] - grid.halo
    hi = grid.core_hi[selected] + grid.halo
    inside = np.zeros(s_pool.shape[0], dtype=bool)
    for j in range(lo.shape[0]):
        inside |= np.all((s_pool[:, :2] >= lo[j]) & (s_pool[:, :2] <= hi[j]), axis=1)
    if grantable is not None:
        inside &= grantable
    return np.flatnonzero(inside)


def refit_arm(selected, granted, base, ctx, cache):
    """Refit the selected tiles with their granted pool points; re-stitch.

    Returns (preds, kept_counts, refit_records). Non-selected tiles keep the
    initial model's predictives. Refits are cached across arms on
    (tile, granted-point set): the targeted, random, and oracle arms often
    select overlapping tiles.
    """
    args, grid, asn, cfg, sp, cp = ctx
    s, c = base["s"], base["c"]
    s_pool, c_pool, pool_w = base["s_pool"], base["c_pool"], base["pool_weight"]
    preds = list(base["preds"])
    kept_counts = list(base["kept_counts"])
    records = []
    for t in selected:
        lo, hi = grid.core_lo[t] - grid.halo, grid.core_hi[t] + grid.halo
        g_in = granted[np.all((s_pool[granted][:, :2] >= lo)
                              & (s_pool[granted][:, :2] <= hi), axis=1)]
        key = (int(t), g_in.tobytes())
        if key not in cache:
            idx = np.asarray(asn.indices[t])
            xs_t = jnp.asarray(np.concatenate([s[idx], s_pool[g_in]]))
            xc_t = jnp.asarray(np.concatenate([c[idx], c_pool[g_in]]))
            w_t = jnp.asarray(np.concatenate([asn.weight[idx], pool_w[g_in]]))
            if xs_t.shape[0] < args.min_tile_points:
                st, cfg_t, kc = pruned_tile_state(None, None, args.n_min, asn.alpha[t], sp)
                cache[key] = (stitch.tile_predictive(st, cfg_t), kc,
                              dict(tile=int(t), n_points=int(xs_t.shape[0]),
                                   n_granted=int(g_in.size), khat=0, path="skipped",
                                   seconds=0.0))
            else:
                cfg_t = dataclasses.replace(cfg, alpha=float(asn.alpha[t]))
                state, counts, rec = fit_one_tile(int(t), xs_t, xc_t, w_t,
                                                  cfg_t, sp, cp, args)
                st, pcfg, kc = pruned_tile_state(state, counts, args.n_min,
                                                 asn.alpha[t], sp)
                rec["n_granted"] = int(g_in.size)
                cache[key] = (stitch.tile_predictive(st, pcfg), kc, rec)
            print(f"[refit tile {t}] n={cache[key][2]['n_points']} "
                  f"(+{g_in.size} granted) K-hat={cache[key][2]['khat']} "
                  f"{cache[key][2]['seconds']:.1f}s", flush=True)
        preds[t], kept_counts[t] = cache[key][0], cache[key][1]
        records.append(cache[key][2])
    # Arm-specific global weights: rebuilt from THIS arm's post-refit pooled
    # counts (refit tiles' new counts + non-selected tiles' initial counts).
    return globalize_weights(preds, kept_counts, args.alpha), kept_counts, records


def score_arm(arm_preds, arm_kept, selected, base, ctx):
    """Deltas vs the initial model on the fixed evaluation set."""
    args, grid = ctx[0], ctx[1]
    xs_e, xc_e = base["xs_eval"], base["xc_eval"]
    nlpd = stitched_nlpd(arm_preds, grid, xs_e, xc_e, args.chunk)
    in_sel = np.isin(base["eval_tile"], selected)
    cov = ev.coverage_per_point(arm_preds, grid, xs_e, xc_e, args.chunk)
    coverage = {f"{q:.2f}": float(cov[q].mean()) for q in LEVELS}
    rng = np.random.default_rng(args.seed + 12345)
    draw = sample_model(arm_preds, arm_kept, rng, args.model_samples)
    d, _ = cKDTree(draw).query(np.asarray(xs_e), k=1)
    b = base["baseline"]
    return dict(
        selected_tiles=[int(t) for t in selected],
        n_eval_in_selected=int(in_sel.sum()),
        mean_nlpd=float(nlpd.mean()),
        delta_mean_nlpd=float(b["mean_nlpd"] - nlpd.mean()),
        mean_nlpd_selected=float(nlpd[in_sel].mean()) if in_sel.any() else None,
        delta_mean_nlpd_selected=(
            float(b["nlpd"][in_sel].mean() - nlpd[in_sel].mean()) if in_sel.any() else None),
        coverage=coverage,
        coverage_shift={q: coverage[q] - b["coverage"][q] for q in coverage},
        completeness=float((d < args.nn_thresh).mean()),
        delta_completeness=float((d < args.nn_thresh).mean() - b["completeness"]),
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(grid, tile_nlpd, arms, name, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    finite = np.isfinite(tile_nlpd)
    vmin, vmax = (tile_nlpd[finite].min(), tile_nlpd[finite].max()) if finite.any() else (0, 1)
    cmap = plt.get_cmap("viridis")
    for t in range(grid.core_lo.shape[0]):
        w, h = grid.core_hi[t] - grid.core_lo[t]
        v = (tile_nlpd[t] - vmin) / max(vmax - vmin, 1e-12) if finite[t] else 1.0
        ax0.add_patch(Rectangle(grid.core_lo[t], w, h, facecolor=cmap(v),
                                edgecolor="0.4", lw=0.5))
        ax0.text(*(grid.core_lo[t] + 0.5 * (grid.core_hi[t] - grid.core_lo[t])),
                 str(t), ha="center", va="center", fontsize=7, color="w")
    for t in arms["targeted"]["selected_tiles"]:
        w, h = grid.core_hi[t] - grid.core_lo[t]
        ax0.add_patch(Rectangle(grid.core_lo[t], w, h, facecolor="none",
                                edgecolor="red", lw=2.0))
    for t in arms["oracle"]["selected_tiles"]:
        w, h = grid.core_hi[t] - grid.core_lo[t]
        ax0.add_patch(Rectangle(grid.core_lo[t], w, h, facecolor="none",
                                edgecolor="k", lw=1.5, linestyle="--"))
    ax0.set_xlim(grid.bbox_lo[0], grid.bbox_hi[0])
    ax0.set_ylim(grid.bbox_lo[1], grid.bbox_hi[1])
    ax0.set_aspect("equal")
    ax0.set_title("Per-tile mean NLPD (initial fit)\n"
                  "red: targeted selection; dashed: oracle")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("z (m)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=ax0, label="mean NLPD (nats)")

    labels = ["targeted", "random", "oracle"]
    overall = [arms["targeted"]["delta_mean_nlpd"],
               arms["random"]["aggregate"]["delta_mean_nlpd_mean"],
               arms["oracle"]["delta_mean_nlpd"]]
    insel = [arms["targeted"]["delta_mean_nlpd_selected"] or 0.0,
             arms["random"]["aggregate"]["delta_mean_nlpd_selected_mean"] or 0.0,
             arms["oracle"]["delta_mean_nlpd_selected"] or 0.0]
    x = np.arange(3)
    err = [0.0, arms["random"]["aggregate"]["delta_mean_nlpd_std"], 0.0]
    ax1.bar(x - 0.2, overall, 0.35, yerr=err, capsize=3, label="overall")
    err_s = [0.0, arms["random"]["aggregate"]["delta_mean_nlpd_selected_std"] or 0.0, 0.0]
    ax1.bar(x + 0.2, insel, 0.35, yerr=err_s, capsize=3, label="within selected tiles")
    ax1.axhline(0.0, color="0.3", lw=0.8)
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("delta mean NLPD on fixed eval set (nats, higher = better)")
    ax1.set_title("Re-flight gain by arm")
    ax1.legend()
    fig.tight_layout()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    t_start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.colmap_model).expanduser()

    # 1. Load, split, reorder axes so the ground plane is (axis0, axis2).
    t0 = time.perf_counter()
    id2name = load_image_names(model_dir)
    xyz, rgb, label, split_info = split_points(
        model_dir, id2name, args.holdback_every, args.min_track,
        mode=args.holdback_mode, frac=args.holdback_frac)
    xyz = xyz[:, [0, 2, 1]]  # (right, forward, down): tiling splits axes 0-1
    s_init, c_init = xyz[label == 0], rgb[label == 0]
    s_pool, c_pool = xyz[label == 1], rgb[label == 1]
    print(f"[split] images {split_info['n_survey_images']} survey / "
          f"{split_info['n_holdback_images']} held-back; points "
          f"{split_info['n_initial']} initial / {split_info['n_pool']} pool / "
          f"{split_info['n_discarded']} discarded ({time.perf_counter() - t0:.1f}s)",
          flush=True)

    # Robust crop from the initial cloud, applied to both sets.
    lo = np.percentile(s_init, 0.5, axis=0)
    hi = np.percentile(s_init, 99.5, axis=0)
    keep_i = np.all((s_init >= lo) & (s_init <= hi), axis=1)
    keep_p = np.all((s_pool >= lo) & (s_pool <= hi), axis=1)
    crop_info = dict(robust_lo=lo.tolist(), robust_hi=hi.tolist(),
                     dropped_initial=int((~keep_i).sum()),
                     dropped_pool=int((~keep_p).sum()))
    s_init, c_init = s_init[keep_i], c_init[keep_i]
    s_pool, c_pool = s_pool[keep_p], c_pool[keep_p]
    if args.subsample is not None and args.subsample < s_init.shape[0]:
        take = np.random.default_rng(args.seed).choice(
            s_init.shape[0], size=args.subsample, replace=False)
        s_init, c_init = s_init[take], c_init[take]
    s_init, origin = io_aerial.center_scene(s_init)
    s_pool = s_pool - origin
    print(f"[crop] initial {s_init.shape[0]} pool {s_pool.shape[0]} "
          f"(dropped {crop_info['dropped_initial']}/{crop_info['dropped_pool']})",
          flush=True)

    # 2. Grid, priors, initial tiled fit.
    grid = tiling.make_grid(s_init, halo=args.halo, target_points=args.target_tile_points)
    asn = tiling.assign_points(grid, s_init, alpha_global=args.alpha)
    n_tiles = grid.core_lo.shape[0]
    print(f"[tiles] grid {grid.shape}, {n_tiles} tiles, halo {grid.halo} m; "
          f"per-tile points {[len(ix) for ix in asn.indices]}", flush=True)
    cfg = cavi.Config(weight_prior="dp", T=args.truncation, alpha=args.alpha,
                      max_iters=args.max_iters, tol=args.tol)
    sp = cavi.default_niw_prior(jnp.asarray(s_init), cfg.T, cfg.kappa0, cfg.nu0_offset)
    cp = cavi.default_niw_prior(jnp.asarray(c_init), cfg.T, cfg.kappa0, cfg.nu0_offset)
    tiles = initial_fit(s_init, c_init, grid, asn, cfg, sp, cp, args)
    preds0, _, kept0 = build_predictives(tiles, asn, args, sp)
    # Global weights across all tiles (alpha_global): the stitched predictive
    # must carry each component's model-wide weight, not per-tile-normalized
    # sticks, to be a normalized density (see globalize_weights).
    preds0 = globalize_weights(preds0, kept0, args.alpha)

    # Pool points outside every dilated tile cannot be gated or granted; drop them.
    pool_lo, pool_hi = grid.core_lo - grid.halo, grid.core_hi + grid.halo
    pool_own = np.zeros(s_pool.shape[0], dtype=np.int64)
    for t in range(n_tiles):
        pool_own += np.all((s_pool[:, :2] >= pool_lo[t])
                           & (s_pool[:, :2] <= pool_hi[t]), axis=1)
    in_grid = pool_own > 0
    n_pool_outside = int((~in_grid).sum())
    s_pool, c_pool, pool_own = s_pool[in_grid], c_pool[in_grid], pool_own[in_grid]
    pool_weight = 1.0 / pool_own

    # 3. Per-tile uncertainty: mean stitched NLPD of own points + MC spatial entropy.
    t0 = time.perf_counter()
    nlpd_init = stitched_nlpd(preds0, grid, jnp.asarray(s_init), jnp.asarray(c_init),
                              args.chunk)
    tile_of_init = core_tile_of(grid, s_init)
    rng = np.random.default_rng(args.seed + 777)
    tile_nlpd = np.full(n_tiles, np.inf)   # skipped tiles: maximally uncertain
    tile_entropy = np.full(n_tiles, np.inf)
    for t in range(n_tiles):
        m = tile_of_init == t
        if m.any() and tiles[t]["state"] is not None:
            tile_nlpd[t] = float(nlpd_init[m].mean())
            tile_entropy[t] = tile_spatial_entropy(preds0[t], kept0[t], rng,
                                                   args.entropy_samples)
    rank_nlpd = rank_tiles_desc(tile_nlpd)
    rank_entropy = rank_tiles_desc(tile_entropy)
    finite = np.isfinite(tile_nlpd) & np.isfinite(tile_entropy)
    rho = (float(spearmanr(tile_nlpd[finite], tile_entropy[finite]).correlation)
           if finite.sum() > 2 else None)
    print(f"[rank] NLPD order {rank_nlpd.tolist()}; entropy order "
          f"{rank_entropy.tolist()}; spearman {rho} "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    # Oracle ranking: mean held-out (pool) NLPD per tile under the initial model.
    nlpd_pool = stitched_nlpd(preds0, grid, jnp.asarray(s_pool), jnp.asarray(c_pool),
                              args.chunk)
    tile_of_pool = core_tile_of(grid, s_pool)
    tile_oracle = np.full(n_tiles, -np.inf)  # no held-out mass -> no contribution
    for t in range(n_tiles):
        m = tile_of_pool == t
        if m.any():
            tile_oracle[t] = float(nlpd_pool[m].mean())

    # 4. Reserved eval split, then selections and grants. The eval set is
    #    drawn from the pool BEFORE any granting (module docstring, step 4):
    #    grants come exclusively from the remaining pool, so no arm ever
    #    refits on an eval point while selected tiles keep eval mass.
    n_eval = max(1, int(round(args.eval_frac * s_pool.shape[0])))
    eval_idx = np.sort(np.random.default_rng(args.seed + 424242).choice(
        s_pool.shape[0], size=n_eval, replace=False))
    grantable = np.ones(s_pool.shape[0], dtype=bool)
    grantable[eval_idx] = False
    k_sel = max(1, int(round(args.top_frac * n_tiles)))
    # Grantable pool mass per tile: used by the yield score and, alone, by the
    # mass-only control arm (grant by recoverable mass, ignoring the model).
    pool_mass = np.array([
        grant_pool(grid, [t], s_pool, grantable).size for t in range(n_tiles)
    ], dtype=float)
    if args.rank_mode == "yield":
        # Expected utility of re-flying tile t: how wrong the model is there
        # times how much new data a re-flight can actually recover there.
        score = np.where(np.isfinite(tile_nlpd), tile_nlpd, 0.0) * pool_mass
        sel_targeted = rank_tiles_desc(score)[:k_sel]
    else:
        sel_targeted = rank_nlpd[:k_sel]
    sel_oracle = rank_tiles_desc(tile_oracle)[:k_sel]
    sel_mass = rank_tiles_desc(pool_mass)[:k_sel]
    sel_random = [np.random.default_rng(args.seed + 100 + i).choice(
        n_tiles, size=k_sel, replace=False) for i in range(args.seeds)]
    grants = {"targeted": grant_pool(grid, sel_targeted, s_pool, grantable),
              "oracle": grant_pool(grid, sel_oracle, s_pool, grantable),
              "mass_only": grant_pool(grid, sel_mass, s_pool, grantable)}
    for i, sel in enumerate(sel_random):
        grants[f"random_{i}"] = grant_pool(grid, sel, s_pool, grantable)
    granted_any = np.zeros(s_pool.shape[0], dtype=bool)
    for g in grants.values():
        granted_any[g] = True
    if bool(granted_any[eval_idx].any()):
        raise RuntimeError("leakage: a reserved eval point was granted to an arm")
    print(f"[eval] pool {s_pool.shape[0]} (outside grid {n_pool_outside}); "
          f"reserved eval set {eval_idx.size}; granted-to-any "
          f"{int(granted_any.sum())} of {int(grantable.sum())} grantable",
          flush=True)

    base = dict(
        s=s_init, c=c_init, s_pool=s_pool, c_pool=c_pool, pool_weight=pool_weight,
        preds=preds0, kept_counts=kept0,
        xs_eval=jnp.asarray(s_pool[eval_idx]), xc_eval=jnp.asarray(c_pool[eval_idx]),
        eval_tile=tile_of_pool[eval_idx],
    )
    ctx = (args, grid, asn, cfg, sp, cp)

    # Baseline scores of the initial model on the eval set.
    b_nlpd = stitched_nlpd(preds0, grid, base["xs_eval"], base["xc_eval"], args.chunk)
    b_cov = ev.coverage_per_point(preds0, grid, base["xs_eval"], base["xc_eval"],
                                  args.chunk)
    rng_b = np.random.default_rng(args.seed + 12345)
    draw0 = sample_model(preds0, kept0, rng_b, args.model_samples)
    d0, _ = cKDTree(draw0).query(np.asarray(base["xs_eval"]), k=1)
    base["baseline"] = dict(
        nlpd=b_nlpd, mean_nlpd=float(b_nlpd.mean()),
        coverage={f"{q:.2f}": float(b_cov[q].mean()) for q in LEVELS},
        completeness=float((d0 < args.nn_thresh).mean()),
    )
    print(f"[baseline] eval mean NLPD {b_nlpd.mean():.4f}; completeness "
          f"{base['baseline']['completeness']:.4f}", flush=True)

    # 5. Refit + score each arm (refits cached across arms on identical grants).
    cache = {}
    arms = {}
    for arm_name, sel in (("targeted", sel_targeted), ("oracle", sel_oracle),
                          ("mass_only", sel_mass)):
        preds_a, kept_a, recs = refit_arm(sel, grants[arm_name], base, ctx, cache)
        arms[arm_name] = dict(n_granted=int(grants[arm_name].size), refits=recs,
                              **score_arm(preds_a, kept_a, sel, base, ctx))
        print(f"[{arm_name}] delta mean NLPD {arms[arm_name]['delta_mean_nlpd']:+.4f} "
              f"(selected: {arms[arm_name]['delta_mean_nlpd_selected']})", flush=True)
    per_seed = []
    for i, sel in enumerate(sel_random):
        preds_a, kept_a, recs = refit_arm(sel, grants[f"random_{i}"], base, ctx, cache)
        r = dict(seed=i, n_granted=int(grants[f"random_{i}"].size), refits=recs,
                 **score_arm(preds_a, kept_a, sel, base, ctx))
        per_seed.append(r)
        print(f"[random {i}] delta mean NLPD {r['delta_mean_nlpd']:+.4f}", flush=True)
    d_all = np.array([r["delta_mean_nlpd"] for r in per_seed])
    d_sel = np.array([r["delta_mean_nlpd_selected"] for r in per_seed
                      if r["delta_mean_nlpd_selected"] is not None])
    arms["random"] = dict(per_seed=per_seed, aggregate=dict(
        delta_mean_nlpd_mean=float(d_all.mean()),
        delta_mean_nlpd_std=float(d_all.std(ddof=1)) if d_all.size > 1 else 0.0,
        delta_mean_nlpd_selected_mean=float(d_sel.mean()) if d_sel.size else None,
        delta_mean_nlpd_selected_std=(
            float(d_sel.std(ddof=1)) if d_sel.size > 1 else 0.0),
    ))

    # 6. Outputs.
    make_figure(grid, tile_nlpd, arms, args.name, args.fig_dir)
    out = dict(
        config=vars(args),
        split=split_info,
        crop=crop_info,
        origin=origin.tolist(),
        axis_note="coordinates reordered (x, z, y): ground plane first, vertical last",
        grid=dict(shape=list(grid.shape), n_tiles=n_tiles, halo=grid.halo,
                  core_lo=grid.core_lo.tolist(), core_hi=grid.core_hi.tolist()),
        n_pool_outside_grid=n_pool_outside,
        initial_tiles=[t["record"] for t in tiles],
        tile_mean_nlpd=[None if not np.isfinite(v) else float(v) for v in tile_nlpd],
        tile_entropy_mc=[None if not np.isfinite(v) else float(v) for v in tile_entropy],
        tile_oracle_nlpd=[None if not np.isfinite(v) else float(v) for v in tile_oracle],
        tile_pool_mass=pool_mass.tolist(),
        ranking=dict(by_nlpd=rank_nlpd.tolist(), by_entropy=rank_entropy.tolist(),
                     spearman_nlpd_vs_entropy=rho, k_selected=k_sel),
        n_eval=int(eval_idx.size),
        baseline=dict(mean_nlpd=base["baseline"]["mean_nlpd"],
                      coverage=base["baseline"]["coverage"],
                      completeness=base["baseline"]["completeness"]),
        arms=arms,
        headline=dict(
            targeted_minus_random=float(
                arms["targeted"]["delta_mean_nlpd"]
                - arms["random"]["aggregate"]["delta_mean_nlpd_mean"]),
            targeted_minus_oracle=float(
                arms["targeted"]["delta_mean_nlpd"]
                - arms["oracle"]["delta_mean_nlpd"]),
        ),
        seconds_total=time.perf_counter() - t_start,
        versions=dict(jax=jax.__version__, numpy=np.__version__),
    )
    out_path = out_dir / f"{args.name}.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[done] {out_path.name}; figures/{args.name}.png|pdf; "
          f"total {out['seconds_total']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

"""Held-out predictive evaluation of a merged tiled DP-Splat model.

Consumes the outputs of experiments/run_tiled_fit.py (step 7) and scores the model
on a fresh subsample of the source cloud, comparing the partition-of-unity stitched
predictive against the two hard-selection seam baselines on identical per-tile
sub-mixtures:

  (a) stitched:  log p(x) = log sum_t g_t(s) p_t(x) with the normalized bump gates
                 of stitch.py over the per-tile Student-t mixtures;
  (b) crop:      baselines.crop_mean_in_tile -- keep a tile's component iff its
                 posterior spatial mean lies in the tile core (VastGaussian-style
                 boundary crop), pool, renormalize;
  (c) voronoi:   baselines.voronoi_ownership -- keep a component iff the owning
                 tile's core center is the nearest core center to its mean
                 (Kerbl-style chunk ownership).

Per-tile sub-mixtures from provenance: the merged model stores, for every global
component k, its source nodes (src_tile, src_comp) in the slice
src_offsets[k]:src_offsets[k+1]. Tile t's sub-mixture contains every global
component with a source node in t. Its NIW parameters are the merged global ones
(the exact pooled posterior; per-tile pre-merge NIWs are not stored -- for the
single-source majority the two coincide).

Weight conventions -- the two consumers need DIFFERENT weight vectors:

  * STITCHED path (a): each tile's sub-mixture carries its components' GLOBAL
    mixture weights model["expected_pi"][k] (the pooled count-rebuilt
    stick-breaking weights of run_tiled_fit step 6), deliberately NOT
    renormalized within the tile. A merged component straddling several tiles
    appears in each with the same global weight, so the stitched density
    sum_t g_t(s) p_t(x) carries it with mass pi_k sum_t g_t = pi_k and the
    stitched predictive integrates to 1 over the scene (the construction
    oracle-verified in tests/test_stitch_oracle.py test (vi); the mass itself
    in tests/test_eval_mass_oracle.py). Renormalizing per tile instead makes
    every p_t a probability density and inflates the stitched mass to ~#tiles.
  * BASELINE paths (b), (c): tile-level weights rebuilt from the tile's own
    pre-merge weighted counts tile_counts[t, src_comp] with the tile's alpha_t
    (merge.rebuild_weights) -- baselines.TileFit expects within-tile E[pi],
    which the baselines scale by tile mass, pool, and renormalize; their
    output is normalized by construction.

Splits: every metric is reported on ALL held-out points and on SEAM points -- points
lying in two or more halo-dilated tile rectangles, i.e. within one halo width of an
interior core boundary, where the three combination rules actually differ.

Coverage construction (d): for each held-out point x = (s, c), responsibilities over
all (tile, component) pairs are computed under the stitched predictive,

    r_{tk}(x) proportional to g_t(s) E[pi_k^t] St(s; m_k, S_k, eta_k) St(c; ...),

normalized over pairs. Each pair's spatial predictive is multivariate Student-t, so
its marginal along axis d is univariate Student-t with location m_{k,d}, scale
sqrt(S_{k,dd}), and the same dof eta_k. The point is covered by pair (t, k) along
axis d at nominal level q iff |s_d - m_{k,d}| <= t_{eta_k}^{-1}((1+q)/2) *
sqrt(S_{k,dd}) (the central q-interval), and the per-point coverage indicator is the
responsibility-weighted average sum_{tk} r_{tk}(x) 1[covered]. Empirical coverage is
the mean over points, reported per axis and axis-averaged at nominal 68/90/95%.
This is a component-conditional calibration check (does a point land in the central
q-interval of the component that claims it), not the mixture-marginal CDF; the
responsibilities condition on the full point, including its color.

Hold-out protocol: the held-out set is a fresh uniform subsample of the input cloud
drawn with --seed, which must differ from the fit's subsample seed. When the fit
consumed ALL points (config.subsample null, e.g. the full_val run), no disjoint
hold-out exists and the evaluation is an in-sample sanity check; the output record
is labeled accordingly (in_sample = true).

Output: experiments/out/<name>_eval.json (next to the record) plus a stdout table.

Run:
  ~/.venvs/dp-splat/bin/python experiments/eval_heldout.py \
      --record experiments/out/full_val_record.json \
      --model experiments/out/full_val_model.npz \
      --input ~/dp-splat-data/h3d/Epoch_March2018/LiDAR/Mar18_val.laz \
      --holdout 100000
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "dp-splat" / "src"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from scipy.stats import t as student_t

from dp_splat import cavi
from dp_splat import niw as dniw
from dp_splat import priors as dpriors
from dp_splat.predictive import StudentT, niw_predictive, student_logpdf
from dp_splat_uav import baselines, io_aerial, merge, stitch, tiling

LEVELS = (0.68, 0.90, 0.95)
AXES = ("x", "y", "z")
_Z_BIG = 1e12  # vertical pseudo-extent: tiling never splits z, so all tiles share it


def parse_args():
    p = argparse.ArgumentParser(description="Held-out evaluation of a merged tiled fit")
    p.add_argument("--record", required=True, help="<name>_record.json from run_tiled_fit")
    p.add_argument("--model", required=True, help="<name>_model.npz from run_tiled_fit")
    p.add_argument("--input", required=True, help="LAS/LAZ or PLY source point cloud")
    p.add_argument("--holdout", type=int, default=200_000,
                   help="held-out subsample size")
    p.add_argument("--seed", type=int, default=1,
                   help="hold-out subsample seed; must differ from the fit's seed")
    p.add_argument("--chunk", type=int, default=131_072,
                   help="evaluation chunk size (points)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------


def load_grid(model):
    """Rebuild the TileGrid from the geometry arrays saved in the model npz."""
    return tiling.TileGrid(
        core_lo=model["core_lo"], core_hi=model["core_hi"], halo=float(model["halo"]),
        bbox_lo=model["bbox_lo"], bbox_hi=model["bbox_hi"],
        shape=tuple(int(v) for v in model["grid_shape"]),
        index=model["tile_index"], morton=model["tile_morton"],
    )


def _niw_take(model, prefix, idx):
    return dniw.NIW(
        m=jnp.asarray(model[f"{prefix}_m"][idx]),
        kappa=jnp.asarray(model[f"{prefix}_kappa"][idx]),
        Psi=jnp.asarray(model[f"{prefix}_Psi"][idx]),
        nu=jnp.asarray(model[f"{prefix}_nu"][idx]),
    )


def _global_expected_pi(model, alpha_t):
    """The merged model's global mixture weights, aligned with component order.

    run_tiled_fit step 6 saves them as model["expected_pi"]
    (dp_expected_pi of rebuild_weights over the pooled counts with
    alpha_global). For minimal model archives without the field they are
    rebuilt here from the stored pooled counts -- the identical construction;
    alpha_global is read from the npz, falling back to sum_t alpha_t (the
    tiling.assign_points invariant) for archives predating that field too.
    """
    if "expected_pi" in model:
        return np.asarray(model["expected_pi"])
    alpha_global = (float(model["alpha_global"]) if "alpha_global" in model
                    else float(np.sum(np.asarray(alpha_t))))
    a, b, order = merge.rebuild_weights(jnp.asarray(model["counts"]), alpha_global)
    pi = np.empty(np.asarray(model["counts"]).shape[0])
    pi[np.asarray(order)] = np.asarray(dpriors.dp_expected_pi(a, b))
    return pi


def tile_submixtures(model, alpha_t):
    """Per-tile sub-mixtures of the merged model, from src_tile provenance.

    Returns per tile (module docstring, "Weight conventions"):

    * pred:  stitch.TilePredictive for the STITCHED path -- the tile's
             components with their GLOBAL weights model["expected_pi"][gk],
             un-renormalized within the tile, so the partition-of-unity
             stitched density integrates to 1 (a straddling merged component
             contributes pi_k sum_t g_t = pi_k). Built directly rather than
             via the State sticks, which can only encode a normalized vector.
    * state/cfg: a cavi.State carrying the same components with tile-level
             weights rebuilt from the tile's own pre-merge counts (alpha_t) --
             the input the BASELINE paths expect (baselines.TileFit). Must not
             be fed to the stitched predictive: per-tile renormalization
             inflates the stitched mass to ~#tiles.
    * counts (tile's pre-merge weighted soft counts) and global_ids, both in
             the same size-biased order as state's sticks and pred.
    """
    src_offsets = model["src_offsets"]
    src_tile = model["src_tile"]
    src_comp = model["src_comp"]
    tile_counts = model["tile_counts"]
    expected_pi = _global_expected_pi(model, alpha_t)  # global weights, component order
    comp_of_node = np.repeat(np.arange(len(src_offsets) - 1), np.diff(src_offsets))

    tiles = []
    for t in range(tile_counts.shape[0]):
        node = src_tile == t
        gk = comp_of_node[node]  # global component ids present in tile t
        counts = tile_counts[t, src_comp[node]]  # tile's own pre-merge soft counts
        if gk.size == 0:
            # Skipped/unfittable tile: an empty sub-mixture, not a one-stick
            # placeholder -- rebuild_weights on empty counts would fabricate a
            # phantom weight with no matching component.
            empty = jnp.zeros((0,))
            niw_s, niw_c = _niw_take(model, "spatial", gk), _niw_take(model, "color", gk)
            pred = stitch.TilePredictive(log_pi=empty,
                                         spatial=niw_predictive(niw_s),
                                         color=niw_predictive(niw_c))
            state = cavi.State(niw_s, niw_c, niw_s, niw_c,
                               dpriors.StickBreakingPosterior(empty, empty), None)
            cfg = cavi.Config(weight_prior="dp", T=0, alpha=float(alpha_t[t]))
            tiles.append(dict(pred=pred, state=state, cfg=cfg,
                              counts=counts, global_ids=gk))
            continue
        a, b, order = merge.rebuild_weights(jnp.asarray(counts), float(alpha_t[t]))
        order = np.asarray(order)
        gk, counts = gk[order], counts[order]
        spatial = _niw_take(model, "spatial", gk)
        color = _niw_take(model, "color", gk)
        pred = stitch.TilePredictive(
            log_pi=jnp.log(jnp.asarray(expected_pi[gk]) + 1e-300),
            spatial=niw_predictive(spatial),
            color=niw_predictive(color),
        )
        # Prior slots are placeholders: nothing on the predictive/baseline path
        # reads them (expected_pi uses weights only; niw_predictive the posterior).
        state = cavi.State(spatial, color, spatial, color,
                           dpriors.StickBreakingPosterior(a, b), None)
        cfg = cavi.Config(weight_prior="dp", T=len(gk), alpha=float(alpha_t[t]))
        tiles.append(dict(pred=pred, state=state, cfg=cfg,
                          counts=counts, global_ids=gk))
    return tiles


def baseline_boxes(grid):
    """Tile cores as 3D baseline boxes; the shared vertical pseudo-extent keeps the
    half-open crop test trivially true in z and the Voronoi centers coplanar."""
    return [
        baselines.Tile(
            lo=jnp.asarray(np.append(grid.core_lo[t], -_Z_BIG)),
            hi=jnp.asarray(np.append(grid.core_hi[t], _Z_BIG)),
        )
        for t in range(grid.core_lo.shape[0])
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def mixture_logpdf(mix, xs, xc, chunk):
    """Per-point log density under a pooled GlobalMixture (baselines convention)."""
    logw = jnp.log(mix.weights + 1e-300)
    st_s = niw_predictive(mix.spatial)
    st_c = niw_predictive(mix.color)
    out = []
    for start in range(0, xs.shape[0], chunk):
        sl = slice(start, start + chunk)
        ll = logw[None, :] + student_logpdf(st_s, xs[sl]) + student_logpdf(st_c, xc[sl])
        out.append(np.asarray(logsumexp(ll, axis=1)))
    return np.concatenate(out)


def stitched_logpdf(preds, grid, xs, xc, chunk):
    out = []
    for start in range(0, xs.shape[0], chunk):
        sl = slice(start, start + chunk)
        out.append(np.asarray(stitch.stitched_logpdf(preds, grid, xs[sl], xc[sl])))
    return np.concatenate(out)


def coverage_per_point(preds, grid, xs, xc, chunk):
    """Responsibility-weighted central-interval coverage (module docstring, item d).

    Returns {level: (N, 3)} per-point per-axis coverage indicators in [0, 1].
    """
    # Flatten all (tile, component) pairs into one Student-t bank.
    pair_tile = np.concatenate(
        [np.full(p.log_pi.shape[0], t) for t, p in enumerate(preds)]
    )
    log_pi = jnp.concatenate([p.log_pi for p in preds])
    st_s = StudentT(*(jnp.concatenate([getattr(p.spatial, f) for p in preds])
                      for f in StudentT._fields))
    st_c = StudentT(*(jnp.concatenate([getattr(p.color, f) for p in preds])
                      for f in StudentT._fields))

    sd = np.sqrt(np.asarray(jnp.diagonal(st_s.scale, axis1=-2, axis2=-1)))  # (P, 3)
    dof = np.asarray(st_s.dof)
    half_width = {
        q: student_t.ppf(0.5 * (1.0 + q), dof)[:, None] * sd for q in LEVELS
    }  # (P, 3) central-interval half-widths per pair and axis
    loc = np.asarray(st_s.loc)

    out = {q: [] for q in LEVELS}
    for start in range(0, xs.shape[0], chunk):
        sl = slice(start, start + chunk)
        g = stitch.gate_weights(grid, xs[sl])[:, pair_tile]  # (n, P)
        log_g = jnp.where(g > 0.0, jnp.log(jnp.where(g > 0.0, g, 1.0)), -jnp.inf)
        lp = (log_g + log_pi[None, :]
              + student_logpdf(st_s, xs[sl]) + student_logpdf(st_c, xc[sl]))
        r = np.asarray(jnp.exp(lp - logsumexp(lp, axis=1, keepdims=True)))  # (n, P)
        dev = np.abs(np.asarray(xs[sl])[:, None, :] - loc[None, :, :])  # (n, P, 3)
        for q in LEVELS:
            covered = dev <= half_width[q][None, :, :]
            out[q].append(np.einsum("np,npd->nd", r, covered))
    return {q: np.concatenate(v) for q, v in out.items()}


def split_stats(values, seam):
    """Mean/median of per-point values on all points and on the seam subset."""
    return {
        "all": {"mean": float(values.mean()), "median": float(np.median(values))},
        "seam": {"mean": float(values[seam].mean()), "median": float(np.median(values[seam]))},
    }


def main():
    args = parse_args()
    t_start = time.perf_counter()
    record = json.loads(Path(args.record).read_text())
    model = np.load(args.model)
    name = record["name"]

    # Hold-out protocol bookkeeping (module docstring).
    fit_subsample = record["config"].get("subsample")
    fit_seed = record["config"].get("seed", 0)
    in_sample = fit_subsample is None
    if in_sample:
        note = ("fit consumed all points (config.subsample null): hold-out overlaps "
                "the training data -- in-sample sanity check, not a held-out score")
    else:
        note = ("hold-out drawn from the exact complement of the recorded fit "
                "subsample (indices reconstructed from config.subsample and "
                "config.seed) -- a true held-out score")

    # Hold-out points, centered with the origin the model was fit in.
    path = Path(args.input).expanduser()
    if path.suffix.lower() in (".las", ".laz"):
        loader = io_aerial.load_laz
    elif path.suffix.lower() == ".ply":
        loader = io_aerial.load_ply
    else:
        raise ValueError(f"unsupported point-cloud format {path.suffix!r}")
    if in_sample:
        s, c, _ = loader(path, subsample=args.holdout, rng=args.seed)
    else:
        # A fresh uniform draw would overlap the fit set in proportion to the
        # fit fraction; the honest hold-out is the complement of the exact
        # subsample the fit consumed.
        s, c, _ = loader(path, subsample=None, rng=None)
        fit_idx = io_aerial._subsample_indices(s.shape[0], fit_subsample, fit_seed)
        keep = np.ones(s.shape[0], dtype=bool)
        keep[fit_idx] = False
        pool = np.flatnonzero(keep)
        take = np.random.default_rng(args.seed).choice(
            pool, size=min(args.holdout, pool.size), replace=False)
        s, c = s[take], c[take]
    s = s - model["origin"]
    xs, xc = jnp.asarray(s), jnp.asarray(c)

    grid = load_grid(model)
    # Seam points: inside >= 2 halo-dilated tiles, i.e. within one halo width of an
    # interior core boundary (where stitching and the hard baselines disagree).
    lo, hi = grid.core_lo - grid.halo, grid.core_hi + grid.halo
    ownership = sum(
        np.all((s[:, :2] >= lo[t]) & (s[:, :2] <= hi[t]), axis=1)
        for t in range(lo.shape[0])
    )
    seam = ownership >= 2

    # Per-tile sub-mixtures -> stitched predictives + baseline fits. The
    # stitched path consumes the GLOBAL-weight predictives (t["pred"]); the
    # baselines consume the tile-normalized states (weight conventions above).
    tiles = tile_submixtures(model, record["alpha_t"])
    preds = [t["pred"] for t in tiles]
    fits = [baselines.TileFit(t["state"], t["cfg"]) for t in tiles]
    masses = [float(t["counts"].sum()) for t in tiles]
    mix_crop = baselines.crop_mean_in_tile(fits, baseline_boxes(grid), masses)
    mix_vor = baselines.voronoi_ownership(fits, baseline_boxes(grid), masses)

    ll = {
        "stitched": stitched_logpdf(preds, grid, xs, xc, args.chunk),
        "crop": mixture_logpdf(mix_crop, xs, xc, args.chunk),
        "voronoi": mixture_logpdf(mix_vor, xs, xc, args.chunk),
    }
    cov = coverage_per_point(preds, grid, xs, xc, args.chunk)

    results = {m: split_stats(v, seam) for m, v in ll.items()}
    coverage = {
        f"{q:.2f}": {
            "all": {"axis_mean": float(cov[q].mean()),
                    **{a: float(cov[q][:, d].mean()) for d, a in enumerate(AXES)}},
            "seam": {"axis_mean": float(cov[q][seam].mean()),
                     **{a: float(cov[q][seam][:, d].mean()) for d, a in enumerate(AXES)}},
        }
        for q in LEVELS
    }

    runtime = time.perf_counter() - t_start
    out = dict(
        name=name,
        config=vars(args),
        in_sample=in_sample,
        holdout_note=note,
        n_holdout=int(s.shape[0]),
        n_seam=int(seam.sum()),
        seam_fraction=float(seam.mean()),
        tile_k=[len(t["global_ids"]) for t in tiles],
        k_global=int(model["spatial_m"].shape[0]),
        k_crop=int(mix_crop.weights.shape[0]),
        k_voronoi=int(mix_vor.weights.shape[0]),
        log_predictive=results,
        coverage=coverage,
        seconds_total=runtime,
        versions=dict(jax=jax.__version__, numpy=np.__version__),
    )
    out_path = Path(args.record).parent / f"{name}_eval.json"
    out_path.write_text(json.dumps(out, indent=1))

    tag = "IN-SAMPLE sanity check" if in_sample else "held-out"
    print(f"[eval] {name}: {s.shape[0]} points ({tag}), seam {int(seam.sum())} "
          f"({100 * seam.mean():.1f}%); K global {out['k_global']}, "
          f"crop {out['k_crop']}, voronoi {out['k_voronoi']}")
    print(f"{'method':<10} {'mean(all)':>10} {'med(all)':>10} {'mean(seam)':>11} {'med(seam)':>10}")
    for m in ("stitched", "crop", "voronoi"):
        r = results[m]
        print(f"{m:<10} {r['all']['mean']:>10.4f} {r['all']['median']:>10.4f} "
              f"{r['seam']['mean']:>11.4f} {r['seam']['median']:>10.4f}")
    print(f"{'coverage':<10} {'nominal':>10} {'all':>10} {'seam':>11}")
    for q in LEVELS:
        cq = coverage[f"{q:.2f}"]
        print(f"{'':<10} {q:>10.2f} {cq['all']['axis_mean']:>10.4f} "
              f"{cq['seam']['axis_mean']:>11.4f}")
    print(f"[done] {out_path.name}; {runtime:.1f}s")


if __name__ == "__main__":
    main()

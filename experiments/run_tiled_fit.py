"""Phase A pipeline runner: tiled weighted DP-Splat fit with exact cross-tile merge.

Pipeline (PHASE_A_NOTES items 1-3 are owned here):

  1. Load an aerial cloud (io_aerial) and center coordinates: UTM-scale offsets
     must be removed in float64 before any natural-parameter algebra
     (io_aerial.center_scene; the origin is saved for mapping back to the CRS).
  2. Tile the ground plane with halo overlap (tiling.make_grid / assign_points):
     ownership weights w_n = 1/#owning-tiles and per-tile concentrations
     alpha_t with sum_t alpha_t = alpha_global.
  3. Build ONE shared global NIW prior per modality from the full-scene data
     scale, using dp_splat's own prior recipe (cavi.default_niw_prior), and
     install it in every tile via the State._replace pattern -- the
     prior-subtracted merge algebra requires all tiles to share lam_0.
  4. Fit each tile against the halo-weighted objective: full-batch weighted
     CAVI (weighted.weighted_cavi_step) for tiles at or below --svi-threshold
     points, otherwise a weighted natural-gradient SVI loop (dp_splat.svi
     schedule, statistics scaled by w_n r_nk) followed by one exact full-batch
     weighted CAVI cycle, so the final posteriors are exactly
     prior + full weighted sufficient statistics, as the merge assumes.
  5. Record per-tile K-hat BEFORE merging (soft-count threshold estimator on
     the weighted counts, plus the entropy estimator; brief section 3.7) --
     this is the spatial complexity map.
  6. Drop numerically empty components (weighted count <= --n-min; the EPPF
     term requires N > 0), match components across halo-adjacent tile pairs
     (merge.hungarian_match, scored with alpha_global -- never alpha_t), take
     transitive merge sets, merge exactly in natural-parameter space, and
     rebuild global stick-breaking weights from the pooled counts.
  7. Save <name>_record.json (config plus all diagnostics) and <name>_model.npz
     (merged global mixture, per-tile pre-merge K-hats, tile geometry). No
     plotting here; figures are separate scripts.

Run (wiring check, minutes):
  ~/.venvs/dp-splat/bin/python experiments/run_tiled_fit.py \
      --input ~/dp-splat-data/h3d/Epoch_March2018/LiDAR/Mar18_val.laz \
      --subsample 2000000 --target-tile-points 500000

Run (full validation split, ~4 tiles via SVI):
  ~/.venvs/dp-splat/bin/python experiments/run_tiled_fit.py \
      --input ~/dp-splat-data/h3d/Epoch_March2018/LiDAR/Mar18_val.laz \
      --target-tile-points 4000000
"""

import argparse
import dataclasses
import json
import math
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

from dp_splat import cavi
from dp_splat import niw as dniw
from dp_splat import priors as dpriors
from dp_splat import prune
from dp_splat import svi as dsvi
from dp_splat_uav import io_aerial, merge, tiling, weighted


def parse_args():
    p = argparse.ArgumentParser(description="Tiled weighted DP-Splat fit + merge")
    p.add_argument("--input", required=True, help="LAS/LAZ or PLY point cloud")
    p.add_argument("--name", default=None, help="output basename (default: derived)")
    p.add_argument("--out-dir", default=str(REPO / "experiments" / "out"))
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-tile-points", type=float, default=1e7)
    p.add_argument("--halo", type=float, default=5.0, help="halo width, meters")
    p.add_argument("--alpha", type=float, default=1.0, help="alpha_global")
    p.add_argument("--truncation", type=int, default=64, help="per-tile T")
    p.add_argument("--max-iters", type=int, default=200, help="CAVI iteration cap")
    p.add_argument("--tol", type=float, default=1e-6, help="relative ELBO tolerance")
    p.add_argument("--svi-threshold", type=float, default=2e6,
                   help="tiles above this point count use the SVI path")
    p.add_argument("--svi-batch", type=int, default=65536)
    p.add_argument("--svi-epochs", type=float, default=3.0,
                   help="passes over the tile per SVI fit")
    p.add_argument("--svi-eval-every", type=int, default=25,
                   help="steps between subsample-ELBO evaluations")
    p.add_argument("--svi-eval-points", type=int, default=100_000,
                   help="fixed subsample size for the SVI ELBO trace")
    p.add_argument("--n-min", type=float, default=1.0,
                   help="weighted-count threshold for K-hat and component pruning")
    p.add_argument("--min-tile-points", type=int, default=1000,
                   help="tiles below this count are recorded as skipped, not fitted "
                        "(elongated scenes leave empty grid cells in the bounding box)")
    p.add_argument("--match-margin", type=float, default=None,
                   help="candidate-mask dilation beyond the shared halo band "
                        "(default: one halo width)")
    p.add_argument("--no-merge", action="store_true",
                   help="ablation: skip cross-tile matching entirely and stitch "
                        "the unmerged tile posteriors (weight rebuild unchanged)")
    p.add_argument("--chunk", type=int, default=262144,
                   help="chunk size for full-data passes on SVI-path tiles")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_cloud(args):
    path = Path(args.input).expanduser()
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        return io_aerial.load_laz(path, subsample=args.subsample, rng=args.seed)
    if suffix == ".ply":
        return io_aerial.load_ply(path, subsample=args.subsample, rng=args.seed)
    raise ValueError(f"unsupported point-cloud format {suffix!r}")


# ---------------------------------------------------------------------------
# Per-tile fits (shared global prior installed in both paths)
# ---------------------------------------------------------------------------


def fit_tile_cavi(seed, xs, xc, w, cfg, sp, cp):
    """Weighted full-batch CAVI under the shared global NIW priors.

    Mirrors weighted.weighted_fit (same init, same |dL|/|L| convergence test)
    but installs the global priors after init -- the merge subtracts a single
    shared lam_0, so the tile-local default prior must not be used.
    """
    state = cavi.init_state(seed, xs, xc, cfg)
    state = state._replace(spatial_prior=sp, color_prior=cp)
    history = []
    prev = -np.inf
    for it in range(cfg.max_iters):
        state = weighted.weighted_cavi_step(state, xs, xc, w, cfg)
        L = float(weighted.weighted_elbo(state, xs, xc, w, cfg))
        history.append(L)
        if it > 0 and abs(L - prev) < cfg.tol * abs(prev):
            break
        prev = L
    counts = np.asarray((state.r * w[:, None]).sum(axis=0))
    return state, counts, history, {"path": "cavi", "iters": len(history)}


def _blend_niw(q, hat, rho):
    """Natural-space blend (1 - rho) lam + rho lam_hat (dp_splat.svi convention)."""
    cur = dsvi.niw_to_natural(q)
    return dsvi.niw_from_natural(
        *[(1.0 - rho) * c + rho * h for c, h in zip(cur, hat)]
    )


def _weighted_full_stats(state, xs, xc, w, cfg, chunk):
    """Full-data weighted sufficient statistics, accumulated in chunks.

    Responsibilities are row-wise independent, so chunked evaluation is exact.
    Returns (Nk, (sum_x, sum_xxT) per modality) with the w_n r_nk scaling of
    weighted.weighted_soft_stats.
    """
    n = xs.shape[0]
    T = state.spatial.m.shape[0]
    Nk = jnp.zeros(T)
    acc = {"s": [jnp.zeros((T, xs.shape[1])), jnp.zeros((T, xs.shape[1], xs.shape[1]))],
           "c": [jnp.zeros((T, xc.shape[1])), jnp.zeros((T, xc.shape[1], xc.shape[1]))]}
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        r = cavi.responsibilities(state, xs[sl], xc[sl], cfg)
        rw = r * w[sl][:, None]
        Nk = Nk + rw.sum(axis=0)
        for key, x in (("s", xs[sl]), ("c", xc[sl])):
            acc[key][0] = acc[key][0] + rw.T @ x
            acc[key][1] = acc[key][1] + jnp.einsum("nk,ni,nj->kij", rw, x, x)
    return Nk, acc


def _mstep_from_sums(prior, Nk, sum_x, sum_xxT):
    """Exact conjugate M-step from accumulated (uncentered) weighted sums."""
    xbar = sum_x / jnp.maximum(Nk, 1e-32)[:, None]
    S = sum_xxT - Nk[:, None, None] * jnp.einsum("ki,kj->kij", xbar, xbar)
    return dniw.posterior_update(prior, Nk, xbar, S)


def _chunked_weighted_elbo(state_e, state_m, xs, xc, w, cfg, chunk):
    """Weighted ELBO with dp_splat's evaluation convention, in chunks.

    Point-sum terms use responsibilities from the E-step on state_e and
    densities/log-weights from the updated state_m -- exactly what
    weighted.weighted_elbo sees after weighted_cavi_step.
    """
    n = xs.shape[0]
    elp = cavi.elogpi(state_m, cfg)
    e_ll = lpz = lqz = 0.0
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        r = cavi.responsibilities(state_e, xs[sl], xc[sl], cfg)
        rw = r * w[sl][:, None]
        ell = dniw.expected_gauss_loglik(state_m.spatial, xs[sl]) + \
            dniw.expected_gauss_loglik(state_m.color, xc[sl])
        e_ll += float((rw * ell).sum())
        lpz += float((rw * elp[None, :]).sum())
        lqz += float(jnp.where(r > 0, rw * jnp.log(jnp.where(r > 0, r, 1.0)), 0.0).sum())
    return (
        e_ll
        + lpz
        + float(cavi.elbo_log_p_weights(state_m, cfg))
        + float(cavi.elbo_log_p_theta(state_m))
        + float(cavi.elbo_log_p_alpha(state_m, cfg))
        - lqz
        - float(cavi.elbo_log_q_weights(state_m, cfg))
        - float(cavi.elbo_log_q_theta(state_m))
        - float(cavi.elbo_log_q_alpha(state_m, cfg))
    )


def fit_tile_svi(seed, xs, xc, w, cfg, sp, cp, args):
    """Weighted natural-gradient SVI, then one exact full-batch weighted cycle.

    The step mirrors dp_splat.svi.svi_step with r_nk -> w_n r_nk in every
    sufficient statistic (the same substitution weighted_cavi_step makes in the
    full-batch updates; with w = 1 the two loops coincide). The closing
    full-batch cycle rebuilds each posterior as prior + full weighted
    statistics, which is what the prior-subtracted merge algebra assumes --
    SVI's blended naturals are only a stochastic approximation of that.
    """
    n = xs.shape[0]
    B = min(args.svi_batch, n)
    scale = n / B
    n_steps = max(1, math.ceil(args.svi_epochs * n / B))
    svicfg = dsvi.SVIConfig(batch_size=B, n_steps=n_steps)
    rng = np.random.default_rng(seed)

    state = cavi.init_state(seed, xs, xc, cfg)
    state = state._replace(spatial_prior=sp, color_prior=cp)
    sp_nat = dsvi.niw_to_natural(sp)
    cp_nat = dsvi.niw_to_natural(cp)

    # Fixed subsample for the ELBO trace (full-data ELBO costs a full pass).
    sub = rng.choice(n, size=min(args.svi_eval_points, n), replace=False)
    xs_sub, xc_sub, w_sub = xs[sub], xc[sub], w[sub]

    def sub_elbo(st):
        r = cavi.responsibilities(st, xs_sub, xc_sub, cfg)
        return float(weighted.weighted_elbo(st._replace(r=r), xs_sub, xc_sub, w_sub, cfg))

    trace = []
    for t in range(n_steps):
        rho = (t + 1 + svicfg.tau0) ** (-svicfg.kappa_sched)
        idx = jnp.asarray(rng.choice(n, size=B, replace=False))
        r = cavi.responsibilities(state, xs[idx], xc[idx], cfg)
        rw = r * w[idx][:, None]
        Nk = scale * rw.sum(axis=0)

        new_niw = {}
        for key, x_b, q, p_nat in (
            ("spatial", xs[idx], state.spatial, sp_nat),
            ("color", xc[idx], state.color, cp_nat),
        ):
            sum_x = scale * (rw.T @ x_b)
            sum_xxT = scale * jnp.einsum("nk,ni,nj->kij", rw, x_b, x_b)
            hat = (p_nat[0] + Nk, p_nat[1] + sum_x, p_nat[2] + Nk, p_nat[3] + sum_xxT)
            new_niw[key] = _blend_niw(q, hat, rho)

        g1_hat, g2_hat = dpriors.dp_update(Nk, cfg.alpha)
        wts = dpriors.StickBreakingPosterior(
            (1.0 - rho) * state.weights.gamma1 + rho * g1_hat,
            (1.0 - rho) * state.weights.gamma2 + rho * g2_hat,
        )
        state = cavi.State(new_niw["spatial"], new_niw["color"], sp, cp, wts, None)
        if (t + 1) % args.svi_eval_every == 0 or t == n_steps - 1:
            trace.append(sub_elbo(state))

    # Closing exact cycle: E-step on the SVI state, full-batch weighted M-step.
    Nk, acc = _weighted_full_stats(state, xs, xc, w, cfg, args.chunk)
    spatial = _mstep_from_sums(sp, Nk, *acc["s"])
    color = _mstep_from_sums(cp, Nk, *acc["c"])
    g1, g2 = dpriors.dp_update(Nk, cfg.alpha)
    final = cavi.State(spatial, color, sp, cp,
                       dpriors.StickBreakingPosterior(g1, g2), None)
    full_elbo = _chunked_weighted_elbo(state, final, xs, xc, w, cfg, args.chunk)
    meta = {"path": "svi", "steps": n_steps, "batch": B,
            "elbo_sub_trace": trace, "elbo_full_final": full_elbo}
    return final, np.asarray(Nk), [full_elbo], meta


# ---------------------------------------------------------------------------
# Matching and merging
# ---------------------------------------------------------------------------


def _niw_slice(q, k):
    return dniw.NIW(m=q.m[k:k + 1], kappa=q.kappa[k:k + 1],
                    Psi=q.Psi[k:k + 1], nu=q.nu[k:k + 1])


def _niw_stack(qs):
    return dniw.NIW(*(jnp.concatenate([getattr(q, f) for q in qs])
                      for f in dniw.NIW._fields))


def tile_components(state, counts, n_min):
    """Components surviving the empty-slot prune (weighted count > n_min)."""
    kept = [k for k in range(counts.size) if counts[k] > n_min]
    comps = [
        merge.Component(
            niws=(_niw_slice(state.spatial, k), _niw_slice(state.color, k)),
            count=float(counts[k]),
        )
        for k in kept
    ]
    return kept, comps


def adjacent_pairs(grid):
    """Tile pairs whose halo-dilated rectangles intersect, with the overlap box."""
    T = grid.core_lo.shape[0]
    pairs = []
    for t in range(T):
        for u in range(t + 1, T):
            lo = np.maximum(grid.core_lo[t], grid.core_lo[u]) - grid.halo
            hi = np.minimum(grid.core_hi[t], grid.core_hi[u]) + grid.halo
            if np.all(lo <= hi):
                pairs.append((t, u, lo, hi))
    return pairs


def candidate_mask(comps_a, comps_b, lo, hi, margin):
    """Candidate pairs: both posterior spatial means inside the shared halo
    band, dilated by ``margin``. A tile's copy of a wide straddling structure
    can sit up to about one halo width outside the strict overlap box (its
    data support is the whole dilated tile), so the mask is deliberately
    generous; the Bayes-factor sign test does the real rejection."""
    lo = lo - margin
    hi = hi + margin

    def inside(comps):
        m = np.array([np.asarray(c.niws[0].m)[0, :2] for c in comps])
        return np.all((m >= lo) & (m <= hi), axis=1)

    return inside(comps_a)[:, None] & inside(comps_b)[None, :]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    t_start = time.perf_counter()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and center.
    t0 = time.perf_counter()
    s_raw, c, extras = load_cloud(args)
    s, origin = io_aerial.center_scene(s_raw)
    del s_raw
    n = s.shape[0]
    t_load = time.perf_counter() - t0
    name = args.name or f"{Path(args.input).stem}_n{n}"
    print(f"[load] {args.input}: {n} points in {t_load:.1f}s; "
          f"rgb range [{c.min():.4f}, {c.max():.4f}]; origin {origin.round(2).tolist()}",
          flush=True)

    # 2. Tiles, ownership weights, per-tile alpha_t.
    grid = tiling.make_grid(s, halo=args.halo, target_points=args.target_tile_points)
    asn = tiling.assign_points(grid, s, alpha_global=args.alpha)
    n_tiles = grid.core_lo.shape[0]
    print(f"[tiles] grid {grid.shape}, {n_tiles} tiles, halo {grid.halo} m; "
          f"per-tile points {[len(ix) for ix in asn.indices]}", flush=True)

    # 3. Shared global NIW priors from the full-scene data scale.
    cfg = cavi.Config(weight_prior="dp", T=args.truncation, alpha=args.alpha,
                      max_iters=args.max_iters, tol=args.tol)
    xs_all = jnp.asarray(s)
    xc_all = jnp.asarray(c)
    sp = cavi.default_niw_prior(xs_all, cfg.T, cfg.kappa0, cfg.nu0_offset)
    cp = cavi.default_niw_prior(xc_all, cfg.T, cfg.kappa0, cfg.nu0_offset)

    # 4-5. Per-tile weighted fits and pre-merge K-hat.
    tiles = []
    for t in range(n_tiles):
        idx = np.asarray(asn.indices[t])
        if idx.size < args.min_tile_points:
            # Placeholder keeps grid-index alignment for the merge stage; an
            # unfittable tile contributes no components and K-hat 0.
            tiles.append(dict(state=None, counts=np.zeros(cfg.T), kept=[], comps=[],
                              record=dict(
                                  tile=t, grid_index=grid.index[t].tolist(),
                                  n_points=int(idx.size), weighted_points=0.0,
                                  alpha_t=float(asn.alpha[t]), seconds=0.0,
                                  khat_count=0, khat_entropy=0.0,
                                  elbo_trace=[], elbo_tail=[], elbo_dips=0,
                                  path="skipped",
                              )))
            print(f"[tile {t}] n={idx.size} SKIPPED (< --min-tile-points)", flush=True)
            continue
        xs_t = xs_all[jnp.asarray(idx)]
        xc_t = xc_all[jnp.asarray(idx)]
        w_t = jnp.asarray(asn.weight[idx])
        cfg_t = dataclasses.replace(cfg, alpha=float(asn.alpha[t]))
        t0 = time.perf_counter()
        if idx.size > args.svi_threshold:
            state, counts, history, meta = fit_tile_svi(
                args.seed + t, xs_t, xc_t, w_t, cfg_t, sp, cp, args)
        else:
            state, counts, history, meta = fit_tile_cavi(
                args.seed + t, xs_t, xc_t, w_t, cfg_t, sp, cp)
        seconds = time.perf_counter() - t0

        # ELBO monotonicity audit (exact for the CAVI path; the SVI subsample
        # trace is stochastic and only reported, not audited).
        dips = sum(
            1 for i in range(1, len(history))
            if history[i] < history[i - 1] - 1e-9 * abs(history[i - 1])
        )
        khat_count = int((counts > args.n_min).sum())  # prune.effective_k on weighted counts
        khat_entropy = prune.entropy_effective_k(state, cfg_t)
        kept, comps = tile_components(state, counts, args.n_min)
        tiles.append(dict(state=state, counts=counts, kept=kept, comps=comps,
                          record=dict(
                              tile=t, grid_index=grid.index[t].tolist(),
                              n_points=int(idx.size),
                              weighted_points=float(asn.weight[idx].sum()),
                              alpha_t=float(asn.alpha[t]),
                              seconds=seconds,
                              khat_count=khat_count,
                              khat_entropy=khat_entropy,
                              elbo_trace=history,
                              elbo_tail=history[-5:],
                              elbo_dips=dips,
                              **meta,
                          )))
        print(f"[tile {t}] n={idx.size} path={meta['path']} "
              f"K-hat={khat_count} (entropy {khat_entropy:.1f}) "
              f"elbo_tail={[round(v, 2) for v in history[-3:]]} "
              f"dips={dips} {seconds:.1f}s", flush=True)

    # 6. Cross-tile matching and exact merging.
    t0 = time.perf_counter()
    margin = args.halo if args.match_margin is None else args.match_margin
    prior_pair = (_niw_slice(sp, 0), _niw_slice(cp, 0))
    edges = []
    match_records = []
    for t, u, lo, hi in ([] if args.no_merge else adjacent_pairs(grid)):
        comps_a, comps_b = tiles[t]["comps"], tiles[u]["comps"]
        if not comps_a or not comps_b:
            continue
        mask = candidate_mask(comps_a, comps_b, lo, hi, margin)
        pairs = merge.hungarian_match(
            comps_a, comps_b, prior_pair, args.alpha, mask)
        for i, j in pairs:
            edges.append(((t, tiles[t]["kept"][i]), (u, tiles[u]["kept"][j])))
        match_records.append(dict(
            tiles=[t, u], candidates=int(mask.sum()), accepted=len(pairs),
            pairs=[[tiles[t]["kept"][i], tiles[u]["kept"][j]] for i, j in pairs],
        ))

    merge_sets = merge.transitive_merge_sets(edges)
    in_set = set().union(*merge_sets) if merge_sets else set()

    comp_by_node = {}
    node_order = []
    for t, tile in enumerate(tiles):
        for kept_i, k in enumerate(tile["kept"]):
            comp_by_node[(t, k)] = tile["comps"][kept_i]
            node_order.append((t, k))

    entries = []  # (spatial NIW, color NIW, pooled count, source nodes)
    for members in merge_sets:
        members = sorted(members)
        entries.append((
            merge.merge_components([comp_by_node[m].niws[0] for m in members],
                                   prior_pair[0]),
            merge.merge_components([comp_by_node[m].niws[1] for m in members],
                                   prior_pair[1]),
            sum(comp_by_node[m].count for m in members),
            members,
        ))
    for node in node_order:
        if node not in in_set:
            cmp_ = comp_by_node[node]
            entries.append((cmp_.niws[0], cmp_.niws[1], cmp_.count, [node]))
    if not entries:
        raise RuntimeError("no components survived pruning in any tile")

    counts_global = jnp.asarray([e[2] for e in entries])
    a, b, order = merge.rebuild_weights(counts_global, args.alpha)
    order = np.asarray(order)
    entries = [entries[i] for i in order]
    counts_global = np.asarray(counts_global)[order]
    exp_pi = np.asarray(dpriors.dp_expected_pi(a, b))
    spatial_global = _niw_stack([e[0] for e in entries])
    color_global = _niw_stack([e[1] for e in entries])

    khat_pre = int(sum(t["record"]["khat_count"] for t in tiles))
    khat_post = int((counts_global > args.n_min).sum())
    p_norm = exp_pi / exp_pi.sum()
    entropy_global = float(np.exp(-(p_norm * np.log(np.where(p_norm > 0, p_norm, 1.0))).sum()))
    t_merge = time.perf_counter() - t0
    print(f"[merge] {len(edges)} accepted pairs -> {len(merge_sets)} merge sets; "
          f"global K-hat {khat_pre} pre -> {khat_post} post; {t_merge:.1f}s", flush=True)

    # 7. Outputs.
    src_offsets = np.cumsum([0] + [len(e[3]) for e in entries])
    src_tile = np.array([m[0] for e in entries for m in e[3]], dtype=np.int32)
    src_comp = np.array([m[1] for e in entries for m in e[3]], dtype=np.int32)
    model_path = out_dir / f"{name}_model.npz"
    np.savez(
        model_path,
        spatial_m=np.asarray(spatial_global.m),
        spatial_kappa=np.asarray(spatial_global.kappa),
        spatial_Psi=np.asarray(spatial_global.Psi),
        spatial_nu=np.asarray(spatial_global.nu),
        color_m=np.asarray(color_global.m),
        color_kappa=np.asarray(color_global.kappa),
        color_Psi=np.asarray(color_global.Psi),
        color_nu=np.asarray(color_global.nu),
        counts=counts_global,
        stick_a=np.asarray(a),
        stick_b=np.asarray(b),
        expected_pi=exp_pi,
        src_offsets=src_offsets,
        src_tile=src_tile,
        src_comp=src_comp,
        tile_khat_count=np.array([t["record"]["khat_count"] for t in tiles]),
        tile_khat_entropy=np.array([t["record"]["khat_entropy"] for t in tiles]),
        tile_counts=np.stack([t["counts"] for t in tiles]),
        core_lo=grid.core_lo, core_hi=grid.core_hi, halo=grid.halo,
        bbox_lo=grid.bbox_lo, bbox_hi=grid.bbox_hi,
        grid_shape=np.array(grid.shape), tile_index=grid.index,
        tile_morton=grid.morton, origin=origin, alpha_global=args.alpha,
    )

    total = time.perf_counter() - t_start
    record = dict(
        config=vars(args),
        name=name,
        n_points=n,
        extras_fields=sorted(extras),
        rgb_range=[float(c.min()), float(c.max())],
        origin=origin.tolist(),
        grid=dict(shape=list(grid.shape), n_tiles=n_tiles, halo=grid.halo,
                  bbox_lo=grid.bbox_lo.tolist(), bbox_hi=grid.bbox_hi.tolist()),
        alpha_t=np.asarray(asn.alpha).tolist(),
        tiles=[t["record"] for t in tiles],
        matches=match_records,
        merge=dict(
            accepted_pairs=len(edges),
            merge_sets=len(merge_sets),
            set_sizes=sorted((len(s_) for s_ in merge_sets), reverse=True),
            khat_pre=khat_pre,
            khat_post=khat_post,
            khat_entropy_global=entropy_global,
            match_seconds=t_merge,
        ),
        seconds_total=total,
        load_seconds=t_load,
        versions=dict(jax=jax.__version__, numpy=np.__version__),
    )
    record_path = out_dir / f"{name}_record.json"
    record_path.write_text(json.dumps(record, indent=1))
    print(f"[done] {record_path.name}, {model_path.name}; total {total:.1f}s", flush=True)


if __name__ == "__main__":
    main()

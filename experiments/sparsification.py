"""Sparsification-error analysis of a merged tiled DP-Splat model.

Implements the adopted H3D evaluation recipe (LIT_REVIEW section 3, items 1-3, 6)
for the partition-of-unity stitched predictive: does the model's own predictive
uncertainty rank its reconstruction errors?

Model-point sampling (exact, by gate rejection)
-----------------------------------------------
The stitched spatial predictive scored by eval_heldout.py is

    p(s) = sum_t g_t(s) p_t(s),      p_t(s) = sum_k pi_k St(s; m_k, S_k, eta_k),

with g_t the normalized partition-of-unity gates of stitch.py and p_t the per-tile
Student-t sub-mixtures rebuilt from merge provenance (eval_heldout.tile_submixtures):
each tile carries its components' GLOBAL mixture weights pi_k = expected_pi[k],
un-renormalized within the tile, so p integrates to 1 (a straddling merged
component contributes pi_k sum_t g_t = pi_k; see eval_heldout's module docstring
and tests/test_eval_mass_oracle.py). Because the gates depend on the location
being generated, p is not an ancestral mixture and cannot be sampled by drawing a
component first. It admits exact rejection sampling with the pooled ungated pair
bank as proposal:

    (t, k) ~ pi (renormalized over ALL (tile, component) pairs),
    s | (t, k) ~ St(m_k, S_k, eta_k),   accept with probability g_t(s) in [0, 1].

The accepted-draw density is proportional to sum_{t,k} pi_{tk} g_t(s) St(s; ...),
i.e. the stitched density up to the proposal mass sum_{t,k} pi_{tk} (>= 1: a
straddling component's global weight is counted once per source tile): accepted
samples follow p(s)/Z exactly, Z = total stitched mass / sum_{t,k} pi_{tk}. The
empirical acceptance rate estimates Z (~ 1 for the normalized construction) and
is reported. Student-t draws use the scale-mixture representation
s = m + L z / sqrt(g), z ~ N(0, I), g ~ chi2(eta)/eta, L = chol(S).

Reconstruction error e_i (Huang & Qin plane-to-reference convention)
--------------------------------------------------------------------
For each sampled point, e_i is the orthogonal distance to the least-squares plane
through its 6 nearest neighbors in the REFERENCE cloud (the full LiDAR epoch):
with neighborhood centroid c and scatter C = sum_j (x_j - c)(x_j - c)^T, the LS
plane passes through c with normal n = eigenvector of C's smallest eigenvalue, and
e_i = |n . (s_i - c)|. Degenerate-neighborhood guard: when the neighborhood is
rank-deficient (near-collinear or coincident: lambda_2 <= 1e-8 * lambda_3 for
eigenvalues lambda_1 <= lambda_2 <= lambda_3, or lambda_3 ~ 0, or a non-finite
eigensolve), fall back to the nearest-neighbor distance; the fallback fraction is
reported.

Predicted uncertainty u_i (three variants, all reported)
--------------------------------------------------------
(a) sqrt_mean_marginal_var: the generating component's spatial predictive is
    St(m_k, S_k, eta_k) with covariance Sigma_k = S_k eta_k / (eta_k - 2)
    (finite: eta_k = nu_k - D + 1 >= 3 under the default prior); u_i =
    sqrt(tr(Sigma_k) / 3), the RMS marginal predictive sd (Huang & Qin convention).
(b) neg_log_predictive: u_i = -log p(s_i) under the full stitched spatial mixture
    (all tiles and components, gated) -- a mixture-level summary that does not
    condition on the generating component.
(c) local_density: u_i = -density_i, the NEGATIVE local reference-cloud density
    already computed for the confound controls (sparse neighborhood = presumed
    bad). A model-free control ranker: it needs no fitted model at all, only the
    reference cloud, so it prices what the sparsification result buys over raw
    point density. Ranked with the identical AUSE/AURG machinery; the cross-rank
    Spearman between this ranker and (b) is reported at the top level.

Sparsification protocol (Ilg et al. 2018; Poggi et al. 2020)
------------------------------------------------------------
Sort by u descending and remove in 2% steps (fractions 0, 0.02, ..., 0.98); at each
step record the mean |e| of the retained set, normalized by the full-set mean |e|.
Oracle curve: sort by |e| itself. Random curve: mean over 5 independent shuffles.
AUSE  = integral over removed fraction of (u-curve - oracle-curve)   [lower better],
AURG  = integral over removed fraction of (random-curve - u-curve)   [higher better],
both by the trapezoid rule on the normalized curves over f in [0, 0.98]. Reported
scene-global and per-tile (samples grouped by generating tile; the global number
exercises cross-tile ordering through the gates, a quantitative seam metric).

Controls (mandatory)
--------------------
(i)  local point density at each sample: with r_6 the distance to the 6th nearest
     reference neighbor, density = 6 / ((4/3) pi r_6^3); report Spearman
     corr(u, density) and density-stratified AUSE in 3 density terciles.
(ii) Spearman corr(u, |e|) directly.

Output: experiments/out/<name>_sparsification.json and a figure
figures/<name>_sparsification.png/.pdf (u-curves for both variants, oracle,
random; AUSE/AURG annotated).

Run:
  ~/.venvs/dp-splat/bin/python experiments/sparsification.py \
      --record experiments/out/mar18_ho_record.json \
      --model experiments/out/mar18_ho_model.npz \
      --input ~/dp-splat-data/h3d/Epoch_March2018/LiDAR/Mar18_train.laz \
      --samples 200000 --seed 3
"""

import argparse
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
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import logsumexp
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from dp_splat.predictive import StudentT, student_logpdf
from dp_splat_uav import io_aerial, stitch

from eval_heldout import load_grid, tile_submixtures

K_NN = 6  # reference neighbors per sample (plane fit and density radius)
FRACTIONS = np.arange(50) * 0.02  # removed fractions, 2% steps
N_SHUFFLES = 5
MIN_TILE_SAMPLES = 100  # per-tile curves need enough points for 2% quantile steps
_trapz = np.trapezoid

# Okabe-Ito (colorblind-safe), matching the repo's figure conventions.
COLOR_A, COLOR_B, COLOR_C = "#0072B2", "#E69F00", "#009E73"


def parse_args():
    p = argparse.ArgumentParser(
        description="Sparsification / AUSE evaluation of a merged tiled fit")
    p.add_argument("--record", required=True, help="<name>_record.json from run_tiled_fit")
    p.add_argument("--model", required=True, help="<name>_model.npz from run_tiled_fit")
    p.add_argument("--input", required=True,
                   help="reference point cloud (full LiDAR epoch, LAS/LAZ or PLY)")
    p.add_argument("--samples", type=int, default=200_000,
                   help="model points to draw from the stitched spatial predictive")
    p.add_argument("--seed", type=int, default=3, help="sampling / shuffle seed")
    p.add_argument("--chunk", type=int, default=131_072,
                   help="evaluation chunk size (points)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model bank: flattened (tile, component) pairs of the stitched predictive
# ---------------------------------------------------------------------------


def pair_bank(model, record):
    """Flatten per-tile sub-mixtures into one spatial Student-t bank.

    Returns (grid, pair_tile (P,), log_pi (P,), st_s StudentT(P,)) where log_pi
    is each pair's GLOBAL log mixture weight log expected_pi[k]
    (eval_heldout.tile_submixtures' stitched-path convention: un-renormalized
    within a tile, so sum_pairs exp(log_pi) >= 1 -- a straddling merged
    component's weight appears once per source tile).
    """
    grid = load_grid(model)
    tiles = tile_submixtures(model, record["alpha_t"])
    preds = [t["pred"] for t in tiles]
    pair_tile = np.concatenate(
        [np.full(p.log_pi.shape[0], t, dtype=np.int64) for t, p in enumerate(preds)])
    log_pi = jnp.concatenate([p.log_pi for p in preds])
    st_s = StudentT(*(jnp.concatenate([getattr(p.spatial, f) for p in preds])
                      for f in StudentT._fields))
    return grid, pair_tile, log_pi, st_s


# ---------------------------------------------------------------------------
# Sampling from the stitched spatial predictive
# ---------------------------------------------------------------------------


def tile_gate(grid, xy, tile_idx):
    """g_{t_i}(s_i): the proposing tile's normalized gate at each point -> (N,).

    Same separable linear-ramp bump as stitch.gate_weights, but returns 0 for
    points outside every halo-dilated tile instead of raising (such proposals
    are simply rejected).
    """
    lo, hi, h = grid.core_lo, grid.core_hi, grid.halo
    d = xy[:, None, :]  # (N, 1, 2) against (T, 2)
    if h > 0:
        left = (d - (lo - h)) / h
        right = ((hi + h) - d) / h
        profile = np.clip(np.minimum(np.minimum(left, right), 1.0), 0.0, 1.0)
    else:
        profile = ((d >= lo) & (d <= hi)).astype(np.float64)
    raw = profile.prod(axis=-1)  # (N, T)
    total = raw.sum(axis=1)
    own = raw[np.arange(xy.shape[0]), tile_idx]
    return np.divide(own, total, out=np.zeros_like(own), where=total > 0)


def sample_stitched_spatial(grid, pair_tile, log_pi, st_s, n, rng):
    """Draw n exact samples from the stitched spatial predictive (module docstring).

    Proposal: pair (t, k) with probability proportional to its GLOBAL weight
    over all (tile, component) pairs, s from the pair's Student-t, accept with
    the proposing tile's gate g_t(s). Accepted draws follow
    sum_{t,k} pi_{tk} g_t(s) St(s; ...) up to normalization -- exactly the
    stitched spatial predictive, self-normalized.

    Returns (s (n, 3), pair (n,) generating pair index, acceptance_rate).
    """
    pi = np.exp(np.asarray(log_pi))
    p_pair = pi / pi.sum()

    loc = np.asarray(st_s.loc)
    dof = np.asarray(st_s.dof)
    chol = np.linalg.cholesky(np.asarray(st_s.scale))  # (P, 3, 3)

    out_s, out_pair = [], []
    n_acc, n_prop = 0, 0
    while n_acc < n:
        batch = max(int(1.5 * (n - n_acc)), 4096)
        k = rng.choice(p_pair.size, size=batch, p=p_pair)
        # Student-t via the scale mixture: s = m + L z / sqrt(g), g ~ chi2(eta)/eta.
        z = rng.standard_normal((batch, 3))
        g = rng.chisquare(dof[k]) / dof[k]
        s = loc[k] + np.einsum("nij,nj->ni", chol[k], z) / np.sqrt(g)[:, None]
        accept = rng.random(batch) < tile_gate(grid, s[:, :2], pair_tile[k])
        out_s.append(s[accept])
        out_pair.append(k[accept])
        n_acc += int(accept.sum())
        n_prop += batch
    s = np.concatenate(out_s)[:n]
    pair = np.concatenate(out_pair)[:n]
    return s, pair, n_acc / n_prop


# ---------------------------------------------------------------------------
# Uncertainty variants
# ---------------------------------------------------------------------------


def pair_sqrt_mean_var(st_s):
    """Per-pair u (variant a): sqrt(tr(Sigma)/3), Sigma = S eta/(eta-2) (eta > 2)."""
    dof = np.asarray(st_s.dof)
    if not (dof > 2.0).all():
        raise ValueError("spatial predictive dof <= 2: Student-t covariance undefined")
    scale = np.asarray(st_s.scale)
    cov = scale * (dof / (dof - 2.0))[:, None, None]
    return np.sqrt(np.trace(cov, axis1=-2, axis2=-1) / 3.0)


def stitched_spatial_nlpd(grid, pair_tile, log_pi, st_s, s, chunk):
    """Per-point u (variant b): -log sum_t g_t(s) sum_k pi_k St(s; ...) -> (N,),
    with pi_k the components' global weights (normalized stitched density)."""
    xs = jnp.asarray(s)
    out = []
    for start in range(0, s.shape[0], chunk):
        sl = slice(start, start + chunk)
        g = stitch.gate_weights(grid, xs[sl])[:, pair_tile]  # (n, P)
        log_g = jnp.where(g > 0.0, jnp.log(jnp.where(g > 0.0, g, 1.0)), -jnp.inf)
        lp = log_g + log_pi[None, :] + student_logpdf(st_s, xs[sl])
        out.append(-np.asarray(logsumexp(lp, axis=1)))
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Reconstruction error against the reference cloud
# ---------------------------------------------------------------------------


def reconstruction_error(s, ref, rel_tol=1e-8, abs_tol=1e-18):
    """e_i = |distance to LS plane through the 6 reference NNs| (module docstring).

    Returns (e (N,), density (N,), fallback_fraction, nn_dist (N,)).
    """
    tree = cKDTree(ref)
    dist, idx = tree.query(s, k=K_NN, workers=-1)
    nbr = ref[idx]  # (N, 6, 3)
    cen = nbr.mean(axis=1)
    x = nbr - cen[:, None, :]
    scatter = np.einsum("nki,nkj->nij", x, x)
    w, v = np.linalg.eigh(scatter)  # eigenvalues ascending
    normal = v[..., 0]
    e_plane = np.abs(np.einsum("ni,ni->n", s - cen, normal))
    # Ill-conditioned plane: neighborhood rank < 2 (coincident or near-collinear).
    ill = (~np.isfinite(w).all(axis=1)
           | (w[:, 2] <= abs_tol)
           | (w[:, 1] <= rel_tol * w[:, 2]))
    e = np.where(ill, dist[:, 0], e_plane)
    r_k = np.maximum(dist[:, -1], 1e-12)
    density = K_NN / ((4.0 / 3.0) * np.pi * r_k**3)
    return e, density, float(ill.mean()), dist[:, 0]


# ---------------------------------------------------------------------------
# Sparsification curves and areas
# ---------------------------------------------------------------------------


def _retained_means(e, removal_order):
    """Mean of e retained after removing the first ceil(f n) points of removal_order,
    at each f in FRACTIONS."""
    es = e[removal_order]
    n = e.shape[0]
    suffix_sum = np.cumsum(es[::-1])[::-1]  # suffix_sum[i] = sum es[i:]
    n_removed = np.minimum(np.ceil(FRACTIONS * n).astype(int), n - 1)
    return suffix_sum[n_removed] / (n - n_removed)


def sparsification(e, u, rng):
    """Normalized sparsification curves and areas for one uncertainty variant.

    Curves are normalized by the full-set mean |e|; AUSE/AURG are trapezoid areas
    between curves over removed fraction f in [0, 0.98] (module docstring).
    """
    base = e.mean()
    curve_u = _retained_means(e, np.argsort(-u, kind="stable")) / base
    curve_or = _retained_means(e, np.argsort(-e, kind="stable")) / base
    curve_rnd = np.mean(
        [_retained_means(e, rng.permutation(e.shape[0])) for _ in range(N_SHUFFLES)],
        axis=0) / base
    return dict(
        ause=float(_trapz(curve_u - curve_or, FRACTIONS)),
        aurg=float(_trapz(curve_rnd - curve_u, FRACTIONS)),
        curve_u=curve_u.tolist(),
        curve_oracle=curve_or.tolist(),
        curve_random=curve_rnd.tolist(),
    )


def variant_report(e, u, density, tile_of_sample, rng):
    """Global + per-tile + density-stratified sparsification, and rank controls."""
    glob = sparsification(e, u, rng)
    rho_e = spearmanr(u, e)
    rho_d = spearmanr(u, density)

    edges = np.quantile(density, [1.0 / 3.0, 2.0 / 3.0])
    stratum = np.digitize(density, edges)  # 0 = sparsest tercile
    terciles = []
    for q in range(3):
        m = stratum == q
        terciles.append(dict(
            n=int(m.sum()),
            density_range=[float(density[m].min()), float(density[m].max())],
            **{k: v for k, v in sparsification(e[m], u[m], rng).items()
               if k in ("ause", "aurg")},
        ))

    per_tile = []
    for t in np.unique(tile_of_sample):
        m = tile_of_sample == t
        if int(m.sum()) < MIN_TILE_SAMPLES:
            per_tile.append(dict(tile=int(t), n=int(m.sum()), skipped=True))
            continue
        sp = sparsification(e[m], u[m], rng)
        per_tile.append(dict(tile=int(t), n=int(m.sum()),
                             ause=sp["ause"], aurg=sp["aurg"]))

    return dict(
        global_=glob,
        spearman_u_abs_e=dict(rho=float(rho_e.statistic), p=float(rho_e.pvalue)),
        spearman_u_density=dict(rho=float(rho_d.statistic), p=float(rho_d.pvalue)),
        density_terciles=terciles,
        per_tile=per_tile,
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(name, n_samples, rep_a, rep_b, rep_c, out_png):
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ga, gb, gc = rep_a["global_"], rep_b["global_"], rep_c["global_"]
    ax.plot(FRACTIONS, ga["curve_oracle"], color="black", ls="--", lw=1.2,
            label="oracle (sort by $|e|$)")
    ax.plot(FRACTIONS, ga["curve_random"], color="0.55", ls=":", lw=1.2,
            label="random (5 shuffles)")
    ax.plot(FRACTIONS, ga["curve_u"], color=COLOR_A, lw=1.6,
            label=(r"model $u$: $\sqrt{\overline{\mathrm{Var}}}$"
                   f"  (AUSE {ga['ause']:.3f}, AURG {ga['aurg']:.3f})"))
    ax.plot(FRACTIONS, gb["curve_u"], color=COLOR_B, lw=1.6,
            label=(r"model $u$: $-\log \hat p(s)$"
                   f"  (AUSE {gb['ause']:.3f}, AURG {gb['aurg']:.3f})"))
    ax.plot(FRACTIONS, gc["curve_u"], color=COLOR_C, lw=1.4, ls="-.",
            label=("model-free $u$: $-$density"
                   f"  (AUSE {gc['ause']:.3f}, AURG {gc['aurg']:.3f})"))
    ax.set_xlabel("fraction removed (by decreasing $u$)")
    ax.set_ylabel(r"mean $|e|$ of retained set / mean $|e|$")
    ax.set_xlim(0.0, 0.98)
    ax.set_title(f"{name}: sparsification, {n_samples:,} model points")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    t_start = time.perf_counter()
    record = json.loads(Path(args.record).read_text())
    model = np.load(args.model)
    name = record["name"]
    rng = np.random.default_rng(args.seed)

    # Model bank + sampling.
    grid, pair_tile, log_pi, st_s = pair_bank(model, record)
    t0 = time.perf_counter()
    s, pair, acc_rate = sample_stitched_spatial(
        grid, pair_tile, log_pi, st_s, args.samples, rng)
    t_sample = time.perf_counter() - t0
    print(f"[sample] {name}: {args.samples} points from {pair_tile.shape[0]} "
          f"(tile, component) pairs; acceptance {acc_rate:.3f}; {t_sample:.1f}s",
          flush=True)

    # Uncertainty variants.
    u_a = pair_sqrt_mean_var(st_s)[pair]
    t0 = time.perf_counter()
    u_b = stitched_spatial_nlpd(grid, pair_tile, log_pi, st_s, s, args.chunk)
    t_nlpd = time.perf_counter() - t0

    # Reference cloud: the full LiDAR epoch, in the model's centered frame.
    t0 = time.perf_counter()
    path = Path(args.input).expanduser()
    loader = io_aerial.load_laz if path.suffix.lower() in (".las", ".laz") else io_aerial.load_ply
    ref, _, _ = loader(path, subsample=None, rng=None)
    ref = ref - model["origin"]
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    e, density, fallback_frac, nn_dist = reconstruction_error(s, ref)
    t_knn = time.perf_counter() - t0
    del ref
    print(f"[error] reference {path.name} loaded ({t_load:.1f}s); 6-NN plane "
          f"distances in {t_knn:.1f}s; plane fallback fraction {fallback_frac:.4f}",
          flush=True)

    # Sparsification + controls for all variants. Order matters: (a) and (b)
    # consume the shared rng stream first, so their random curves (and hence
    # AUSE/AURG) reproduce runs made before variant (c) existed.
    tile_of_sample = pair_tile[pair]
    rep_a = variant_report(e, u_a, density, tile_of_sample, rng)
    rep_b = variant_report(e, u_b, density, tile_of_sample, rng)
    u_c = -density  # model-free control ranker (variant c): sparse = presumed bad
    rep_c = variant_report(e, u_c, density, tile_of_sample, rng)
    rho_cb = spearmanr(u_c, u_b)  # rank agreement, density ranker vs NLPD ranker

    fig_path = REPO / "figures" / f"{name}_sparsification.png"
    make_figure(name, args.samples, rep_a, rep_b, rep_c, fig_path)

    runtime = time.perf_counter() - t_start
    out = dict(
        name=name,
        config=vars(args),
        n_samples=int(s.shape[0]),
        n_pairs=int(pair_tile.shape[0]),
        acceptance_rate=float(acc_rate),
        samples_per_tile={int(t): int((tile_of_sample == t).sum())
                          for t in np.unique(tile_of_sample)},
        plane_fallback_fraction=fallback_frac,
        error_summary=dict(mean=float(e.mean()), median=float(np.median(e)),
                           p95=float(np.quantile(e, 0.95))),
        nn_dist_summary=dict(median=float(np.median(nn_dist)),
                             p95=float(np.quantile(nn_dist, 0.95))),
        fractions=FRACTIONS.tolist(),
        u_variants=dict(
            sqrt_mean_marginal_var=rep_a,
            neg_log_predictive=rep_b,
            local_density=rep_c,
        ),
        spearman_density_ranker_vs_nlpd=dict(
            rho=float(rho_cb.statistic), p=float(rho_cb.pvalue)),
        seconds=dict(sample=t_sample, nlpd=t_nlpd, ref_load=t_load, knn=t_knn,
                     total=runtime),
        versions=dict(jax=jax.__version__, numpy=np.__version__),
    )
    out_path = Path(args.record).parent / f"{name}_sparsification.json"
    out_path.write_text(json.dumps(out, indent=1))

    for label, rep in (("sqrt-mean-var", rep_a), ("neg-log-pred", rep_b),
                       ("local-density", rep_c)):
        g = rep["global_"]
        print(f"[{label:>13}] AUSE {g['ause']:.4f}  AURG {g['aurg']:.4f}  "
              f"rho(u,|e|) {rep['spearman_u_abs_e']['rho']:.3f}  "
              f"rho(u,density) {rep['spearman_u_density']['rho']:.3f}")
    print(f"[rank-agree] rho(density-ranker, nlpd-ranker) {rho_cb.statistic:.3f}")
    print(f"[done] {out_path.name}, {fig_path.name}; {runtime:.1f}s")


if __name__ == "__main__":
    main()

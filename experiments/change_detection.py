"""Uncertainty-aware multi-epoch change detection with a model-derived LoD.

Compares two independently fitted tiled DP-Splat models of the same scene
(epoch A minus epoch B, e.g. Mar19 - Nov18) on a regular 2D grid, flags cells
whose vertical surface change exceeds a 95% level of detection propagated from
the posterior MEAN covariance of the spatial components (M3C2-EP-style,
LIT_REVIEW section 3 item 5), and validates against per-point semantic labels.

Frame reconciliation
--------------------
Each fit is performed in its own centered frame with a recorded absolute UTM
origin (record["origin"] == model["origin"]). All grid geometry here lives in
absolute UTM: the shared extent is the intersection of the two models' data
bounding boxes (each model's centered bbox_lo/hi plus its origin's xy),
optionally intersected with --bbox. Cell centers are converted to each model's
frame by subtracting that model's origin xy, and the per-epoch surface heights
are lifted back to absolute elevation by adding the origin z, so
d(s) = mu_z^A(s) - mu_z^B(s) is a difference of absolute elevations.

Per-epoch surface estimate at a cell center s (exact construction)
------------------------------------------------------------------
The model bank is the flattened set of (tile, component) pairs of the merged
model's per-tile sub-mixtures (eval_heldout.tile_submixtures). The cell weight
of pair (t, k) is the gated-mixture responsibility under the FULL 3D spatial
Student-t predictive evaluated at the cell's mean ground point
q = (cx, cy, zbar), where zbar is the mean z of THAT epoch's raw points in the
cell (each epoch anchors to its own observed surface, so a cell whose surface
moved queries each model where its data actually lie):

    w_tk(q) proportional to g_t(q) pi_k St_3(q; m_k, S_k, eta_k),

with pi_k the component's GLOBAL mixture weight (eval_heldout.tile_submixtures'
stitched-path convention -- un-renormalized within a tile, so the gated bank is
a normalized density and cross-tile responsibilities in seam bands are weighted
by the components' true global masses, not per-tile-renormalized ones),
normalized over all pairs (g_t are stitch.py's partition-of-unity gates; a safe
variant returns zero outside every halo-dilated tile instead of raising, and
such cells are marked unsupported for that epoch). The alternative xy-marginal
weighting St_2(s_xy) was tried first and rejected: it distributes
responsibility over the whole vertical column, so elevated structure whose xy
footprint overlaps the cell (canopy, roof overhangs) contaminates mu_z of
ground cells and inflates the between-component variance to meters even on
pure impervious cells. Then

    mu_z(s)  = sum_tk w_tk m_k,z                       (posterior mean heights)
    var_z(s) = sum_tk w_tk [Sigma_mu,k]_zz             (within-component)
             + sum_tk w_tk (m_k,z - mu_z(s))^2         (between-component)

which is the law of total variance applied to the ESTIMATED mean surface: the
first term propagates each component's uncertainty about its own mean location,
the second the ambiguity over which component's surface the cell belongs to.
Sigma_mu,k = Psi_k / (kappa_k (nu_k - D - 1)) with D = 3 is the marginal
posterior covariance of the NIW spatial mean -- NOT the predictive covariance
S_k (kappa+1)/(kappa eta): the predictive adds the surface-roughness term
Psi/(nu - D - 1), and using it would double-count roughness in the LoD (the
exact error M3C2-EP corrects; LIT_REVIEW pitfall (i)). Color is marginalized
out throughout (spatial factors only).

Change statistic and LoD
------------------------
    d(s)     = mu_z^A(s) - mu_z^B(s)
    LoD95(s) = 1.96 sqrt(var_z^A(s) + var_z^B(s) + sigma_reg^2)

flag change where |d| > LoD95. sigma_reg (m, one term for the relative
registration of the two georeferenced epochs; LIT_REVIEW pitfall (ii)) is a CLI
argument, default 0; results are additionally reported at sigma_reg = 0.02 as a
sensitivity block regardless of the headline value.

Classical DoD baseline (--classical)
------------------------------------
With --classical, a standard DEM-of-difference arm is evaluated on the SAME
cells, masks, and null/detection protocol, from the raw points alone (Lane et
al. 2003 propagated-error form): per-cell mean-z difference

    d_cl(s)     = zbar_A(s) - zbar_B(s)
    LoD95_cl(s) = 1.96 sqrt(s_A^2/n_A + s_B^2/n_B + sigma_reg^2)

with s^2 the within-cell sample variance (ddof = 1) and n the raw point count
of that epoch's cell. This is the empirical referee baseline: it needs both
epochs' raw points in every cell at comparison time, whereas the model arm
queries only the two fitted mixtures (plus the per-cell anchor height).

Semantic proxy evaluation (per-point LAS classification, full clouds)
---------------------------------------------------------------------
Per epoch and cell: class histogram of ALL raw points falling in the cell
(chunked laspy read of the full LAZ, not the fit subsample), dominant class,
purity = dominant fraction, point count. Cells need >= --min-cell-points in
BOTH epochs (and gate support in both models) to be evaluated at all.

1. NULL TEST: cells whose dominant class is Impervious Surface (1) in BOTH
   epochs with purity > 0.9 in both are physically stable; the flag rate there
   is the empirical exceedance of the nominal-5% LoD.
2. DETECTION: among cells with purity > 0.6 in each epoch, positives are cells
   whose dominant class CHANGED between epochs, negatives those where it did
   not. Score = |d| / LoD95; ROC AUC plus precision/recall at score > 1. The
   headline numbers EXCLUDE cells whose dominant class in EITHER epoch is
   vegetation (0 Low Veg, 6 Shrub, 7 Tree): Nov->Mar foliage change is real but
   trivially detectable; veg-included numbers are reported as secondary.

Output: experiments/out/<name>.json, per-cell arrays in
experiments/out/<name>_cells.npz (for figure regeneration), and a two-panel
map figures/<name>.{png,pdf} (|d| heatmap; flagged mask over the
class-transition proxy).

Run (smoke, 200 x 200 m crop):
  ~/.venvs/dp-splat/bin/python experiments/change_detection.py \
      --record-a experiments/out/mar19_rich_record.json \
      --model-a  experiments/out/mar19_rich_model.npz \
      --input-a  ~/dp-splat-data/h3d/Epoch_March2019/LiDAR/Mar19_train.laz \
      --record-b experiments/out/nov18_rich_record.json \
      --model-b  experiments/out/nov18_rich_model.npz \
      --input-b  ~/dp-splat-data/h3d/Epoch_November2018/LiDAR/Nov18_train.laz \
      --bbox 513850 5426800 514050 5427000 --name change_smoke
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import rankdata

from dp_splat.predictive import StudentT, student_logpdf
from dp_splat_uav import stitch

from eval_heldout import load_grid, tile_submixtures

D_SPATIAL = 3
N_CLASS_BINS = 32  # LAS classification is 5 bits in legacy formats; H3D uses 0-10
VEG_CLASSES = (0, 6, 7)  # Low Vegetation, Shrub, Tree
IMPERVIOUS = 1
CLASS_NAMES = {
    0: "Low Vegetation", 1: "Impervious Surface", 2: "Vehicle",
    3: "Urban Furniture", 4: "Roof", 5: "Facade", 6: "Shrub", 7: "Tree",
    8: "Soil/Gravel", 9: "Vertical Surface", 10: "Chimney",
}
Z95 = 1.96
SIGMA_REG_SENSITIVITY = 0.02  # m; always-reported sensitivity value

# Okabe-Ito (colorblind-safe), matching the repo's figure conventions.
C_DETECTED, C_MISS, C_FALSE = "#0072B2", "#D55E00", "#E69F00"


def parse_args():
    p = argparse.ArgumentParser(
        description="LoD-based change detection between two tiled DP-Splat fits")
    p.add_argument("--record-a", required=True, help="epoch A record (e.g. Mar19)")
    p.add_argument("--model-a", required=True, help="epoch A model npz")
    p.add_argument("--input-a", required=True, help="epoch A full LAZ (labels)")
    p.add_argument("--record-b", required=True, help="epoch B record (e.g. Nov18)")
    p.add_argument("--model-b", required=True, help="epoch B model npz")
    p.add_argument("--input-b", required=True, help="epoch B full LAZ (labels)")
    p.add_argument("--cell", type=float, default=1.0, help="grid cell size (m)")
    p.add_argument("--sigma-reg", type=float, default=0.0,
                   help="registration sd along z (m), added once to the LoD variance")
    p.add_argument("--seed", type=int, default=5,
                   help="protocol seed (recorded; the pipeline is deterministic)")
    p.add_argument("--chunk", type=int, default=32_768,
                   help="cell-evaluation chunk size")
    p.add_argument("--bbox", type=float, nargs=4, default=None,
                   metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                   help="absolute UTM crop; default: full shared extent")
    p.add_argument("--min-cell-points", type=int, default=20,
                   help="min raw points per cell per epoch for evaluation")
    p.add_argument("--classical", action="store_true",
                   help="also evaluate the classical DoD baseline "
                        "(raw per-cell mean-z difference, empirical LoD95) "
                        "on the same cells and masks")
    p.add_argument("--name", default="change_nov18_mar19", help="output basename")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Grid over the shared absolute extent
# ---------------------------------------------------------------------------


def shared_grid(model_a, model_b, bbox, cell):
    """Cell edges/centers (absolute UTM) over the intersected model extents."""
    lo_a = model_a["bbox_lo"] + model_a["origin"][:2]
    hi_a = model_a["bbox_hi"] + model_a["origin"][:2]
    lo_b = model_b["bbox_lo"] + model_b["origin"][:2]
    hi_b = model_b["bbox_hi"] + model_b["origin"][:2]
    lo = np.maximum(lo_a, lo_b)
    hi = np.minimum(hi_a, hi_b)
    if bbox is not None:
        lo = np.maximum(lo, [bbox[0], bbox[1]])
        hi = np.minimum(hi, [bbox[2], bbox[3]])
    if not (hi > lo).all():
        raise ValueError(f"empty shared extent: lo {lo}, hi {hi}")
    n = np.maximum(np.floor((hi - lo) / cell).astype(int), 1)  # (nx, ny)
    centers_x = lo[0] + cell * (np.arange(n[0]) + 0.5)
    centers_y = lo[1] + cell * (np.arange(n[1]) + 0.5)
    cx, cy = np.meshgrid(centers_x, centers_y)  # (ny, nx), row-major in y
    centers = np.column_stack([cx.ravel(), cy.ravel()])
    return lo, lo + cell * n, tuple(int(v) for v in n), centers


# ---------------------------------------------------------------------------
# Per-epoch model surface at cell centers
# ---------------------------------------------------------------------------


def safe_gate_weights(grid, xy):
    """Normalized partition-of-unity gates -> (N, T); zero rows outside coverage.

    Same separable linear-ramp bump as stitch.gate_weights, but cells outside
    every halo-dilated tile get an all-zero row (marked unsupported upstream)
    instead of raising -- the shared extent can graze the models' bbox edges.
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
    total = raw.sum(axis=1, keepdims=True)
    return np.divide(raw, total, out=np.zeros_like(raw), where=total > 0)


def epoch_bank(model, record):
    """Flattened (tile, component) bank: gates geometry, weights, 3D spatial
    Student-t predictive, mean heights, and the zz posterior-MEAN variance.

    Sigma_mu = Psi_N / (kappa_N (nu_N - D - 1)) is the marginal covariance of
    the NIW mean (mean-location uncertainty only; module docstring).
    """
    grid = load_grid(model)
    tiles = tile_submixtures(model, record["alpha_t"])
    preds = [t["pred"] for t in tiles]  # GLOBAL weights (module docstring)
    pair_tile = np.concatenate(
        [np.full(p.log_pi.shape[0], t, dtype=np.int64) for t, p in enumerate(preds)])
    log_pi = jnp.concatenate([p.log_pi for p in preds])
    st_s = StudentT(*(jnp.concatenate([getattr(p.spatial, f) for p in preds])
                      for f in StudentT._fields))

    m = jnp.concatenate([t["state"].spatial.m for t in tiles])
    kappa = jnp.concatenate([t["state"].spatial.kappa for t in tiles])
    Psi = jnp.concatenate([t["state"].spatial.Psi for t in tiles])
    nu = jnp.concatenate([t["state"].spatial.nu for t in tiles])
    denom = kappa * (nu - D_SPATIAL - 1.0)
    if not bool((denom > 0.0).all()):
        raise ValueError("nu_N <= D + 1: posterior mean covariance undefined")
    m_z = np.asarray(m[:, 2])
    sig_mu_zz = np.asarray(Psi[:, 2, 2] / denom)
    return dict(grid=grid, pair_tile=pair_tile, log_pi=log_pi, st_s=st_s,
                m_z=m_z, sig_mu_zz=sig_mu_zz)


def epoch_surface(bank, origin, centers_utm, zbar_abs, chunk):
    """mu_z (absolute), var_z, support mask, and log density at the cells'
    mean ground points q = (center_xy, zbar) in the model's centered frame."""
    n = centers_utm.shape[0]
    has_z = np.isfinite(zbar_abs)
    q = np.column_stack([centers_utm, np.where(has_z, zbar_abs, 0.0)]) - origin
    mu = np.full(n, np.nan)
    var = np.full(n, np.nan)
    logdens = np.full(n, -np.inf)
    supported = np.zeros(n, dtype=bool)
    m_z = jnp.asarray(bank["m_z"])
    sig = jnp.asarray(bank["sig_mu_zz"])
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        g = safe_gate_weights(bank["grid"], q[sl, :2])[:, bank["pair_tile"]]  # (c, P)
        ok = (g.sum(axis=1) > 0.0) & has_z[sl]
        gj = jnp.asarray(g)
        log_g = jnp.where(gj > 0.0, jnp.log(jnp.where(gj > 0.0, gj, 1.0)), -jnp.inf)
        lp = log_g + bank["log_pi"][None, :] + student_logpdf(bank["st_s"], jnp.asarray(q[sl]))
        norm = logsumexp(lp, axis=1)
        w = jnp.exp(lp - norm[:, None])  # (c, P) responsibilities
        mu_c = w @ m_z
        within = w @ sig
        between = (w * (m_z[None, :] - mu_c[:, None]) ** 2).sum(axis=1)
        mu[sl] = np.where(ok, np.asarray(mu_c), np.nan)
        var[sl] = np.where(ok, np.asarray(within + between), np.nan)
        logdens[sl] = np.where(ok, np.asarray(norm), -np.inf)
        supported[sl] = ok
    return mu + origin[2], var, supported, logdens


# ---------------------------------------------------------------------------
# Semantic proxy from the raw labeled clouds
# ---------------------------------------------------------------------------


def class_histogram(path, lo, shape, cell):
    """Per-cell class counts (n_cells, N_CLASS_BINS), z sums, and z^2 sums
    (n_cells,) from the full LAZ, chunked."""
    import laspy

    nx, ny = shape
    hist = np.zeros(ny * nx * N_CLASS_BINS, dtype=np.int64)
    zsum = np.zeros(ny * nx)
    z2sum = np.zeros(ny * nx)
    with laspy.open(str(Path(path).expanduser())) as f:
        for pts in f.chunk_iterator(10_000_000):
            x = np.asarray(pts.x)
            y = np.asarray(pts.y)
            ix = np.floor((x - lo[0]) / cell).astype(np.int64)
            iy = np.floor((y - lo[1]) / cell).astype(np.int64)
            keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            if not keep.any():
                continue
            cell_idx = iy[keep] * nx + ix[keep]
            cls = np.minimum(np.asarray(pts.classification)[keep], N_CLASS_BINS - 1)
            hist += np.bincount(cell_idx * N_CLASS_BINS + cls, minlength=hist.size)
            z = np.asarray(pts.z)[keep]
            zsum += np.bincount(cell_idx, weights=z, minlength=zsum.size)
            z2sum += np.bincount(cell_idx, weights=z * z, minlength=z2sum.size)
    return hist.reshape(ny * nx, N_CLASS_BINS), zsum, z2sum


def cell_semantics(hist, zsum, z2sum):
    """(dominant class, purity, point count, mean z, sample z-variance) per
    cell; purity 0 on empty cells, mean z NaN on empty cells, variance NaN
    below 2 points (ddof = 1 undefined)."""
    total = hist.sum(axis=1)
    dom = hist.argmax(axis=1)
    top = hist.max(axis=1)
    purity = np.divide(top, total, out=np.zeros(total.shape), where=total > 0)
    zbar = np.divide(zsum, total, out=np.full(zsum.shape, np.nan), where=total > 0)
    # Sample variance from the moment sums; clip tiny negative float residue.
    ss = np.maximum(z2sum - total * np.where(total > 0, zbar, 0.0) ** 2, 0.0)
    s2 = np.divide(ss, total - 1, out=np.full(zsum.shape, np.nan),
                   where=total >= 2)
    return dom, purity, total, zbar, s2


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def roc_auc(scores, labels):
    """Rank-based (Mann-Whitney) AUC of scores for binary labels; ties averaged."""
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def detection_metrics(score, positive):
    """AUC of |d|/LoD95 plus precision/recall/false-alarm at the LoD threshold."""
    flagged = score > 1.0
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    fn = int((~flagged & positive).sum())
    return dict(
        n_pos=int(positive.sum()),
        n_neg=int((~positive).sum()),
        auc=roc_auc(score, positive),
        precision=(tp / (tp + fp)) if (tp + fp) > 0 else None,
        recall=(tp / (tp + fn)) if (tp + fn) > 0 else None,
        flag_rate_pos=float(flagged[positive].mean()) if positive.any() else None,
        flag_rate_neg=float(flagged[~positive].mean()) if (~positive).any() else None,
    )


def evaluate(d, var_sum, sigma_reg, masks):
    """Null test + detection blocks (veg-excluded headline, veg-included) at
    one registration sd. masks: dict of boolean cell masks (see main)."""
    lod = Z95 * np.sqrt(var_sum + sigma_reg**2)
    score = np.abs(d) / np.maximum(lod, 1e-12)
    flagged = score > 1.0

    null = masks["null"]
    out = dict(
        sigma_reg=float(sigma_reg),
        null_test=dict(
            n_cells=int(null.sum()),
            exceedance_rate=float(flagged[null].mean()) if null.any() else None,
            median_abs_d_m=float(np.median(np.abs(d[null]))) if null.any() else None,
            median_lod_m=float(np.median(lod[null])) if null.any() else None,
        ),
    )
    for label, elig in (("veg_excluded", masks["elig_noveg"]),
                        ("veg_included", masks["elig"])):
        out[label] = detection_metrics(score[elig], masks["positive"][elig])
    return out, lod, flagged


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def make_figure(name, shape, lo, cell, d, valid, flagged, masks, out_png):
    nx, ny = shape
    extent = [0.0, nx * cell, 0.0, ny * cell]  # local meters from the grid corner

    abs_d = np.abs(np.where(valid, d, np.nan)).reshape(ny, nx)
    vmax = np.nanquantile(abs_d, 0.99) if np.isfinite(abs_d).any() else 1.0

    # Panel-2 categories over the veg-excluded (headline) proxy sets.
    cat = np.zeros(nx * ny, dtype=np.int8)          # 0: not evaluated
    elig = masks["elig_noveg"]
    pos = masks["positive"]
    cat[valid & ~elig] = 1                          # evaluated, proxy-ineligible
    cat[elig & ~pos & ~flagged] = 2                 # stable, not flagged
    cat[elig & ~pos & flagged] = 3                  # stable, flagged (false alarm)
    cat[elig & pos & flagged] = 4                   # transition, flagged (detected)
    cat[elig & pos & ~flagged] = 5                  # transition, missed
    colors = ["white", "#dcdcdc", "#bcd8ec", C_FALSE, C_DETECTED, C_MISS]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(7) - 0.5, cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 1.0 + 4.4 * max(ny / nx, 0.5)),
                             sharex=True, sharey=True, layout="constrained")
    cmap_d = matplotlib.colormaps["viridis"].copy()
    cmap_d.set_bad("white")
    im = axes[0].imshow(abs_d, origin="lower", extent=extent, cmap=cmap_d,
                        vmin=0.0, vmax=max(vmax, 1e-6), interpolation="nearest")
    axes[0].set_title(r"$|d|$: absolute vertical change (m)", fontsize=9)
    fig.colorbar(im, ax=axes[0], shrink=0.85, pad=0.02)

    axes[1].imshow(cat.reshape(ny, nx), origin="lower", extent=extent,
                   cmap=cmap, norm=norm, interpolation="nearest")
    axes[1].set_title(r"$|d| > \mathrm{LoD}_{95}$ vs class-transition proxy", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.6", lw=0.4) for c in
               (colors[4], colors[5], colors[3], colors[2], colors[1])]
    axes[1].legend(handles,
                   ["transition, flagged", "transition, missed",
                    "stable, flagged", "stable, not flagged", "ineligible"],
                   fontsize=6.5, loc="upper right", framealpha=0.9)
    for ax in axes:
        ax.set_xlabel(f"easting $-$ {lo[0]:.0f} (m)", fontsize=8)
        ax.set_aspect("equal")
    axes[0].set_ylabel(f"northing $-$ {lo[1]:.0f} (m)", fontsize=8)
    fig.suptitle(name, fontsize=10)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    t_start = time.perf_counter()
    record_a = json.loads(Path(args.record_a).read_text())
    record_b = json.loads(Path(args.record_b).read_text())
    model_a = np.load(args.model_a)
    model_b = np.load(args.model_b)
    origin_a = np.asarray(model_a["origin"])
    origin_b = np.asarray(model_b["origin"])

    lo, hi, shape, centers = shared_grid(model_a, model_b, args.bbox, args.cell)
    nx, ny = shape
    print(f"[grid] shared extent x [{lo[0]:.1f}, {hi[0]:.1f}] "
          f"y [{lo[1]:.1f}, {hi[1]:.1f}] UTM; {nx} x {ny} = {nx * ny} cells "
          f"({args.cell} m); origin gap A-B "
          f"{np.array2string(origin_a - origin_b, precision=3)}", flush=True)

    # Semantic histograms and per-cell mean ground points from the full
    # labeled clouds (the mean points anchor the model queries below).
    sem = {}
    for tag, path in (("a", args.input_a), ("b", args.input_b)):
        t0 = time.perf_counter()
        dom, purity, count, zbar, s2 = cell_semantics(
            *class_histogram(path, lo, shape, args.cell))
        sem[tag] = dict(dom=dom, purity=purity, count=count, zbar=zbar, s2=s2)
        print(f"[labels {tag}] {Path(path).name}: {int((count > 0).sum())} occupied "
              f"cells; {time.perf_counter() - t0:.1f}s", flush=True)

    # Model surfaces (each queried in its own centered frame at its own epoch's
    # mean ground points, lifted to absolute z).
    surf = {}
    for tag, model, record, origin in (("a", model_a, record_a, origin_a),
                                       ("b", model_b, record_b, origin_b)):
        t0 = time.perf_counter()
        bank = epoch_bank(model, record)
        mu, var, ok, logdens = epoch_surface(
            bank, origin, centers, sem[tag]["zbar"], args.chunk)
        surf[tag] = dict(mu=mu, var=var, ok=ok, logdens=logdens,
                         n_pairs=int(bank["pair_tile"].shape[0]))
        print(f"[model {tag}] {record['name']}: {surf[tag]['n_pairs']} pairs, "
              f"{int(ok.sum())}/{nx * ny} cells supported; "
              f"{time.perf_counter() - t0:.1f}s", flush=True)

    d = surf["a"]["mu"] - surf["b"]["mu"]
    var_sum = surf["a"]["var"] + surf["b"]["var"]
    valid = (surf["a"]["ok"] & surf["b"]["ok"]
             & (sem["a"]["count"] >= args.min_cell_points)
             & (sem["b"]["count"] >= args.min_cell_points)
             & np.isfinite(d) & np.isfinite(var_sum))

    dom_a, pur_a = sem["a"]["dom"], sem["a"]["purity"]
    dom_b, pur_b = sem["b"]["dom"], sem["b"]["purity"]
    veg = np.isin(dom_a, VEG_CLASSES) | np.isin(dom_b, VEG_CLASSES)
    masks = dict(
        null=(valid & (dom_a == IMPERVIOUS) & (dom_b == IMPERVIOUS)
              & (pur_a > 0.9) & (pur_b > 0.9)),
        elig=valid & (pur_a > 0.6) & (pur_b > 0.6),
        positive=dom_a != dom_b,
    )
    masks["elig_noveg"] = masks["elig"] & ~veg

    results, lod, flagged = evaluate(d, var_sum, args.sigma_reg, masks)
    sensitivity, _, _ = evaluate(d, var_sum, SIGMA_REG_SENSITIVITY, masks)

    classical = None
    if args.classical:
        d_cl = sem["a"]["zbar"] - sem["b"]["zbar"]
        var_cl = (sem["a"]["s2"] / sem["a"]["count"]
                  + sem["b"]["s2"] / sem["b"]["count"])
        cl_results, lod_cl, flagged_cl = evaluate(d_cl, var_cl,
                                                  args.sigma_reg, masks)
        cl_sensitivity, _, _ = evaluate(d_cl, var_cl,
                                        SIGMA_REG_SENSITIVITY, masks)
        classical = dict(
            construction=dict(
                d="zbar_A - zbar_B (raw per-cell mean z, both epochs)",
                lod=("1.96 sqrt(s_A^2/n_A + s_B^2/n_B + sigma_reg^2), "
                     "s^2 the within-cell sample z-variance (ddof=1) of the "
                     "raw points; standard propagated-error DoD form"),
                cells="identical grid, validity masks, and proxy sets as "
                      "the model arm",
            ),
            d_summary=dict(
                median_abs_m=float(np.median(np.abs(d_cl[valid]))) if valid.any() else None,
                p95_abs_m=float(np.quantile(np.abs(d_cl[valid]), 0.95)) if valid.any() else None,
                median_lod_m=float(np.median(lod_cl[valid])) if valid.any() else None,
                flag_rate_valid=float(flagged_cl[valid].mean()) if valid.any() else None,
            ),
            results=cl_results,
            sensitivity_sigma_reg=cl_sensitivity,
        )

    fig_path = REPO / "figures" / f"{args.name}.png"
    make_figure(args.name, shape, lo, args.cell, d, valid, flagged, masks, fig_path)

    out_dir = Path(args.record_a).parent
    cells_path = out_dir / f"{args.name}_cells.npz"
    extra = {}
    if classical is not None:
        extra = dict(d_classical=d_cl, lod_classical=lod_cl,
                     flagged_classical=flagged_cl,
                     s2_a=sem["a"]["s2"], s2_b=sem["b"]["s2"])
    np.savez_compressed(
        cells_path, grid_lo=lo, grid_hi=hi, shape=np.asarray(shape), cell=args.cell,
        d=d, var_a=surf["a"]["var"], var_b=surf["b"]["var"], lod=lod,
        valid=valid, flagged=flagged, dom_a=dom_a, dom_b=dom_b,
        purity_a=pur_a, purity_b=pur_b, count_a=sem["a"]["count"],
        count_b=sem["b"]["count"], zbar_a=sem["a"]["zbar"],
        zbar_b=sem["b"]["zbar"], logdens_a=surf["a"]["logdens"],
        logdens_b=surf["b"]["logdens"], **extra)

    runtime = time.perf_counter() - t_start
    out = dict(
        name=args.name,
        config=vars(args),
        epochs=dict(a=record_a["name"], b=record_b["name"]),
        frame=dict(
            origin_a=origin_a.tolist(), origin_b=origin_b.tolist(),
            grid_lo_utm=lo.tolist(), grid_hi_utm=hi.tolist(),
            grid_shape=list(shape), cell_m=args.cell,
            note=("grid in absolute UTM; cell centers converted per epoch by "
                  "subtracting that model's recorded origin xy; heights lifted "
                  "by origin z before differencing"),
        ),
        construction=dict(
            weights=("gated GLOBAL E[pi] (un-renormalized within tiles) times "
                     "full 3D spatial Student-t predictive at the cell's "
                     "per-epoch mean ground point (cx, cy, zbar); xy-marginal "
                     "weighting rejected, see docstring"),
            variance=("law of total variance on the mean estimate: "
                      "sum_k w_k [Psi_N/(kappa_N(nu_N-D-1))]_zz "
                      "+ sum_k w_k (m_kz - mu_z)^2; posterior MEAN covariance, "
                      "not predictive (M3C2-EP double-counting pitfall)"),
            lod="1.96 sqrt(var_a + var_b + sigma_reg^2)",
        ),
        class_names=CLASS_NAMES,
        n_cells=dict(
            grid=int(nx * ny), valid=int(valid.sum()),
            null=int(masks["null"].sum()),
            eligible=int(masks["elig"].sum()),
            eligible_noveg=int(masks["elig_noveg"].sum()),
            positive_noveg=int((masks["elig_noveg"] & masks["positive"]).sum()),
        ),
        d_summary=dict(
            median_abs_m=float(np.median(np.abs(d[valid]))) if valid.any() else None,
            p95_abs_m=float(np.quantile(np.abs(d[valid]), 0.95)) if valid.any() else None,
            median_lod_m=float(np.median(lod[valid])) if valid.any() else None,
            flag_rate_valid=float(flagged[valid].mean()) if valid.any() else None,
        ),
        results=results,
        sensitivity_sigma_reg=sensitivity,
        classical=classical,
        seconds_total=runtime,
        versions=dict(jax=jax.__version__, numpy=np.__version__),
    )
    out_path = out_dir / f"{args.name}.json"
    out_path.write_text(json.dumps(out, indent=1))

    nt = results["null_test"]
    print(f"[null] {nt['n_cells']} stable impervious cells: exceedance "
          f"{nt['exceedance_rate'] if nt['exceedance_rate'] is not None else float('nan'):.4f}, "
          f"median |d| {nt['median_abs_d_m']:.4f} m, median LoD {nt['median_lod_m']:.4f} m"
          if nt["n_cells"] else "[null] no qualifying cells", flush=True)
    for label in ("veg_excluded", "veg_included"):
        r = results[label]
        auc = f"{r['auc']:.4f}" if r["auc"] is not None else "n/a"
        prec = f"{r['precision']:.3f}" if r["precision"] is not None else "n/a"
        rec = f"{r['recall']:.3f}" if r["recall"] is not None else "n/a"
        print(f"[{label}] pos {r['n_pos']} / neg {r['n_neg']}: AUC {auc}, "
              f"precision {prec}, recall {rec} at LoD95", flush=True)
    if classical is not None:
        nt = classical["results"]["null_test"]
        print(f"[classical null] exceedance {nt['exceedance_rate']:.4f}, "
              f"median |d| {nt['median_abs_d_m']:.4f} m, "
              f"median LoD {nt['median_lod_m']:.4f} m", flush=True)
        for label in ("veg_excluded", "veg_included"):
            r = classical["results"][label]
            auc = f"{r['auc']:.4f}" if r["auc"] is not None else "n/a"
            prec = f"{r['precision']:.3f}" if r["precision"] is not None else "n/a"
            rec = f"{r['recall']:.3f}" if r["recall"] is not None else "n/a"
            print(f"[classical {label}] AUC {auc}, precision {prec}, "
                  f"recall {rec} at LoD95", flush=True)
    print(f"[done] {out_path.name}, {cells_path.name}, {fig_path.name}; "
          f"{runtime:.1f}s")


if __name__ == "__main__":
    main()

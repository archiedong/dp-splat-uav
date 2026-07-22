"""Absolute geometric accuracy of a merged tiled DP-Splat model against the
H3D reference mesh (submission-audit item A3).

Scores the stitched spatial predictive -- and the crop / Voronoi seam baselines
built from the identical per-tile sub-mixtures -- against the Hessigheim 3D
reference surface in meters, in both directions:

  ACCURACY   (model -> reference): unsigned exact point-to-mesh distance of
             samples drawn from each model's spatial predictive; mean / median /
             RMS / P90 / P95 and accuracy@{0.10, 0.25, 0.50} m fractions.
  COMPLETENESS (reference -> model): area-weighted samples of the mesh surface,
             distance to the nearest of a dense bank of model samples;
             completeness@{0.25, 0.50, 1.00} m fractions.

Both directions are additionally stratified by the stitched model's own
per-sample predictive NLPD (-log p(s) under the gated spatial mixture,
sparsification.stitched_spatial_nlpd) in terciles: does surface the model is
confident about sit closer to the independent reference?

Reference surface and parser choice (documented decision)
---------------------------------------------------------
The reference is the ifp H3D March-2018 mesh, shipped as 51 per-tile OBJ pairs
(43 train + 8 val tiles; tiles named <E/10>_<N/10>_<slab> where the third field
indexes a VERTICAL slab -- e.g. 51382_542673_20 spans z 241-258 m and _25 the
tower slab above it -- so all requested tiles are merged, not de-duplicated by
name). Each tile carries a textured OBJ and a "_labeled" OBJ with identical
triangle count; the labeled variant stores plain "f i j k" triangles, no
texture indirection, and 4-decimal (0.1 mm) vertex coordinates -- quantization
three orders of magnitude below the 0.1 m evaluation threshold. We parse it
with a minimal reader for that fixed grammar rather than trimesh: trimesh's
proximity queries require the rtree extension and evaluate per point in
Python (unusable at 2e5 queries x 1e7 faces), its OBJ loader material-splits
the per-face "usemtl <class>" labeled files, and the oracle contract below
requires an independently tested exact point-to-triangle routine anyway.
Identical faces duplicated on slab boundaries are dropped by exact key
(centroid, area) rounded to 0.1 mm; the dropped count is reported.

Frame reconciliation
--------------------
Mesh vertices are in the same UTM32N-like H3D local frame as the LiDAR LAZ.
Exactly as eval_heldout.py does for held-out points, all mesh coordinates are
centered by model["origin"] (the fit's stored centroid shift) immediately
after loading; float64 throughout, so the ~5e6-magnitude raw coordinates cost
~1e-9 m of rounding at worst. The centered mesh xy bounding box must overlap
the fit's tile-grid bounding box by at least --min-frame-overlap of the grid
area (hard failure otherwise); the overlap fraction and both boxes land in the
output JSON.

Exact point-to-mesh distance
----------------------------
d(p, mesh) = min over faces of the exact point-to-triangle distance (interior
orthogonal projection when its barycentric coordinates admit it, else the
minimum of the three point-segment distances -- the closest point of a
triangle is its plane projection or lies on its boundary). The minimum over
1e7 faces is found by candidate search: a cKDTree over face centroids proposes
the k0 nearest faces, giving an upper bound d*; the result is certified exact
when the k0-th centroid distance is >= d* + r_max (r_max = the largest
face circumradius: any other face is then at distance >= d*). Points failing
the certificate are re-solved over EVERY face whose centroid lies within
d* + r_max (a superset of all faces that could beat d*, and always
non-empty: it contains the incumbent). A handful of outlier-sized faces would
inflate r_max for everyone, so they are split off and brute-forced exactly
(MeshNearest docstring); the escalated fraction is reported.
The routine is oracle-tested against a closed-form point-plane computation
(1e-12) and brute force over random triangle soups (tests/test_geoacc_oracle.py).

Support restriction (both directions, honest per the adopted H3D recipe)
------------------------------------------------------------------------
* ACCURACY: model samples outside the union of the loaded mesh tiles' xy
  bounding rectangles have no reference support (the val strip is a hole in
  the train tiles) and are excluded; the excluded fraction is reported.
* COMPLETENESS: the mesh extends beyond the training LiDAR (e.g. ~26 m west
  of the March-2018 train cloud), where "incompleteness" would measure data
  coverage, not the model. Mesh-surface samples are therefore restricted to
  xy cells (--domain-cell, default 1 m) holding >= --domain-min-count points
  of the fit's own input cloud; the kept fraction is reported. Pass
  --no-domain-mask to skip (then completeness includes never-surveyed area).
* Completeness is sample-discretized: distance to the nearest of --n-dense
  model samples over-estimates distance to the model's surface by about the
  model-sample spacing; the median nearest-neighbor spacing within the dense
  bank is reported next to the fractions as the resolution floor.

Model sampling machinery
------------------------
Stitched samples reuse sparsification.sample_stitched_spatial verbatim: pairs
proposed by GLOBAL mixture weight, gate-rejected by the proposing tile's
partition-of-unity gate (exact; the corrected un-renormalized convention of
eval_heldout, mass oracle tests/test_eval_mass_oracle.py). The crop and
Voronoi baselines are the ancestral mixtures of baselines.crop_mean_in_tile /
voronoi_ownership on the identical sub-mixtures (eval_heldout machinery),
sampled component-first (their weights are normalized by construction). RNG
streams are seeded np.random.default_rng([seed, k]) with a distinct documented
k per draw (0 stitched, 1 crop, 2 voronoi, 3 mesh surface, 4-6 dense banks).

Output: experiments/out/<name>.json and figures/<name>.{png,pdf}
(IEEE column width, Type-42 fonts, print-size type -- audit item B).

Run (production March-2018 fit):
  ~/.venvs/dp-splat/bin/python experiments/geometric_accuracy.py \
      --record experiments/out/mar18_rich_record.json \
      --model experiments/out/mar18_rich_model.npz \
      --input ~/dp-splat-data/h3d/Epoch_March2018/LiDAR/Mar18_train.laz \
      --name geoacc_mar18
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

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,          # TrueType, not Type 3 (IEEE PDF eXpress)
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from dp_splat.predictive import niw_predictive
from dp_splat_uav import baselines, io_aerial

from eval_heldout import baseline_boxes, tile_submixtures
from sparsification import pair_bank, sample_stitched_spatial, stitched_spatial_nlpd

ACC_THRESH = (0.10, 0.25, 0.50)    # accuracy@tau fractions (m)
COMP_THRESH = (0.25, 0.50, 1.00)   # completeness@tau fractions (m)
TERCILE_LABELS = ("low_nlpd", "mid_nlpd", "high_nlpd")
DEDUP_QUANTUM = 1e-4               # face-duplicate key resolution (m / m^2)

COLW = 252.0 / 72.27  # IEEEtran \columnwidth in inches

# Okabe-Ito (colorblind-safe), the repo's fixed series assignment:
# stitched = blue everywhere; baselines take orange / bluish-green.
C_BLUE, C_ORANGE, C_GREEN, C_SKY = "#0072B2", "#E69F00", "#009E73", "#56B4E9"


def parse_args():
    p = argparse.ArgumentParser(
        description="Geometric accuracy / completeness vs the H3D reference mesh")
    p.add_argument("--record", required=True, help="<name>_record.json from run_tiled_fit")
    p.add_argument("--model", required=True, help="<name>_model.npz from run_tiled_fit")
    p.add_argument("--input", default=None,
                   help="the fit's input cloud (LAS/LAZ/PLY), used only for the "
                        "completeness domain mask; required unless --no-domain-mask")
    p.add_argument("--mesh-dir",
                   default="~/dp-splat-data/h3d/Epoch_March2018/Mesh/per-tile",
                   help="root of the per-tile reference mesh (contains train/ val/)")
    p.add_argument("--splits", default="train",
                   help="comma-separated mesh splits to merge (default: train, the "
                        "region the model's training cloud covers)")
    p.add_argument("--variant", choices=("labeled", "textured"), default="labeled",
                   help="OBJ variant per tile (identical geometry; see docstring)")
    p.add_argument("--n-model", type=int, default=200_000,
                   help="model samples per arm for the accuracy direction")
    p.add_argument("--n-ref", type=int, default=200_000,
                   help="area-weighted mesh-surface samples (completeness direction)")
    p.add_argument("--n-dense", type=int, default=500_000,
                   help="dense model samples per arm serving as completeness target")
    p.add_argument("--seed", type=int, default=3, help="base seed for all RNG streams")
    p.add_argument("--chunk", type=int, default=131_072, help="NLPD evaluation chunk")
    p.add_argument("--k0", type=int, default=32,
                   help="candidate faces per point before the exactness certificate")
    p.add_argument("--domain-cell", type=float, default=1.0,
                   help="occupancy-mask cell size (m) for the completeness domain")
    p.add_argument("--domain-min-count", type=int, default=5,
                   help="min input-cloud points per cell to count as surveyed")
    p.add_argument("--no-domain-mask", action="store_true",
                   help="skip the input-cloud occupancy restriction")
    p.add_argument("--min-frame-overlap", type=float, default=0.5,
                   help="required overlap of mesh xy bbox with the tile-grid bbox, "
                        "as a fraction of grid bbox area (frame-reconciliation guard)")
    p.add_argument("--no-baselines", action="store_true",
                   help="score the stitched model only")
    p.add_argument("--name", default=None,
                   help="output stem (default: geoacc_<record name>)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reference mesh: minimal OBJ reader for the fixed H3D per-tile grammar
# ---------------------------------------------------------------------------


def parse_obj(path):
    """Vertices (V, 3) float64 and 0-based triangle faces (F, 3) int64.

    Accepts the two H3D grammars: labeled ("f i j k") and textured
    ("f i/ti j/tj k/tk" -- position index taken, texture index dropped).
    Any non-triangular face is a hard error.
    """
    v_rows, f_rows = [], []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith("v "):
                v_rows.append(ln[2:].split())
            elif ln.startswith("f "):
                f_rows.append([tok.partition("/")[0] for tok in ln[2:].split()])
    verts = np.asarray(v_rows, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"{path}: expected 'v x y z' vertices")
    faces = np.asarray(f_rows)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path}: non-triangular or ragged faces")
    faces = faces.astype(np.int64) - 1  # OBJ is 1-based
    if faces.min() < 0 or faces.max() >= verts.shape[0]:
        raise ValueError(f"{path}: face index out of range")
    return verts, faces


def load_mesh_tiles(mesh_dir, splits, variant):
    """Merge every requested tile into one (vertices, faces, per-tile info) triple.

    Coordinates stay in the raw H3D frame here; the caller centers by the
    model origin. per_tile records each tile's xy bounding rectangle (raw
    frame) for the accuracy-direction support test.
    """
    mesh_dir = Path(mesh_dir).expanduser()
    suffix = "_labeled.obj" if variant == "labeled" else ".obj"
    verts_all, faces_all, per_tile = [], [], []
    n_verts = 0
    for split in splits:
        split_dir = mesh_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"mesh split directory not found: {split_dir}")
        for tile_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            obj = tile_dir / f"{tile_dir.name}{suffix}"
            if not obj.is_file():
                raise FileNotFoundError(f"missing OBJ variant: {obj}")
            v, f = parse_obj(obj)
            verts_all.append(v)
            faces_all.append(f + n_verts)
            n_verts += v.shape[0]
            per_tile.append(dict(
                split=split, tile=tile_dir.name,
                n_vertices=int(v.shape[0]), n_faces=int(f.shape[0]),
                xy_lo=v[:, :2].min(axis=0).tolist(),
                xy_hi=v[:, :2].max(axis=0).tolist(),
                z_range=[float(v[:, 2].min()), float(v[:, 2].max())],
            ))
    return np.concatenate(verts_all), np.concatenate(faces_all), per_tile


def face_areas(tri):
    """Triangle areas (F,) from a (F, 3, 3) vertex array."""
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.sqrt((cross * cross).sum(axis=-1))


def dedup_faces(tri):
    """Keep the first copy of exactly duplicated faces (slab-boundary repeats).

    Key = (centroid, area) rounded to DEDUP_QUANTUM (0.1 mm -- the labeled
    variant's own coordinate quantum, so identical faces collide exactly and
    distinct faces collide with negligible probability). Returns (keep_index
    ascending, n_dropped).
    """
    key = np.column_stack([tri.mean(axis=1), face_areas(tri)])
    key = np.round(key / DEDUP_QUANTUM).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    keep = np.sort(first)
    return keep, tri.shape[0] - keep.shape[0]


# ---------------------------------------------------------------------------
# Exact point-to-mesh distance
# ---------------------------------------------------------------------------


def _point_segment_dist2(p, a, b):
    """Squared distance from p (N, 3) to segments (a, b) (N, 3 each)."""
    ab = b - a
    denom = (ab * ab).sum(axis=-1)
    t = np.divide(((p - a) * ab).sum(axis=-1), denom,
                  out=np.zeros_like(denom), where=denom > 0.0)
    q = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return ((p - q) ** 2).sum(axis=-1)


def point_triangle_dist2(p, tri):
    """Exact squared point-to-triangle distance, elementwise: p (N, 3), tri (N, 3, 3).

    The closest point of a triangle is the orthogonal projection when that
    projection's barycentric coordinates are admissible, otherwise a point on
    the boundary; the boundary is covered exactly by the three point-segment
    distances (which also handle degenerate triangles: a zero-area triangle IS
    its edges).
    """
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    e0, e1 = v1 - v0, v2 - v0
    w = p - v0
    a = (e0 * e0).sum(-1)
    b = (e0 * e1).sum(-1)
    c = (e1 * e1).sum(-1)
    d = (e0 * w).sum(-1)
    e = (e1 * w).sum(-1)
    det = a * c - b * b
    u = c * d - b * e      # unnormalized barycentric coords of the projection
    v = a * e - b * d
    inside = (u >= 0.0) & (v >= 0.0) & (u + v <= det) & (det > 0.0)
    safe_det = np.where(det > 0.0, det, 1.0)
    foot = (u[:, None] * e0 + v[:, None] * e1) / safe_det[:, None]  # foot - v0
    d2_interior = ((w - foot) ** 2).sum(-1)
    d2_edges = np.minimum(
        _point_segment_dist2(p, v0, v1),
        np.minimum(_point_segment_dist2(p, v0, v2),
                   _point_segment_dist2(p, v1, v2)))
    return np.where(inside, np.minimum(d2_interior, d2_edges), d2_edges)


class MeshNearest:
    """Exact unsigned distance to a triangle soup (module docstring algorithm).

    Outlier-face robustness: the certificate margin is the LARGEST face
    circumradius among tree-indexed faces, so a handful of oversized faces
    (decimated roofs, water) would force huge escalation balls for everyone.
    Faces with circumradius > max(8 x median, OUTLIER_CAP largest) are
    therefore split off and solved by direct vectorized brute force for every
    query point (exact, cheap for a small set); the kNN certificate runs on
    the homogeneous remainder with its own tight rmax. The final distance is
    the minimum of the two exact parts.
    """

    OUTLIER_CAP = 2048  # at most this many faces on the brute-force path

    def __init__(self, tri, k0=32):
        tri = np.ascontiguousarray(tri, dtype=np.float64)
        centroid = tri.mean(axis=1)
        radius = np.sqrt(((tri - centroid[:, None, :]) ** 2).sum(-1)).max(axis=1)
        cut = 8.0 * float(np.median(radius))
        if (radius > cut).sum() > self.OUTLIER_CAP:
            cut = float(np.partition(radius, -self.OUTLIER_CAP)[-self.OUTLIER_CAP])
        large = radius > cut
        self.tri_large = tri[large]
        self.tri = tri[~large]
        self.centroid = centroid[~large]
        self.rmax = float(radius[~large].max()) if (~large).any() else 0.0
        self.k0 = int(min(k0, max(self.tri.shape[0], 1)))
        self.tree = cKDTree(self.centroid) if self.tri.shape[0] else None

    def _candidate_min(self, p, idx):
        """min over candidate faces idx (N, k) for points p (N, 3) -> (N,)."""
        n, k = idx.shape
        d2 = point_triangle_dist2(
            np.repeat(p, k, axis=0), self.tri[idx.ravel()]).reshape(n, k)
        return np.sqrt(d2.min(axis=1))

    def _refine(self, p, upper):
        """Exact small-face distance for points p given upper bounds: solve over
        every face whose centroid lies within upper + rmax (always contains the
        true nearest face AND the incumbent), batched under a face budget."""
        out = np.empty(p.shape[0])
        balls = self.tree.query_ball_point(p, upper + self.rmax + 1e-9, workers=-1)
        start = 0
        budget = 4_000_000  # faces per vectorized batch
        while start < p.shape[0]:
            stop = start
            total = 0
            while stop < p.shape[0] and (total == 0 or total + len(balls[stop]) <= budget):
                total += len(balls[stop])
                stop += 1
            counts = [len(balls[i]) for i in range(start, stop)]
            flat = np.concatenate([balls[i] for i in range(start, stop)]).astype(np.int64)
            rep = np.repeat(np.arange(start, stop), counts)
            d2 = point_triangle_dist2(p[rep], self.tri[flat])
            best = np.full(stop - start, np.inf)
            np.minimum.at(best, rep - start, d2)
            out[start:stop] = np.sqrt(best)
            start = stop
        return out

    def _large_min(self, p):
        """Exact distance to the outlier faces by direct brute force -> (N,)."""
        best = np.full(p.shape[0], np.inf)
        for f0 in range(0, self.tri_large.shape[0], 512):
            block = self.tri_large[f0:f0 + 512]
            for s0 in range(0, p.shape[0], 4096):
                sl = slice(s0, s0 + 4096)
                n, m = p[sl].shape[0], block.shape[0]
                d2 = point_triangle_dist2(
                    np.repeat(p[sl], m, axis=0),
                    np.tile(block, (n, 1, 1))).reshape(n, m)
                best[sl] = np.minimum(best[sl], d2.min(axis=1))
        return np.sqrt(best)

    def distance(self, p, chunk=8192):
        """Exact d(p_i, mesh) (N,) and certificate stats."""
        p = np.asarray(p, dtype=np.float64)
        out = np.full(p.shape[0], np.inf)
        n_escalated = 0
        if self.tree is not None:
            query_all = self.k0 >= self.tri.shape[0]
            for start in range(0, p.shape[0], chunk):
                sl = slice(start, min(start + chunk, p.shape[0]))
                dc, idx = self.tree.query(p[sl], k=self.k0, workers=-1)
                if self.k0 == 1:
                    dc, idx = dc[:, None], idx[:, None]
                best = self._candidate_min(p[sl], idx)
                if query_all:
                    out[sl] = best
                    continue
                # Certificate: every non-candidate face has centroid distance
                # >= dc[:, -1], hence face distance >= dc[:, -1] - rmax.
                ok = dc[:, -1] - self.rmax >= best
                res = np.where(ok, best, 0.0)
                if not ok.all():
                    bad = ~ok
                    res[bad] = self._refine(p[sl][bad], best[bad])
                    n_escalated += int(bad.sum())
                out[sl] = res
        if self.tri_large.shape[0]:
            out = np.minimum(out, self._large_min(p))
        return out, dict(k0=self.k0, rmax=self.rmax,
                         n_outlier_faces=int(self.tri_large.shape[0]),
                         n_escalated=n_escalated,
                         escalated_fraction=n_escalated / max(p.shape[0], 1))


# ---------------------------------------------------------------------------
# Sampling: mesh surface (area-weighted) and baseline mixtures (ancestral)
# ---------------------------------------------------------------------------


def sample_surface(tri, n, rng, areas=None):
    """Area-weighted uniform samples of a triangle soup -> (points (n, 3), face (n,))."""
    if areas is None:
        areas = face_areas(tri)
    f = rng.choice(areas.shape[0], size=n, p=areas / areas.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    pts = (tri[f, 0]
           + u[:, None] * (tri[f, 1] - tri[f, 0])
           + v[:, None] * (tri[f, 2] - tri[f, 0]))
    return pts, f


def sample_mixture(mix, n, rng):
    """Ancestral samples of a baselines.GlobalMixture's spatial predictive.

    Component-first is exact here: baseline weights are normalized and carry no
    location-dependent gates. Student-t via the same scale-mixture construction
    as sparsification.sample_stitched_spatial.
    """
    st = niw_predictive(mix.spatial)
    w = np.asarray(mix.weights)
    k = rng.choice(w.shape[0], size=n, p=w / w.sum())
    loc, dof = np.asarray(st.loc), np.asarray(st.dof)
    chol = np.linalg.cholesky(np.asarray(st.scale))
    z = rng.standard_normal((n, 3))
    g = rng.chisquare(dof[k]) / dof[k]
    return loc[k] + np.einsum("nij,nj->ni", chol[k], z) / np.sqrt(g)[:, None]


def nn_distance(from_pts, to_pts):
    """Distance from each of from_pts to the nearest of to_pts."""
    d, _ = cKDTree(to_pts).query(from_pts, k=1, workers=-1)
    return d


# ---------------------------------------------------------------------------
# Support restriction
# ---------------------------------------------------------------------------


def in_any_rect(xy, rect_lo, rect_hi):
    """Point-in-union-of-rectangles: xy (N, 2) vs rects (T, 2) -> bool (N,)."""
    inside = (xy[:, None, :] >= rect_lo[None]) & (xy[:, None, :] <= rect_hi[None])
    return inside.all(axis=-1).any(axis=-1)


class OccupancyMask:
    """xy cells of a reference cloud holding >= min_count points (surveyed area)."""

    def __init__(self, cloud_xy, cell, min_count):
        self.cell = float(cell)
        self.lo = cloud_xy.min(axis=0) - self.cell
        shape = np.ceil((cloud_xy.max(axis=0) + self.cell - self.lo) / self.cell
                        ).astype(np.int64) + 1
        self.shape = shape
        ij = ((cloud_xy - self.lo) / self.cell).astype(np.int64)
        counts = np.bincount(ij[:, 0] * shape[1] + ij[:, 1],
                             minlength=int(shape[0] * shape[1]))
        self.occupied = counts >= int(min_count)

    def __call__(self, xy):
        ij = np.floor((xy - self.lo) / self.cell).astype(np.int64)
        valid = ((ij >= 0) & (ij < self.shape[None, :])).all(axis=1)
        code = np.where(valid, ij[:, 0] * self.shape[1] + ij[:, 1], 0)
        return valid & self.occupied[code]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def dist_stats(d, thresholds):
    """mean/median/RMS/P90/P95 (m) and fraction-below-threshold table for |d|."""
    d = np.asarray(d)
    if d.size == 0:
        return dict(n=0)
    return dict(
        n=int(d.size),
        mean=float(d.mean()),
        median=float(np.median(d)),
        rms=float(np.sqrt(np.mean(d ** 2))),
        p90=float(np.quantile(d, 0.90)),
        p95=float(np.quantile(d, 0.95)),
        **{f"frac_le_{t:g}m": float((d <= t).mean()) for t in thresholds},
    )


def tercile_report(d, nlpd, thresholds):
    """dist_stats per predictive-NLPD tercile (low = most confident)."""
    edges = np.quantile(nlpd, [1.0 / 3.0, 2.0 / 3.0])
    which = np.digitize(nlpd, edges)
    rows = []
    for q, label in enumerate(TERCILE_LABELS):
        m = which == q
        rows.append(dict(
            tercile=label,
            nlpd_range=[float(nlpd[m].min()), float(nlpd[m].max())],
            **dist_stats(d[m], thresholds)))
    return dict(edges=edges.tolist(), rows=rows)


# ---------------------------------------------------------------------------
# Figure (IEEE column width, Type-42 fonts -- audit item B)
# ---------------------------------------------------------------------------


def make_figure(name, acc, comp, out_png):
    """(a) model->reference error CDFs; (b) tercile bars, both directions."""
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(COLW, 3.55), layout="constrained",
        gridspec_kw={"height_ratios": [1.05, 1.0]})

    # (a) empirical CDF of unsigned distance, log-x (errors span mm to m).
    # A density histogram over geometric bins normalizes by *linear* bin
    # width, so the narrow leftmost bins render as spikes carrying almost
    # no samples; the CDF is spike-free and reads the threshold fractions
    # of Table III directly.
    arms = [("stitched", C_BLUE), ("crop", C_ORANGE), ("voronoi", C_GREEN)]
    lo = min(max(np.quantile(acc[a]["d"], 0.001), 1e-4) for a, _ in arms if a in acc)
    hi = max(np.quantile(acc[a]["d"], 0.999) for a, _ in arms if a in acc)
    for arm, color in arms:
        if arm not in acc:
            continue
        st = acc[arm]["stats"]
        d = np.sort(acc[arm]["d"])
        y = np.arange(1, d.size + 1) / d.size
        label = "Voronoi" if arm == "voronoi" else arm
        ax0.semilogx(d, y, color=color, lw=1.1,
                     label=f"{label} (RMS {st['rms']:.2f} m)")
    ax0.set_xlim(0.8 * lo, 1.25 * hi)
    ax0.set_ylim(0.0, 1.0)
    tf = matplotlib.transforms.blended_transform_factory(ax0.transData, ax0.transAxes)
    for t in ACC_THRESH:
        ax0.axvline(t, color="0.75", lw=0.6, ls=":", zorder=0)
        ax0.text(t, 1.01, f"{t:g}", color="0.45", transform=tf,
                 ha="center", va="bottom", fontsize=6)
    ax0.set_xlabel("unsigned distance to reference mesh (m)")
    ax0.set_ylabel(r"fraction of samples $\leq d$")
    ax0.legend(loc="upper left", frameon=False, handlelength=1.4)
    ax0.grid(alpha=0.25, lw=0.4)

    # (b) stitched-arm NLPD terciles: mean distance, both directions.
    t_acc = [r["mean"] for r in acc["stitched"]["terciles"]["rows"]]
    t_comp = [r["mean"] for r in comp["stitched"]["terciles"]["rows"]]
    x = np.arange(3)
    wd = 0.34
    b0 = ax1.bar(x - 0.19, t_acc, wd, color=C_BLUE, label="model $\\to$ mesh")
    b1 = ax1.bar(x + 0.19, t_comp, wd, color=C_SKY, label="mesh $\\to$ model")
    for bars in (b0, b1):
        ax1.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["low NLPD\n(confident)", "mid NLPD", "high NLPD"])
    ax1.set_ylabel("mean distance (m)")
    ax1.set_ylim(0, 1.15 * max(t_acc + t_comp))
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=2,
               frameon=False, borderaxespad=0.0)
    ax1.grid(axis="y", alpha=0.25, lw=0.4)

    fig.savefig(out_png, dpi=400)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    t_start = time.perf_counter()
    record = json.loads(Path(args.record).read_text())
    model = np.load(args.model)
    name = args.name or f"geoacc_{record['name']}"
    origin = np.asarray(model["origin"], dtype=np.float64)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if args.input is None and not args.no_domain_mask:
        raise SystemExit("--input is required unless --no-domain-mask is set")
    seconds = {}

    def rng(stream):
        return np.random.default_rng([args.seed, stream])

    # ---- Model banks and samples (small memory; before any bulk loading). ----
    t0 = time.perf_counter()
    grid, pair_tile, log_pi, st_s = pair_bank(model, record)
    s_mod, _, acc_rate = sample_stitched_spatial(
        grid, pair_tile, log_pi, st_s, args.n_model, rng(0))
    s_dense, _, _ = sample_stitched_spatial(
        grid, pair_tile, log_pi, st_s, args.n_dense, rng(4))
    arms = {"stitched": (s_mod, s_dense)}
    if not args.no_baselines:
        tiles = tile_submixtures(model, record["alpha_t"])
        fits = [baselines.TileFit(t["state"], t["cfg"]) for t in tiles]
        masses = [float(t["counts"].sum()) for t in tiles]
        boxes = baseline_boxes(grid)
        mix_crop = baselines.crop_mean_in_tile(fits, boxes, masses)
        mix_vor = baselines.voronoi_ownership(fits, boxes, masses)
        arms["crop"] = (sample_mixture(mix_crop, args.n_model, rng(1)),
                        sample_mixture(mix_crop, args.n_dense, rng(5)))
        arms["voronoi"] = (sample_mixture(mix_vor, args.n_model, rng(2)),
                           sample_mixture(mix_vor, args.n_dense, rng(6)))
    nlpd_mod = stitched_spatial_nlpd(grid, pair_tile, log_pi, st_s, s_mod, args.chunk)
    seconds["model_samples"] = time.perf_counter() - t0
    print(f"[model] {len(arms)} arm(s); stitched acceptance {acc_rate:.3f}; "
          f"{seconds['model_samples']:.1f}s", flush=True)

    # ---- Completeness domain mask from the fit's input cloud. ----
    occupancy = None
    if not args.no_domain_mask:
        t0 = time.perf_counter()
        path = Path(args.input).expanduser()
        loader = (io_aerial.load_laz if path.suffix.lower() in (".las", ".laz")
                  else io_aerial.load_ply)
        cloud_s, _, _ = loader(path, subsample=None, rng=None)
        n_cloud = cloud_s.shape[0]
        occupancy = OccupancyMask(cloud_s[:, :2] - origin[None, :2],
                                  args.domain_cell, args.domain_min_count)
        del cloud_s
        seconds["domain_mask"] = time.perf_counter() - t0
        print(f"[domain] {path.name}: {n_cloud} points -> "
              f"{int(occupancy.occupied.sum())} occupied {args.domain_cell:g} m "
              f"cells; {seconds['domain_mask']:.1f}s", flush=True)

    # ---- Reference mesh. ----
    t0 = time.perf_counter()
    verts, faces, per_tile = load_mesh_tiles(args.mesh_dir, splits, args.variant)
    verts -= origin[None, :]
    tri = verts[faces]
    rect_lo = np.asarray([t["xy_lo"] for t in per_tile]) - origin[None, :2]
    rect_hi = np.asarray([t["xy_hi"] for t in per_tile]) - origin[None, :2]
    n_faces_raw = faces.shape[0]
    keep, n_dup = dedup_faces(tri)
    tri = tri[keep]
    mesh_info = dict(
        mesh_dir=str(Path(args.mesh_dir).expanduser()), splits=splits,
        variant=args.variant, n_tiles=len(per_tile),
        n_vertices=int(verts.shape[0]), n_faces_raw=int(n_faces_raw),
        n_duplicate_faces_dropped=int(n_dup), n_faces=int(tri.shape[0]),
        total_area_m2=float(face_areas(tri).sum()), per_tile=per_tile)
    del verts, faces
    seconds["mesh_load"] = time.perf_counter() - t0
    print(f"[mesh] {mesh_info['n_tiles']} tiles ({'+'.join(splits)}, "
          f"{args.variant}): {mesh_info['n_faces']} faces "
          f"({n_dup} slab-boundary duplicates dropped), "
          f"area {mesh_info['total_area_m2']:.0f} m^2; "
          f"{seconds['mesh_load']:.1f}s", flush=True)

    # ---- Frame reconciliation guard. ----
    mesh_lo, mesh_hi = rect_lo.min(axis=0), rect_hi.max(axis=0)
    grid_lo, grid_hi = np.asarray(grid.bbox_lo), np.asarray(grid.bbox_hi)
    inter = np.maximum(np.minimum(mesh_hi, grid_hi) - np.maximum(mesh_lo, grid_lo), 0.0)
    grid_area = float(np.prod(grid_hi - grid_lo))
    overlap = float(np.prod(inter)) / grid_area
    frame = dict(
        origin=origin.tolist(),
        mesh_bbox_xy=[mesh_lo.tolist(), mesh_hi.tolist()],
        grid_bbox_xy=[grid_lo.tolist(), grid_hi.tolist()],
        mesh_z_range=[float(tri[..., 2].min()), float(tri[..., 2].max())],
        overlap_fraction_of_grid=overlap)
    print(f"[frame] centered mesh xy bbox {np.round(mesh_lo, 1)}..{np.round(mesh_hi, 1)}"
          f" vs grid {np.round(grid_lo, 1)}..{np.round(grid_hi, 1)}: "
          f"overlap {overlap:.3f} of grid area", flush=True)
    if overlap < args.min_frame_overlap:
        raise SystemExit(
            f"frame reconciliation failed: mesh/grid bbox overlap {overlap:.3f} < "
            f"{args.min_frame_overlap} -- mesh and model do not share a frame?")

    # ---- ACCURACY: model -> reference. ----
    t0 = time.perf_counter()
    mn = MeshNearest(tri, k0=args.k0)
    accuracy, acc_fig = {}, {}
    for arm, (pts, _) in arms.items():
        support = in_any_rect(pts[:, :2], rect_lo, rect_hi)
        d, cert = mn.distance(pts[support])
        entry = dict(
            n_sampled=int(pts.shape[0]),
            excluded_no_reference_support=int((~support).sum()),
            excluded_fraction=float((~support).mean()),
            stats=dist_stats(d, ACC_THRESH),
            certificate=cert)
        if arm == "stitched":
            entry["terciles"] = tercile_report(d, nlpd_mod[support], ACC_THRESH)
        accuracy[arm] = entry
        acc_fig[arm] = dict(d=d, stats=entry["stats"],
                            terciles=entry.get("terciles"))
        print(f"[accuracy] {arm}: {d.shape[0]} in-support samples "
              f"(excl {entry['excluded_fraction']:.4f}); mean {entry['stats']['mean']:.3f} "
              f"RMS {entry['stats']['rms']:.3f} m; escalated "
              f"{cert['escalated_fraction']:.4f}", flush=True)
    seconds["accuracy"] = time.perf_counter() - t0

    # ---- COMPLETENESS: reference -> model. ----
    t0 = time.perf_counter()
    ref_pts, _ = sample_surface(tri, args.n_ref, rng(3))
    if occupancy is not None:
        dom = occupancy(ref_pts[:, :2])
        domain = dict(cell=args.domain_cell, min_count=args.domain_min_count,
                      kept_fraction=float(dom.mean()))
        ref_pts = ref_pts[dom]
    else:
        domain = dict(skipped=True)
    nlpd_ref = stitched_spatial_nlpd(grid, pair_tile, log_pi, st_s,
                                     ref_pts, args.chunk)
    # A surface sample outside every halo-dilated tile has zero gated density
    # (NLPD = inf): keep it, ranked least-confident, and report the count
    # (also keeps the output strict-JSON serializable).
    nonfinite = ~np.isfinite(nlpd_ref)
    if nonfinite.any():
        nlpd_ref[nonfinite] = nlpd_ref[~nonfinite].max() + 1.0
    spacing = float(np.median(
        cKDTree(arms["stitched"][1]).query(
            arms["stitched"][1][:50_000], k=2, workers=-1)[0][:, 1]))
    completeness = dict(domain_mask=domain,
                        n_surface_samples=int(ref_pts.shape[0]),
                        n_zero_gate_support=int(nonfinite.sum()),
                        dense_bank_size=args.n_dense,
                        model_sample_spacing_median=spacing)
    comp_fig = {}
    for arm, (_, dense) in arms.items():
        d = nn_distance(ref_pts, dense)
        entry = dict(stats=dist_stats(d, COMP_THRESH))
        if arm == "stitched":
            entry["terciles"] = tercile_report(d, nlpd_ref, COMP_THRESH)
        completeness[arm] = entry
        comp_fig[arm] = dict(d=d, stats=entry["stats"],
                             terciles=entry.get("terciles"))
        c = {t: entry["stats"][f"frac_le_{t:g}m"] for t in COMP_THRESH}
        print(f"[completeness] {arm}: mean {entry['stats']['mean']:.3f} m; "
              + "  ".join(f"@{t:g}m {v:.3f}" for t, v in c.items()), flush=True)
    print(f"[completeness] dense-bank median self-spacing {spacing:.3f} m "
          f"(resolution floor of the completeness fractions)", flush=True)
    seconds["completeness"] = time.perf_counter() - t0

    # ---- Figure + JSON. ----
    fig_path = REPO / "figures" / f"{name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(name, acc_fig, comp_fig, fig_path)

    seconds["total"] = time.perf_counter() - t_start
    out = dict(
        name=name,
        record=record["name"],
        config=vars(args),
        rng_streams=dict(stitched=0, crop=1, voronoi=2, mesh_surface=3,
                         dense_stitched=4, dense_crop=5, dense_voronoi=6),
        machinery=dict(n_pairs=int(pair_tile.shape[0]),
                       k_global=int(model["spatial_m"].shape[0]),
                       stitched_acceptance_rate=float(acc_rate)),
        frame=frame,
        mesh=mesh_info,
        accuracy=accuracy,
        completeness=completeness,
        seconds=seconds,
        versions=dict(jax=jax.__version__, numpy=np.__version__),
    )
    out_path = Path(args.record).parent / f"{name}.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"[done] {out_path.name}, {fig_path.name}; {seconds['total']:.1f}s")


if __name__ == "__main__":
    main()

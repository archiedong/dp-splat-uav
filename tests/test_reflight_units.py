"""Unit and adversarial tests for experiments/reflight.py.

Covered here:

  1. Track-membership survey split on a hand-built toy COLMAP text model:
     every label rule (survey / re-flight pool / discard), track image-id
     deduplication, name-sorted temporal hold-back, and the images.txt parser's
     immunity to long numeric POINTS2D lines.
  2. Tile ranking: deterministic descending order with the documented infinity
     conventions (+inf skipped tiles first in the uncertainty ranking, -inf
     no-pool tiles last in the oracle ranking) and index-ordered tie-breaking.
  3. The leakage crux: the reserved eval set is disjoint from every arm's
     granted set, and no eval point ever reaches a refit -- proven by capturing
     the exact point arrays passed to fit_one_tile for all arms.
  4. Non-vacuity of the within-selection metric: selected tiles retain eval
     mass under the reserved-eval protocol.
  5. Scoring symmetry: score_arm is a pure function of (model, selection) --
     identical models yield identical global metrics regardless of selection.
  6. Sanity on a fitted toy scene with a strong density gradient: the
     NLPD-targeted selection is deterministic and differs from the random
     controls (overlap reported), and the tile-NLPD vs point-count Spearman
     rho is reported (documented, no numeric target).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import spearmanr

from dp_splat import cavi
from dp_splat_uav import stitch, tiling

import reflight


# ---------------------------------------------------------------------------
# 1. Survey split on a hand-built COLMAP model
# ---------------------------------------------------------------------------

# 10 images; image ids deliberately NOT in name order so the temporal hold-back
# must come from the sorted names, not the ids. sorted(names)[4::5] with
# --holdback-every 5 holds back exactly {000005.jpg, 000010.jpg} = ids {3, 21}.
_IMAGES = {
    7: "000003.jpg", 2: "000001.jpg", 21: "000010.jpg", 5: "000002.jpg",
    3: "000005.jpg", 11: "000004.jpg", 13: "000006.jpg", 17: "000007.jpg",
    19: "000008.jpg", 23: "000009.jpg",
}
_HOLDBACK_IDS = {3, 21}
_SURVEY_IDS = set(_IMAGES) - _HOLDBACK_IDS

# Points: (xyz, rgb, track image ids, expected label). Labels: 0 initial
# (>= 2 survey), 1 pool (< 2 survey, >= 2 held-back), 2 discarded.
_POINTS = [
    ((0.0, 1.0, 2.0), (10, 20, 30), [2, 5], 0),          # two survey
    ((1.0, 2.0, 3.0), (40, 50, 60), [3, 21], 1),         # two held-back
    ((2.0, 3.0, 4.0), (70, 80, 90), [2, 3], 2),          # one of each
    ((3.0, 4.0, 5.0), (1, 2, 3), [5, 5, 5], 2),          # dup survey -> ns = 1
    ((4.0, 5.0, 6.0), (4, 5, 6), [2, 5, 3, 21], 0),      # survey wins
    ((5.0, 6.0, 7.0), (7, 8, 9), [7, 3, 21], 1),         # 1 survey, 2 held-back
    ((6.0, 7.0, 8.0), (11, 12, 13), [7, 11, 13], 0),     # three survey
    ((7.0, 8.0, 9.0), (14, 15, 16), [3, 3, 21, 21], 1),  # dup held-back -> nh = 2
]


@pytest.fixture()
def toy_colmap(tmp_path):
    lines = ["# Image list with two lines of data per image:",
             "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for iid, name in _IMAGES.items():
        lines.append(f"{iid} 1.0 0.0 0.0 0.0 0.1 0.2 0.3 1 {name}")
        # Adversarial POINTS2D line: >= 10 numeric fields; must not parse as a
        # header line (the parser keys on the trailing image-name suffix).
        lines.append(" ".join(f"{v}.5 {v}.25 -1" for v in range(4)))
    (tmp_path / "images.txt").write_text("\n".join(lines) + "\n")

    lines = ["# 3D point list with one line of data per point:"]
    for i, (xyz, rgb, track, _) in enumerate(_POINTS):
        pairs = " ".join(f"{iid} {j}" for j, iid in enumerate(track))
        lines.append(f"{i} {xyz[0]} {xyz[1]} {xyz[2]} "
                     f"{rgb[0]} {rgb[1]} {rgb[2]} 0.5 {pairs}")
    (tmp_path / "points3D.txt").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_load_image_names_parses_headers_only(toy_colmap):
    id2name = reflight.load_image_names(toy_colmap)
    assert id2name == _IMAGES


def test_split_points_labels_match_hand_assignment(toy_colmap):
    id2name = reflight.load_image_names(toy_colmap)
    xyz, rgb, label, info = reflight.split_points(toy_colmap, id2name,
                                                  every=5, min_track=2)
    expected = np.array([p[3] for p in _POINTS])
    np.testing.assert_array_equal(label, expected)
    np.testing.assert_allclose(xyz, np.array([p[0] for p in _POINTS]))
    np.testing.assert_allclose(rgb, np.array([p[1] for p in _POINTS]) / 255.0)
    assert info["n_images"] == 10
    assert info["n_holdback_images"] == 2
    assert info["n_survey_images"] == 8
    assert info["n_initial"] == int((expected == 0).sum())
    assert info["n_pool"] == int((expected == 1).sum())
    assert info["n_discarded"] == int((expected == 2).sum())


def test_split_points_is_deterministic(toy_colmap):
    id2name = reflight.load_image_names(toy_colmap)
    out1 = reflight.split_points(toy_colmap, id2name, every=5, min_track=2)
    out2 = reflight.split_points(toy_colmap, id2name, every=5, min_track=2)
    for a, b in zip(out1[:3], out2[:3]):
        np.testing.assert_array_equal(a, b)
    assert out1[3] == out2[3]


def test_holdback_set_is_temporal_interleaving(toy_colmap):
    # A point seen by exactly {000005.jpg, 000010.jpg} lands in the pool: those
    # are sorted_names[4::5], regardless of their (shuffled) integer ids.
    id2name = reflight.load_image_names(toy_colmap)
    _, _, label, _ = reflight.split_points(toy_colmap, id2name,
                                           every=5, min_track=2)
    assert label[1] == 1  # track [3, 21] = the two held-back names


# ---------------------------------------------------------------------------
# 2. Tile ranking determinism and infinity conventions
# ---------------------------------------------------------------------------


def test_rank_tiles_desc_conventions_and_determinism():
    # Skipped tiles (+inf mean NLPD) must head the uncertainty ranking; ties
    # resolve by tile index; repeated calls agree exactly.
    nlpd = np.array([1.0, np.inf, 0.5, 1.0, np.inf, 2.0])
    r1 = reflight.rank_tiles_desc(nlpd)
    r2 = reflight.rank_tiles_desc(nlpd)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(r1, [1, 4, 5, 0, 3, 2])

    # Oracle convention: tiles with no pool mass (-inf) rank last.
    oracle = np.array([-np.inf, 3.0, -np.inf, 1.0])
    np.testing.assert_array_equal(reflight.rank_tiles_desc(oracle), [1, 3, 0, 2])


def test_core_tile_of_partitions_and_clips():
    rng = np.random.default_rng(0)
    s = np.column_stack([rng.uniform(0, 40, 4000), rng.uniform(0, 20, 4000),
                         rng.uniform(0, 3, 4000)])
    grid = tiling.make_grid(s, halo=1.0, target_points=500.0)
    t_of = reflight.core_tile_of(grid, s)
    # Oracle: a point's core tile is the unique tile whose core rectangle
    # contains it (up to shared boundaries, where either neighbor is valid).
    inside = (s[:, None, :2] >= grid.core_lo[None]) & (s[:, None, :2] <= grid.core_hi[None])
    ok = inside.all(axis=2)[np.arange(s.shape[0]), t_of]
    assert ok.all()
    # Out-of-box points clip to an edge tile deterministically.
    far = np.array([[-5.0, -5.0, 0.0], [100.0, 100.0, 0.0]])
    t_far = reflight.core_tile_of(grid, far)
    np.testing.assert_array_equal(t_far, reflight.core_tile_of(grid, far))
    assert (t_far >= 0).all() and (t_far < grid.core_lo.shape[0]).all()


# ---------------------------------------------------------------------------
# Toy fitted scene shared by the leakage, symmetry, and sanity tests
# ---------------------------------------------------------------------------

_T_TOY = 8


def _toy_args(**over):
    base = dict(seed=0, alpha=4.0, truncation=_T_TOY, halo=1.0,
                max_iters=40, tol=1e-6, n_min=1.0, min_tile_points=30,
                svi_threshold=1e9, svi_batch=1024, svi_epochs=1.0,
                svi_eval_every=10, svi_eval_points=1000, chunk=8192,
                top_frac=0.34, seeds=4, eval_frac=0.25,
                entropy_samples=2000, model_samples=4000, nn_thresh=0.5)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture(scope="module")
def toy_scene():
    """2x3 tile scene with a strong left-to-right density gradient.

    Six ground-plane blocks of 12 x 10 m; per-block point counts fall from
    1400 to 90, so the sparse blocks should fit worse (higher mean NLPD) --
    the correlation the sanity check documents.
    """
    rng = np.random.default_rng(20260717)
    counts = [1400, 900, 500, 260, 140, 90]
    s_parts, c_parts = [], []
    for b, n in enumerate(counts):
        bx, by = b % 3, b // 3
        centers = np.array([[bx * 12 + 4, by * 10 + 3], [bx * 12 + 8, by * 10 + 7]])
        comp = rng.integers(0, 2, n)
        xy = centers[comp] + rng.standard_normal((n, 2)) * 1.1
        z = 0.4 * comp + rng.standard_normal(n) * 0.15
        s_parts.append(np.column_stack([xy, z]))
        c_parts.append(np.clip(
            np.array([[0.8, 0.3, 0.2], [0.2, 0.4, 0.8]])[comp]
            + rng.standard_normal((n, 3)) * 0.05, 0.0, 1.0))
    s = np.vstack(s_parts)
    c = np.vstack(c_parts)

    args = _toy_args()
    grid = tiling.make_grid(s, halo=args.halo, target_points=600.0)
    asn = tiling.assign_points(grid, s, alpha_global=args.alpha)
    cfg = cavi.Config(weight_prior="dp", T=args.truncation, alpha=args.alpha,
                      max_iters=args.max_iters, tol=args.tol)
    sp = cavi.default_niw_prior(jnp.asarray(s), cfg.T, cfg.kappa0, cfg.nu0_offset)
    cp = cavi.default_niw_prior(jnp.asarray(c), cfg.T, cfg.kappa0, cfg.nu0_offset)
    tiles = reflight.initial_fit(s, c, grid, asn, cfg, sp, cp, args)
    preds, _, kept = reflight.build_predictives(tiles, asn, args, sp)

    # Pool: fresh draws from the same blocks (plus deterministic z-tags used by
    # the leakage test to identify pool points inside captured fit inputs).
    pool_parts = []
    for b, n in enumerate(counts):
        bx, by = b % 3, b // 3
        centers = np.array([[bx * 12 + 4, by * 10 + 3], [bx * 12 + 8, by * 10 + 7]])
        comp = rng.integers(0, 2, max(40, n // 4))
        xy = centers[comp] + rng.standard_normal((comp.size, 2)) * 1.1
        pool_parts.append(np.column_stack([xy, np.zeros(comp.size)]))
    s_pool = np.vstack(pool_parts)
    keep = np.all((s_pool[:, :2] >= grid.bbox_lo - grid.halo)
                  & (s_pool[:, :2] <= grid.bbox_hi + grid.halo), axis=1)
    s_pool = s_pool[keep]
    s_pool[:, 2] = 1000.0 + np.arange(s_pool.shape[0])  # unique per-point tag
    c_pool = np.tile([0.5, 0.5, 0.5], (s_pool.shape[0], 1))

    return dict(s=s, c=c, grid=grid, asn=asn, cfg=cfg, sp=sp, cp=cp,
                tiles=tiles, preds=preds, kept=kept, args=args,
                s_pool=s_pool, c_pool=c_pool)


def _selections_and_grants(sc):
    """Reserved eval split, per-arm selections, and grants (main, step 4)."""
    args, grid = sc["args"], sc["grid"]
    n_tiles = grid.core_lo.shape[0]
    s_pool = sc["s_pool"]
    n_eval = max(1, int(round(args.eval_frac * s_pool.shape[0])))
    eval_idx = np.sort(np.random.default_rng(args.seed + 424242).choice(
        s_pool.shape[0], size=n_eval, replace=False))
    grantable = np.ones(s_pool.shape[0], dtype=bool)
    grantable[eval_idx] = False

    tile_nlpd = _tile_mean_nlpd(sc)
    k_sel = max(1, int(round(args.top_frac * n_tiles)))
    sels = {"targeted": reflight.rank_tiles_desc(tile_nlpd)[:k_sel],
            "oracle": reflight.rank_tiles_desc(-tile_nlpd)[:k_sel]}
    for i in range(args.seeds):
        sels[f"random_{i}"] = np.random.default_rng(args.seed + 100 + i).choice(
            n_tiles, size=k_sel, replace=False)
    grants = {name: reflight.grant_pool(grid, sel, s_pool, grantable)
              for name, sel in sels.items()}
    return eval_idx, grantable, sels, grants, tile_nlpd


def _tile_mean_nlpd(sc):
    grid, args = sc["grid"], sc["args"]
    nlpd = reflight.stitched_nlpd(sc["preds"], grid, jnp.asarray(sc["s"]),
                                  jnp.asarray(sc["c"]), args.chunk)
    t_of = reflight.core_tile_of(grid, sc["s"])
    n_tiles = grid.core_lo.shape[0]
    out = np.full(n_tiles, np.inf)
    for t in range(n_tiles):
        m = t_of == t
        if m.any() and sc["tiles"][t]["state"] is not None:
            out[t] = float(nlpd[m].mean())
    return out


# ---------------------------------------------------------------------------
# 3. Leakage crux: eval disjoint from every grant; no eval point in any refit
# ---------------------------------------------------------------------------


def test_eval_set_disjoint_from_all_grants(toy_scene):
    eval_idx, grantable, _, grants, _ = _selections_and_grants(toy_scene)
    eval_set = set(eval_idx.tolist())
    assert not grantable[eval_idx].any()
    for name, g in grants.items():
        assert eval_set.isdisjoint(g.tolist()), f"arm {name} was granted eval points"


def test_no_eval_point_enters_any_refit(toy_scene, monkeypatch):
    sc = toy_scene
    args, grid = sc["args"], sc["grid"]
    eval_idx, _, sels, grants, _ = _selections_and_grants(sc)
    eval_tags = set(sc["s_pool"][eval_idx, 2].tolist())

    captured = []  # (tile, spatial array) for every attempted refit

    def fake_fit(t, xs_t, xc_t, w_t, cfg_t, sp, cp, a):
        captured.append((t, np.asarray(xs_t)))
        return None, np.zeros(cfg_t.T), dict(tile=t, n_points=int(xs_t.shape[0]),
                                             khat=0, path="captured", seconds=0.0)

    monkeypatch.setattr(reflight, "fit_one_tile", fake_fit)

    pool_lo = grid.core_lo - grid.halo
    pool_hi = grid.core_hi + grid.halo
    own = np.zeros(sc["s_pool"].shape[0], dtype=np.int64)
    for t in range(grid.core_lo.shape[0]):
        own += np.all((sc["s_pool"][:, :2] >= pool_lo[t])
                      & (sc["s_pool"][:, :2] <= pool_hi[t]), axis=1)
    assert (own > 0).all()

    base = dict(s=sc["s"], c=sc["c"], s_pool=sc["s_pool"], c_pool=sc["c_pool"],
                pool_weight=1.0 / own, preds=list(sc["preds"]),
                kept_counts=list(sc["kept"]))
    ctx = (args, grid, sc["asn"], sc["cfg"], sc["sp"], sc["cp"])

    cache = {}
    for name, sel in sels.items():
        preds_a, kept_a, recs = reflight.refit_arm(sel, grants[name], base, ctx, cache)
        assert len(recs) == len(sel)
        assert len(preds_a) == grid.core_lo.shape[0]

    assert captured, "no refit was exercised"
    granted_all = set()
    for t, xs_t in captured:
        tags = xs_t[xs_t[:, 2] >= 1000.0, 2]
        granted_all.update(tags.tolist())
        assert eval_tags.isdisjoint(tags.tolist()), \
            f"eval point leaked into the refit of tile {t}"
        # Every granted point in this refit lies in the tile's dilated rect.
        lo = grid.core_lo[t] - grid.halo
        hi = grid.core_hi[t] + grid.halo
        in_rect = np.all((xs_t[:, :2] >= lo) & (xs_t[:, :2] <= hi), axis=1)
        assert in_rect[xs_t[:, 2] >= 1000.0].all()
    # And the union of refit-consumed pool points is exactly the union of grants.
    grant_tags = set()
    for g in grants.values():
        grant_tags.update(sc["s_pool"][g, 2].tolist())
    assert granted_all == grant_tags


def test_selected_tiles_retain_eval_mass(toy_scene):
    # The reserved-eval protocol keeps eval points inside selected tiles, so
    # the within-selection delta is measurable (it was vacuously empty when
    # the eval set was the complement of the grants).
    sc = toy_scene
    eval_idx, _, sels, _, _ = _selections_and_grants(sc)
    eval_tile = reflight.core_tile_of(sc["grid"], sc["s_pool"][eval_idx])
    for name, sel in sels.items():
        assert np.isin(eval_tile, sel).any(), \
            f"arm {name}: no eval mass in its selected tiles"


# ---------------------------------------------------------------------------
# 5. Scoring symmetry: identical models -> identical global metrics
# ---------------------------------------------------------------------------


def test_score_arm_depends_only_on_model_not_selection(toy_scene):
    sc = toy_scene
    args, grid = sc["args"], sc["grid"]
    eval_idx, _, sels, _, _ = _selections_and_grants(sc)
    s_pool, c_pool = sc["s_pool"], sc["c_pool"]
    xs_e = jnp.asarray(np.column_stack([s_pool[eval_idx, :2],
                                        np.zeros(eval_idx.size)]))
    xc_e = jnp.asarray(c_pool[eval_idx])

    b_nlpd = reflight.stitched_nlpd(sc["preds"], grid, xs_e, xc_e, args.chunk)
    base = dict(s=sc["s"], c=sc["c"], preds=sc["preds"], kept_counts=sc["kept"],
                xs_eval=xs_e, xc_eval=xc_e,
                eval_tile=reflight.core_tile_of(grid, np.asarray(xs_e)),
                baseline=dict(nlpd=b_nlpd, mean_nlpd=float(b_nlpd.mean()),
                              coverage={f"{q:.2f}": 0.0 for q in reflight.LEVELS},
                              completeness=0.0))
    ctx = (args, grid, sc["asn"], sc["cfg"], sc["sp"], sc["cp"])

    r1 = reflight.score_arm(sc["preds"], sc["kept"], sels["targeted"], base, ctx)
    r2 = reflight.score_arm(sc["preds"], sc["kept"], sels["random_0"], base, ctx)
    # Same model: every selection-independent metric agrees exactly.
    assert r1["mean_nlpd"] == r2["mean_nlpd"]
    assert r1["delta_mean_nlpd"] == r2["delta_mean_nlpd"]
    assert r1["coverage"] == r2["coverage"]
    assert r1["completeness"] == r2["completeness"]
    # The unchanged model shows zero global gain (baseline consistency).
    np.testing.assert_allclose(r1["delta_mean_nlpd"], 0.0, atol=1e-12)
    # Selection-dependent slices differ only through the selection.
    assert r1["selected_tiles"] != r2["selected_tiles"]


def test_score_arm_is_deterministic(toy_scene):
    sc = toy_scene
    args, grid = sc["args"], sc["grid"]
    eval_idx, _, sels, _, _ = _selections_and_grants(sc)
    xs_e = jnp.asarray(np.column_stack([sc["s_pool"][eval_idx, :2],
                                        np.zeros(eval_idx.size)]))
    xc_e = jnp.asarray(sc["c_pool"][eval_idx])
    b_nlpd = reflight.stitched_nlpd(sc["preds"], grid, xs_e, xc_e, args.chunk)
    base = dict(s=sc["s"], c=sc["c"], preds=sc["preds"], kept_counts=sc["kept"],
                xs_eval=xs_e, xc_eval=xc_e,
                eval_tile=reflight.core_tile_of(grid, np.asarray(xs_e)),
                baseline=dict(nlpd=b_nlpd, mean_nlpd=float(b_nlpd.mean()),
                              coverage={f"{q:.2f}": 0.0 for q in reflight.LEVELS},
                              completeness=0.0))
    ctx = (args, grid, sc["asn"], sc["cfg"], sc["sp"], sc["cp"])
    r1 = reflight.score_arm(sc["preds"], sc["kept"], sels["targeted"], base, ctx)
    r2 = reflight.score_arm(sc["preds"], sc["kept"], sels["targeted"], base, ctx)
    assert r1 == r2  # includes the seeded completeness draw


# ---------------------------------------------------------------------------
# 6. Sanity: targeted vs random overlap; NLPD-vs-sparsity correlation
# ---------------------------------------------------------------------------


def test_targeted_selection_differs_from_random_and_reports_overlap(toy_scene):
    sc = toy_scene
    _, _, sels, _, tile_nlpd = _selections_and_grants(sc)
    targeted = set(sels["targeted"].tolist())
    overlaps = []
    for i in range(sc["args"].seeds):
        rnd = set(sels[f"random_{i}"].tolist())
        overlaps.append(len(targeted & rnd) / len(targeted))
    print(f"[sanity] targeted {sorted(targeted)}; "
          f"random overlap fractions {overlaps}")
    # The targeted selection is a deterministic function of the fit; at least
    # one random replicate must differ from it (all-equal would mean the
    # "targeted" arm degenerates to the control).
    assert any(set(sels[f"random_{i}"].tolist()) != targeted
               for i in range(sc["args"].seeds))
    # Determinism: recomputing the ranking from the same fit reproduces it.
    again = reflight.rank_tiles_desc(_tile_mean_nlpd(sc))[:len(targeted)]
    assert set(again.tolist()) == targeted


def test_tile_nlpd_tracks_point_sparsity(toy_scene):
    # Documented sanity: sparser tiles should fit worse. Spearman rho between
    # per-tile mean NLPD and per-tile core point count is reported; the only
    # hard assertions are that the ranking is complete and finite for fitted
    # tiles and the correlation is computable.
    sc = toy_scene
    tile_nlpd = _tile_mean_nlpd(sc)
    t_of = reflight.core_tile_of(sc["grid"], sc["s"])
    n_pts = np.bincount(t_of, minlength=sc["grid"].core_lo.shape[0])
    fitted = np.isfinite(tile_nlpd)
    assert fitted.sum() >= 4
    rho = float(spearmanr(tile_nlpd[fitted], n_pts[fitted]).correlation)
    print(f"[sanity] per-tile n {n_pts.tolist()}; mean NLPD "
          f"{np.round(tile_nlpd, 3).tolist()}; spearman(nlpd, n) rho = {rho:.3f}")
    assert np.isfinite(rho)
    # Direction (documented, generous margin): higher NLPD on sparser tiles.
    assert rho < 0.5

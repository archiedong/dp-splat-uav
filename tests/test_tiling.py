"""Tiling and partition-of-unity stitching tests.

Every quantity with new math (gates, ownership weights, stitched density) is checked
against an independent NumPy oracle written from the definitions, not from the
implementation.
"""

import math

import numpy as np
import jax.numpy as jnp
import pytest
from scipy.special import gammaln

from dp_splat_uav import stitch, tiling

RNG = np.random.default_rng(20260715)

# Scene: 100 x 50 ground plane, z in [0, 10]; target forces a multi-tile grid.
N = 20000
TARGET = 2000.0
HALO = 2.0
ALPHA_GLOBAL = 5.0


@pytest.fixture(scope="module")
def cloud():
    s = np.column_stack(
        [
            RNG.uniform(0.0, 100.0, N),
            RNG.uniform(0.0, 50.0, N),
            RNG.uniform(0.0, 10.0, N),
        ]
    )
    return s


@pytest.fixture(scope="module")
def grid(cloud):
    return tiling.make_grid(cloud, halo=HALO, target_points=TARGET)


@pytest.fixture(scope="module")
def assignment(cloud, grid):
    return tiling.assign_points(grid, cloud, alpha_global=ALPHA_GLOBAL)


# ---------------------------------------------------------------------------
# Grid geometry and Morton ordering
# ---------------------------------------------------------------------------


def test_grid_shape_meets_target(cloud, grid):
    nx, ny = grid.shape
    assert nx * ny >= math.ceil(N / TARGET)
    assert N / (nx * ny) <= TARGET


def test_cores_partition_bbox(grid):
    # Core x-edges and y-edges tile the bounding box without gaps or overlap.
    nx, ny = grid.shape
    assert grid.core_lo.shape == (nx * ny, 2)
    for t in range(grid.core_lo.shape[0]):
        assert np.all(grid.core_lo[t] < grid.core_hi[t])
    assert np.isclose(grid.core_lo[:, 0].min(), grid.bbox_lo[0])
    assert np.isclose(grid.core_hi[:, 0].max(), grid.bbox_hi[0])
    # Total core area equals bbox area (exact partition).
    areas = np.prod(grid.core_hi - grid.core_lo, axis=1)
    assert np.isclose(areas.sum(), np.prod(grid.bbox_hi - grid.bbox_lo))


def test_morton_known_values():
    assert int(tiling.morton_key(0, 0)) == 0
    assert int(tiling.morton_key(1, 0)) == 1
    assert int(tiling.morton_key(0, 1)) == 2
    assert int(tiling.morton_key(1, 1)) == 3
    # (3, 5): x = 011 in even bits, y = 101 in odd bits -> 100111b = 39.
    assert int(tiling.morton_key(3, 5)) == 39


def test_tiles_in_morton_order(grid):
    m = np.asarray(grid.morton, dtype=np.uint64)
    assert np.all(m[1:] > m[:-1])
    assert np.array_equal(m, tiling.morton_key(grid.index[:, 0], grid.index[:, 1]))


# ---------------------------------------------------------------------------
# Point assignment and ownership weights
# ---------------------------------------------------------------------------


def test_every_point_in_some_tile(assignment):
    covered = np.zeros(N, dtype=bool)
    for idx in assignment.indices:
        covered[idx] = True
    assert covered.all()
    assert np.all(assignment.ownership >= 1)


def test_ownership_oracle(cloud, grid, assignment):
    # Independent scalar-loop count of halo-dilated rectangles containing each point.
    sub = RNG.choice(N, size=300, replace=False)
    for n in sub:
        x, y = cloud[n, 0], cloud[n, 1]
        k = 0
        for t in range(grid.core_lo.shape[0]):
            if (
                grid.core_lo[t, 0] - HALO <= x <= grid.core_hi[t, 0] + HALO
                and grid.core_lo[t, 1] - HALO <= y <= grid.core_hi[t, 1] + HALO
            ):
                k += 1
        assert assignment.ownership[n] == k
        assert assignment.weight[n] == 1.0 / k


def test_core_interior_points_have_weight_one(grid):
    # Tile core centers are farther than HALO from every foreign core boundary.
    centers = (grid.core_lo + grid.core_hi) / 2.0
    a = tiling.assign_points(grid, centers)
    assert np.all(a.ownership == 1)
    assert np.all(a.weight == 1.0)


def test_halo_points_weight_one_over_k(grid):
    nx, ny = grid.shape
    assert nx >= 2 and ny >= 2, "scene/target must produce a 2D grid for this test"
    # Interior vertical core edge, mid-height: inside exactly 2 dilated tiles.
    x_edge = grid.core_hi[:, 0][grid.core_hi[:, 0] < grid.bbox_hi[0]].min()
    y_edge = grid.core_hi[:, 1][grid.core_hi[:, 1] < grid.bbox_hi[1]].min()
    y_mid = (grid.bbox_lo[1] + y_edge) / 2.0
    edge_pt = np.array([[x_edge, y_mid]])
    a2 = tiling.assign_points(grid, edge_pt)
    assert a2.ownership[0] == 2 and a2.weight[0] == 0.5
    # Interior 4-tile corner: inside exactly 4 dilated tiles.
    corner_pt = np.array([[x_edge, y_edge]])
    a4 = tiling.assign_points(grid, corner_pt)
    assert a4.ownership[0] == 4 and a4.weight[0] == 0.25


def test_alpha_sums_to_global(assignment):
    assert np.isclose(assignment.alpha.sum(), ALPHA_GLOBAL, rtol=0, atol=1e-12)
    assert np.all(assignment.alpha > 0)


def test_tile_counts_near_target(grid, assignment):
    # Uniform cloud: core counts concentrate at N / (nx * ny) <= TARGET; halo dilation
    # inflates counts by the dilated-to-core area ratio, well under 2x here.
    counts = np.array([len(idx) for idx in assignment.indices])
    mean_core = N / (grid.shape[0] * grid.shape[1])
    assert np.all(counts >= 0.5 * mean_core)
    assert np.all(counts <= 2.0 * TARGET)


# ---------------------------------------------------------------------------
# Partition-of-unity gates
# ---------------------------------------------------------------------------


def _gate_oracle(grid, xy):
    """Scalar NumPy oracle for the gates, straight from the definition: per-dim ramp
    (1 on the core, linear to 0 across the halo band), product over dims, normalize."""
    T = grid.core_lo.shape[0]
    out = np.zeros((xy.shape[0], T))
    for n in range(xy.shape[0]):
        raw = np.zeros(T)
        for t in range(T):
            p = 1.0
            for d in range(2):
                lo, hi, h = grid.core_lo[t, d], grid.core_hi[t, d], grid.halo
                v = xy[n, d]
                if lo <= v <= hi:
                    r = 1.0
                elif lo - h < v < lo:
                    r = (v - (lo - h)) / h
                elif hi < v < hi + h:
                    r = ((hi + h) - v) / h
                else:
                    r = 0.0
                p *= r
            raw[t] = p
        out[n] = raw / raw.sum()
    return out


def test_gates_match_oracle(grid):
    xy = np.column_stack(
        [
            RNG.uniform(grid.bbox_lo[0], grid.bbox_hi[0], 200),
            RNG.uniform(grid.bbox_lo[1], grid.bbox_hi[1], 200),
        ]
    )
    g = np.asarray(stitch.gate_weights(grid, xy))
    np.testing.assert_allclose(g, _gate_oracle(grid, xy), atol=1e-12)


def test_gates_sum_to_one_dense(grid):
    # Dense grid including the bbox corners, edges, and interior tile-corner lines.
    gx = np.linspace(grid.bbox_lo[0], grid.bbox_hi[0], 101)
    gy = np.linspace(grid.bbox_lo[1], grid.bbox_hi[1], 101)
    xx, yy = np.meshgrid(gx, gy)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    # Add exact interior 4-tile corners.
    ux = np.unique(grid.core_hi[:, 0])[:-1]
    uy = np.unique(grid.core_hi[:, 1])[:-1]
    if ux.size and uy.size:
        cx, cy = np.meshgrid(ux, uy)
        pts = np.vstack([pts, np.column_stack([cx.ravel(), cy.ravel()])])
    g = np.asarray(stitch.gate_weights(grid, pts))
    assert np.all(g >= 0.0)
    np.testing.assert_allclose(g.sum(axis=1), 1.0, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Stitched predictive
# ---------------------------------------------------------------------------


def _student_logpdf_np(x, loc, scale, dof):
    """NumPy oracle: multivariate Student-t log density from the standard formula."""
    D = loc.shape[0]
    diff = x - loc
    quad = diff @ np.linalg.solve(scale, diff)
    _, logdet = np.linalg.slogdet(scale)
    return (
        gammaln((dof + D) / 2.0)
        - gammaln(dof / 2.0)
        - 0.5 * D * np.log(dof * np.pi)
        - 0.5 * logdet
        - 0.5 * (dof + D) * np.log1p(quad / dof)
    )


def _random_predictive(K, Ds, Dc):
    def spd(D):
        A = RNG.normal(size=(D, D))
        return A @ A.T + D * np.eye(D)

    pi = RNG.dirichlet(np.ones(K))
    return stitch.TilePredictive(
        log_pi=jnp.log(jnp.asarray(pi)),
        spatial=stitch.StudentT(
            loc=jnp.asarray(RNG.normal(size=(K, Ds)) + 5.0),
            scale=jnp.asarray(np.stack([spd(Ds) for _ in range(K)])),
            dof=jnp.asarray(RNG.uniform(4.0, 10.0, K)),
        ),
        color=stitch.StudentT(
            loc=jnp.asarray(RNG.normal(size=(K, Dc))),
            scale=jnp.asarray(np.stack([spd(Dc) for _ in range(K)])),
            dof=jnp.asarray(RNG.uniform(4.0, 10.0, K)),
        ),
    )


def _mixture_logpdf_np(tp, s, c):
    log_pi = np.asarray(tp.log_pi)
    out = np.zeros(s.shape[0])
    for n in range(s.shape[0]):
        terms = [
            log_pi[k]
            + _student_logpdf_np(
                s[n], np.asarray(tp.spatial.loc[k]), np.asarray(tp.spatial.scale[k]),
                float(tp.spatial.dof[k]),
            )
            + _student_logpdf_np(
                c[n], np.asarray(tp.color.loc[k]), np.asarray(tp.color.scale[k]),
                float(tp.color.dof[k]),
            )
            for k in range(log_pi.shape[0])
        ]
        m = max(terms)
        out[n] = m + np.log(sum(np.exp(t - m) for t in terms))
    return out


def _one_tile_grid():
    return tiling.TileGrid(
        core_lo=np.array([[0.0, 0.0]]),
        core_hi=np.array([[10.0, 10.0]]),
        halo=1.0,
        bbox_lo=np.array([0.0, 0.0]),
        bbox_hi=np.array([10.0, 10.0]),
        shape=(1, 1),
        index=np.array([[0, 0]]),
        morton=tiling.morton_key(np.array([0]), np.array([0])),
    )


def _two_tile_grid():
    return tiling.TileGrid(
        core_lo=np.array([[0.0, 0.0], [5.0, 0.0]]),
        core_hi=np.array([[5.0, 10.0], [10.0, 10.0]]),
        halo=1.0,
        bbox_lo=np.array([0.0, 0.0]),
        bbox_hi=np.array([10.0, 10.0]),
        shape=(2, 1),
        index=np.array([[0, 0], [1, 0]]),
        morton=tiling.morton_key(np.array([0, 1]), np.array([0, 0])),
    )


def test_stitched_single_tile_matches_mixture_oracle():
    tp = _random_predictive(K=3, Ds=3, Dc=3)
    s = np.column_stack([RNG.uniform(0, 10, 50), RNG.uniform(0, 10, 50),
                         RNG.normal(size=50)])
    c = RNG.normal(size=(50, 3))
    got = np.asarray(stitch.stitched_logpdf([tp], _one_tile_grid(), jnp.asarray(s),
                                            jnp.asarray(c)))
    np.testing.assert_allclose(got, _mixture_logpdf_np(tp, s, c), rtol=1e-10)


def test_stitched_identical_tiles_reduce_to_one():
    # With p_1 = p_2, sum_t g_t p_t = p regardless of the gates: exercises the
    # gate/log-density combination across a real seam, including g = 0 regions.
    tp = _random_predictive(K=2, Ds=3, Dc=3)
    s = np.column_stack([RNG.uniform(0, 10, 80), RNG.uniform(0, 10, 80),
                         RNG.normal(size=80)])
    c = RNG.normal(size=(80, 3))
    got = np.asarray(
        stitch.stitched_logpdf([tp, tp], _two_tile_grid(), jnp.asarray(s), jnp.asarray(c))
    )
    np.testing.assert_allclose(got, _mixture_logpdf_np(tp, s, c), rtol=1e-10)


def test_tile_predictive_empty_state_yields_empty_mixture():
    """A zero-component tile must contribute nothing to any predictive bank:
    stick-breaking E[pi] on empty sticks would otherwise put mass 1 on the
    remainder with no matching component."""
    import jax.numpy as jnp
    from dp_splat import cavi, niw as dniw, priors as dpriors
    from dp_splat_uav import stitch

    empty_niw = dniw.NIW(m=jnp.zeros((0, 3)), kappa=jnp.zeros((0,)),
                         Psi=jnp.zeros((0, 3, 3)), nu=jnp.zeros((0,)))
    empty_sticks = dpriors.StickBreakingPosterior(jnp.zeros((0,)), jnp.zeros((0,)))
    state = cavi.State(empty_niw, empty_niw, empty_niw, empty_niw, empty_sticks, None)
    cfg = cavi.Config(weight_prior="dp", T=0, alpha=1.0)
    tp = stitch.tile_predictive(state, cfg)
    assert tp.log_pi.shape == (0,)
    assert tp.spatial.loc.shape[0] == 0 and tp.color.loc.shape[0] == 0

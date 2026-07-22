"""Oracle tests (v) and (vi): stitched-density mass and two-tile end-to-end merge.

(v)  When every tile carries the same predictive p — the post-merge contract:
     tile predictives agree wherever gates overlap — the stitched density is
     sum_t g_t(s) p(x) = p(x) because sum_t g_t(s) = 1 across the scene, so it
     must integrate to 1. Verified two independent ways: panelwise
     Gauss-Legendre quadrature (panels split at the gate kink lines, where the
     integrand is only piecewise smooth) and importance-sampling Monte Carlo
     from an overdispersed proposal, plus a 1e-12 partition-of-unity check of
     the gates on grids concentrated in the halo bands.

(vi) Two tiles fit independently (shared global NIW priors, halo
     down-weighting, alpha_t from the point assignment), then matched and
     merged, must reproduce a pooled single fit on a 3-cluster scene with one
     cluster straddling the tile boundary: exactly the straddling pair is
     accepted, the merged component matches the pooled fit's straddling
     component, and the post-merge stitched predictive scores held-out data
     like the pooled predictive.
"""

import dataclasses

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaln

from dp_splat import cavi
from dp_splat import niw as dniw
from dp_splat import predictive
from dp_splat import priors as dpriors
from dp_splat_uav import merge, stitch, tiling, weighted


def _two_tile_grid(bbox_lo, bbox_hi, seam_x, halo):
    """Manual 2x1 tile grid with a vertical seam at x = seam_x."""
    bbox_lo = np.asarray(bbox_lo, dtype=np.float64)
    bbox_hi = np.asarray(bbox_hi, dtype=np.float64)
    return tiling.TileGrid(
        core_lo=np.array([[bbox_lo[0], bbox_lo[1]], [seam_x, bbox_lo[1]]]),
        core_hi=np.array([[seam_x, bbox_hi[1]], [bbox_hi[0], bbox_hi[1]]]),
        halo=float(halo),
        bbox_lo=bbox_lo,
        bbox_hi=bbox_hi,
        shape=(2, 1),
        index=np.array([[0, 0], [1, 0]]),
        morton=tiling.morton_key(np.array([0, 1]), np.array([0, 0])),
    )


# ---------------------------------------------------------------------------
# (v) stitched density integrates to 1 (2-tile toy scene, Ds = 2, Dc = 1)
# ---------------------------------------------------------------------------

# Scene [0,10] x [0,6], seam x = 5, halo 1 => gate kink lines at x in {4, 5, 6}.
_MASS_GRID = _two_tile_grid((0.0, 0.0), (10.0, 6.0), 5.0, 1.0)

# Component placement for the mass tests: every location is >= 2.5 length units
# (>= 6 Student-t scale units) from each bounding-box face and all dof >= 25,
# so the predictive mass outside the integration domain is < 1e-5 — far below
# the 1e-3 assertion. (The gates are undefined outside the halo-dilated scene,
# so the integral necessarily runs over the bounding box only.)
_MASS_PI = np.array([0.3, 0.4, 0.3])
_MASS_S_LOC = np.array([[2.5, 3.0], [5.0, 3.0], [7.5, 3.0]])
_MASS_S_SCALE = np.array(
    [
        [[0.16, 0.04], [0.04, 0.09]],
        [[0.09, -0.02], [-0.02, 0.16]],
        [[0.12, 0.00], [0.00, 0.12]],
    ]
)
_MASS_S_DOF = np.array([30.0, 35.0, 25.0])
_MASS_C_LOC = np.array([[0.2], [0.5], [0.8]])
_MASS_C_SCALE = np.array([[[0.04]], [[0.03]], [[0.04]]])
_MASS_C_DOF = np.array([25.0, 30.0, 35.0])


def _mass_predictive() -> stitch.TilePredictive:
    return stitch.TilePredictive(
        log_pi=jnp.log(jnp.asarray(_MASS_PI)),
        spatial=stitch.StudentT(
            loc=jnp.asarray(_MASS_S_LOC),
            scale=jnp.asarray(_MASS_S_SCALE),
            dof=jnp.asarray(_MASS_S_DOF),
        ),
        color=stitch.StudentT(
            loc=jnp.asarray(_MASS_C_LOC),
            scale=jnp.asarray(_MASS_C_SCALE),
            dof=jnp.asarray(_MASS_C_DOF),
        ),
    )


def _gl_panels(edges, n):
    """Gauss-Legendre nodes/weights for consecutive panels [edges_i, edges_{i+1}]."""
    x, w = np.polynomial.legendre.leggauss(n)
    nodes, weights = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        nodes.append(0.5 * (b - a) * x + 0.5 * (a + b))
        weights.append(0.5 * (b - a) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def test_stitched_mass_integrates_to_one_quadrature():
    # x panels split at the gate kink lines {4, 5, 6}: the integrand is analytic
    # within each panel, so 40-node Gauss-Legendre per panel is exact to
    # working precision and the 1e-3 tolerance is dominated by tail truncation.
    tp = _mass_predictive()
    xn, xw = _gl_panels([0.0, 4.0, 5.0, 6.0, 10.0], 40)
    yn, yw = _gl_panels([0.0, 3.0, 6.0], 40)
    cn, cw = _gl_panels([-1.5, 0.5, 2.5], 40)

    xx, yy = np.meshgrid(xn, yn, indexing="ij")
    s = jnp.asarray(np.column_stack([xx.ravel(), yy.ravel()]))
    w_xy = np.outer(xw, yw).ravel()

    mass = 0.0
    for c_val, c_w in zip(cn, cw):
        c = jnp.full((s.shape[0], 1), c_val)
        lp = np.asarray(stitch.stitched_logpdf([tp, tp], _MASS_GRID, s, c))
        mass += c_w * float(np.sum(w_xy * np.exp(lp)))
    np.testing.assert_allclose(mass, 1.0, rtol=0, atol=1e-3)


def _student_logpdf_np(x, loc, scale, dof):
    """NumPy multivariate Student-t log density from the standard formula,
    vectorized over rows of x."""
    d = loc.size
    diff = x - loc
    sol = np.linalg.solve(scale, diff.T).T
    quad = (diff * sol).sum(axis=1)
    _, logdet = np.linalg.slogdet(scale)
    return (
        gammaln((dof + d) / 2.0)
        - gammaln(dof / 2.0)
        - 0.5 * d * np.log(dof * np.pi)
        - 0.5 * logdet
        - 0.5 * (dof + d) * np.log1p(quad / dof)
    )


def _sample_student(rng, n, loc, scale, dof):
    """Draws from the multivariate Student-t: loc + L z sqrt(dof / chi2_dof)."""
    L = np.linalg.cholesky(scale)
    z = rng.standard_normal((n, loc.size))
    g = rng.chisquare(dof, size=n)
    return loc + (z @ L.T) * np.sqrt(dof / g)[:, None]


def test_stitched_mass_integrates_to_one_importance_mc():
    # Importance sampling with proposal q = the same mixture with every scale
    # matrix inflated by 1.44 (sd x 1.2): the weights f/q are bounded and close
    # to 1, giving a standard error of a few 1e-4 at this sample size; the run
    # is deterministic under the fixed seed. Samples outside the bounding box
    # (where the gates, hence f, are zero) contribute zero.
    rng = np.random.default_rng(9)
    n = 800_000
    inflate = 1.44
    tp = _mass_predictive()

    comp = rng.choice(_MASS_PI.size, size=n, p=_MASS_PI)
    s = np.empty((n, 2))
    c = np.empty((n, 1))
    for k in range(_MASS_PI.size):
        idx = np.nonzero(comp == k)[0]
        s[idx] = _sample_student(
            rng, idx.size, _MASS_S_LOC[k], inflate * _MASS_S_SCALE[k], _MASS_S_DOF[k]
        )
        c[idx] = _sample_student(
            rng, idx.size, _MASS_C_LOC[k], inflate * _MASS_C_SCALE[k], _MASS_C_DOF[k]
        )

    log_terms = np.stack(
        [
            np.log(_MASS_PI[k])
            + _student_logpdf_np(s, _MASS_S_LOC[k], inflate * _MASS_S_SCALE[k], _MASS_S_DOF[k])
            + _student_logpdf_np(c, _MASS_C_LOC[k], inflate * _MASS_C_SCALE[k], _MASS_C_DOF[k])
            for k in range(_MASS_PI.size)
        ],
        axis=1,
    )
    mx = log_terms.max(axis=1)
    log_q = mx + np.log(np.exp(log_terms - mx[:, None]).sum(axis=1))

    inside = np.all(
        (s >= _MASS_GRID.bbox_lo) & (s <= _MASS_GRID.bbox_hi), axis=1
    )
    log_f = np.full(n, -np.inf)
    log_f[inside] = np.asarray(
        stitch.stitched_logpdf(
            [tp, tp], _MASS_GRID, jnp.asarray(s[inside]), jnp.asarray(c[inside])
        )
    )
    est = float(np.mean(np.where(inside, np.exp(log_f - log_q), 0.0)))
    np.testing.assert_allclose(est, 1.0, rtol=0, atol=1e-3)


def test_gate_partition_of_unity_on_halo_grids():
    # Grid concentrated in the halo band [4, 6] of the 2-tile scene, including
    # the exact kink lines x in {4, 5, 6}.
    xs = np.concatenate([np.linspace(4.0, 6.0, 401), np.array([4.0, 5.0, 6.0])])
    ys = np.linspace(0.0, 6.0, 41)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    g = np.asarray(stitch.gate_weights(_MASS_GRID, pts))
    assert np.all(g >= 0.0)
    np.testing.assert_allclose(g.sum(axis=1), 1.0, rtol=0, atol=1e-12)

    # Multi-tile grid from make_grid: points along every interior core edge,
    # offset across the full halo band, plus the exact 4-tile corner points.
    rng = np.random.default_rng(11)
    cloud = np.column_stack(
        [rng.uniform(0.0, 60.0, 6000), rng.uniform(0.0, 30.0, 6000)]
    )
    grid = tiling.make_grid(cloud, halo=2.0, target_points=1000.0)
    assert grid.shape[0] >= 2 and grid.shape[1] >= 2
    ex = np.unique(grid.core_hi[:, 0])[:-1]  # interior vertical edges
    ey = np.unique(grid.core_hi[:, 1])[:-1]  # interior horizontal edges
    offs = np.linspace(-2.0, 2.0, 21)
    pts = [np.column_stack([np.meshgrid(ex, ey)[0].ravel(), np.meshgrid(ex, ey)[1].ravel()])]
    for x0 in ex:
        for off in offs:
            y = rng.uniform(grid.bbox_lo[1], grid.bbox_hi[1], 25)
            pts.append(np.column_stack([np.full(25, x0 + off), y]))
    for y0 in ey:
        for off in offs:
            x = rng.uniform(grid.bbox_lo[0], grid.bbox_hi[0], 25)
            pts.append(np.column_stack([x, np.full(25, y0 + off)]))
    pts = np.vstack(pts)
    pts = pts[
        np.all((pts >= grid.bbox_lo) & (pts <= grid.bbox_hi), axis=1)
    ]
    g = np.asarray(stitch.gate_weights(grid, pts))
    assert np.all(g >= 0.0)
    np.testing.assert_allclose(g.sum(axis=1), 1.0, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# (vi) two-tile end-to-end: fit, match, merge vs pooled single fit
# ---------------------------------------------------------------------------

# 3 well-separated clusters; the middle one straddles the seam at x = 10.
# Spatial separation is 12 cluster standard deviations, so responsibilities are
# effectively hard (cross-cluster leakage ~ exp(-70)) and the per-tile and
# pooled fits assign the same points to corresponding components.
_SEAM_X = 10.0
_HALO = 1.5
_ALPHA_GLOBAL = 1.0
_N_PER = 400
_T_TRUNC = 6
_S_CENTERS = np.array([[4.0, 3.0, 1.0], [10.0, 3.0, 1.0], [16.0, 3.0, 1.0]])
_S_SIGMA = np.array([0.5, 0.5, 0.3])
_C_CENTERS = np.array([[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]])
_C_SIGMA = 0.05


def _sample_scene(rng, n_per):
    s_parts, c_parts = [], []
    for j in range(3):
        s_parts.append(_S_CENTERS[j] + rng.standard_normal((n_per, 3)) * _S_SIGMA)
        c_parts.append(_C_CENTERS[j] + rng.standard_normal((n_per, 3)) * _C_SIGMA)
    return np.vstack(s_parts), np.vstack(c_parts)


def _fit_tile(seed, xs_t, xc_t, w_t, cfg_t, sp, cp):
    """Weighted CAVI to convergence under the shared global NIW priors.

    The merge arithmetic subtracts a single shared prior, so every tile must be
    fit under the same global prior rather than the tile-local default that
    init_state derives from the tile's own points.
    """
    state = cavi.init_state(seed, xs_t, xc_t, cfg_t)
    state = state._replace(spatial_prior=sp, color_prior=cp)
    prev = -np.inf
    for _ in range(cfg_t.max_iters):
        state = weighted.weighted_cavi_step(state, xs_t, xc_t, w_t, cfg_t)
        elbo = float(weighted.weighted_elbo(state, xs_t, xc_t, w_t, cfg_t))
        if np.isfinite(prev) and abs(elbo - prev) < cfg_t.tol * abs(prev):
            break
        prev = elbo
    return state


def _niw_slice(q: dniw.NIW, k: int) -> dniw.NIW:
    return dniw.NIW(
        m=q.m[k : k + 1], kappa=q.kappa[k : k + 1],
        Psi=q.Psi[k : k + 1], nu=q.nu[k : k + 1],
    )


def _niw_stack(qs) -> dniw.NIW:
    return dniw.NIW(
        m=jnp.concatenate([q.m for q in qs]),
        kappa=jnp.concatenate([q.kappa for q in qs]),
        Psi=jnp.concatenate([q.Psi for q in qs]),
        nu=jnp.concatenate([q.nu for q in qs]),
    )


def _significant_components(state, w, min_count=5.0):
    """Component indices with weighted soft count above min_count, with counts."""
    counts = np.asarray((state.r * w[:, None]).sum(axis=0))
    keep = [k for k in range(counts.size) if counts[k] > min_count]
    return keep, counts


@pytest.fixture(scope="module")
def two_tile_scene():
    rng = np.random.default_rng(42)
    s, c = _sample_scene(rng, _N_PER)
    grid = _two_tile_grid(s[:, :2].min(axis=0), s[:, :2].max(axis=0), _SEAM_X, _HALO)
    asn = tiling.assign_points(grid, s, alpha_global=_ALPHA_GLOBAL)

    xs, xc = jnp.asarray(s), jnp.asarray(c)
    cfg = cavi.Config(
        weight_prior="dp", T=_T_TRUNC, alpha=_ALPHA_GLOBAL, max_iters=400, tol=1e-9
    )
    # Shared global priors: identical to the ones cavi.fit derives internally
    # for the pooled fit (same data, same construction).
    sp = cavi.default_niw_prior(xs, _T_TRUNC, cfg.kappa0, cfg.nu0_offset)
    cp = cavi.default_niw_prior(xc, _T_TRUNC, cfg.kappa0, cfg.nu0_offset)

    pooled_state, _ = cavi.fit(0, xs, xc, cfg)

    tiles = []
    for t in range(2):
        idx = np.asarray(asn.indices[t])
        cfg_t = dataclasses.replace(cfg, alpha=float(asn.alpha[t]))
        w_t = jnp.asarray(asn.weight[idx])
        state = _fit_tile(0, xs[idx], xc[idx], w_t, cfg_t, sp, cp)
        keep, counts = _significant_components(state, w_t)
        tiles.append(dict(state=state, cfg=cfg_t, keep=keep, counts=counts))

    return dict(
        s=s, c=c, grid=grid, cfg=cfg, sp=sp, cp=cp,
        pooled=pooled_state, tiles=tiles, rng_heldout=np.random.default_rng(43),
    )


def _tile_components(tile):
    return [
        merge.Component(
            niws=(
                _niw_slice(tile["state"].spatial, k),
                _niw_slice(tile["state"].color, k),
            ),
            count=float(tile["counts"][k]),
        )
        for k in tile["keep"]
    ]


def _mean_x(comp):
    return float(comp.niws[0].m[0, 0])


def test_each_tile_finds_its_two_clusters(two_tile_scene):
    sc = two_tile_scene
    for t, (lo_x, hi_x) in enumerate([(4.0, 10.0), (10.0, 16.0)]):
        comps = _tile_components(sc["tiles"][t])
        assert len(comps) == 2
        mean_xs = sorted(_mean_x(cmp) for cmp in comps)
        np.testing.assert_allclose(mean_xs, [lo_x, hi_x], atol=0.2)
        # The straddling cluster's points are shared by both tiles at weight
        # 1/2, so each tile's straddling soft count is ~ N_PER / 2, while the
        # exclusive cluster keeps its full count.
        counts = sorted(cmp.count for cmp in comps)
        np.testing.assert_allclose(counts, [_N_PER / 2, _N_PER], rtol=0.05)


def test_match_accepts_exactly_the_straddling_pair(two_tile_scene):
    sc = two_tile_scene
    comps_a = _tile_components(sc["tiles"][0])
    comps_b = _tile_components(sc["tiles"][1])
    prior = (_niw_slice(sc["sp"], 0), _niw_slice(sc["cp"], 0))
    mask = np.ones((len(comps_a), len(comps_b)), dtype=bool)
    pairs = merge.hungarian_match(comps_a, comps_b, prior, _ALPHA_GLOBAL, mask)
    assert len(pairs) == 1
    i, j = pairs[0]
    assert abs(_mean_x(comps_a[i]) - _SEAM_X) < 0.5
    assert abs(_mean_x(comps_b[j]) - _SEAM_X) < 0.5


def _merged_and_pooled(sc):
    """Merged straddling component (spatial, color, count) and the pooled fit's
    straddling component index."""
    comps_a = _tile_components(sc["tiles"][0])
    comps_b = _tile_components(sc["tiles"][1])
    prior = (_niw_slice(sc["sp"], 0), _niw_slice(sc["cp"], 0))
    mask = np.ones((len(comps_a), len(comps_b)), dtype=bool)
    (i, j), = merge.hungarian_match(comps_a, comps_b, prior, _ALPHA_GLOBAL, mask)
    merged_s = merge.merge_components(
        [comps_a[i].niws[0], comps_b[j].niws[0]], prior[0]
    )
    merged_c = merge.merge_components(
        [comps_a[i].niws[1], comps_b[j].niws[1]], prior[1]
    )
    merged_count = comps_a[i].count + comps_b[j].count

    pooled = sc["pooled"]
    pooled_counts = np.asarray(pooled.r.sum(axis=0))
    sig = [k for k in range(pooled_counts.size) if pooled_counts[k] > 5.0]
    k_strad = min(sig, key=lambda k: abs(float(pooled.spatial.m[k, 0]) - _SEAM_X))
    return (merged_s, merged_c, merged_count), (pooled, pooled_counts, k_strad), (i, j)


def test_merged_component_matches_pooled_fit(two_tile_scene):
    sc = two_tile_scene
    (merged_s, merged_c, merged_count), (pooled, pooled_counts, k), _ = (
        _merged_and_pooled(sc)
    )

    # Ownership weights satisfy sum_{t owning n} w_n = 1 for every point, so
    # the merged natural parameters equal the pooled sufficient statistics
    # weighted by the per-tile responsibilities. With 12-sigma cluster
    # separation those responsibilities are hard (cross-cluster leakage
    # ~exp(-70)): the tile-wise and pooled fits assign identical effective
    # point sets and the two sides agree to floating-point accumulation order
    # (observed ~1e-14 relative). Tolerances of 1e-3 leave ~1e11 margin while
    # still failing on any real defect (a single misassigned point shifts the
    # count by 1/400 = 2.5e-3).
    np.testing.assert_allclose(merged_count, pooled_counts[k], rtol=1e-3)
    np.testing.assert_allclose(
        float(merged_s.kappa[0]), float(pooled.spatial.kappa[k]), rtol=1e-3
    )
    np.testing.assert_allclose(
        float(merged_s.nu[0]), float(pooled.spatial.nu[k]), rtol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(merged_s.m[0]), np.asarray(pooled.spatial.m[k]), atol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(merged_c.m[0]), np.asarray(pooled.color.m[k]), atol=1e-3
    )
    for got, ref in (
        (merged_s.Psi[0], pooled.spatial.Psi[k]),
        (merged_c.Psi[0], pooled.color.Psi[k]),
    ):
        rel = np.linalg.norm(np.asarray(got) - np.asarray(ref)) / np.linalg.norm(
            np.asarray(ref)
        )
        assert rel < 1e-3


def test_stitched_heldout_matches_pooled_predictive(two_tile_scene):
    sc = two_tile_scene
    (merged_s, merged_c, merged_count), (pooled, _, _), (i, j) = _merged_and_pooled(sc)
    comps_a = _tile_components(sc["tiles"][0])
    comps_b = _tile_components(sc["tiles"][1])

    # Post-merge global mixture: each tile's exclusive component plus the
    # merged straddling component; global weights rebuilt from pooled counts.
    (i_left,) = [k for k in range(len(comps_a)) if k != i]
    (j_right,) = [k for k in range(len(comps_b)) if k != j]
    left, right = comps_a[i_left], comps_b[j_right]
    counts = jnp.asarray([left.count, merged_count, right.count])
    a, b, order = merge.rebuild_weights(counts, _ALPHA_GLOBAL)
    w_global = np.empty(3)
    w_global[np.asarray(order)] = np.asarray(dpriors.dp_expected_pi(a, b))

    # Post-merge tile predictives: each tile carries its own components with
    # their GLOBAL mixture weights (unnormalized within the tile). The tiles
    # then agree on the shared component in the halo, and each tile's
    # sub-mixture omits only the far tile's component, whose density on the
    # tile's gate support is a Student-t tail at >= 12 sd (~e-30) — so the
    # stitched density equals the merged global mixture up to that tail.
    def tile_predictive(niws_s, niws_c, weights):
        return stitch.TilePredictive(
            log_pi=jnp.log(jnp.asarray(weights)),
            spatial=predictive.niw_predictive(_niw_stack(niws_s)),
            color=predictive.niw_predictive(_niw_stack(niws_c)),
        )

    tp1 = tile_predictive(
        [left.niws[0], merged_s], [left.niws[1], merged_c],
        [w_global[0], w_global[1]],
    )
    tp2 = tile_predictive(
        [merged_s, right.niws[0]], [merged_c, right.niws[1]],
        [w_global[1], w_global[2]],
    )

    # Held-out points from the same generative mixture, restricted to the
    # scene bounding box (the gates are undefined outside it); both sides of
    # the comparison are evaluated on the identical restricted set.
    s_ho, c_ho = _sample_scene(sc["rng_heldout"], 200)
    grid = sc["grid"]
    inside = np.all(
        (s_ho[:, :2] >= grid.bbox_lo) & (s_ho[:, :2] <= grid.bbox_hi), axis=1
    )
    s_ho, c_ho = jnp.asarray(s_ho[inside]), jnp.asarray(c_ho[inside])
    assert s_ho.shape[0] > 400

    stitched = np.asarray(stitch.stitched_logpdf([tp1, tp2], grid, s_ho, c_ho))

    # Pooled per-point log predictive from the same public predictive algebra.
    from dp_splat.prune import expected_pi
    from jax.scipy.special import logsumexp

    log_w = jnp.log(expected_pi(pooled, sc["cfg"]) + 1e-300)
    pooled_pts = np.asarray(
        logsumexp(
            log_w[None, :]
            + predictive.student_logpdf(predictive.niw_predictive(pooled.spatial), s_ho)
            + predictive.student_logpdf(predictive.niw_predictive(pooled.color), c_ho),
            axis=1,
        )
    )
    pooled_mean = float(
        predictive.heldout_loglik(pooled, sc["cfg"], s_ho, c_ho)
    )
    np.testing.assert_allclose(pooled_pts.mean(), pooled_mean, rtol=1e-12)

    # Both sides are 3-component Student-t mixtures over the same clusters and
    # their NIW parameters agree to ~1e-14 (merged-component test), so the
    # remaining gap comes only from the weight vectors: rebuilt 3-component
    # stick-breaking weights versus the pooled fit's T=6 truncated posterior
    # (observed: 8e-4 mean, 4e-3 max pointwise). 0.05 / 0.1 nats leave a wide
    # margin yet fail on any normalization defect (a missing partition-of-unity
    # or weight renormalization shifts the mean by ~log 2 = 0.69); the
    # pointwise bound catches localized seam artifacts a mean could average out.
    assert abs(float(stitched.mean()) - pooled_mean) < 0.05
    assert float(np.max(np.abs(stitched - pooled_pts))) < 0.1

"""Independent oracles for the statistical constructions in experiments/sparsification.py.

(i)  Sparsification curves and AUSE/AURG recomputed from the definitions
     (Ilg et al. 2018 convention, as documented in the script): sort by u
     descending (stable in the original index on ties), remove the first
     ceil(f n) points at each fraction f in {0, 0.02, ..., 0.98}, record the
     mean |e| of the retained set normalized by the full-set mean, and take
     trapezoid areas AUSE = int (curve_u - curve_oracle) df and
     AURG = int (curve_random - curve_u) df. Checked for exact agreement on
     random inputs including heavy ties, plus structural facts: the oracle
     curve is monotone non-increasing and is the pointwise lower envelope of
     every removal ordering (removing the ceil(f n) largest |e| minimizes the
     retained mean over all removal sets of that size); a perfect ranking
     u = |e| gives AUSE = 0 exactly; an inverted ranking gives negative AURG;
     the random curve is flat at 1 in expectation.

(ii) The gate-rejection sampler against the density it claims to draw from.
     Single component: per-axis marginals against the closed-form univariate
     Student-t (the axis-d marginal of St(m, S, eta) is univariate Student-t
     with location m_d, scale sqrt(S_dd), and the same dof eta), and the
     empirical covariance against S eta / (eta - 2). Two tiles with
     location-dependent gates: binned chi-square of the sampled (x, y) against
     midpoint-quadrature cell masses of the self-normalized xy marginal

         p(xy) / Z,   p(xy) = sum_t g_t(xy) sum_k E[pi_k^t] St2(xy; m_xy, S_xy, eta),

     using that the xy marginal of a multivariate Student-t is Student-t with
     the corresponding scale sub-block and unchanged dof; per-(tile, component)
     draw frequencies are checked against the quadrature pair masses, which
     exercises the within-tile weight draw and the gate acceptance jointly.
     The sampler's own-tile gate is checked against stitch.gate_weights.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import sparsification as sp
from dp_splat.predictive import StudentT
from dp_splat_uav import stitch, tiling

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# (i) AUSE / AURG against an independent implementation of the definitions
# ---------------------------------------------------------------------------


def _retained_mean_curve(e, order):
    """Mean of e after removing the first ceil(f n) entries of ``order``, per f."""
    n = e.shape[0]
    means = []
    for f in sp.FRACTIONS:
        removed = min(int(np.ceil(f * n)), n - 1)
        means.append(float(np.mean(e[order[removed:]])))
    return np.array(means)


def _reference_sparsification(e, u):
    """Definition-level u-ordered and oracle curves plus AUSE (no numpy argsort).

    Ties in u are broken by the original index (the stable-sort convention the
    script documents); the oracle sorts by |e| itself.
    """
    n = e.shape[0]
    order_u = np.array(sorted(range(n), key=lambda i: (-u[i], i)))
    order_or = np.array(sorted(range(n), key=lambda i: (-e[i], i)))
    base = float(np.mean(e))
    curve_u = _retained_mean_curve(e, order_u) / base
    curve_or = _retained_mean_curve(e, order_or) / base
    ause = float(np.trapezoid(curve_u - curve_or, sp.FRACTIONS))
    return curve_u, curve_or, ause


@pytest.mark.parametrize("n,seed,ties", [(53, 0, False), (200, 1, True),
                                         (1024, 2, False), (997, 3, True)])
def test_curves_and_ause_match_definition(n, seed, ties):
    rng = np.random.default_rng(seed)
    e = rng.lognormal(0.0, 1.0, size=n)
    if ties:
        u = rng.integers(0, 5, size=n).astype(np.float64)  # heavy ties
        e[rng.choice(n, size=n // 4, replace=False)] = 1.0  # ties in e too
    else:
        u = rng.standard_normal(n)
    res = sp.sparsification(e, u, np.random.default_rng(seed + 100))
    curve_u, curve_or, ause = _reference_sparsification(e, u)
    np.testing.assert_allclose(res["curve_u"], curve_u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res["curve_oracle"], curve_or, rtol=0, atol=1e-12)
    assert res["ause"] == pytest.approx(ause, abs=1e-12)
    # AURG is wired as the area between the reported random and u curves.
    aurg = float(np.trapezoid(np.array(res["curve_random"]) - curve_u, sp.FRACTIONS))
    assert res["aurg"] == pytest.approx(aurg, abs=1e-12)


def test_oracle_is_monotone_lower_envelope():
    rng = np.random.default_rng(7)
    e = rng.lognormal(0.0, 0.8, size=800)
    u = rng.standard_normal(800)
    res = sp.sparsification(e, u, rng)
    oracle = np.array(res["curve_oracle"])
    assert np.all(np.diff(oracle) <= 1e-12)  # monotone non-increasing
    # Lower envelope: any removal ordering retains at least the oracle mean.
    base = e.mean()
    for k in range(20):
        order = np.random.default_rng(k).permutation(e.shape[0])
        curve = _retained_mean_curve(e, order) / base
        assert np.all(curve >= oracle - 1e-12)
    assert np.all(np.array(res["curve_u"]) >= oracle - 1e-12)


def test_perfect_and_inverted_rankings():
    rng = np.random.default_rng(11)
    e = rng.lognormal(0.0, 1.0, size=600)
    perfect = sp.sparsification(e, e.copy(), np.random.default_rng(0))
    assert perfect["ause"] == pytest.approx(0.0, abs=1e-12)
    assert perfect["aurg"] > 0.0
    inverted = sp.sparsification(e, -e, np.random.default_rng(0))
    assert inverted["ause"] > perfect["aurg"]  # worse than random by construction
    assert inverted["aurg"] < 0.0
    independent = sp.sparsification(e, rng.standard_normal(600), np.random.default_rng(0))
    assert 0.0 < independent["ause"] < inverted["ause"]


def test_random_curve_flat_in_expectation():
    """E[mean of a uniformly random retained subset] equals the full mean, so the
    normalized random curve is 1 at every fraction; average many orderings."""
    rng = np.random.default_rng(5)
    e = rng.lognormal(0.0, 0.5, size=4000)
    base = e.mean()
    curves = [
        _retained_mean_curve(e, np.random.default_rng(k).permutation(4000)) / base
        for k in range(200)
    ]
    mean_curve = np.mean(curves, axis=0)
    # Retained sets shrink to 2% of n at f = 0.98; the widest standard error of
    # the 200-ordering average is ~ (cv/sqrt(80))/sqrt(200) ~ 0.005.
    assert np.all(np.abs(mean_curve - 1.0) < 0.02)


# ---------------------------------------------------------------------------
# (ii) The gate-rejection sampler against the stitched density
# ---------------------------------------------------------------------------


def _one_tile_grid(half=60.0):
    lo = np.array([[-half, -half]])
    hi = np.array([[half, half]])
    return tiling.TileGrid(core_lo=lo, core_hi=hi, halo=0.0,
                           bbox_lo=lo[0], bbox_hi=hi[0], shape=(1, 1),
                           index=np.array([[0, 0]]),
                           morton=np.array([0], dtype=np.uint64))


def _two_tile_grid():
    """2x1 grid on [0, 20] x [0, 10], seam at x = 10, halo 2."""
    return tiling.TileGrid(
        core_lo=np.array([[0.0, 0.0], [10.0, 0.0]]),
        core_hi=np.array([[10.0, 10.0], [20.0, 10.0]]),
        halo=2.0,
        bbox_lo=np.array([0.0, 0.0]),
        bbox_hi=np.array([20.0, 10.0]),
        shape=(2, 1),
        index=np.array([[0, 0], [1, 0]]),
        morton=np.array([0, 1], dtype=np.uint64),
    )


def _spd(rng, d=3, scale=1.0):
    a = rng.standard_normal((d, d))
    return scale * (a @ a.T + d * np.eye(d))


def test_single_component_student_t_marginals_and_covariance():
    """Draws must follow St(m, S, eta): axis marginals are univariate Student-t
    (same dof, scale sqrt(S_dd)) and the covariance is S eta/(eta - 2)."""
    rng_np = np.random.default_rng(42)
    dof = 7.0
    m = np.array([1.5, -2.0, 0.5])
    S = _spd(np.random.default_rng(3), scale=0.5)
    st = StudentT(loc=jnp.asarray(m[None]), scale=jnp.asarray(S[None]),
                  dof=jnp.asarray([dof]))
    grid = _one_tile_grid()
    s, pair, acc = sp.sample_stitched_spatial(
        grid, np.array([0]), jnp.log(jnp.ones(1)), st, 40_000, rng_np)
    assert np.all(pair == 0)
    assert acc > 0.999  # one tile, gate 1 on the core; only far tails rejected
    for d in range(3):
        ks = stats.kstest(s[:, d], stats.t(df=dof, loc=m[d], scale=np.sqrt(S[d, d])).cdf)
        assert ks.pvalue > 1e-3, f"axis {d} marginal mismatch: {ks}"
    cov_true = S * dof / (dof - 2.0)
    cov_emp = np.cov(s.T)
    np.testing.assert_allclose(cov_emp, cov_true, rtol=0.08, atol=0.02)
    # Variant (a) summary equals the analytic RMS marginal predictive sd and
    # matches the empirical one.
    u_a = sp.pair_sqrt_mean_var(st)
    assert u_a[0] == pytest.approx(np.sqrt(np.trace(cov_true) / 3.0), rel=1e-12)
    assert u_a[0] == pytest.approx(np.sqrt(np.trace(cov_emp) / 3.0), rel=0.05)


def test_tile_gate_matches_stitch_gate_weights():
    grid = _two_tile_grid()
    rng = np.random.default_rng(0)
    xy = rng.uniform([-2.0, -2.0], [22.0, 12.0], size=(2000, 2))  # dilated bbox
    g_all = np.asarray(stitch.gate_weights(grid, jnp.asarray(xy)))
    for t in range(2):
        own = sp.tile_gate(grid, xy, np.full(2000, t))
        np.testing.assert_allclose(own, g_all[:, t], atol=1e-12)
    # Outside every dilated tile the sampler's gate is 0 (proposal rejected).
    far = np.array([[30.0, 5.0], [-10.0, 5.0], [5.0, 20.0]])
    assert np.all(sp.tile_gate(grid, far, np.zeros(3, dtype=int)) == 0.0)


def _two_tile_bank():
    """Two tiles, two components each, distinct scales/dofs/weights."""
    rng = np.random.default_rng(9)
    loc = np.array([[4.0, 5.0, 0.0], [9.0, 3.0, 2.0],
                    [13.0, 6.0, -1.0], [17.0, 4.0, 1.0]])
    scale = np.stack([_spd(rng, scale=s) for s in (0.8, 0.4, 0.6, 1.0)])
    dof = np.array([5.0, 9.0, 6.0, 12.0])
    st = StudentT(loc=jnp.asarray(loc), scale=jnp.asarray(scale), dof=jnp.asarray(dof))
    pair_tile = np.array([0, 0, 1, 1])
    log_pi = jnp.log(jnp.asarray([0.6, 0.4, 0.5, 0.5]))  # sums to 1 per tile
    return pair_tile, log_pi, st


def test_rejection_sampler_matches_stitched_xy_marginal():
    """Binned chi-square of sampled (x, y) against quadrature masses of the
    self-normalized stitched xy marginal, plus pair-frequency agreement."""
    grid = _two_tile_grid()
    pair_tile, log_pi, st = _two_tile_bank()
    rng = np.random.default_rng(123)
    n = 60_000
    s, pair, acc = sp.sample_stitched_spatial(grid, pair_tile, log_pi, st, n, rng)
    assert 0.0 < acc < 1.0  # gates genuinely reject on this geometry

    # Support of the stitched density: the union of halo-dilated tiles.
    lo = np.array([-2.0, -2.0])
    hi = np.array([22.0, 12.0])
    assert np.all(s[:, :2] >= lo - 1e-9) and np.all(s[:, :2] <= hi + 1e-9)

    # Midpoint quadrature of p(xy) = sum_p g_{t(p)}(xy) pi_p St2(xy; sub-block)
    # on nb x nb cells with sub x sub midpoints each.
    nb, sub = (12, 7), 24
    xs_edges = np.linspace(lo[0], hi[0], nb[0] + 1)
    ys_edges = np.linspace(lo[1], hi[1], nb[1] + 1)
    xg = (np.arange(nb[0] * sub) + 0.5) * (hi[0] - lo[0]) / (nb[0] * sub) + lo[0]
    yg = (np.arange(nb[1] * sub) + 0.5) * (hi[1] - lo[1]) / (nb[1] * sub) + lo[1]
    pts = np.stack(np.meshgrid(xg, yg, indexing="ij"), axis=-1).reshape(-1, 2)

    st2 = StudentT(loc=st.loc[:, :2], scale=st.scale[:, :2, :2], dof=st.dof)
    g = np.asarray(stitch.gate_weights(grid, jnp.asarray(pts)))[:, pair_tile]  # (M, P)
    dens_pairs = g * np.exp(
        np.asarray(log_pi)[None, :]
        + np.asarray(sp.student_logpdf(st2, jnp.asarray(pts))))  # (M, P)
    dens_pairs = dens_pairs.reshape(nb[0], sub, nb[1], sub, 4)
    cell_pair_mass = dens_pairs.sum(axis=(1, 3))  # (nbx, nby, P), up to a constant
    z = cell_pair_mass.sum()

    # Pair frequencies: the sampler's (tile, component) draw filtered by the
    # gate must reproduce each pair's share of the total mass.
    pair_prob = cell_pair_mass.sum(axis=(0, 1)) / z
    pair_freq = np.bincount(pair, minlength=4) / n
    se = np.sqrt(pair_prob * (1.0 - pair_prob) / n)
    assert np.all(np.abs(pair_freq - pair_prob) < 5.0 * se + 1e-3)

    # Chi-square over spatial cells (pool cells below 20 expected counts).
    probs = cell_pair_mass.sum(axis=-1).ravel() / z
    ix = np.clip(np.searchsorted(xs_edges, s[:, 0], side="right") - 1, 0, nb[0] - 1)
    iy = np.clip(np.searchsorted(ys_edges, s[:, 1], side="right") - 1, 0, nb[1] - 1)
    obs = np.bincount(ix * nb[1] + iy, minlength=nb[0] * nb[1]).astype(float)
    exp = probs * n
    big = exp >= 20.0
    obs_p = np.append(obs[big], obs[~big].sum())
    exp_p = np.append(exp[big], exp[~big].sum())
    chi2 = float(((obs_p - exp_p) ** 2 / np.maximum(exp_p, 1e-12)).sum())
    crit = stats.chi2.ppf(0.9995, df=obs_p.size - 1)
    assert chi2 < crit, f"chi2 {chi2:.1f} exceeds {crit:.1f} on {obs_p.size} bins"

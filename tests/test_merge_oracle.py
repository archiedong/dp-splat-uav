"""Independent NumPy oracles for the cross-tile merge math (oracle tests (i)-(iii)).

Every oracle below is written from the standard NIW conjugate equations and the
NIW density normalization, not from the implementation. dp_splat containers are
used only to feed the public API under test and to read back its results.

(i)   prior-subtracted natural-parameter addition: merging NIW posteriors built
      from disjoint point sets under a shared prior must equal the NIW posterior
      computed on the concatenated points, for m = 2 and m = 3.
(ii)  match score: Delta L for component pairs must equal the directly computed
      log-normalizer combination A(pooled) - A(A) - A(B) + A(prior).
(iii) rebuild_weights: against a from-scratch truncated stick-breaking
      construction on hand counts.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln

from dp_splat import niw as dniw
from dp_splat_uav import merge

D = 3


# ---------------------------------------------------------------------------
# Independent NumPy NIW machinery, written from the standard equations
# ---------------------------------------------------------------------------


def _spd(rng, d, scale=1.0):
    A = rng.normal(size=(d, d))
    return scale * (A @ A.T + d * np.eye(d))


def _niw_update_np(m0, kappa0, Psi0, nu0, x):
    """Standard NIW conjugate update for fully assigned points x (n, D):

        kappa_n = kappa0 + n
        m_n     = (kappa0 m0 + n xbar) / kappa_n
        nu_n    = nu0 + n
        Psi_n   = Psi0 + S + kappa0 n / (kappa0 + n) (xbar - m0)(xbar - m0)^T,

    with xbar the sample mean and S the scatter about xbar.
    """
    n = x.shape[0]
    xbar = x.mean(axis=0)
    S = (x - xbar).T @ (x - xbar)
    kappa_n = kappa0 + n
    m_n = (kappa0 * m0 + n * xbar) / kappa_n
    nu_n = nu0 + n
    dev = xbar - m0
    Psi_n = Psi0 + S + (kappa0 * n / kappa_n) * np.outer(dev, dev)
    return m_n, kappa_n, Psi_n, nu_n


def _log_normalizer_np(kappa, nu, Psi):
    """NIW log normalizer from the density normalization constants directly.

    NIW(mu, Lambda | m, kappa, Psi, nu) = N(mu | m, (kappa Lambda)^{-1})
    x Wishart(Lambda | Psi^{-1}, nu). Integrating the unnormalized natural form:
    the Gaussian mean integral contributes (2 pi / kappa)^{D/2} |Lambda|^{-1/2},
    whose |Lambda|^{-1/2} folds into the Wishart integral with scale Psi, dof nu,
    which contributes 2^{nu D / 2} Gamma_D(nu / 2) |Psi|^{-nu/2}. Hence

        A = (D/2) log(2 pi) - (D/2) log kappa + (nu D / 2) log 2
            + log Gamma_D(nu / 2) - (nu / 2) log |Psi|.
    """
    d = Psi.shape[0]
    sign, logdet = np.linalg.slogdet(Psi)
    assert sign > 0
    mvlgamma = (d * (d - 1) / 4.0) * np.log(np.pi) + sum(
        gammaln((nu + 1.0 - i) / 2.0) for i in range(1, d + 1)
    )
    return (
        0.5 * d * np.log(2.0 * np.pi)
        - 0.5 * d * np.log(kappa)
        + 0.5 * nu * d * np.log(2.0)
        + mvlgamma
        - 0.5 * nu * logdet
    )


def _to_container(m, kappa, Psi, nu) -> dniw.NIW:
    """Wrap scalar-component NumPy parameters as a K = 1 dp_splat NIW container."""
    return dniw.NIW(
        m=jnp.asarray(m)[None, :],
        kappa=jnp.asarray([float(kappa)]),
        Psi=jnp.asarray(Psi)[None, :, :],
        nu=jnp.asarray([float(nu)]),
    )


def _random_prior(rng):
    """(m0, kappa0, Psi0, nu0) with non-trivial mean, scale, and non-integer dof."""
    return rng.normal(size=D), 2.3, _spd(rng, D, 0.8), D + 3.5


def _posterior_container(prior_np, x):
    return _to_container(*_niw_update_np(*prior_np, x))


# ---------------------------------------------------------------------------
# (i) prior-subtracted lambda-addition == posterior on concatenated points
# ---------------------------------------------------------------------------


def _assert_niw_equal(got: dniw.NIW, m_e, kappa_e, Psi_e, nu_e):
    np.testing.assert_allclose(np.asarray(got.m[0]), m_e, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(float(got.kappa[0]), kappa_e, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(got.Psi[0]), Psi_e, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(float(got.nu[0]), nu_e, rtol=1e-9)


def test_two_way_merge_equals_concatenated_posterior():
    rng = np.random.default_rng(1)
    prior = _random_prior(rng)
    xa = rng.normal(loc=2.0, scale=0.7, size=(37, D))
    xb = rng.normal(loc=2.4, scale=0.9, size=(23, D))
    merged = merge.merge_components(
        [_posterior_container(prior, xa), _posterior_container(prior, xb)],
        _to_container(*prior),
    )
    _assert_niw_equal(merged, *_niw_update_np(*prior, np.vstack([xa, xb])))


def test_three_way_merge_equals_concatenated_posterior():
    rng = np.random.default_rng(2)
    prior = _random_prior(rng)
    parts = [
        rng.normal(loc=-1.0, scale=0.6, size=(41, D)),
        rng.normal(loc=-0.8, scale=1.1, size=(17, D)),
        rng.normal(loc=-1.2, scale=0.8, size=(29, D)),
    ]
    merged = merge.merge_components(
        [_posterior_container(prior, x) for x in parts], _to_container(*prior)
    )
    _assert_niw_equal(merged, *_niw_update_np(*prior, np.vstack(parts)))


# ---------------------------------------------------------------------------
# (ii) match-score matrix == brute-force log-normalizer differences
# ---------------------------------------------------------------------------


def _delta_L_np(prior_np, xa, xb):
    """Delta L computed only from the NumPy update and NumPy log normalizer:
    A(posterior on concatenated points) - A(post A) - A(post B) + A(prior)."""
    m0, kappa0, Psi0, nu0 = prior_np

    def A_post(x):
        _, kappa, Psi, nu = _niw_update_np(m0, kappa0, Psi0, nu0, x)
        return _log_normalizer_np(kappa, nu, Psi)

    A_prior = _log_normalizer_np(kappa0, nu0, Psi0)
    return A_post(np.vstack([xa, xb])) - A_post(xa) - A_post(xb) + A_prior


def test_match_score_matrix_vs_bruteforce():
    rng = np.random.default_rng(3)
    prior = _random_prior(rng)
    prior_c = _to_container(*prior)
    a_sets = [
        rng.normal(loc=c, scale=s, size=(n, D))
        for c, s, n in ((0.0, 0.5, 20), (1.5, 0.8, 45), (-2.0, 1.2, 33))
    ]
    b_sets = [
        rng.normal(loc=c, scale=s, size=(n, D))
        for c, s, n in ((0.1, 0.6, 28), (4.0, 0.4, 52), (-2.2, 0.9, 19))
    ]
    for xa in a_sets:
        qa = _posterior_container(prior, xa)
        for xb in b_sets:
            qb = _posterior_container(prior, xb)
            got = float(merge.match_score((qa,), (qb,), (prior_c,)))
            expected = _delta_L_np(prior, xa, xb)
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


def test_match_score_two_modalities_vs_bruteforce():
    rng = np.random.default_rng(4)
    prior_s = _random_prior(rng)
    prior_c = _random_prior(rng)
    xs_a = rng.normal(loc=1.0, scale=0.7, size=(40, D))
    xs_b = rng.normal(loc=1.1, scale=0.6, size=(35, D))
    xc_a = rng.normal(loc=0.4, scale=0.1, size=(40, D))
    xc_b = rng.normal(loc=0.5, scale=0.1, size=(35, D))
    got = float(
        merge.match_score(
            (_posterior_container(prior_s, xs_a), _posterior_container(prior_c, xc_a)),
            (_posterior_container(prior_s, xs_b), _posterior_container(prior_c, xc_b)),
            (_to_container(*prior_s), _to_container(*prior_c)),
        )
    )
    expected = _delta_L_np(prior_s, xs_a, xs_b) + _delta_L_np(prior_c, xc_a, xc_b)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


# ---------------------------------------------------------------------------
# (iii) rebuild_weights == from-scratch truncated stick-breaking construction
# ---------------------------------------------------------------------------


def _stick_breaking_oracle(counts, alpha):
    """From the truncated stick-breaking posterior definition: components in
    size-biased (descending-count) order, q(v_k) = Beta(1 + N_k,
    alpha + sum_{j>k} N_j) for k = 1..K-1, v_K := 1 (scalar loop)."""
    order = np.argsort(-counts, kind="stable")
    srt = counts[order]
    K = srt.size
    a = np.empty(K - 1)
    b = np.empty(K - 1)
    for k in range(K - 1):
        a[k] = 1.0 + srt[k]
        b[k] = alpha + srt[k + 1 :].sum()
    return a, b, order


def test_rebuild_weights_matches_scratch_stick_breaking():
    cases = [
        (np.array([4.0, 1.5, 7.0, 2.5]), 1.7),
        (np.array([120.0, 3.25, 40.5, 0.75, 88.0, 11.0]), 0.6),
    ]
    for counts, alpha in cases:
        a_got, b_got, order_got = merge.rebuild_weights(jnp.asarray(counts), alpha)
        a_e, b_e, order_e = _stick_breaking_oracle(counts, alpha)
        np.testing.assert_array_equal(np.asarray(order_got), order_e)
        np.testing.assert_allclose(np.asarray(a_got), a_e, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(b_got), b_e, rtol=1e-12)

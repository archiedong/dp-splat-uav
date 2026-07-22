"""Independent NumPy oracle for one weighted CAVI cycle, plus the two exact identities.

The oracle below is written directly from the weighted-likelihood math — each point n
counted w_n times (fractional allowed) — using plain NumPy/SciPy primitives (explicit
inverses, slogdet, per-component loops), never dp_splat code paths. Every posterior
quantity of one cycle (responsibilities, both NIW modalities, stick/Dirichlet weight
parameters, q(alpha)) must match dp_splat_uav.weighted.weighted_cavi_step to <= 1e-8.

Identities:
  * w_n = 1 for all n: weighted_fit reproduces dp_splat.cavi.fit (ELBO trace to 1e-12);
  * integer weight w on point i equals physically duplicating point i w times
    (identical updates up to floating-point summation order).
"""

import numpy as np
import pytest
from scipy.special import digamma

import jax.numpy as jnp

from dp_splat import cavi as _cavi
from dp_splat.cavi import Config

from dp_splat_uav import weighted as _w


# ---------------------------------------------------------------------------
# NumPy oracle: one weighted CAVI cycle from the math
# ---------------------------------------------------------------------------


def _oracle_e_gauss_loglik(m, kappa, Psi, nu, x):
    """E[log N(x_n | mu_k, Lambda_k^{-1})] under NIW q, (N, K).

    E[log|Lambda_k|] = sum_i psi((nu_k+1-i)/2) + D log 2 - log|Psi_k|
    E[(x-mu)^T Lambda (x-mu)] = D/kappa_k + nu_k (x-m_k)^T Psi_k^{-1} (x-m_k)
    """
    N, D = x.shape
    K = m.shape[0]
    out = np.empty((N, K))
    for k in range(K):
        elogdet = (
            digamma((nu[k] + 1.0 - np.arange(1, D + 1)) / 2.0).sum()
            + D * np.log(2.0)
            - np.linalg.slogdet(Psi[k])[1]
        )
        dev = x - m[k]
        quad = np.einsum("ni,ij,nj->n", dev, np.linalg.inv(Psi[k]), dev)
        maha = D / kappa[k] + nu[k] * quad
        out[:, k] = 0.5 * elogdet - 0.5 * D * np.log(2.0 * np.pi) - 0.5 * maha
    return out


def _oracle_elogpi_dp(g1, g2):
    """E[log pi_k] = E[log v_k] + sum_{j<k} E[log(1-v_j)], v_T := 1, (T,)."""
    dg = digamma(g1 + g2)
    elogv = digamma(g1) - dg
    elog1mv = digamma(g2) - dg
    T = g1.shape[0] + 1
    out = np.empty(T)
    acc = 0.0
    for k in range(T - 1):
        out[k] = elogv[k] + acc
        acc += elog1mv[k]
    out[T - 1] = acc
    return out


def _oracle_niw_update(m0, kappa0, Psi0, nu0, Nk, xbar, S):
    """Conjugate NIW update, one component at a time.

    kappa = kappa0 + N;  m = (kappa0 m0 + N xbar)/kappa;  nu = nu0 + N;
    Psi = Psi0 + S + kappa0 N/(kappa0+N) (xbar-m0)(xbar-m0)^T.
    """
    K, D = xbar.shape
    m = np.empty((K, D))
    kappa = np.empty(K)
    Psi = np.empty((K, D, D))
    nu = np.empty(K)
    for k in range(K):
        kappa[k] = kappa0[k] + Nk[k]
        m[k] = (kappa0[k] * m0[k] + Nk[k] * xbar[k]) / kappa[k]
        nu[k] = nu0[k] + Nk[k]
        dev = xbar[k] - m0[k]
        Psi[k] = Psi0[k] + S[k] + (kappa0[k] * Nk[k] / (kappa0[k] + Nk[k])) * np.outer(
            dev, dev
        )
    return m, kappa, Psi, nu


def oracle_weighted_cycle(state, xs, xc, w, cfg):
    """One weighted CAVI cycle: softmax E-step, then w_n-weighted M-steps.

    Each point counted w_n times: N_k = sum_n w_n r_nk, weighted mean and
    centered scatter, weight-prior counts likewise weighted.
    """
    xs, xc, w = np.asarray(xs), np.asarray(xc), np.asarray(w)

    # E-step: log rho_nk = E[log pi_k] + sum_m E[log N(x_nm | ...)], row-softmax.
    if cfg.weight_prior == "dp":
        elogpi = _oracle_elogpi_dp(
            np.asarray(state.weights.gamma1), np.asarray(state.weights.gamma2)
        )
    else:
        a = np.asarray(state.weights.alpha_post)
        elogpi = digamma(a) - digamma(a.sum())
    log_rho = (
        elogpi[None, :]
        + _oracle_e_gauss_loglik(
            np.asarray(state.spatial.m),
            np.asarray(state.spatial.kappa),
            np.asarray(state.spatial.Psi),
            np.asarray(state.spatial.nu),
            xs,
        )
        + _oracle_e_gauss_loglik(
            np.asarray(state.color.m),
            np.asarray(state.color.kappa),
            np.asarray(state.color.Psi),
            np.asarray(state.color.nu),
            xc,
        )
    )
    log_rho -= log_rho.max(axis=1, keepdims=True)
    r = np.exp(log_rho)
    r /= r.sum(axis=1, keepdims=True)

    # Weighted sufficient statistics and NIW updates, one modality at a time.
    rw = w[:, None] * r
    Nk = rw.sum(axis=0)
    out = {"r": r}
    for name, x, prior in (
        ("spatial", xs, state.spatial_prior),
        ("color", xc, state.color_prior),
    ):
        K = Nk.shape[0]
        D = x.shape[1]
        xbar = np.empty((K, D))
        S = np.empty((K, D, D))
        for k in range(K):
            xbar[k] = (rw[:, k : k + 1] * x).sum(axis=0) / max(Nk[k], 1e-32)
            dev = x - xbar[k]
            S[k] = (rw[:, k : k + 1] * dev).T @ dev
        out[name] = _oracle_niw_update(
            np.asarray(prior.m),
            np.asarray(prior.kappa),
            np.asarray(prior.Psi),
            np.asarray(prior.nu),
            Nk,
            xbar,
            S,
        )

    # Weight-prior update on the weighted counts.
    if cfg.weight_prior == "dp":
        T = Nk.shape[0]
        if cfg.learn_alpha and state.weights.w1 is not None:
            e_alpha = float(state.weights.w1) / float(state.weights.w2)
        else:
            e_alpha = cfg.alpha
        g1 = np.empty(T - 1)
        g2 = np.empty(T - 1)
        for k in range(T - 1):
            g1[k] = 1.0 + Nk[k]
            g2[k] = e_alpha + Nk[k + 1 :].sum()
        out["gamma1"], out["gamma2"] = g1, g2
        if cfg.learn_alpha:
            elog1mv = digamma(g2) - digamma(g1 + g2)
            out["w1"] = cfg.a0 + (T - 1)
            out["w2"] = cfg.b0 - elog1mv.sum()
    else:
        out["alpha_post"] = cfg.e0 + Nk
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_data(rng, n=90):
    z = rng.integers(0, 3, size=n)
    mu_s = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 0.0], [-4.0, 2.0, 3.0]])
    mu_c = np.array([[0.2, 0.3, 0.4], [0.7, 0.5, 0.2], [0.4, 0.8, 0.6]])
    xs = mu_s[z] + 0.5 * rng.standard_normal((n, 3))
    xc = mu_c[z] + 0.05 * rng.standard_normal((n, 3))
    return jnp.asarray(xs), jnp.asarray(xc)


def _generic_state(seed, xs, xc, cfg):
    """A non-degenerate state: init + one unweighted CAVI cycle."""
    state = _cavi.init_state(seed, xs, xc, cfg)
    return _cavi.cavi_step(state, xs, xc, cfg)


# ---------------------------------------------------------------------------
# Oracle comparison: every posterior quantity of one cycle to <= 1e-8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weight_prior,learn_alpha",
    [("dp", False), ("dp", True), ("dir", False), ("sparse_dir", False)],
)
def test_weighted_cycle_matches_numpy_oracle(weight_prior, learn_alpha):
    rng = np.random.default_rng(0)
    xs, xc = _make_data(rng)
    cfg = Config(weight_prior=weight_prior, T=5, alpha=1.5, learn_alpha=learn_alpha,
                 e0=0.2)
    state = _generic_state(0, xs, xc, cfg)
    w = jnp.asarray(rng.uniform(0.2, 2.5, size=xs.shape[0]))

    got = _w.weighted_cavi_step(state, xs, xc, w, cfg)
    want = oracle_weighted_cycle(state, xs, xc, w, cfg)

    tol = dict(rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(got.r), want["r"], **tol)
    for name in ("spatial", "color"):
        m, kappa, Psi, nu = want[name]
        q = getattr(got, name)
        np.testing.assert_allclose(np.asarray(q.m), m, **tol)
        np.testing.assert_allclose(np.asarray(q.kappa), kappa, **tol)
        np.testing.assert_allclose(np.asarray(q.Psi), Psi, **tol)
        np.testing.assert_allclose(np.asarray(q.nu), nu, **tol)
    if weight_prior == "dp":
        np.testing.assert_allclose(np.asarray(got.weights.gamma1), want["gamma1"], **tol)
        np.testing.assert_allclose(np.asarray(got.weights.gamma2), want["gamma2"], **tol)
        if learn_alpha:
            np.testing.assert_allclose(float(got.weights.w1), want["w1"], **tol)
            np.testing.assert_allclose(float(got.weights.w2), want["w2"], **tol)
    else:
        np.testing.assert_allclose(
            np.asarray(got.weights.alpha_post), want["alpha_post"], **tol
        )


# ---------------------------------------------------------------------------
# Identity 1: w == 1 reduces exactly to the unweighted fit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight_prior", ["dp", "dir"])
def test_unit_weights_reproduce_unweighted_fit(weight_prior):
    rng = np.random.default_rng(1)
    xs, xc = _make_data(rng)
    cfg = Config(weight_prior=weight_prior, T=6, max_iters=30, tol=1e-8)

    state_u, hist_u = _cavi.fit(3, xs, xc, cfg)
    state_w, hist_w = _w.weighted_fit(3, xs, xc, jnp.ones(xs.shape[0]), cfg)

    assert len(hist_w) == len(hist_u)
    np.testing.assert_allclose(hist_w, hist_u, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(state_w.r), np.asarray(state_u.r),
                               rtol=0.0, atol=1e-12)
    for name in ("spatial", "color"):
        qw, qu = getattr(state_w, name), getattr(state_u, name)
        for field in ("m", "kappa", "Psi", "nu"):
            np.testing.assert_allclose(
                np.asarray(getattr(qw, field)), np.asarray(getattr(qu, field)),
                rtol=0.0, atol=1e-12,
            )


def test_unit_weights_single_step_and_elbo():
    rng = np.random.default_rng(2)
    xs, xc = _make_data(rng)
    cfg = Config(weight_prior="dp", T=5, learn_alpha=True)
    state = _generic_state(4, xs, xc, cfg)
    ones = jnp.ones(xs.shape[0])

    su = _cavi.cavi_step(state, xs, xc, cfg)
    sw = _w.weighted_cavi_step(state, xs, xc, ones, cfg)
    np.testing.assert_array_equal(np.asarray(sw.r), np.asarray(su.r))
    np.testing.assert_allclose(
        float(_w.weighted_elbo(sw, xs, xc, ones, cfg)),
        float(_cavi.elbo(su, xs, xc, cfg)),
        rtol=0.0, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Identity 2: integer weight w on point i == duplicating point i w times
# ---------------------------------------------------------------------------


def test_integer_weight_equals_duplication():
    rng = np.random.default_rng(5)
    xs, xc = _make_data(rng, n=60)
    n = xs.shape[0]
    cfg = Config(weight_prior="dp", T=5, alpha=1.0)
    state = _generic_state(7, xs, xc, cfg)

    i, mult = 17, 4
    w = np.ones(n)
    w[i] = float(mult)
    xs_dup = jnp.concatenate([xs, jnp.tile(xs[i : i + 1], (mult - 1, 1))], axis=0)
    xc_dup = jnp.concatenate([xc, jnp.tile(xc[i : i + 1], (mult - 1, 1))], axis=0)

    got = _w.weighted_cavi_step(state, xs, xc, jnp.asarray(w), cfg)
    dup = _w.weighted_cavi_step(state, xs_dup, xc_dup, jnp.ones(n + mult - 1), cfg)

    # Copies of point i get exactly point i's responsibility row.
    np.testing.assert_array_equal(np.asarray(dup.r[n:]), np.tile(np.asarray(dup.r[i]),
                                                                 (mult - 1, 1)))

    # Updates agree exactly up to floating-point summation order.
    tol = dict(rtol=1e-12, atol=1e-12)
    for name in ("spatial", "color"):
        qg, qd = getattr(got, name), getattr(dup, name)
        for field in ("m", "kappa", "Psi", "nu"):
            np.testing.assert_allclose(
                np.asarray(getattr(qg, field)), np.asarray(getattr(qd, field)), **tol
            )
    np.testing.assert_allclose(np.asarray(got.weights.gamma1),
                               np.asarray(dup.weights.gamma1), **tol)
    np.testing.assert_allclose(np.asarray(got.weights.gamma2),
                               np.asarray(dup.weights.gamma2), **tol)

    # The weighted ELBO obeys the same identity.
    np.testing.assert_allclose(
        float(_w.weighted_elbo(got, xs, xc, jnp.asarray(w), cfg)),
        float(_w.weighted_elbo(dup, xs_dup, xc_dup, jnp.ones(n + mult - 1), cfg)),
        rtol=1e-12, atol=0.0,
    )

"""Per-point-weighted full-batch CAVI for DP-Splat (halo down-weighting).

Weighted model: point n contributes its likelihood and assignment factors raised
to the power w_n >= 0, i.e. it is counted w_n times, with fractional counts
allowed. The variational family is unchanged, so the E-step is the standard
responsibility softmax (every fractional copy of a point has the same optimal
q(z_n)), and the weights enter only where the objective sums over points:

    N_k    = sum_n w_n r_nk
    xbar_k = sum_n w_n r_nk x_n / N_k
    S_k    = sum_n w_n r_nk (x_n - xbar_k)(x_n - xbar_k)^T

feed the NIW updates and the weight-prior counts, and the ELBO terms
E[log p(X|Z,theta)], E[log p(Z|v)], E[log q(Z)] each carry a factor w_n.
Parameter-space terms (weights, theta, alpha) are unaffected.

Two exact identities pin the construction down (both tested in
tests/test_weighted_oracle.py):
  * w_n = 1 for all n reproduces dp_splat.cavi bitwise — every array below is
    computed by the same dp_splat function on identical inputs;
  * integer w_i equals physically duplicating point i w_i times, up to
    floating-point summation order.

All conjugate updates are delegated to dp_splat (niw.posterior_update,
priors.dp_update / dir_update / alpha_update); only the w_n-scaling of the
sufficient statistics and the point-sum ELBO terms are implemented here.
"""

import jax.numpy as jnp

from dp_splat import cavi as _cavi
from dp_splat import niw as _niw
from dp_splat import priors as _pr
from dp_splat.cavi import Config, State


def weighted_soft_stats(x: jnp.ndarray, r: jnp.ndarray, w: jnp.ndarray):
    """Weighted soft counts and moments: niw.soft_stats with r_nk -> w_n r_nk.

    x: (N, D) one modality; r: (N, K) responsibilities; w: (N,) point weights.
    Returns (Nk, xbar, S) with S the centered weighted scatter.
    """
    return _niw.soft_stats(x, r * w[:, None])


def weighted_cavi_step(state: State, xs, xc, w, cfg: Config) -> State:
    """One weighted CAVI cycle: dp_splat.cavi.cavi_step with w_n-scaled statistics.

    The E-step (responsibilities) is unweighted; w enters only through the
    sufficient statistics of the NIW and weight-prior updates.
    """
    r = _cavi.responsibilities(state, xs, xc, cfg)
    rw = r * w[:, None]

    Nk_s, xbar_s, S_s = _niw.soft_stats(xs, rw)
    Nk_c, xbar_c, S_c = _niw.soft_stats(xc, rw)
    spatial = _niw.posterior_update(state.spatial_prior, Nk_s, xbar_s, S_s)
    color = _niw.posterior_update(state.color_prior, Nk_c, xbar_c, S_c)
    if cfg.fixed_color_precision:
        color = color._replace(Psi=state.color_prior.Psi, nu=state.color_prior.nu)

    Nk = Nk_s  # same weighted responsibilities, same counts
    if cfg.weight_prior == "dp":
        e_alpha, _ = _cavi._e_alpha_pair(state, cfg)
        g1, g2 = _pr.dp_update(Nk, e_alpha)
        w1 = w2 = None
        if cfg.learn_alpha:
            _, elog1mv = _pr.beta_expectations(g1, g2)
            w1, w2 = _pr.alpha_update(elog1mv, cfg.a0, cfg.b0)
        weights: _cavi.Weights = _pr.StickBreakingPosterior(g1, g2, w1, w2)
    else:
        weights = _pr.DirichletPosterior(_pr.dir_update(Nk, cfg.e0))

    return State(spatial, color, state.spatial_prior, state.color_prior, weights, r)


def weighted_elbo(state: State, xs, xc, w, cfg: Config) -> jnp.ndarray:
    """Weighted ELBO: point-sum terms carry w_n; parameter-space terms are unchanged.

    E[log p(X|Z,theta)] = sum_{n,k} w_n r_nk E[log N(x_n | ...)]
    E[log p(Z|v)]       = sum_{n,k} w_n r_nk E[log pi_k]
    E[log q(Z)]         = sum_{n,k} w_n r_nk log r_nk   (0 log 0 := 0)
    """
    if state.r is None:
        raise ValueError("ELBO needs responsibilities; run weighted_cavi_step first")
    r = state.r
    rw = r * w[:, None]
    ell = _niw.expected_gauss_loglik(state.spatial, xs) + _niw.expected_gauss_loglik(
        state.color, xc
    )
    e_loglik = (rw * ell).sum()
    log_p_z = (rw * _cavi.elogpi(state, cfg)[None, :]).sum()
    log_q_z = jnp.where(r > 0, rw * jnp.log(jnp.where(r > 0, r, 1.0)), 0.0).sum()
    return (
        e_loglik
        + log_p_z
        + _cavi.elbo_log_p_weights(state, cfg)
        + _cavi.elbo_log_p_theta(state)
        + _cavi.elbo_log_p_alpha(state, cfg)
        - log_q_z
        - _cavi.elbo_log_q_weights(state, cfg)
        - _cavi.elbo_log_q_theta(state)
        - _cavi.elbo_log_q_alpha(state, cfg)
    )


def weighted_fit(seed: int, xs, xc, w, cfg: Config, verbose: bool = False):
    """Weighted full-batch CAVI until relative ELBO change < cfg.tol or cfg.max_iters.

    Mirrors dp_splat.cavi.fit exactly (same init, same |dL| / |L| convergence
    test on the weighted ELBO); with w == 1 the state and ELBO history are
    identical to the unweighted fit. Returns (state, elbo_history).
    """
    xs = jnp.asarray(xs)
    xc = jnp.asarray(xc)
    w = jnp.asarray(w, dtype=xs.dtype)
    state = _cavi.init_state(seed, xs, xc, cfg)
    history = []
    prev = -jnp.inf
    for it in range(cfg.max_iters):
        state = weighted_cavi_step(state, xs, xc, w, cfg)
        L = float(weighted_elbo(state, xs, xc, w, cfg))
        history.append(L)
        if verbose:
            print(f"  iter {it:4d}  elbo {L:.6f}")
        if it > 0 and abs(L - prev) < cfg.tol * abs(prev):
            break
        prev = L
    return state, history

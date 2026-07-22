"""Unit-level sanity tests for dp_splat_uav.merge.

Shape/finiteness, symmetry, ordering, and hand-computed cases only; independent
NumPy oracles for the update equations live in the dedicated oracle test modules.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.special import gammaln as sp_gammaln

from dp_splat import niw as dniw
from dp_splat_uav import merge

D = 3


def make_prior(K=1):
    return dniw.make_prior(
        m0=np.zeros(D), kappa0=1.0, Psi0=2.0, nu0=float(D + 3), K=K
    )


def posterior_from_data(x, prior):
    """Hard-assign all points to the single component (r = 1)."""
    r = jnp.ones((x.shape[0], 1))
    Nk, xbar, S = dniw.soft_stats(jnp.asarray(x), r)
    return dniw.posterior_update(prior, Nk, xbar, S)


def sample_cluster(rng, center, n, scale=0.1):
    return center + scale * rng.standard_normal((n, D))


def make_component(rng, spatial_center, color_center, n, prior):
    xs = sample_cluster(rng, spatial_center, n)
    xc = sample_cluster(rng, color_center, n)
    return (
        merge.Component(
            niws=(posterior_from_data(xs, prior), posterior_from_data(xc, prior)),
            count=float(n),
        ),
        xs,
        xc,
    )


# ---------------------------------------------------------------------------
# niw_log_normalizer
# ---------------------------------------------------------------------------


def test_log_normalizer_shape_and_finite():
    prior = make_prior(K=4)
    A = merge.niw_log_normalizer(prior)
    assert A.shape == (4,)
    assert np.all(np.isfinite(np.asarray(A)))


def test_log_normalizer_closed_form_numpy():
    """Check against a direct NumPy evaluation of the documented formula."""
    rng = np.random.default_rng(0)
    prior = make_prior()
    x = sample_cluster(rng, np.array([1.0, -2.0, 0.5]), 50)
    q = posterior_from_data(x, prior)

    kappa = float(q.kappa[0])
    nu = float(q.nu[0])
    Psi = np.asarray(q.Psi[0])
    sign, logdet = np.linalg.slogdet(Psi)
    assert sign > 0
    mvlgamma = (D * (D - 1) / 4.0) * np.log(np.pi) + sum(
        sp_gammaln((nu + 1.0 - i) / 2.0) for i in range(1, D + 1)
    )
    expected = (
        0.5 * D * np.log(2.0 * np.pi)
        - 0.5 * D * np.log(kappa)
        + 0.5 * nu * D * np.log(2.0)
        + mvlgamma
        - 0.5 * nu * logdet
    )
    got = float(merge.niw_log_normalizer(q)[0])
    np.testing.assert_allclose(got, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# match_score
# ---------------------------------------------------------------------------


def test_match_score_symmetry():
    rng = np.random.default_rng(1)
    prior = make_prior()
    qa = posterior_from_data(sample_cluster(rng, np.zeros(D), 40), prior)
    qb = posterior_from_data(sample_cluster(rng, np.ones(D), 60), prior)
    s_ab = float(merge.match_score((qa,), (qb,), (prior,)))
    s_ba = float(merge.match_score((qb,), (qa,), (prior,)))
    assert np.isfinite(s_ab)
    np.testing.assert_allclose(s_ab, s_ba, rtol=1e-12)


def test_match_score_same_cluster_beats_distant():
    rng = np.random.default_rng(2)
    prior = make_prior()
    center = np.array([1.0, 2.0, 3.0])
    q_same_a = posterior_from_data(sample_cluster(rng, center, 100), prior)
    q_same_b = posterior_from_data(sample_cluster(rng, center, 100), prior)
    q_far = posterior_from_data(sample_cluster(rng, center + 50.0, 100), prior)

    s_same = float(merge.match_score((q_same_a,), (q_same_b,), (prior,)))
    s_far = float(merge.match_score((q_same_a,), (q_far,), (prior,)))
    assert s_same > s_far
    assert s_far < 0.0  # distant components must fail the sign test outright


def test_match_score_sums_modalities():
    rng = np.random.default_rng(3)
    prior = make_prior()
    qs_a = posterior_from_data(sample_cluster(rng, np.zeros(D), 30), prior)
    qs_b = posterior_from_data(sample_cluster(rng, np.zeros(D), 30), prior)
    qc_a = posterior_from_data(sample_cluster(rng, 0.5 * np.ones(D), 30), prior)
    qc_b = posterior_from_data(sample_cluster(rng, 0.5 * np.ones(D), 30), prior)
    s_spatial = float(merge.match_score((qs_a,), (qs_b,), (prior,)))
    s_color = float(merge.match_score((qc_a,), (qc_b,), (prior,)))
    s_both = float(merge.match_score((qs_a, qc_a), (qs_b, qc_b), (prior, prior)))
    np.testing.assert_allclose(s_both, s_spatial + s_color, rtol=1e-10)


# ---------------------------------------------------------------------------
# eppf_penalty
# ---------------------------------------------------------------------------


def test_eppf_penalty_hand_computation():
    got = float(merge.eppf_penalty(3.0, 4.0, 2.0))
    expected = sp_gammaln(7.0) - sp_gammaln(3.0) - sp_gammaln(4.0) - np.log(2.0)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_eppf_penalty_smaller_alpha_favors_merge():
    lo = float(merge.eppf_penalty(10.0, 10.0, 0.1))
    hi = float(merge.eppf_penalty(10.0, 10.0, 10.0))
    assert lo > hi


# ---------------------------------------------------------------------------
# merge_components
# ---------------------------------------------------------------------------


def test_merge_two_equals_pooled_posterior():
    rng = np.random.default_rng(4)
    prior = make_prior()
    xa = sample_cluster(rng, np.array([0.5, -1.0, 2.0]), 70)
    xb = sample_cluster(rng, np.array([0.6, -1.1, 2.1]), 30)
    merged = merge.merge_components(
        [posterior_from_data(xa, prior), posterior_from_data(xb, prior)], prior
    )
    pooled = posterior_from_data(np.vstack([xa, xb]), prior)
    for got, expected in zip(merged, pooled, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-9)


def test_merge_three_way_exact():
    rng = np.random.default_rng(5)
    prior = make_prior()
    parts = [sample_cluster(rng, np.zeros(D), n) for n in (25, 40, 15)]
    merged = merge.merge_components(
        [posterior_from_data(x, prior) for x in parts], prior
    )
    pooled = posterior_from_data(np.vstack(parts), prior)
    for got, expected in zip(merged, pooled, strict=True):
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-9)


# ---------------------------------------------------------------------------
# transitive_merge_sets
# ---------------------------------------------------------------------------


def test_transitive_merge_three_tiles():
    matches = [
        (("t1", 0), ("t2", 1)),
        (("t2", 1), ("t3", 2)),  # chains with the first edge -> 3-way set
        (("t1", 5), ("t3", 7)),
    ]
    sets = merge.transitive_merge_sets(matches)
    assert len(sets) == 2
    assert {("t1", 0), ("t2", 1), ("t3", 2)} in sets
    assert {("t1", 5), ("t3", 7)} in sets


def test_transitive_merge_empty():
    assert merge.transitive_merge_sets([]) == []


# ---------------------------------------------------------------------------
# rebuild_weights
# ---------------------------------------------------------------------------


def test_rebuild_weights_hand_computation():
    a, b, order = merge.rebuild_weights(jnp.array([2.0, 5.0, 3.0]), alpha=1.5)
    np.testing.assert_array_equal(np.asarray(order), [1, 2, 0])
    # sorted counts [5, 3, 2]; a_k = 1 + N_k, b_k = alpha + sum_{j>k} N_j; last stick v_K := 1
    np.testing.assert_allclose(np.asarray(a), [6.0, 4.0])
    np.testing.assert_allclose(np.asarray(b), [1.5 + 5.0, 1.5 + 2.0])
    assert a.shape == b.shape == (2,)


# ---------------------------------------------------------------------------
# hungarian_match
# ---------------------------------------------------------------------------


def test_hungarian_matches_corresponding_components():
    rng = np.random.default_rng(6)
    prior = make_prior()
    priors = (prior, prior)
    near = np.zeros(D)
    far = 20.0 * np.ones(D)
    red = np.array([0.9, 0.1, 0.1])
    blue = np.array([0.1, 0.1, 0.9])

    a0, _, _ = make_component(rng, near, red, 150, prior)
    a1, _, _ = make_component(rng, far, blue, 150, prior)
    b0, _, _ = make_component(rng, near, red, 150, prior)
    b1, _, _ = make_component(rng, far, blue, 150, prior)

    mask = np.ones((2, 2), dtype=bool)
    pairs = merge.hungarian_match([a0, a1], [b0, b1], priors, 1.0, mask)
    assert sorted(pairs) == [(0, 0), (1, 1)]


def test_hungarian_adjacency_mask_blocks_pairs():
    rng = np.random.default_rng(7)
    prior = make_prior()
    priors = (prior, prior)
    near = np.zeros(D)
    far = 20.0 * np.ones(D)
    gray = 0.5 * np.ones(D)

    a0, _, _ = make_component(rng, near, gray, 150, prior)
    a1, _, _ = make_component(rng, far, gray, 150, prior)
    b0, _, _ = make_component(rng, near, gray, 150, prior)
    b1, _, _ = make_component(rng, far, gray, 150, prior)

    # Only the (1, 1) pair is a halo-adjacent candidate.
    mask = np.array([[False, False], [False, True]])
    pairs = merge.hungarian_match([a0, a1], [b0, b1], priors, 1.0, mask)
    assert pairs == [(1, 1)]


def test_hungarian_sign_test_rejects_distant_pairs():
    rng = np.random.default_rng(8)
    prior = make_prior()
    priors = (prior, prior)
    gray = 0.5 * np.ones(D)

    a0, _, _ = make_component(rng, np.zeros(D), gray, 150, prior)
    b0, _, _ = make_component(rng, 50.0 * np.ones(D), gray, 150, prior)

    mask = np.ones((1, 1), dtype=bool)
    pairs = merge.hungarian_match([a0], [b0], priors, 1.0, mask)
    assert pairs == []


def test_hungarian_empty_inputs():
    prior = make_prior()
    assert merge.hungarian_match([], [], (prior, prior), 1.0, np.ones((0, 0))) == []

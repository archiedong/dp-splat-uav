"""Cross-tile component matching and exact NIW merging.

Tiles are fit independently with dp_splat; components describing the same surface
across a tile boundary are identified by a conjugate-marginal match score and merged
exactly in natural-parameter space.

Because NIW posterior naturals are prior naturals plus sufficient statistics
(lam_k = lam_0 + s_k), the natural-space combination

    lam_A + lam_B - lam_0 = lam_0 + s_A + s_B

is exactly the posterior a single component would have reached on the pooled data,
for any number of operands. The match score

    Delta L = A(lam_A + lam_B - lam_0) - A(lam_A) - A(lam_B) + A(lam_0)

is therefore the log Bayes factor of "one component generated both statistic blocks"
against "two independent components" (base-measure terms cancel since counts add).
The partition side of the decision is the DP EPPF ratio of the merged versus separate
clusterings (Campbell et al. 2015, "Streaming, Distributed Variational Inference for
Bayesian Nonparametrics", Eqs. 13-14); see `eppf_penalty`. A candidate pair is
accepted iff Delta L (summed over modalities) plus the EPPF term is positive.

All NIW containers use dp_splat's convention: Inverse-Wishart scale Psi stored, Wishart
on the precision with W = Psi^{-1}, leading component axis K. Run in float64; the
natural-space round trip subtracts a rank-one term (see dp_splat.svi module docstring).
"""

from typing import NamedTuple, Sequence

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from scipy.optimize import linear_sum_assignment

from dp_splat import niw as _niw
from dp_splat import priors as _pr
from dp_splat import svi as _svi

# Finite stand-in for -inf in the assignment matrix: linear_sum_assignment rejects
# matrices it deems infeasible when entries are infinite, and any pair carrying this
# score fails the sign test regardless.
_MASKED_SCORE = -1e12


class Component(NamedTuple):
    """One fitted mixture component of a tile, as used for cross-tile matching.

    niws:  per-modality NIW posteriors (spatial, color), each with leading axis K = 1
    count: soft count N_k = sum_n r_nk (halo down-weighting already applied)
    """

    niws: tuple
    count: float


def niw_log_normalizer(q: _niw.NIW) -> jnp.ndarray:
    """Log normalizer A(lambda) of the NIW in dp_splat's parameterization, (K,).

    With p(mu, Lambda) = N(mu | m, (kappa Lambda)^{-1}) Wishart(Lambda | Psi^{-1}, nu)
    and naturals (n1, n2, n3, n4) = (kappa, kappa m, nu, Psi + kappa m m^T), the
    unnormalized exponential-family density is

        |Lambda|^{(n3 - D)/2} exp(-1/2 tr(Lambda n4) - 1/2 n1 mu^T Lambda mu
                                  + mu^T Lambda n2).

    Integrating mu (Gaussian with precision n1 Lambda) and then Lambda (Wishart
    integral with scale n4 - n2 n2^T / n1 = Psi, dof n3 = nu) gives the closed form

        A = (D/2) log(2 pi) - (D/2) log kappa + (nu D / 2) log 2
            + log Gamma_D(nu / 2) - (nu / 2) log |Psi|,

    where log Gamma_D(a) = D(D-1)/4 log pi + sum_{i=1}^{D} lgamma(a + (1 - i)/2)
    is the multivariate log-gamma. log|Psi| is computed with the same ridged
    Cholesky as the rest of dp_splat (niw.logdet_psi).
    """
    D = q.dim
    i = jnp.arange(1, D + 1)
    mvlgamma = (D * (D - 1) / 4.0) * jnp.log(jnp.pi) + gammaln(
        (q.nu[..., None] + 1.0 - i) / 2.0
    ).sum(-1)
    return (
        0.5 * D * jnp.log(2.0 * jnp.pi)
        - 0.5 * D * jnp.log(q.kappa)
        + 0.5 * q.nu * D * jnp.log(2.0)
        + mvlgamma
        - 0.5 * q.nu * _niw.logdet_psi(q)
    )


def match_score(
    niw_A: Sequence[_niw.NIW],
    niw_B: Sequence[_niw.NIW],
    niw_prior: Sequence[_niw.NIW],
) -> jnp.ndarray:
    """Log Bayes factor Delta L for merging two components, summed over modalities.

    Delta L = A(lam_A + lam_B - lam_0) - A(lam_A) - A(lam_B) + A(lam_0)

    per modality, with the combination done in natural space (svi.niw_to_natural /
    niw_from_natural), so lam_A + lam_B - lam_0 is exactly the pooled posterior.
    Arguments are matching sequences of NIW containers, one per modality (spatial,
    color), each with leading axis K = 1; returns a scalar.
    """
    total = jnp.asarray(0.0)
    for q_a, q_b, q_0 in zip(niw_A, niw_B, niw_prior, strict=True):
        nat = [
            a + b - p
            for a, b, p in zip(
                _svi.niw_to_natural(q_a),
                _svi.niw_to_natural(q_b),
                _svi.niw_to_natural(q_0),
                strict=True,
            )
        ]
        q_ab = _svi.niw_from_natural(*nat)
        total = total + (
            niw_log_normalizer(q_ab)
            - niw_log_normalizer(q_a)
            - niw_log_normalizer(q_b)
            + niw_log_normalizer(q_0)
        ).sum()
    return total


def eppf_penalty(N_A, N_B, alpha) -> jnp.ndarray:
    """DP partition term of the merge decision: log EPPF(merged) - log EPPF(separate).

    The DP EPPF is p(partition) proportional to alpha^{#clusters} prod_c Gamma(N_c)
    (Antoniak 1974), so replacing two clusters of sizes N_A, N_B by one of size
    N_A + N_B changes the log EPPF by

        lgamma(N_A + N_B) - lgamma(N_A) - lgamma(N_B) - log(alpha);

    equivalently log(alpha) + lgamma(N_A) + lgamma(N_B) - lgamma(N_A + N_B) enters
    the "keep separate" side of the sign test. This is the merge/no-merge EPPF ratio
    of Campbell et al. (2015), Eqs. 13-14, specialized to the DP. Soft counts may be
    non-integer; the Gamma form extends the EPPF continuously (requires N > 0).
    """
    return gammaln(N_A + N_B) - gammaln(N_A) - gammaln(N_B) - jnp.log(alpha)


def hungarian_match(
    tileA_components: Sequence[Component],
    tileB_components: Sequence[Component],
    prior: Sequence[_niw.NIW],
    alpha,
    halo_adjacency_mask,
) -> list:
    """Optimal one-to-one matching of components across two adjacent tiles.

    Builds the score matrix S[i, j] = match_score(A_i, B_j, prior)
    + eppf_penalty(N_i, N_j, alpha), with pairs whose halos do not overlap
    (halo_adjacency_mask[i, j] == False) excluded from consideration, solves the
    assignment with scipy.optimize.linear_sum_assignment (maximize), and keeps an
    assigned pair iff its total score is strictly positive (sign test).

    prior: per-modality NIW containers matching Component.niws.
    halo_adjacency_mask: boolean (len(A), len(B)) candidate mask.
    Returns a list of (i, j) index pairs.
    """
    n_a, n_b = len(tileA_components), len(tileB_components)
    if n_a == 0 or n_b == 0:
        return []
    mask = np.asarray(halo_adjacency_mask, dtype=bool)
    if mask.shape != (n_a, n_b):
        raise ValueError(f"mask shape {mask.shape} != ({n_a}, {n_b})")
    scores = np.full((n_a, n_b), _MASKED_SCORE)
    for i, comp_a in enumerate(tileA_components):
        for j, comp_b in enumerate(tileB_components):
            if mask[i, j]:
                scores[i, j] = float(
                    match_score(comp_a.niws, comp_b.niws, prior)
                    + eppf_penalty(comp_a.count, comp_b.count, alpha)
                )
    rows, cols = linear_sum_assignment(scores, maximize=True)
    return [
        (int(i), int(j))
        for i, j in zip(rows, cols)
        if mask[i, j] and scores[i, j] > 0.0
    ]


def merge_components(niw_list: Sequence[_niw.NIW], prior: _niw.NIW) -> _niw.NIW:
    """Exact m-way merge of NIW posteriors (one modality) in natural space.

    lam_merged = sum_i lam_i - (m - 1) lam_0.

    Each posterior is lam_0 plus its own sufficient statistics, so the prior is
    counted once and the merged parameters equal the single-component posterior on
    the pooled statistics — exact for any m >= 2 (identity for m = 1).
    """
    if not niw_list:
        raise ValueError("niw_list must contain at least one component")
    nats = [_svi.niw_to_natural(q) for q in niw_list]
    prior_nat = _svi.niw_to_natural(prior)
    m = len(nats)
    merged = [
        sum(component[k] for component in nats) - (m - 1) * prior_nat[k]
        for k in range(4)
    ]
    return _svi.niw_from_natural(*merged)


def transitive_merge_sets(pairwise_matches) -> list:
    """Connected components of the cross-tile match graph (m-way merge sets).

    pairwise_matches: iterable of (node_a, node_b) edges with hashable node ids,
    e.g. (tile_id, component_index). Pairwise matches are transitive by
    construction — the natural-space merge is exact for any m — so all nodes in one
    connected component merge together. Returns a list of sets (each of size >= 2),
    ordered by first appearance of a member edge.
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    order: list = []
    for a, b in pairwise_matches:
        for node in (a, b):
            if node not in parent:
                order.append(node)
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    groups: dict = {}
    for node in order:
        groups.setdefault(find(node), set()).add(node)
    return [members for members in groups.values() if len(members) >= 2]


def rebuild_weights(pooled_counts, alpha):
    """Rebuild global truncated stick-breaking weights from pooled soft counts.

    Components are put in size-biased order (pooled N_k descending), then the
    stick-breaking posterior is recomputed from scratch with priors.dp_update:

        a_k = 1 + N_k,   b_k = alpha + sum_{j > k} N_j,   k = 1..K-1,  v_K := 1.

    Beta parameters are functions of the pooled counts only and are never added
    across tiles (Beta parameters are not natural parameters of the joint model).

    Returns (a, b, order): a and b of shape (K-1,) in dp_splat's truncation
    convention, and order the (K,) permutation that sorts counts descending —
    apply it to the merged component list so weights and components stay aligned.
    """
    counts = jnp.asarray(pooled_counts)
    order = jnp.argsort(-counts, stable=True)
    a, b = _pr.dp_update(counts[order], alpha)
    return a, b, order

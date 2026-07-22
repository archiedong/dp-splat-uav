"""Partition-of-unity stitching of per-tile DP-Splat predictive densities.

Gating: each tile carries a separable bump over the ground plane,

    raw_t(s) = prod_{d in {x,y}} ramp_d(s_d),

where ramp_d is 1 on the tile's core interval [lo_d, hi_d], falls linearly to 0
across the halo band of width h on either side, and is 0 beyond. The gates are the
normalized bumps g_t(s) = raw_t(s) / sum_u raw_u(s), so sum_t g_t(s) = 1 at every
point of the scene bounding box by construction — including edges and the corner
regions where four dilated tiles overlap. Inside the bounding box the denominator
is >= 1 because every point lies in at least one core (raw = 1 there).

The stitched predictive is the gated mixture of the per-tile mixture-of-Student-t
predictives (dp_splat.predictive):

    log p(x) = log sum_t g_t(s) p_t(x),
    p_t(x) = sum_k E[pi_k^t] St(s; ...) St(c; ...).
"""

from typing import NamedTuple, Sequence

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from dp_splat.cavi import Config, State
from dp_splat.predictive import StudentT, niw_predictive, student_logpdf
from dp_splat.prune import expected_pi

from .tiling import TileGrid


class TilePredictive(NamedTuple):
    """One tile's posterior predictive: log mixture weights and per-modality Student-t."""

    log_pi: jnp.ndarray  # (K,) log E[pi_k]
    spatial: StudentT
    color: StudentT


def tile_predictive(state: State, cfg: Config) -> TilePredictive:
    """Extract a tile's predictive mixture from its fitted DP-Splat state.

    A zero-component state (a tile skipped as unfittable) yields a zero-length
    predictive: stick-breaking E[pi] on empty sticks would otherwise place all
    mass on the remainder and fabricate a weight with no matching component.
    """
    if state.spatial.m.shape[0] == 0:
        return TilePredictive(
            log_pi=jnp.zeros((0,)),
            spatial=niw_predictive(state.spatial),
            color=niw_predictive(state.color),
        )
    log_pi = jnp.log(expected_pi(state, cfg) + 1e-300)
    return TilePredictive(
        log_pi=log_pi,
        spatial=niw_predictive(state.spatial),
        color=niw_predictive(state.color),
    )


def gate_weights(grid: TileGrid, s: jnp.ndarray) -> jnp.ndarray:
    """Normalized partition-of-unity gates g_t(s) -> (N, T); only s[:, :2] is used.

    Rows sum to 1 for every point of the scene bounding box. Points outside every
    dilated tile have no defined gate and raise.
    """
    xy = jnp.asarray(s)[:, :2]  # (N, 2)
    lo = jnp.asarray(grid.core_lo)  # (T, 2)
    hi = jnp.asarray(grid.core_hi)
    h = grid.halo

    d = xy[:, None, :]  # (N, 1, 2) against (T, 2)
    if h > 0:
        left = (d - (lo - h)) / h
        right = ((hi + h) - d) / h
        profile = jnp.clip(jnp.minimum(jnp.minimum(left, right), 1.0), 0.0, 1.0)
    else:
        profile = ((d >= lo) & (d <= hi)).astype(xy.dtype)
    raw = profile.prod(axis=-1)  # (N, T)

    total = raw.sum(axis=1, keepdims=True)
    if bool(jnp.any(total <= 0.0)):
        raise ValueError("some points fall outside every halo-dilated tile")
    return raw / total


def stitched_logpdf(
    tiles_predictives: Sequence[TilePredictive],
    grid: TileGrid,
    s: jnp.ndarray,
    c: jnp.ndarray,
) -> jnp.ndarray:
    """log p(x) = log sum_t g_t(s) p_t(s, c) at query points (s (N, Ds), c (N, Dc)) -> (N,).

    tiles_predictives must be in the grid's tile order (Morton order).
    """
    if len(tiles_predictives) != grid.core_lo.shape[0]:
        raise ValueError("one predictive per tile required, in grid order")
    g = gate_weights(grid, s)  # (N, T)
    log_pt = jnp.stack(
        [
            logsumexp(
                tp.log_pi[None, :]
                + student_logpdf(tp.spatial, s)
                + student_logpdf(tp.color, c),
                axis=1,
            )
            for tp in tiles_predictives
        ],
        axis=1,
    )  # (N, T)
    log_g = jnp.where(g > 0.0, jnp.log(jnp.where(g > 0.0, g, 1.0)), -jnp.inf)
    return logsumexp(log_g + log_pt, axis=1)

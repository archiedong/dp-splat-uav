"""Seam baselines for tiled fits: mean-in-tile crop and Voronoi ownership.

Each per-tile DP-Splat fit covers its tile core plus a halo, so components near a
seam are represented by more than one tile. Both baselines resolve the duplication
by hard selection on the posterior spatial means m_k, then pool the survivors into
one global mixture with renormalized weights:

  crop_mean_in_tile:  keep component k of tile t iff m_k lies in t's core
                      (VastGaussian-style boundary crop).
  voronoi_ownership:  keep component k of tile t iff t's core center is the nearest
                      of all core centers to m_k (Kerbl-style chunk ownership);
                      duplicates of the same structure held by other tiles are
                      dropped because their means resolve to a single owner.

Pooled weights are w_{tk} proportional to mass_t * E[pi_{tk}], where mass_t defaults
to the tile's soft point count (sum of responsibilities) so tiles contribute in
proportion to the data they explain. The result is a GlobalMixture whose components
are NIW posteriors, scored with the same mixture-of-Students predictive as
dp_splat.predictive (mixture_heldout_loglik below).
"""

from typing import NamedTuple, Optional, Sequence

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from dp_splat import niw as _niw
from dp_splat import predictive as _pred
from dp_splat.cavi import Config, State
from dp_splat.prune import expected_pi


class Tile(NamedTuple):
    """Axis-aligned tile core (halo excluded), half-open: lo <= s < hi per axis."""

    lo: jnp.ndarray  # (Ds,)
    hi: jnp.ndarray  # (Ds,)

    @property
    def center(self) -> jnp.ndarray:
        return 0.5 * (self.lo + self.hi)


class TileFit(NamedTuple):
    """One tile's converged DP-Splat fit and the Config it was fit with."""

    state: State
    cfg: Config


class GlobalMixture(NamedTuple):
    """Stitched global mixture of K retained components.

    weights sum to 1; tile_index[k] / component_index[k] record which tile fit and
    which local component each retained component came from.
    """

    spatial: _niw.NIW
    color: _niw.NIW
    weights: jnp.ndarray  # (K,)
    tile_index: jnp.ndarray  # (K,) int
    component_index: jnp.ndarray  # (K,) int


def _take(q: _niw.NIW, idx: jnp.ndarray) -> _niw.NIW:
    return _niw.NIW(m=q.m[idx], kappa=q.kappa[idx], Psi=q.Psi[idx], nu=q.nu[idx])


def _default_masses(tile_fits: Sequence[TileFit]) -> jnp.ndarray:
    """Soft point count per tile when every fit carries responsibilities, else uniform."""
    if all(f.state.r is not None for f in tile_fits):
        return jnp.asarray([float(f.state.r.sum()) for f in tile_fits])
    return jnp.ones(len(tile_fits))


def _assemble(
    tile_fits: Sequence[TileFit],
    keep_per_tile: Sequence[jnp.ndarray],
    tile_masses: Optional[Sequence[float]],
) -> GlobalMixture:
    masses = (
        jnp.asarray(tile_masses, dtype=jnp.float64)
        if tile_masses is not None
        else _default_masses(tile_fits)
    )
    if masses.shape[0] != len(tile_fits):
        raise ValueError(f"expected {len(tile_fits)} tile masses, got {masses.shape[0]}")

    spatial, color, weights, tidx, cidx = [], [], [], [], []
    for t, (fit, keep) in enumerate(zip(tile_fits, keep_per_tile)):
        if keep.shape[0] == 0:
            continue
        pi = expected_pi(fit.state, fit.cfg)
        spatial.append(_take(fit.state.spatial, keep))
        color.append(_take(fit.state.color, keep))
        weights.append(masses[t] * pi[keep])
        tidx.append(jnp.full(keep.shape[0], t, dtype=jnp.int32))
        cidx.append(keep.astype(jnp.int32))
    if not weights:
        raise ValueError("no components retained by any tile")

    w = jnp.concatenate(weights)
    return GlobalMixture(
        spatial=_niw.NIW(*(jnp.concatenate([getattr(q, f) for q in spatial]) for f in _niw.NIW._fields)),
        color=_niw.NIW(*(jnp.concatenate([getattr(q, f) for q in color]) for f in _niw.NIW._fields)),
        weights=w / w.sum(),
        tile_index=jnp.concatenate(tidx),
        component_index=jnp.concatenate(cidx),
    )


def crop_mean_in_tile(
    tile_fits: Sequence[TileFit],
    tiles: Sequence[Tile],
    tile_masses: Optional[Sequence[float]] = None,
) -> GlobalMixture:
    """Mean-in-tile crop: keep components whose posterior spatial mean lies in the
    owning tile's core, pool across tiles, renormalize weights.

    Cores are half-open boxes, so a mean exactly on a shared face belongs to the
    tile whose core starts there — with non-overlapping cores each mean has at most
    one owner. tile_masses overrides the per-tile pooling mass (default: soft point
    counts when available, else uniform).
    """
    if len(tile_fits) != len(tiles):
        raise ValueError(f"{len(tile_fits)} fits vs {len(tiles)} tiles")
    keep_per_tile = []
    for fit, tile in zip(tile_fits, tiles):
        m = fit.state.spatial.m  # (K, Ds)
        inside = jnp.all((m >= tile.lo) & (m < tile.hi), axis=1)
        keep_per_tile.append(jnp.nonzero(inside)[0])
    return _assemble(tile_fits, keep_per_tile, tile_masses)


def voronoi_ownership(
    tile_fits: Sequence[TileFit],
    tiles: Sequence[Tile],
    tile_masses: Optional[Sequence[float]] = None,
) -> GlobalMixture:
    """Voronoi ownership: assign every component to the tile whose core center is
    nearest its posterior spatial mean; a tile retains only the components it owns.

    Halo duplicates of the same structure resolve to a single owning tile and the
    non-owners' copies are dropped. Distance ties break toward the lowest tile
    index (argmin convention). tile_masses as in crop_mean_in_tile.
    """
    if len(tile_fits) != len(tiles):
        raise ValueError(f"{len(tile_fits)} fits vs {len(tiles)} tiles")
    centers = jnp.stack([t.center for t in tiles])  # (T, Ds)
    keep_per_tile = []
    for t, fit in enumerate(tile_fits):
        m = fit.state.spatial.m  # (K, Ds)
        d2 = ((m[:, None, :] - centers[None, :, :]) ** 2).sum(-1)  # (K, T)
        owner = jnp.argmin(d2, axis=1)
        keep_per_tile.append(jnp.nonzero(owner == t)[0])
    return _assemble(tile_fits, keep_per_tile, tile_masses)


def mixture_heldout_loglik(mix: GlobalMixture, xs: jnp.ndarray, xc: jnp.ndarray) -> jnp.ndarray:
    """Mean per-point held-out log predictive density under the stitched mixture.

    Identical algebra to dp_splat.predictive.heldout_loglik — each retained NIW
    posterior contributes its Student-t predictive per modality — with the pooled
    renormalized weights in place of a single fit's E[pi].
    """
    logw = jnp.log(mix.weights + 1e-300)
    ll = (
        logw[None, :]
        + _pred.student_logpdf(_pred.niw_predictive(mix.spatial), xs)
        + _pred.student_logpdf(_pred.niw_predictive(mix.color), xc)
    )
    return logsumexp(ll, axis=1).mean()

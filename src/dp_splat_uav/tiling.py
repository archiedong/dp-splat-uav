"""Axis-aligned ground-plane tiling with halo overlap for distributed DP-Splat fits.

The scene's spatial extent is partitioned in the (x, y) ground plane only — the
vertical axis is never split. The core rectangles of an m-by-n grid partition the
bounding box exactly; each tile's point set is its core dilated by a halo margin of
physical width ``halo``, so neighboring tiles share the points in the halo bands.

Ownership weights: a point contained in k dilated tiles carries weight w_n = 1/k in
every tile that contains it. Then

    sum_t sum_{n in tile t} w_n = sum_n k_n * (1/k_n) = N,

so allocating per-tile DP concentrations proportionally to the weighted point count,

    alpha_t = alpha_global * (sum_{n in tile t} w_n) / N,

gives sum_t alpha_t = alpha_global exactly.

Tiles are ordered by the Morton (Z-order) code of their integer grid index, which is
deterministic and preserves 2D locality in the 1D tile ordering.
"""

import math
from typing import NamedTuple

import numpy as np


class TileGrid(NamedTuple):
    """Tile geometry, in Morton order. Cores partition [bbox_lo, bbox_hi] in (x, y)."""

    core_lo: np.ndarray  # (T, 2) core rectangle lower corners
    core_hi: np.ndarray  # (T, 2) core rectangle upper corners
    halo: float  # halo margin width, physical units, >= 0
    bbox_lo: np.ndarray  # (2,) scene bounding box, ground plane
    bbox_hi: np.ndarray  # (2,)
    shape: tuple  # (nx, ny) grid dimensions
    index: np.ndarray  # (T, 2) integer grid index (ix, iy) of each tile
    morton: np.ndarray  # (T,) Morton code of (ix, iy), uint64, strictly increasing


class TileAssignment(NamedTuple):
    """Point-to-tile assignment for one point cloud against a TileGrid."""

    indices: tuple  # length-T tuple of int arrays: point indices owned by each tile
    ownership: np.ndarray  # (N,) number of tiles containing each point, >= 1
    weight: np.ndarray  # (N,) w_n = 1 / ownership_n
    alpha: np.ndarray  # (T,) per-tile DP concentration, sums to alpha_global


def _interleave_zeros(v: np.ndarray) -> np.ndarray:
    """Spread the low 32 bits of v so bit i moves to bit 2i (zeros interleaved)."""
    v = np.asarray(v, dtype=np.uint64) & np.uint64(0xFFFFFFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v << np.uint64(2))) & np.uint64(0x3333333333333333)
    v = (v | (v << np.uint64(1))) & np.uint64(0x5555555555555555)
    return v


def morton_key(ix, iy) -> np.ndarray:
    """Morton (Z-order) code of grid index (ix, iy): x bits in even positions, y in odd."""
    return _interleave_zeros(ix) | (_interleave_zeros(iy) << np.uint64(1))


def choose_grid_shape(n_points: int, extent, target_points: float) -> tuple:
    """Grid dimensions (nx, ny) with nx*ny >= ceil(n_points / target_points) tiles,
    split between the axes in proportion to the ground-plane aspect ratio so tiles
    stay roughly square. Deterministic in its inputs."""
    n_tiles = max(1, math.ceil(n_points / float(target_points)))
    ex, ey = float(extent[0]), float(extent[1])
    if ex <= 0.0 and ey <= 0.0:
        return (1, 1)
    if ey <= 0.0:
        return (n_tiles, 1)
    if ex <= 0.0:
        return (1, n_tiles)
    nx = int(round(math.sqrt(n_tiles * ex / ey)))
    nx = min(max(nx, 1), n_tiles)
    ny = math.ceil(n_tiles / nx)
    return (nx, ny)


def make_grid(s: np.ndarray, halo: float, target_points: float = 1e7) -> TileGrid:
    """Build the tile grid for spatial points s (N, Ds), Ds >= 2; only s[:, :2] is used.

    The grid shape is chosen so the expected per-tile core point count is at most
    ``target_points``. ``halo`` is the physical dilation width applied to every core
    rectangle when assigning points (see assign_points).
    """
    if halo < 0:
        raise ValueError("halo must be >= 0")
    xy = np.asarray(s, dtype=np.float64)[:, :2]
    bbox_lo = xy.min(axis=0)
    bbox_hi = xy.max(axis=0)
    nx, ny = choose_grid_shape(xy.shape[0], bbox_hi - bbox_lo, target_points)

    xe = np.linspace(bbox_lo[0], bbox_hi[0], nx + 1)
    ye = np.linspace(bbox_lo[1], bbox_hi[1], ny + 1)
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    ix, iy = ix.ravel(), iy.ravel()
    order = np.argsort(morton_key(ix, iy))
    ix, iy = ix[order], iy[order]

    core_lo = np.stack([xe[ix], ye[iy]], axis=1)
    core_hi = np.stack([xe[ix + 1], ye[iy + 1]], axis=1)
    return TileGrid(
        core_lo=core_lo,
        core_hi=core_hi,
        halo=float(halo),
        bbox_lo=bbox_lo,
        bbox_hi=bbox_hi,
        shape=(nx, ny),
        index=np.stack([ix, iy], axis=1),
        morton=morton_key(ix, iy),
    )


def assign_points(grid: TileGrid, s: np.ndarray, alpha_global: float = 1.0) -> TileAssignment:
    """Assign spatial points s (N, Ds) to tiles; only s[:, :2] is used.

    A point belongs to tile t iff it lies in t's halo-dilated core rectangle
    (inclusive boundaries). Weights are w_n = 1 / #owning-tiles and per-tile
    concentrations are alpha_t = alpha_global * (sum of w_n over the tile) / N.
    Every point must fall in at least one tile (guaranteed for the cloud the grid
    was built from; arbitrary query points outside all dilated tiles raise).
    """
    xy = np.asarray(s, dtype=np.float64)[:, :2]
    n = xy.shape[0]
    lo = grid.core_lo - grid.halo  # (T, 2)
    hi = grid.core_hi + grid.halo

    ownership = np.zeros(n, dtype=np.int64)
    indices = []
    for t in range(lo.shape[0]):
        inside = np.all((xy >= lo[t]) & (xy <= hi[t]), axis=1)
        ownership += inside
        indices.append(np.nonzero(inside)[0])
    if np.any(ownership == 0):
        raise ValueError("some points fall outside every halo-dilated tile")

    weight = 1.0 / ownership
    alpha = alpha_global * np.array([weight[idx].sum() for idx in indices]) / n
    return TileAssignment(
        indices=tuple(indices), ownership=ownership, weight=weight, alpha=alpha
    )

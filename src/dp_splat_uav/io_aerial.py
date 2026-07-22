"""Aerial point-cloud loaders (LAS/LAZ via laspy, PLY via plyfile) and scene centering.

Every loader returns the same triple:

  s:      (N, 3) float64 spatial coordinates in meters (native CRS axis order x, y, z)
  c:      (N, 3) float64 RGB in [0, 1]
  extras: dict of (N,) numpy arrays with auxiliary per-point fields
          (LAS classification, PLY label/class/scalar fields), subsampled
          consistently with s and c.

Loaders are deliberately numpy-only; conversion to jnp happens at fit time so that
subsampling and centering never round-trip through the accelerator.
"""

import pathlib
from typing import Optional, Union

import numpy as np

RngLike = Union[None, int, np.random.Generator]

_SPATIAL_FIELDS = ("x", "y", "z")
_COLOR_FIELDS = ("red", "green", "blue")


def _as_rng(rng: RngLike) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _subsample_indices(n: int, subsample: Optional[int], rng: RngLike) -> Optional[np.ndarray]:
    """Uniform without-replacement indices, or None when no subsampling applies."""
    if subsample is None or subsample >= n:
        return None
    return _as_rng(rng).choice(n, size=subsample, replace=False)


def _rgb_to_unit(rgb: np.ndarray) -> np.ndarray:
    """Map integer RGB to [0, 1].

    The LAS spec stores color in 16-bit fields, but many producers write 8-bit
    values into them; the divisor is therefore chosen from the observed range
    (max > 255 means genuine 16-bit).
    """
    rgb = rgb.astype(np.float64)
    denom = 65535.0 if rgb.max(initial=0.0) > 255.0 else 255.0
    return rgb / denom


def load_laz(path, subsample: Optional[int] = None, rng: RngLike = None):
    """Load a LAS/LAZ point cloud (H3D, STPLS3D conventions).

    Returns (s, c, extras); see module docstring. Coordinates come from laspy's
    scaled float64 x/y/z (offset + scale already applied). RGB is required and
    mapped to [0, 1]; the LAS classification field is placed in
    extras["classification"] when the point format carries it.

    subsample: keep this many points, sampled uniformly without replacement.
    rng: seed or numpy Generator for the subsample draw.
    """
    import laspy

    las = laspy.read(str(path))
    dims = set(las.point_format.dimension_names)
    if not set(_COLOR_FIELDS) <= dims:
        raise ValueError(
            f"{path}: point format {las.point_format.id} has no RGB fields; "
            "colored input is required"
        )

    s = np.column_stack(
        [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]
    ).astype(np.float64)
    c = _rgb_to_unit(
        np.column_stack([np.asarray(las[f], dtype=np.uint32) for f in _COLOR_FIELDS])
    )
    extras = {}
    if "classification" in dims:
        extras["classification"] = np.asarray(las.classification).copy()

    idx = _subsample_indices(s.shape[0], subsample, rng)
    if idx is not None:
        s, c = s[idx], c[idx]
        extras = {k: v[idx] for k, v in extras.items()}
    return s, c, extras


def load_ply(path, subsample: Optional[int] = None, rng: RngLike = None):
    """Load a PLY point cloud (H3D / GauU-Scene conventions).

    Expects vertex properties x, y, z plus red, green, blue. Integer color is
    scaled by its dtype maximum (uint8 -> 255, uint16 -> 65535); float color is
    assumed to be in [0, 1] already. All remaining vertex properties (label,
    class, scalar_* exports, intensity, ...) are passed through in extras.

    Returns (s, c, extras); see module docstring.
    """
    from plyfile import PlyData

    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = v.data.dtype.names
    missing = [f for f in _SPATIAL_FIELDS if f not in names]
    if missing:
        raise ValueError(f"{path}: vertex element lacks spatial fields {missing}")
    if not set(_COLOR_FIELDS) <= set(names):
        raise ValueError(f"{path}: vertex element has no red/green/blue; colored input is required")

    s = np.column_stack([np.asarray(v[f]) for f in _SPATIAL_FIELDS]).astype(np.float64)
    rgb = np.column_stack([np.asarray(v[f]) for f in _COLOR_FIELDS])
    if np.issubdtype(rgb.dtype, np.integer):
        c = rgb.astype(np.float64) / float(np.iinfo(rgb.dtype).max)
    else:
        c = rgb.astype(np.float64)
        if c.min(initial=0.0) < 0.0 or c.max(initial=0.0) > 1.0:
            raise ValueError(f"{path}: float RGB outside [0, 1]")

    extras = {
        name: np.asarray(v[name]).copy()
        for name in names
        if name not in _SPATIAL_FIELDS and name not in _COLOR_FIELDS
    }

    idx = _subsample_indices(s.shape[0], subsample, rng)
    if idx is not None:
        s, c = s[idx], c[idx]
        extras = {k: v_[idx] for k, v_ in extras.items()}
    return s, c, extras


def center_scene(s: np.ndarray):
    """Subtract the centroid from spatial coordinates; return (s_centered, origin).

    Aerial clouds carry projected CRS coordinates — e.g. EPSG:32650 eastings are
    ~1e5 m and northings ~1e6 m. A float32 mantissa (24 bits, ~7 decimal digits)
    resolves only ~0.1 m at 1e6, coarser than the centimeter-scale structure the
    mixture must represent, so raw UTM coordinates must never be cast to float32.
    Centering also matters in float64: the default NIW prior places m0 at the data
    centroid with a weak kappa0, and natural-parameter terms of the form
    kappa * m and kappa * m m^T (svi.py) lose precision when |m| ~ 1e6 dwarfs the
    scene extent. Keep the returned origin to map results back to the CRS.
    """
    s = np.asarray(s, dtype=np.float64)
    origin = s.mean(axis=0)
    return s - origin, origin

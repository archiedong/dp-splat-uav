"""Tests for io_aerial loaders (LAZ/PLY round-trips) and the two seam baselines.

Baseline expectations are hand-derived from the fixture geometry; the stitched
predictive is checked against an independent NumPy Student-t oracle written from
the density formula, not from the implementation.
"""

import numpy as np
import pytest

import jax.numpy as jnp
import laspy
from plyfile import PlyData, PlyElement
from scipy.special import gammaln

from dp_splat import niw as _niw
from dp_splat.cavi import Config, State
from dp_splat.priors import DirichletPosterior

from dp_splat_uav import baselines, io_aerial


# ---------------------------------------------------------------------------
# io_aerial: LAZ
# ---------------------------------------------------------------------------


def _write_laz(path, s, rgb16, classification):
    header = laspy.LasHeader(version="1.2", point_format=3)  # format 3 carries RGB
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = s.min(axis=0)
    las = laspy.LasData(header)
    las.x, las.y, las.z = s[:, 0], s[:, 1], s[:, 2]
    las.red, las.green, las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    las.classification = classification
    las.write(str(path))


def test_load_laz_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    # UTM-scale coordinates (EPSG:32650-like magnitudes)
    s = np.array([500_000.0, 5_500_000.0, 60.0]) + rng.uniform(0, 30, size=(n, 3))
    rgb16 = rng.integers(0, 65536, size=(n, 3), dtype=np.uint16)
    classification = rng.integers(0, 6, size=n, dtype=np.uint8)
    path = tmp_path / "tiny.laz"
    _write_laz(path, s, rgb16, classification)

    s2, c2, extras = io_aerial.load_laz(path)
    assert s2.dtype == np.float64 and s2.shape == (n, 3)
    assert c2.dtype == np.float64 and c2.shape == (n, 3)
    np.testing.assert_allclose(s2, s, atol=0.0011)  # LAS quantization at scale 0.001
    np.testing.assert_allclose(c2, rgb16.astype(np.float64) / 65535.0, atol=1e-12)
    assert c2.min() >= 0.0 and c2.max() <= 1.0
    np.testing.assert_array_equal(extras["classification"], classification)


def test_load_laz_8bit_stuffed_rgb(tmp_path):
    # 8-bit values written into the 16-bit fields must still land in [0, 1]
    n = 10
    s = np.zeros((n, 3)) + [1.0, 2.0, 3.0]
    rgb = np.stack([np.arange(n)] * 3, axis=1).astype(np.uint16) * 25  # max 225
    _write_laz(tmp_path / "t.laz", s, rgb, np.zeros(n, dtype=np.uint8))
    _, c, _ = io_aerial.load_laz(tmp_path / "t.laz")
    np.testing.assert_allclose(c, rgb / 255.0, atol=1e-12)


def test_load_laz_subsample(tmp_path):
    n = 40
    s = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    rgb16 = np.full((n, 3), 1000, dtype=np.uint16)
    _write_laz(tmp_path / "t.laz", s, rgb16, np.arange(n, dtype=np.uint8) % 8)
    s2, c2, extras = io_aerial.load_laz(tmp_path / "t.laz", subsample=7, rng=123)
    assert s2.shape == (7, 3) and c2.shape == (7, 3)
    assert extras["classification"].shape == (7,)
    # extras stay aligned with coordinates: classification was (row index) % 8
    rows = np.round(s2[:, 0] / 3.0).astype(int)
    np.testing.assert_array_equal(extras["classification"], rows % 8)


def test_load_laz_missing_rgb(tmp_path):
    header = laspy.LasHeader(version="1.2", point_format=0)  # format 0: no RGB
    las = laspy.LasData(header)
    las.x = las.y = las.z = np.arange(5, dtype=np.float64)
    path = tmp_path / "norgb.las"
    las.write(str(path))
    with pytest.raises(ValueError, match="RGB"):
        io_aerial.load_laz(path)


# ---------------------------------------------------------------------------
# io_aerial: PLY
# ---------------------------------------------------------------------------


def _write_ply(path, s, rgb8, label=None, text=True):
    fields = [("x", "f8"), ("y", "f8"), ("z", "f8")]
    cols = [s[:, 0], s[:, 1], s[:, 2]]
    if rgb8 is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        cols += [rgb8[:, 0], rgb8[:, 1], rgb8[:, 2]]
    if label is not None:
        fields += [("label", "i4")]
        cols += [label]
    vertex = np.empty(s.shape[0], dtype=fields)
    for (name, _), col in zip(fields, cols):
        vertex[name] = col
    PlyData([PlyElement.describe(vertex, "vertex")], text=text).write(str(path))


def test_load_ply_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    n = 30
    s = np.array([500_000.0, 5_500_000.0, 100.0]) + rng.uniform(0, 20, size=(n, 3))
    rgb8 = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    label = rng.integers(0, 11, size=n).astype(np.int32)  # H3D-style class labels
    path = tmp_path / "tiny.ply"
    _write_ply(path, s, rgb8, label)

    s2, c2, extras = io_aerial.load_ply(path)
    assert s2.dtype == np.float64 and c2.dtype == np.float64
    np.testing.assert_allclose(s2, s, rtol=0, atol=1e-9)
    np.testing.assert_allclose(c2, rgb8 / 255.0, atol=1e-12)
    np.testing.assert_array_equal(extras["label"], label)
    assert set(extras) == {"label"}


def test_load_ply_subsample_and_missing_rgb(tmp_path):
    n = 25
    s = np.arange(n * 3, dtype=np.float64).reshape(n, 3)
    _write_ply(tmp_path / "c.ply", s, np.full((n, 3), 128, dtype=np.uint8))
    s2, c2, extras = io_aerial.load_ply(tmp_path / "c.ply", subsample=5, rng=0)
    assert s2.shape == (5, 3) and extras == {}

    _write_ply(tmp_path / "norgb.ply", s, None)
    with pytest.raises(ValueError, match="red/green/blue"):
        io_aerial.load_ply(tmp_path / "norgb.ply")


def test_center_scene():
    rng = np.random.default_rng(2)
    s = np.array([499_980.0, 5_499_990.0, 55.0]) + rng.normal(0, 10, size=(100, 3))
    sc, origin = io_aerial.center_scene(s)
    np.testing.assert_allclose(origin, s.mean(0))
    # float64 residual at 5.5e6 m magnitude is ~1e-9 m; float32 would leave ~0.1 m
    np.testing.assert_allclose(sc.mean(0), 0.0, atol=1e-6)
    np.testing.assert_allclose(sc + origin, s, atol=1e-9)
    assert sc.dtype == np.float64


# ---------------------------------------------------------------------------
# Baselines: hand-built two-tile fixture
# ---------------------------------------------------------------------------

# Geometry (spatial x-axis is the seam axis; y, z shared):
#   tile A core [0, 16) x [0, 10)^2, center x = 8
#   tile B core [16, 20) x [0, 10)^2, center x = 18
# Tile A fit (K = 3), component mean x-coords: 5, 14, 17
#   x=5:  in A core, nearest A          -> kept by both baselines
#   x=14: in A core, but nearest B (6>4) -> kept by crop, dropped by voronoi
#   x=17: in B core (halo duplicate)     -> dropped by both
# Tile B fit (K = 2), component mean x-coords: 17.5, 14
#   x=17.5: in B core, nearest B         -> kept by both
#   x=14:   in A core (halo duplicate), nearest B -> dropped by crop, kept by voronoi


def _mk_fit(mx, kappa_val, alpha_post):
    K = len(mx)
    ms = jnp.stack([jnp.array([x, 5.0, 5.0]) for x in mx])
    mc = jnp.linspace(0.2, 0.8, K)[:, None] * jnp.ones(3)
    prior_s = _niw.make_prior(jnp.zeros(3), 1e-3, jnp.eye(3), 5.0, K)
    prior_c = _niw.make_prior(0.5 * jnp.ones(3), 1e-3, 0.1 * jnp.eye(3), 5.0, K)
    spatial = _niw.NIW(m=ms, kappa=jnp.full(K, kappa_val),
                       Psi=jnp.broadcast_to(2.0 * jnp.eye(3), (K, 3, 3)), nu=jnp.full(K, 8.0))
    color = _niw.NIW(m=mc, kappa=jnp.full(K, kappa_val),
                     Psi=jnp.broadcast_to(0.05 * jnp.eye(3), (K, 3, 3)), nu=jnp.full(K, 8.0))
    cfg = Config(weight_prior="dir", T=K)
    state = State(spatial, color, prior_s, prior_c,
                  DirichletPosterior(jnp.asarray(alpha_post, dtype=jnp.float64)), None)
    return baselines.TileFit(state, cfg)


@pytest.fixture
def two_tile():
    tiles = [
        baselines.Tile(lo=jnp.array([0.0, 0.0, 0.0]), hi=jnp.array([16.0, 10.0, 10.0])),
        baselines.Tile(lo=jnp.array([16.0, 0.0, 0.0]), hi=jnp.array([20.0, 10.0, 10.0])),
    ]
    fit_a = _mk_fit([5.0, 14.0, 17.0], kappa_val=2.0, alpha_post=[4.0, 3.0, 1.0])
    fit_b = _mk_fit([17.5, 14.0], kappa_val=3.0, alpha_post=[3.0, 1.0])
    return [fit_a, fit_b], tiles


def test_crop_mean_in_tile(two_tile):
    fits, tiles = two_tile
    mix = baselines.crop_mean_in_tile(fits, tiles, tile_masses=[2.0, 1.0])
    assert sorted(np.asarray(mix.spatial.m)[:, 0].tolist()) == [5.0, 14.0, 17.5]
    np.testing.assert_array_equal(np.asarray(mix.tile_index), [0, 0, 1])
    np.testing.assert_array_equal(np.asarray(mix.component_index), [0, 1, 0])
    # E[pi] A = [0.5, 0.375, 0.125], B = [0.75, 0.25]; masses [2, 1]
    # kept: 2*[0.5, 0.375] and 1*[0.75] -> normalized [0.4, 0.3, 0.3]
    np.testing.assert_allclose(np.asarray(mix.weights), [0.4, 0.3, 0.3], atol=1e-12)
    assert float(mix.weights.sum()) == pytest.approx(1.0, abs=1e-12)
    # crop keeps A's copy of the x=14 structure (kappa distinguishes provenance)
    np.testing.assert_allclose(np.asarray(mix.spatial.kappa), [2.0, 2.0, 3.0])


def test_voronoi_ownership(two_tile):
    fits, tiles = two_tile
    mix = baselines.voronoi_ownership(fits, tiles, tile_masses=[2.0, 1.0])
    assert sorted(np.asarray(mix.spatial.m)[:, 0].tolist()) == [5.0, 14.0, 17.5]
    np.testing.assert_array_equal(np.asarray(mix.tile_index), [0, 1, 1])
    np.testing.assert_array_equal(np.asarray(mix.component_index), [0, 0, 1])
    # kept: 2*[0.5] and 1*[0.75, 0.25] -> normalized [0.5, 0.375, 0.125]
    np.testing.assert_allclose(np.asarray(mix.weights), [0.5, 0.375, 0.125], atol=1e-12)
    assert float(mix.weights.sum()) == pytest.approx(1.0, abs=1e-12)
    # voronoi keeps B's copy of the x=14 structure
    np.testing.assert_allclose(np.asarray(mix.spatial.kappa), [2.0, 3.0, 3.0])


def test_default_masses_from_soft_counts(two_tile):
    fits, tiles = two_tile
    # attach responsibilities: tile A explains 30 points, tile B 10
    fits = [
        baselines.TileFit(f.state._replace(r=jnp.ones((n, len(f.state.spatial.kappa))) / len(f.state.spatial.kappa)), f.cfg)
        for f, n in zip(fits, (30, 10))
    ]
    mix = baselines.crop_mean_in_tile(fits, tiles)
    ref = baselines.crop_mean_in_tile(fits, tiles, tile_masses=[3.0, 1.0])
    np.testing.assert_allclose(np.asarray(mix.weights), np.asarray(ref.weights), atol=1e-12)


def test_boundary_mean_owned_by_upper_tile(two_tile):
    fits, tiles = two_tile
    # a mean exactly on the shared face x = 16 belongs to tile B only (half-open cores)
    fit = _mk_fit([16.0], kappa_val=1.0, alpha_post=[1.0])
    mix = baselines.crop_mean_in_tile([fits[0], fit], [tiles[0], tiles[1]], tile_masses=[1.0, 1.0])
    kept_b = np.asarray(mix.tile_index) == 1
    assert kept_b.sum() == 1 and np.asarray(mix.spatial.m)[kept_b][0, 0] == 16.0


def test_empty_selection_raises():
    tiles = [baselines.Tile(lo=jnp.zeros(3), hi=jnp.ones(3))]
    fit = _mk_fit([50.0], kappa_val=1.0, alpha_post=[1.0])  # mean far outside the core
    with pytest.raises(ValueError, match="no components retained"):
        baselines.crop_mean_in_tile([fit], tiles)


# ---------------------------------------------------------------------------
# Stitched predictive vs independent NumPy Student-t oracle
# ---------------------------------------------------------------------------


def _st_logpdf_np(x, loc, scale, dof):
    """log St(x; loc, scale, dof), written directly from the density formula."""
    D = loc.shape[0]
    dev = x - loc
    q = dev @ np.linalg.solve(scale, dev)
    _, logdet = np.linalg.slogdet(scale)
    return (
        gammaln((dof + D) / 2.0)
        - gammaln(dof / 2.0)
        - 0.5 * D * np.log(dof * np.pi)
        - 0.5 * logdet
        - 0.5 * (dof + D) * np.log1p(q / dof)
    )


def _niw_predictive_np(m, kappa, Psi, nu):
    D = m.shape[0]
    eta = nu - D + 1.0
    return m, Psi * (kappa + 1.0) / (kappa * eta), eta


def test_mixture_heldout_loglik_vs_numpy_oracle(two_tile):
    fits, tiles = two_tile
    mix = baselines.crop_mean_in_tile(fits, tiles, tile_masses=[2.0, 1.0])

    rng = np.random.default_rng(3)
    xs = rng.uniform(0, 20, size=(6, 3))
    xc = rng.uniform(0, 1, size=(6, 3))

    w = np.asarray(mix.weights)
    K = w.shape[0]
    m_s, k_s, P_s, n_s = (np.asarray(a) for a in mix.spatial)
    m_c, k_c, P_c, n_c = (np.asarray(a) for a in mix.color)
    per_point = []
    for i in range(xs.shape[0]):
        terms = []
        for k in range(K):
            loc, sc, dof = _niw_predictive_np(m_s[k], k_s[k], P_s[k], n_s[k])
            ls = _st_logpdf_np(xs[i], loc, sc, dof)
            loc, sc, dof = _niw_predictive_np(m_c[k], k_c[k], P_c[k], n_c[k])
            lc = _st_logpdf_np(xc[i], loc, sc, dof)
            terms.append(np.log(w[k]) + ls + lc)
        m = np.max(terms)
        per_point.append(m + np.log(np.sum(np.exp(np.array(terms) - m))))
    expected = float(np.mean(per_point))

    got = float(baselines.mixture_heldout_loglik(mix, jnp.asarray(xs), jnp.asarray(xc)))
    assert got == pytest.approx(expected, rel=1e-8)
    assert np.isfinite(got)

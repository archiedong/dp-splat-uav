"""Synthetic oracles for experiments/geometric_accuracy.py (audit item A3).

All tests are small, data-free, and check the exact routines the experiment
runs, against independent closed-form or brute-force computations:

(i)   point_triangle_dist2 vs an independent point-PLANE closed form
      (|n . (p - v0)| / |n|) for points whose orthogonal projection lands
      strictly inside a generic (non-axis-aligned) triangle -- agreement to
      1e-12 -- plus hand-constructed vertex- and edge-region cases.
(ii)  MeshNearest (candidate kNN + exactness certificate + ball escalation)
      vs brute force over every triangle of a random soup, with k0 small so
      the escalation path is actually exercised -- agreement to 1e-12.
(iii) The planar-offset oracle demanded by the audit: a unit-square mesh at a
      known height with model samples at a known offset h -- every distance
      and the mean must equal h (well inside the <= 1% requirement).
(iv)  Completeness of a half-covered plane: area-weighted surface samples of
      the unit square scored against a dense model bank covering only
      x <= 1/2 must give completeness 0.5 +/- 0.02 at a threshold well above
      the bank spacing and far below the plane size.
(v)   sample_surface area weighting: face pick frequencies proportional to
      area; samples lie on their faces (plane residual ~ 0, inside bbox).
(vi)  Support helpers and tercile stratification against direct computations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import geometric_accuracy as ga


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def brute_force_mesh_distance(p, tri):
    """min over ALL triangles, elementwise per point (independent of MeshNearest)."""
    out = np.empty(p.shape[0])
    for i in range(p.shape[0]):
        d2 = ga.point_triangle_dist2(
            np.repeat(p[i][None, :], tri.shape[0], axis=0), tri)
        out[i] = np.sqrt(d2.min())
    return out


def unit_square_mesh(z=0.0):
    """The unit square [0,1]^2 at height z as two triangles."""
    v = np.array([[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    return v[f]  # (2, 3, 3)


# ---------------------------------------------------------------------------
# (i) point-to-triangle vs closed-form point-to-plane, 1e-12
# ---------------------------------------------------------------------------


def test_point_triangle_matches_plane_closed_form():
    rng = np.random.default_rng(0)
    v0 = np.array([0.3, -1.2, 0.7])
    v1 = np.array([4.1, 0.4, 1.9])
    v2 = np.array([-0.8, 3.6, 2.4])  # generic tilted triangle
    e0, e1 = v1 - v0, v2 - v0
    n = np.cross(e0, e1)

    # Points with interior projection: p = v0 + a e0 + b e1 + h n_unit,
    # a, b in the strict interior of the barycentric simplex.
    m = 500
    a = rng.uniform(0.05, 0.55, m)
    b = rng.uniform(0.05, np.minimum(0.9 - a, 0.55))
    h = rng.uniform(-3.0, 3.0, m)
    p = v0 + a[:, None] * e0 + b[:, None] * e1 + h[:, None] * (n / np.linalg.norm(n))

    d = np.sqrt(ga.point_triangle_dist2(p, np.broadcast_to((v0, v1, v2), (m, 3, 3))))
    # Independent closed form: distance to the triangle's supporting plane.
    d_plane = np.abs((p - v0) @ n) / np.linalg.norm(n)
    np.testing.assert_allclose(d, d_plane, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(d, np.abs(h), rtol=0.0, atol=1e-12)


def test_point_triangle_vertex_and_edge_regions():
    tri = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    # Beyond vertex v1: closest point is v1 itself.
    p = np.array([[3.0, -1.0, 2.0]])
    expect = np.linalg.norm(p[0] - tri[0, 1])
    np.testing.assert_allclose(
        np.sqrt(ga.point_triangle_dist2(p, tri))[0], expect, atol=1e-12)
    # Off the v0-v1 edge interior: closest point is the foot (0.7, 0, 0).
    p = np.array([[0.7, -2.0, 1.0]])
    expect = np.sqrt(2.0 ** 2 + 1.0 ** 2)
    np.testing.assert_allclose(
        np.sqrt(ga.point_triangle_dist2(p, tri))[0], expect, atol=1e-12)
    # Degenerate (zero-area) triangle degrades to its edges.
    dg = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    p = np.array([[1.5, 1.0, 0.0]])
    np.testing.assert_allclose(
        np.sqrt(ga.point_triangle_dist2(p, dg))[0], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# (ii) MeshNearest vs brute force on a random soup (escalation exercised)
# ---------------------------------------------------------------------------


def test_mesh_nearest_matches_brute_force():
    rng = np.random.default_rng(1)
    centers = rng.uniform(0.0, 1.0, (300, 1, 3))
    tri = centers + rng.uniform(-0.15, 0.15, (300, 3, 3))  # mixed-size soup
    p = np.concatenate([
        rng.uniform(-0.3, 1.3, (150, 3)),      # around and inside the soup
        rng.uniform(1.5, 2.5, (20, 3)),        # far outside
    ])
    mn = ga.MeshNearest(tri, k0=4)  # tiny k0: certificate must escalate often
    d, cert = mn.distance(p, chunk=64)
    np.testing.assert_allclose(d, brute_force_mesh_distance(p, tri),
                               rtol=0.0, atol=1e-12)
    assert cert["n_escalated"] > 0  # the escalation path was really exercised


def test_mesh_nearest_outlier_faces_split_off():
    """A giant face among small ones goes to the brute-force path, still exact."""
    rng = np.random.default_rng(5)
    centers = rng.uniform(0.0, 1.0, (200, 1, 3))
    small = centers + rng.uniform(-0.02, 0.02, (200, 3, 3))
    giant = np.array([[[-5.0, -5.0, 2.0], [8.0, -5.0, 2.0], [-5.0, 8.0, 2.0]]])
    tri = np.concatenate([small, giant])
    p = rng.uniform(-0.5, 2.5, (120, 3))
    mn = ga.MeshNearest(tri, k0=6)
    assert mn.tri_large.shape[0] == 1  # the giant face took the outlier path
    d, _ = mn.distance(p, chunk=32)
    np.testing.assert_allclose(d, brute_force_mesh_distance(p, tri),
                               rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# (iii) planar-offset oracle: mean distance == h
# ---------------------------------------------------------------------------


def test_planar_mesh_known_offset():
    z0, h = 0.6, 0.37
    tri = unit_square_mesh(z=z0)
    g = np.linspace(0.05, 0.95, 40)
    xx, yy = np.meshgrid(g, g)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z0 + h)])
    d, _ = ga.MeshNearest(tri, k0=8).distance(pts)
    # Every interior sample is exactly h from the plane; audit demands <= 1%.
    np.testing.assert_allclose(d, h, rtol=1e-9)
    assert abs(d.mean() - h) / h <= 0.01
    stats = ga.dist_stats(d, ga.ACC_THRESH)
    np.testing.assert_allclose(
        [stats["mean"], stats["median"], stats["rms"]], h, rtol=1e-9)
    assert stats["frac_le_0.25m"] == 0.0 and stats["frac_le_0.5m"] == 1.0


# ---------------------------------------------------------------------------
# (iv) completeness of a half-covered plane == 0.5 +/- 0.02
# ---------------------------------------------------------------------------


def test_half_plane_completeness():
    rng = np.random.default_rng(2)
    tri = unit_square_mesh(z=0.0)
    ref, _ = ga.sample_surface(tri, 20_000, rng)
    # Model bank: dense grid on the x <= 1/2 half only (spacing 0.002 << tau).
    gx = np.arange(0.0, 0.5 + 1e-12, 0.002)
    gy = np.arange(0.0, 1.0 + 1e-12, 0.002)
    xx, yy = np.meshgrid(gx, gy)
    bank = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    d = ga.nn_distance(ref, bank)
    comp = ga.dist_stats(d, thresholds=(0.01, 2.0))
    assert abs(comp["frac_le_0.01m"] - 0.5) <= 0.02
    assert comp["frac_le_2m"] == 1.0  # the whole square is within 2 m of the bank


# ---------------------------------------------------------------------------
# (v) area-weighted surface sampling
# ---------------------------------------------------------------------------


def test_sample_surface_area_weighting():
    # Two coplanar triangles with area ratio 1 : 3.
    tri = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],       # area 1/2
        [[2.0, 0.0, 0.0], [2.0 + np.sqrt(3.0), 0.0, 0.0], [2.0, np.sqrt(3.0), 0.0]],
    ])  # area 3/2
    areas = ga.face_areas(tri)
    np.testing.assert_allclose(areas, [0.5, 1.5], atol=1e-12)
    pts, face = ga.sample_surface(tri, 50_000, np.random.default_rng(3))
    assert abs((face == 1).mean() - 0.75) <= 0.01
    assert np.all(pts[:, 2] == 0.0)  # on the z = 0 plane
    # Barycentric containment: every sample within its own face's triangle.
    d, _ = ga.MeshNearest(tri, k0=2).distance(pts[:2000])
    np.testing.assert_allclose(d, 0.0, atol=1e-12)


def test_dedup_faces_drops_exact_duplicates():
    tri = unit_square_mesh()
    doubled = np.concatenate([tri, tri[:1]])  # face 0 duplicated
    keep, n_dropped = ga.dedup_faces(doubled)
    assert n_dropped == 1
    np.testing.assert_array_equal(keep, [0, 1])


# ---------------------------------------------------------------------------
# (vi) support helpers and NLPD-tercile stratification
# ---------------------------------------------------------------------------


def test_in_any_rect():
    lo = np.array([[0.0, 0.0], [2.0, 0.0]])
    hi = np.array([[1.0, 1.0], [3.0, 1.0]])
    xy = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5], [1.0, 1.0]])
    np.testing.assert_array_equal(
        ga.in_any_rect(xy, lo, hi), [True, False, True, False, True])


def test_occupancy_mask():
    rng = np.random.default_rng(4)
    # Dense cloud on [0,1]^2 only; queries on both halves of [0,2] x [0,1].
    cloud = rng.uniform(0.0, 1.0, (5000, 2))
    mask = ga.OccupancyMask(cloud, cell=0.25, min_count=5)
    q = np.array([[0.5, 0.5], [1.7, 0.5], [-4.0, 0.5]])
    np.testing.assert_array_equal(mask(q), [True, False, False])


def test_tercile_report_matches_direct_grouping():
    nlpd = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    d = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4])
    rep = ga.tercile_report(d, nlpd, thresholds=(0.25,))
    assert [r["tercile"] for r in rep["rows"]] == list(ga.TERCILE_LABELS)
    assert [r["n"] for r in rep["rows"]] == [3, 3, 3]
    np.testing.assert_allclose([r["mean"] for r in rep["rows"]], [0.1, 0.2, 0.4])
    np.testing.assert_allclose(
        [r["frac_le_0.25m"] for r in rep["rows"]], [1.0, 1.0, 0.0])

"""Independent oracles for experiments/change_detection.py.

(i)   Sigma_mu formula. Under the NIW posterior (m_N, kappa_N, Psi_N, nu_N) the
      spatial mean satisfies mu | Sigma ~ N(m_N, Sigma / kappa_N) with
      Sigma ~ IW(Psi_N, nu_N), so marginally mu is multivariate Student-t and
      Cov[mu] = E[Sigma] / kappa_N = Psi_N / (kappa_N (nu_N - D - 1)) for
      nu_N > D + 1. Verified two ways: by Monte Carlo sampling of the NIW
      itself (formula-level check, no project code), and against the
      sig_mu_zz the script's epoch_bank extracts from a hand-built
      single-tile model (code-level check), including the dof guard at
      nu_N = D + 1 where the covariance is undefined.

(ii)  Law-of-total-variance assembly. epoch_surface on a hand-built
      2-component model against a from-scratch NumPy/SciPy computation:
      stick-breaking E[pi] from the counts (a_k = 1 + N_k,
      b_k = alpha + sum_{j>k} N_j), responsibilities from scipy's
      multivariate_t at the NIW predictive parameterization
      St(m, Psi (kappa+1)/(kappa eta), eta = nu - D + 1), then
      mu_z = sum w m_kz and var_z = sum w [Sigma_mu]_zz
      + sum w (m_kz - mu_z)^2, with the origin lift to absolute elevation.
      Also: safe_gate_weights equals stitch.gate_weights inside coverage and
      returns zero rows (instead of raising) outside.

(iii) LoD95 assembly (evaluate): LoD = 1.96 sqrt(var_A + var_B + sigma_reg^2)
      with the registration term entering once, squared, inside the root;
      flags, null exceedance, precision/recall, and the rank-based AUC against
      a brute-force pairwise oracle (ties at 1/2).

(iv)  Semantic proxy on a hand-computed toy label grid written to a real LAS
      file: per-cell dominant class, purity, count, mean z, out-of-grid point
      rejection, empty-cell conventions, the vegetation class set, and the
      classical arm's per-cell sample z-variance against np.var (ddof = 1).

(v)   SYNTHETIC TRUTH (end-to-end, module-scoped): two epochs of a flat plane,
      epoch B carrying a known +0.30 m slab (relabeled Soil/Gravel) over
      x >= slab edge, different xy extents so the recorded origins differ.
      Both epochs are fitted with run_tiled_fit.py (multi-tile) and compared
      with change_detection.py; the slab must be detected (recall > 0.9), the
      unchanged region's flag rate must stay <= 7%, the recovered d in the
      slab interior must be within 20% of -0.30 m, and the reported cell
      counts must match masks recomputed from the saved per-cell arrays.
      The run passes --classical: the standard DoD baseline on the same cells
      must also detect the slab (cm-scale empirical LoD), hold its null rate,
      and its saved arrays must reproduce the documented formula.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import invwishart, multivariate_t

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import jax.numpy as jnp

import change_detection as cd
from dp_splat_uav import stitch, tiling

D = cd.D_SPATIAL


# ---------------------------------------------------------------------------
# Shared constructions
# ---------------------------------------------------------------------------


def _spd(rng, d, scale=1.0):
    a = rng.normal(size=(d, d))
    return scale * (a @ a.T + d * np.eye(d))


def _single_tile_model(spatial_nu):
    """Minimal one-tile, two-component merged-model dict (npz field layout)."""
    rng = np.random.default_rng(3)
    K = 2
    m = np.array([[3.0, 3.0, 0.10], [7.0, 7.0, 0.55]])
    kappa = np.array([50.0, 80.0])
    nu = np.asarray(spatial_nu, dtype=float)
    Psi = np.stack([_spd(rng, D, 4.0), _spd(rng, D, 6.0)])
    counts = np.array([120.0, 80.0])  # descending: rebuild_weights keeps order
    return dict(
        spatial_m=m, spatial_kappa=kappa, spatial_Psi=Psi, spatial_nu=nu,
        color_m=np.full((K, 3), 0.5), color_kappa=np.full(K, 30.0),
        color_Psi=np.stack([np.eye(3) * 0.2] * K), color_nu=np.full(K, 12.0),
        counts=counts,
        src_offsets=np.array([0, 1, 2]), src_tile=np.array([0, 0]),
        src_comp=np.array([0, 1]), tile_counts=counts[None, :],
        core_lo=np.array([[0.0, 0.0]]), core_hi=np.array([[10.0, 10.0]]),
        halo=np.float64(2.0), bbox_lo=np.array([0.0, 0.0]),
        bbox_hi=np.array([10.0, 10.0]), grid_shape=np.array([1, 1]),
        tile_index=np.array([[0, 0]]), tile_morton=np.array([0], dtype=np.uint64),
        origin=np.array([100.0, 200.0, 50.0]),
    )


def _stick_expected_pi(counts, alpha):
    """E[pi] under the truncated stick-breaking posterior, from scratch:
    a_k = 1 + N_k, b_k = alpha + sum_{j>k} N_j for k < K, v_K := 1."""
    counts = np.asarray(counts, dtype=float)
    K = counts.size
    tail = np.concatenate([np.cumsum(counts[::-1])[::-1][1:], [0.0]])
    a = 1.0 + counts[: K - 1]
    b = alpha + tail[: K - 1]
    ev = np.concatenate([a / (a + b), [1.0]])
    one_minus = np.concatenate([[1.0], np.cumprod(b / (a + b))])
    return ev * one_minus


def _write_las(path, xyz, cls, rgb=None):
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.floor(xyz.min(axis=0))
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is None:
        rgb = np.full((xyz.shape[0], 3), 0.5)
    rgb16 = np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16)
    las.red, las.green, las.blue = rgb16[:, 0], rgb16[:, 1], rgb16[:, 2]
    las.classification = cls.astype(np.uint8)
    las.write(str(path))


# ---------------------------------------------------------------------------
# (i) Sigma_mu = Psi_N / (kappa_N (nu_N - D - 1))
# ---------------------------------------------------------------------------


def test_sigma_mu_formula_monte_carlo():
    """NIW draws: Cov[mu] must match Psi/(kappa (nu - D - 1)) — the marginal
    posterior covariance of the mean, NOT the predictive covariance."""
    rng = np.random.default_rng(0)
    m0 = np.array([1.0, -2.0, 0.5])
    kappa, nu = 2.0, 14.0
    Psi = _spd(rng, D, 2.0)
    n = 200_000
    Sigma = invwishart(df=nu, scale=Psi).rvs(size=n, random_state=rng)
    L = np.linalg.cholesky(Sigma / kappa)
    mu = m0 + np.einsum("nij,nj->ni", L, rng.normal(size=(n, D)))
    expected = Psi / (kappa * (nu - D - 1.0))
    emp = np.cov(mu.T)
    assert np.allclose(emp, expected, rtol=0.04, atol=0.01 * np.abs(expected).max())
    # The predictive covariance S eta/(eta-2) with S = Psi (kappa+1)/(kappa eta)
    # is strictly larger (adds the surface-roughness term): the empirical
    # mean-covariance must NOT match it (double-counting pitfall).
    eta = nu - D + 1.0
    predictive = Psi * (kappa + 1.0) / (kappa * (eta - 2.0))
    assert not np.allclose(emp, predictive, rtol=0.25)


def test_epoch_bank_sig_mu_zz_and_dof_guard():
    model = _single_tile_model(spatial_nu=[40.0, 60.0])
    record = {"alpha_t": [1.0]}
    bank = cd.epoch_bank(model, record)
    expected = model["spatial_Psi"][:, 2, 2] / (
        model["spatial_kappa"] * (model["spatial_nu"] - D - 1.0))
    assert np.allclose(np.asarray(bank["sig_mu_zz"]), expected, rtol=1e-12)
    assert np.allclose(np.asarray(bank["m_z"]), model["spatial_m"][:, 2])

    # nu = D + 1 makes the denominator zero: the covariance is undefined and
    # the guard must refuse rather than emit inf/negative variances.
    bad = _single_tile_model(spatial_nu=[float(D + 1), 60.0])
    with pytest.raises(ValueError, match="posterior mean covariance"):
        cd.epoch_bank(bad, record)


# ---------------------------------------------------------------------------
# (ii) epoch_surface: weights, law of total variance, origin lift
# ---------------------------------------------------------------------------


def test_epoch_surface_against_scratch_two_component():
    model = _single_tile_model(spatial_nu=[40.0, 60.0])
    alpha = 1.0
    bank = cd.epoch_bank(model, {"alpha_t": [alpha]})
    origin = model["origin"]

    q_model = np.array([  # in the model's centered frame, inside the tile
        [3.0, 3.0, 0.1],
        [7.0, 7.0, 0.5],
        [5.0, 5.0, 0.3],
        [4.4, 6.1, 0.2],
    ])
    centers_utm = q_model[:, :2] + origin[:2]
    zbar_abs = q_model[:, 2] + origin[2]
    # Two degenerate cells: outside every dilated tile / no raw points.
    centers_utm = np.vstack([centers_utm, [[130.0, 230.0]], [[105.0, 205.0]]])
    zbar_abs = np.concatenate([zbar_abs, [50.0, np.nan]])

    mu, var, ok, logdens = cd.epoch_surface(bank, origin, centers_utm,
                                            zbar_abs, chunk=2)
    assert ok.tolist() == [True] * 4 + [False, False]
    assert np.isnan(mu[4:]).all() and np.isnan(var[4:]).all()
    assert np.all(np.isneginf(logdens[4:]))

    # From-scratch oracle: stick E[pi], scipy Student-t responsibilities,
    # law of total variance, origin lift.
    e_pi = _stick_expected_pi(model["counts"], alpha)
    kappa, nu, Psi, m = (model["spatial_kappa"], model["spatial_nu"],
                         model["spatial_Psi"], model["spatial_m"])
    eta = nu - D + 1.0
    sig_zz = Psi[:, 2, 2] / (kappa * (nu - D - 1.0))
    for i in range(4):
        pdf = np.array([
            multivariate_t(loc=m[k], shape=Psi[k] * (kappa[k] + 1.0)
                           / (kappa[k] * eta[k]), df=eta[k]).pdf(q_model[i])
            for k in range(2)
        ])
        w = e_pi * pdf
        dens = w.sum()
        w = w / dens
        mu_model = w @ m[:, 2]
        var_exp = w @ sig_zz + w @ (m[:, 2] - mu_model) ** 2
        assert np.isclose(mu[i], mu_model + origin[2], rtol=0, atol=1e-9)
        assert np.isclose(var[i], var_exp, rtol=1e-8)
        assert np.isclose(logdens[i], np.log(dens), rtol=1e-8)


def test_safe_gate_weights_matches_stitch_and_zeroes_outside():
    grid = tiling.TileGrid(
        core_lo=np.array([[0.0, 0.0], [10.0, 0.0]]),
        core_hi=np.array([[10.0, 20.0], [20.0, 20.0]]),
        halo=2.0, bbox_lo=np.array([0.0, 0.0]), bbox_hi=np.array([20.0, 20.0]),
        shape=(2, 1), index=np.array([[0, 0], [1, 0]]),
        morton=np.array([0, 1], dtype=np.uint64),
    )
    rng = np.random.default_rng(7)
    inside = rng.uniform([0.0, 0.0], [20.0, 20.0], size=(64, 2))
    inside[:8, 0] = rng.uniform(8.5, 11.5, size=8)  # force seam-band coverage
    g_safe = cd.safe_gate_weights(grid, inside)
    g_ref = np.asarray(stitch.gate_weights(grid, jnp.asarray(inside)))
    assert np.allclose(g_safe, g_ref, atol=1e-12)
    assert np.allclose(g_safe.sum(axis=1), 1.0)

    outside = np.array([[25.0, 5.0], [-3.5, 10.0], [10.0, 24.0]])
    g_out = cd.safe_gate_weights(grid, outside)
    assert np.all(g_out == 0.0)
    with pytest.raises(ValueError):
        stitch.gate_weights(grid, jnp.asarray(outside))


# ---------------------------------------------------------------------------
# (iii) LoD95 assembly, flags, and detection metrics
# ---------------------------------------------------------------------------


def test_evaluate_lod_formula_and_null_exceedance():
    d = np.array([0.05, 0.60, -0.02, -0.90, 0.30, 0.01, 0.45, -0.10])
    var_a = np.array([0.010, 0.020, 0.005, 0.040, 0.010, 0.008, 0.015, 0.020])
    var_b = np.array([0.008, 0.015, 0.004, 0.030, 0.012, 0.010, 0.010, 0.015])
    var_sum = var_a + var_b
    sigma_reg = 0.10
    positive = np.array([False, True, False, True, True, False, True, False])
    elig = np.ones(8, dtype=bool)
    null = ~positive
    masks = dict(null=null, elig=elig, elig_noveg=elig, positive=positive)

    out, lod, flagged = cd.evaluate(d, var_sum, sigma_reg, masks)

    lod_expected = 1.96 * np.sqrt(var_a + var_b + sigma_reg**2)  # one reg term
    assert np.allclose(lod, lod_expected, rtol=1e-12)
    assert np.array_equal(flagged, np.abs(d) > lod_expected)
    assert out["sigma_reg"] == sigma_reg
    assert out["null_test"]["n_cells"] == 4
    assert np.isclose(out["null_test"]["exceedance_rate"],
                      flagged[null].mean())
    assert np.isclose(out["null_test"]["median_abs_d_m"],
                      np.median(np.abs(d[null])))
    assert np.isclose(out["null_test"]["median_lod_m"], np.median(lod_expected[null]))

    # Hand-counted confusion at the LoD threshold.
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    fn = int((~flagged & positive).sum())
    r = out["veg_excluded"]
    assert r["n_pos"] == 4 and r["n_neg"] == 4
    assert np.isclose(r["precision"], tp / (tp + fp))
    assert np.isclose(r["recall"], tp / (tp + fn))
    assert out["veg_included"] == r  # identical masks here


def test_roc_auc_against_pairwise_oracle():
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 12, size=200).astype(float)  # heavy ties
    labels = rng.random(200) < 0.3
    pos, neg = scores[labels], scores[~labels]
    oracle = ((pos[:, None] > neg[None, :]).sum()
              + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (pos.size * neg.size)
    assert np.isclose(cd.roc_auc(scores, labels), oracle, rtol=1e-12)
    assert cd.roc_auc(scores, np.ones(200, dtype=bool)) is None
    perfect = np.concatenate([np.zeros(5), np.ones(5)])
    assert cd.roc_auc(perfect, perfect > 0.5) == 1.0


# ---------------------------------------------------------------------------
# (iv) Toy label grid: class_histogram + cell_semantics + vegetation set
# ---------------------------------------------------------------------------


def test_cell_semantics_toy_grid(tmp_path):
    lo = np.array([10.0, 20.0])
    shape, cell = (3, 2), 1.0  # (nx, ny)
    pts = [  # (x, y, z, class)
        (10.2, 20.3, 1.0, 1), (10.7, 20.6, 2.0, 1), (10.4, 20.9, 3.0, 1),
        (11.2, 20.2, 5.0, 7), (11.8, 20.8, 6.0, 7), (11.5, 20.5, 4.0, 1),
        (12.3, 20.4, 2.0, 0), (12.6, 20.7, 4.0, 4),
        (11.1, 21.1, 7.0, 8), (11.9, 21.9, 8.0, 8),
        (11.5, 21.5, 9.0, 8), (11.2, 21.8, 10.0, 8),
        (12.5, 21.5, 1.5, 2), (12.1, 21.1, 2.5, 2),
        (13.5, 21.5, 9.9, 5), (9.5, 20.5, 9.9, 5), (12.5, 22.5, 9.9, 5),
    ]  # last three fall outside the grid and must be dropped
    xyz = np.array([[p[0], p[1], p[2]] for p in pts])
    cls = np.array([p[3] for p in pts])
    path = tmp_path / "toy.las"
    _write_las(path, xyz, cls)

    hist, zsum, z2sum = cd.class_histogram(path, lo, shape, cell)
    dom, purity, count, zbar, s2 = cd.cell_semantics(hist, zsum, z2sum)
    assert hist.shape == (6, cd.N_CLASS_BINS)
    assert int(hist.sum()) == 14  # out-of-grid points rejected

    # Flat index iy * nx + ix.
    assert count.tolist() == [3, 3, 2, 0, 4, 2]
    assert dom[0] == 1 and purity[0] == 1.0 and np.isclose(zbar[0], 2.0)
    assert dom[1] == 7 and np.isclose(purity[1], 2.0 / 3.0) and np.isclose(zbar[1], 5.0)
    assert dom[2] == 0 and purity[2] == 0.5  # tie resolves to the lowest class id
    assert purity[3] == 0.0 and np.isnan(zbar[3])  # empty-cell conventions
    assert dom[4] == 8 and purity[4] == 1.0 and np.isclose(zbar[4], 8.5)
    assert dom[5] == 2 and np.isclose(zbar[5], 2.0)

    # Classical-arm sample variance (ddof = 1) against np.var per cell; NaN on
    # cells with fewer than 2 points. LAS z is stored at 1 mm resolution, so
    # the oracle quantizes the raw z the same way before np.var.
    z_las = np.round(xyz[:, 2], 3)
    for c, members in enumerate(([0, 1, 2], [3, 4, 5], [6, 7], [],
                                 [8, 9, 10, 11], [12, 13])):
        if len(members) >= 2:
            assert np.isclose(s2[c], np.var(z_las[members], ddof=1), atol=1e-9)
        else:
            assert np.isnan(s2[c])

    # Vegetation set is exactly H3D {Low Vegetation, Shrub, Tree}.
    assert cd.VEG_CLASSES == (0, 6, 7)
    veg = np.isin(dom, cd.VEG_CLASSES)
    assert veg.tolist() == [False, True, True, True, False, False]  # dom 0 on empty


# ---------------------------------------------------------------------------
# (v) Synthetic truth: fitted flat plane + 0.30 m slab in epoch B
# ---------------------------------------------------------------------------

X0, Y0, Z0 = 500_000.0, 5_400_000.0, 80.0
SLAB_X = X0 + 22.0  # slab occupies x >= SLAB_X in epoch B
SLAB_DZ = 0.30
SYNTH_NAME = "change_synth_test"


def _synth_epoch(rng, x_lo, x_hi, y_lo, y_hi, slab):
    n = int((x_hi - x_lo) * (y_hi - y_lo) * 30)  # 30 pts / m^2
    x = rng.uniform(x_lo, x_hi, n)
    y = rng.uniform(y_lo, y_hi, n)
    z = Z0 + rng.normal(0.0, 0.03, n)
    cls = np.full(n, cd.IMPERVIOUS, dtype=np.uint8)
    rgb = np.clip(rng.normal(0.5, 0.02, (n, 3)), 0.0, 1.0)
    if slab:
        on = x >= SLAB_X
        z[on] += SLAB_DZ
        cls[on] = 8  # Soil/Gravel: non-vegetation dominant-class transition
        rgb[on] = np.clip(rng.normal([0.75, 0.35, 0.30], 0.02, (int(on.sum()), 3)),
                          0.0, 1.0)
    return np.column_stack([x, y, z]), cls, rgb


@pytest.fixture(scope="module")
def synth_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("change_synth")
    rng = np.random.default_rng(11)
    # Different extents so the two recorded origins genuinely differ in xy.
    xyz_a, cls_a, rgb_a = _synth_epoch(rng, X0, X0 + 44.0, Y0, Y0 + 44.0, False)
    xyz_b, cls_b, rgb_b = _synth_epoch(rng, X0 - 5.0, X0 + 44.0, Y0, Y0 + 44.0, True)
    las = {}
    for tag, (xyz, cls, rgb) in (("a", (xyz_a, cls_a, rgb_a)),
                                 ("b", (xyz_b, cls_b, rgb_b))):
        las[tag] = root / f"synth_{tag}.las"
        _write_las(las[tag], xyz, cls, rgb)

    exp = REPO / "experiments"
    for tag in ("a", "b"):
        run = subprocess.run(
            [sys.executable, str(exp / "run_tiled_fit.py"),
             "--input", str(las[tag]), "--name", f"synth_{tag}",
             "--out-dir", str(root), "--target-tile-points", "30000",
             "--truncation", "24", "--max-iters", "60", "--alpha", "5.0"],
            capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-3000:]
    run = subprocess.run(
        [sys.executable, str(exp / "change_detection.py"),
         "--record-a", str(root / "synth_a_record.json"),
         "--model-a", str(root / "synth_a_model.npz"),
         "--input-a", str(las["a"]),
         "--record-b", str(root / "synth_b_record.json"),
         "--model-b", str(root / "synth_b_model.npz"),
         "--input-b", str(las["b"]),
         "--classical",
         "--name", SYNTH_NAME],
        capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-3000:]

    result = json.loads((root / f"{SYNTH_NAME}.json").read_text())
    cells = dict(np.load(root / f"{SYNTH_NAME}_cells.npz"))
    yield dict(result=result, cells=cells)
    for ext in (".png", ".pdf"):
        (REPO / "figures" / f"{SYNTH_NAME}{ext}").unlink(missing_ok=True)


def _cell_centers_x(cells):
    nx = int(cells["shape"][0])
    n = cells["d"].size
    ix = np.arange(n) % nx
    return cells["grid_lo"][0] + float(cells["cell"]) * (ix + 0.5)


def test_synth_origins_differ_and_grid_shared(synth_run):
    frame = synth_run["result"]["frame"]
    gap = np.abs(np.array(frame["origin_a"]) - np.array(frame["origin_b"]))
    assert gap[0] > 1.0  # extents were offset by 5 m in x
    lo, hi = frame["grid_lo_utm"], frame["grid_hi_utm"]
    assert lo[0] >= X0 - 1e-6 and hi[0] <= X0 + 44.0 + 1e-6


def test_synth_slab_recall(synth_run):
    r = synth_run["result"]["results"]["veg_excluded"]
    assert r["n_pos"] > 500, "slab cells should dominate the positive set"
    assert r["recall"] > 0.9, f"slab recall {r['recall']} (n_pos {r['n_pos']})"


def test_synth_unchanged_flag_rate(synth_run):
    res = synth_run["result"]["results"]
    assert res["null_test"]["n_cells"] > 500
    assert res["null_test"]["exceedance_rate"] <= 0.07
    # Geometry-based (label-free) check on the same criterion: all valid cells
    # strictly left of the slab edge are physically unchanged.
    cells = synth_run["cells"]
    unchanged = cells["valid"] & (_cell_centers_x(cells) < SLAB_X - 1.0)
    assert unchanged.sum() > 500
    assert cells["flagged"][unchanged].mean() <= 0.07


def test_synth_d_accuracy(synth_run):
    cells = synth_run["cells"]
    slab = cells["valid"] & (_cell_centers_x(cells) >= SLAB_X + 1.0)
    assert slab.sum() > 500
    med = np.median(cells["d"][slab])
    # Epoch B gained the slab, so d = mu_A - mu_B ~ -0.30 m; within 20%.
    assert abs(med - (-SLAB_DZ)) <= 0.2 * SLAB_DZ, f"median slab d {med}"


def test_synth_classical_arm(synth_run):
    """--classical: the standard DoD baseline on the same cells must also see
    the slab (recall > 0.9 with a cm-scale LoD), hold its nominal null rate,
    and its saved per-cell arrays must reproduce the documented formula."""
    cl = synth_run["result"]["classical"]
    assert cl["results"]["null_test"]["exceedance_rate"] <= 0.10
    assert cl["results"]["null_test"]["median_lod_m"] < 0.05  # cm-scale
    assert cl["results"]["veg_excluded"]["recall"] > 0.9
    cells = synth_run["cells"]
    valid = cells["valid"]
    d_cl = cells["zbar_a"] - cells["zbar_b"]
    var_cl = cells["s2_a"] / cells["count_a"] + cells["s2_b"] / cells["count_b"]
    sigma = synth_run["result"]["config"]["sigma_reg"]
    lod_expected = 1.96 * np.sqrt(var_cl + sigma**2)
    assert np.allclose(cells["d_classical"][valid], d_cl[valid], rtol=1e-10)
    assert np.allclose(cells["lod_classical"][valid], lod_expected[valid],
                       rtol=1e-10)
    slab = cells["valid"] & (_cell_centers_x(cells) >= SLAB_X + 1.0)
    med = np.median(cells["d_classical"][slab])
    assert abs(med - (-SLAB_DZ)) <= 0.1 * SLAB_DZ


def test_synth_reported_counts_match_saved_masks(synth_run):
    """The JSON cell counts must be reproducible from the saved per-cell
    arrays with the documented mask definitions (null / eligibility / veg)."""
    cells = synth_run["cells"]
    valid = cells["valid"]
    dom_a, dom_b = cells["dom_a"], cells["dom_b"]
    pur_a, pur_b = cells["purity_a"], cells["purity_b"]
    null = (valid & (dom_a == cd.IMPERVIOUS) & (dom_b == cd.IMPERVIOUS)
            & (pur_a > 0.9) & (pur_b > 0.9))
    elig = valid & (pur_a > 0.6) & (pur_b > 0.6)
    veg = np.isin(dom_a, cd.VEG_CLASSES) | np.isin(dom_b, cd.VEG_CLASSES)
    positive = dom_a != dom_b
    n = synth_run["result"]["n_cells"]
    assert n["valid"] == int(valid.sum())
    assert n["null"] == int(null.sum())
    assert n["eligible"] == int(elig.sum())
    assert n["eligible_noveg"] == int((elig & ~veg).sum())
    assert n["positive_noveg"] == int((elig & ~veg & positive).sum())
    # LoD consistency on the saved arrays.
    sigma = synth_run["result"]["config"]["sigma_reg"]
    lod_expected = 1.96 * np.sqrt(cells["var_a"] + cells["var_b"] + sigma**2)
    assert np.allclose(cells["lod"][valid], lod_expected[valid], rtol=1e-10)

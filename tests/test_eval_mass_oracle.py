"""Mass oracle for the EVALUATION-SIDE stitched predictive on a real saved model.

The stitched predictive is p(x) = sum_t g_t(s) p_t(s, c) with partition-of-unity
gates g_t. It is a probability density only if each tile's sub-mixture p_t
carries its components' GLOBAL mixture weights un-renormalized within the tile
(the construction oracle-verified against a pooled fit in
tests/test_stitch_oracle.py test (vi)): a merged component straddling several
tiles then contributes pi_k * sum_t g_t(s) = pi_k of total mass. The historical
evaluation-side reconstruction instead renormalized every tile's weights from
the tile's own pre-merge counts (each p_t a probability density on its own),
which inflates the stitched mass to ~ #tiles. This test pins both facts
numerically on the real mar18_rich production model (2x3 grid, 6 tiles,
K = 841 global components, 880 (tile, component) pairs).

Estimator
---------
The color marginal integrates out analytically: each (tile, component) pair's
density is a product of independent spatial and color Student-t densities, and
the gates depend on s only, so integrating over color leaves the gated SPATIAL
Student-t mixture f(s) = sum_{t,k} pi_{tk} g_t(s) St(s; m_k, S_k, eta_k) with
exactly the total mass of the full predictive. That 3-D integral is estimated
by importance sampling:

    proposal q(s) = sum_pairs rho_pair St(s; m, INFLATE * S, eta),
    rho ~ fixed global pair weights (renormalized), INFLATE = 1.5^2 on the
    scale matrix (sd x 1.5, dof unchanged);   mass = E_q[f / q].

The weight f/q is bounded: per pair the density ratio St(S)/St(2.25 S) is
maximized near the center at 2.25^{3/2} ~ 3.4 (the wider proposal has the
heavier tail for every dof), so Var_q(f/q) is small and the standard error is
std(f/q)/sqrt(N) ~ 2e-3 at N = 200k for the fixed construction (~ 6x that for
the old one, whose f is ~ #tiles larger). The SE is asserted, not just
documented, so the mass assertions cannot pass on estimator noise. Draws
falling outside every halo-dilated tile have f = 0 by construction (the gates'
support); they contribute zero, not an error. Runs in well under 60 s.

The old construction's weights are taken from the literal historical code path:
stitch.tile_predictive on the tile-normalized cavi.State that
eval_heldout.tile_submixtures still builds for the seam baselines -- component
order inside each tile is identical to the fixed bank's, so the two weight
vectors index one shared Student-t bank.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "experiments"))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from scipy.special import logsumexp

from dp_splat.predictive import StudentT, student_logpdf
from dp_splat_uav import stitch

import eval_heldout

RECORD = REPO / "experiments" / "out" / "mar18_rich_record.json"
MODEL = REPO / "experiments" / "out" / "mar18_rich_model.npz"

N_DRAWS = 200_000
SEED = 7
INFLATE = 1.5 ** 2  # proposal scale-matrix inflation (sd x 1.5)
CHUNK = 16_384

needs_model = pytest.mark.skipif(
    not (RECORD.exists() and MODEL.exists()),
    reason="mar18_rich production artifacts not present",
)


def _flat_bank(preds):
    """Flatten per-tile predictives into (pair_tile, log_pi, spatial bank)."""
    pair_tile = np.concatenate(
        [np.full(p.log_pi.shape[0], t, dtype=np.int64) for t, p in enumerate(preds)]
    )
    log_pi = np.concatenate([np.asarray(p.log_pi) for p in preds])
    st_s = StudentT(
        *(jnp.concatenate([getattr(p.spatial, f) for p in preds])
          for f in StudentT._fields)
    )
    return pair_tile, log_pi, st_s


@pytest.fixture(scope="module")
def banks():
    record = json.loads(RECORD.read_text())
    model = np.load(MODEL)
    grid = eval_heldout.load_grid(model)
    tiles = eval_heldout.tile_submixtures(model, record["alpha_t"])

    # Fixed construction: global weights, un-renormalized within tiles.
    pair_tile, log_pi_fixed, st_s = _flat_bank([t["pred"] for t in tiles])
    # Old construction: per-tile stick weights (the states tile_submixtures
    # builds for the baselines are exactly the pre-fix stitched inputs). The
    # within-tile component order matches t["pred"], so the banks share
    # pair_tile and st_s.
    log_pi_old = np.concatenate([
        np.asarray(stitch.tile_predictive(t["state"], t["cfg"]).log_pi)
        for t in tiles
    ])
    assert log_pi_old.shape == log_pi_fixed.shape
    return grid, pair_tile, log_pi_fixed, log_pi_old, st_s


def _stitched_masses(grid, pair_tile, log_pi_list, st_s, rng):
    """Importance-sampling mass estimates (one per weight vector) and their SEs.

    One shared draw and one shared per-pair log-density pass serve every weight
    vector; the proposal mixes over pairs with the FIRST vector's weights.
    """
    P = pair_tile.shape[0]
    loc = np.asarray(st_s.loc)
    scale = np.asarray(st_s.scale)
    dof = np.asarray(st_s.dof)
    rho = np.exp(log_pi_list[0])
    rho = rho / rho.sum()

    # Draws from q: pair ~ rho, then the pair's Student-t with inflated scale
    # via the scale-mixture representation.
    k = rng.choice(P, size=N_DRAWS, p=rho)
    chol_infl = np.linalg.cholesky(INFLATE * scale)
    z = rng.standard_normal((N_DRAWS, 3))
    g = rng.chisquare(dof[k]) / dof[k]
    s = loc[k] + np.einsum("nij,nj->ni", chol_infl[k], z) / np.sqrt(g)[:, None]

    st_infl = StudentT(loc=jnp.asarray(loc), scale=jnp.asarray(INFLATE * scale),
                       dof=jnp.asarray(dof))
    log_rho = np.log(rho)

    dil_lo = grid.core_lo - grid.halo  # gate support: union of dilated tiles
    dil_hi = grid.core_hi + grid.halo

    weights = [np.zeros(N_DRAWS) for _ in log_pi_list]
    for start in range(0, N_DRAWS, CHUNK):
        sl = slice(start, min(start + CHUNK, N_DRAWS))
        s_c = s[sl]
        xs = jnp.asarray(s_c)
        log_q = logsumexp(
            log_rho[None, :] + np.asarray(student_logpdf(st_infl, xs)), axis=1
        )
        inside = np.zeros(s_c.shape[0], dtype=bool)
        for t in range(dil_lo.shape[0]):
            inside |= np.all((s_c[:, :2] >= dil_lo[t]) & (s_c[:, :2] <= dil_hi[t]),
                             axis=1)
        if not inside.any():
            continue  # f = 0 outside the gates' support: zero weight
        xs_in = jnp.asarray(s_c[inside])
        g_pair = np.asarray(stitch.gate_weights(grid, xs_in))[:, pair_tile]
        log_g = np.where(g_pair > 0.0,
                         np.log(np.where(g_pair > 0.0, g_pair, 1.0)), -np.inf)
        lp = np.asarray(student_logpdf(st_s, xs_in))  # shared target bank pass
        for i, log_pi in enumerate(log_pi_list):
            log_f = logsumexp(log_g + log_pi[None, :] + lp, axis=1)
            w = weights[i][sl]
            w[inside] = np.exp(log_f - log_q[inside])
            weights[i][sl] = w
    return [(float(w.mean()), float(w.std() / np.sqrt(N_DRAWS))) for w in weights]


@needs_model
def test_bank_weight_conventions(banks):
    grid, pair_tile, log_pi_fixed, log_pi_old, st_s = banks
    pi_fixed = np.exp(log_pi_fixed)
    # Global weights sum to 1 over components; the pair bank counts a
    # straddling component once per source tile, so its total is slightly
    # above 1 (mar18_rich: 880 pairs for 841 components).
    assert 1.0 - 1e-9 <= pi_fixed.sum() <= 1.2
    # The old construction normalizes within every non-empty tile.
    for t in range(grid.core_lo.shape[0]):
        m = pair_tile == t
        if m.any():
            np.testing.assert_allclose(np.exp(log_pi_old[m]).sum(), 1.0, atol=1e-9)


@needs_model
def test_stitched_mass_fixed_is_one_and_old_is_tiles(banks):
    grid, pair_tile, log_pi_fixed, log_pi_old, st_s = banks
    rng = np.random.default_rng(SEED)
    (mass_fixed, se_fixed), (mass_old, se_old) = _stitched_masses(
        grid, pair_tile, [log_pi_fixed, log_pi_old], st_s, rng
    )
    # Estimator resolution first: the mass assertions must not be satisfiable
    # by Monte Carlo noise.
    assert se_fixed < 0.01, f"IS estimator too noisy: SE {se_fixed:.4f}"
    assert se_old < 0.06, f"IS estimator too noisy (old): SE {se_old:.4f}"
    # Fixed construction: mass 1 up to gate-support/tail leakage of edge
    # components (their Student-t mass outside the halo-dilated footprint).
    assert 0.90 <= mass_fixed <= 1.02, (
        f"stitched mass {mass_fixed:.4f} (SE {se_fixed:.1e}) not ~ 1"
    )
    # Regression guard: per-tile-renormalized weights (the pre-fix eval-side
    # construction) inflate the mass to ~ #tiles (6 here).
    assert mass_old > 3.0, (
        f"old construction mass {mass_old:.4f} (SE {se_old:.1e}); expected ~ 6"
    )

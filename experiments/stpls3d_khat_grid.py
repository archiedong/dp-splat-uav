"""STPLS3D K-hat vs scene-complexity grid: one untiled DP-Splat fit per cell.

For every input tile (STPLS3D ships ~25 synthetic 500 m x 500 m tiles plus four
real-world scans, all PLY) the script fits a SINGLE DP-Splat model per
(tile, alpha, seed) cell -- no spatial tiling; the point of this grid is to ask
whether the posterior complexity K-hat tracks scene content, so each tile is
fitted whole against its own data-scale prior. The fit machinery is shared with
the Phase-A pipeline (run_tiled_fit.fit_tile_cavi / fit_tile_svi with unit
weights, including its SVI-threshold dispatch), so the estimators here are the
same ones reported on H3D.

Alongside the two K-hat estimators (soft-count threshold and entropy/perplexity,
brief section 3.7) each record carries complexity proxies computed from the raw
points, independent of any fit:

  class_entropy      Shannon entropy (nats) of the semantic class distribution
                     (STPLS3D ground-truth labels; field name recorded per tile)
  n_classes          number of distinct semantic classes present
  density_per_m2     raw point count / area of the 2D (x, y) bounding box
  surface_variation  mean local PCA smallest-eigenvalue ratio
                     lambda_1 / (lambda_1 + lambda_2 + lambda_3) over a point
                     subsample with k nearest neighbors (Pauly et al. 2002)
  z_range            vertical extent in meters

Proxies use the (subsampled) loaded points except density_per_m2, whose
numerator is the vertex count from the PLY header; the subsample is uniform,
so the distributional proxies are unbiased and n_classes can at most drop
classes too rare to survive the cap.

Output: one JSON record per cell appended to <out-prefix>_grid.json (a list).
Resume-safe: cells already present in that file are skipped, so an interrupted
grid can be re-launched with the same command line. Figures are separate
(fig_khat_vs_complexity.py).

Run (single-tile wiring check, minutes):
  nice -n 15 ~/.venvs/dp-splat/bin/python experiments/stpls3d_khat_grid.py \
      --inputs "~/dp-splat-data/stpls3d/STPLS3D/Synthetic_v3/1_points_GTv3.ply" \
      --subsample 500000 --alphas 1 --seeds 0 \
      --out-prefix experiments/out/stpls3d_smoke

Run (full synthetic grid):
  nice -n 15 ~/.venvs/dp-splat/bin/python experiments/stpls3d_khat_grid.py \
      --inputs "~/dp-splat-data/stpls3d/STPLS3D/Synthetic_v3/*.ply" \
      --out-prefix experiments/out/stpls3d_synth
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_tiled_fit as rtf  # noqa: E402  (sets up sys.path and float64 JAX)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from dp_splat import cavi, prune  # noqa: E402
from dp_splat_uav import io_aerial  # noqa: E402

# Candidate names for the per-point semantic ground-truth field, in preference
# order. STPLS3D PLY exports use "class" (uint8, labels 0-19).
_CLASS_FIELDS = ("class", "label", "classification", "scalar_class", "sem_class")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-tile single-model DP-Splat fits + complexity proxies (STPLS3D)")
    p.add_argument("--inputs", required=True,
                   help="glob of PLY tiles (quote it so the shell does not expand)")
    p.add_argument("--out-prefix", required=True,
                   help="records go to <out-prefix>_grid.json")
    p.add_argument("--subsample", type=float, default=3e6,
                   help="per-tile point cap (uniform without replacement)")
    p.add_argument("--alphas", default="0.1,1,10",
                   help="comma-separated DP concentrations")
    p.add_argument("--seeds", default="0,1", help="comma-separated fit seeds")
    p.add_argument("--truncation", type=int, default=64, help="truncation level T")
    p.add_argument("--prior-scale", type=float, default=1.0,
                   help="multiplier on the spatial prior length-scale (Psi0 scales "
                        "with its square); < 1 lets components resolve sub-structure")
    p.add_argument("--max-iters", type=int, default=200, help="CAVI iteration cap")
    p.add_argument("--tol", type=float, default=1e-6, help="relative ELBO tolerance")
    p.add_argument("--n-min", type=float, default=1.0,
                   help="soft-count threshold for the count K-hat estimator")
    p.add_argument("--svi-threshold", type=float, default=2e6,
                   help="tiles above this loaded point count use the SVI path")
    p.add_argument("--svi-batch", type=int, default=65536)
    p.add_argument("--svi-epochs", type=float, default=3.0,
                   help="passes over the tile per SVI fit")
    p.add_argument("--svi-eval-every", type=int, default=25,
                   help="steps between subsample-ELBO evaluations")
    p.add_argument("--svi-eval-points", type=int, default=100_000,
                   help="fixed subsample size for the SVI ELBO trace")
    p.add_argument("--chunk", type=int, default=262144,
                   help="chunk size for full-data passes on SVI-path tiles")
    p.add_argument("--proxy-points", type=int, default=50_000,
                   help="subsample size for the surface-variation proxy")
    p.add_argument("--proxy-neighbors", type=int, default=16,
                   help="k nearest neighbors for the surface-variation proxy")
    p.add_argument("--proxy-seed", type=int, default=0,
                   help="seed for proxy subsampling (fixed so proxies are "
                        "identical across grid cells of a tile)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Complexity proxies (raw points only; no fit involved)
# ---------------------------------------------------------------------------


def ply_vertex_count(path: Path) -> int:
    """Vertex count from the PLY header (cheap; no payload read)."""
    with open(path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"{path}: no 'element vertex' line in header")


def find_class_field(extras: dict) -> str:
    for name in _CLASS_FIELDS:
        if name in extras:
            return name
    raise ValueError(
        f"no semantic class field among extras {sorted(extras)}; "
        f"expected one of {_CLASS_FIELDS}")


def class_entropy(labels: np.ndarray):
    """(Shannon entropy in nats, number of distinct classes)."""
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum()), int(counts.size)


def surface_variation(s: np.ndarray, n_sample: int, k: int, rng) -> float:
    """Mean local PCA smallest-eigenvalue ratio lambda_1 / sum(lambda).

    Neighborhoods are the k nearest neighbors within one uniform subsample of
    the tile (self included), so the proxy measures shape at the subsample's
    own spacing -- comparable across tiles as long as n_sample is fixed.
    """
    n = s.shape[0]
    if n > n_sample:
        idx = rng.choice(n, size=n_sample, replace=False)
        pts = np.ascontiguousarray(s[idx])
    else:
        pts = np.ascontiguousarray(s)
    tree = cKDTree(pts)
    _, nbr = tree.query(pts, k=k + 1)
    nb = pts[nbr]
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("mki,mkj->mij", nb, nb) / (k + 1)
    ev = np.linalg.eigvalsh(cov)  # ascending eigenvalues, shape (m, 3)
    tot = ev.sum(axis=1)
    ok = tot > 0  # degenerate neighborhoods (all-duplicate points) carry no shape
    return float(np.mean(ev[ok, 0] / tot[ok]))


def tile_proxies(path: Path, s: np.ndarray, extras: dict, args) -> dict:
    field = find_class_field(extras)
    ent, n_cls = class_entropy(extras[field])
    lo = s.min(axis=0)
    hi = s.max(axis=0)
    area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
    n_raw = ply_vertex_count(path)
    rng = np.random.default_rng(args.proxy_seed)
    return dict(
        class_field=field,
        class_entropy=ent,
        n_classes=n_cls,
        density_per_m2=n_raw / area,
        surface_variation=surface_variation(
            s, args.proxy_points, args.proxy_neighbors, rng),
        z_range=float(hi[2] - lo[2]),
        bbox_area_m2=area,
        n_raw=n_raw,
    )


# ---------------------------------------------------------------------------
# Grid bookkeeping
# ---------------------------------------------------------------------------


def natural_key(path: str):
    """Sort '2_points' before '10_points'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", Path(path).name)]


def load_records(grid_path: Path) -> list:
    if grid_path.exists():
        return json.loads(grid_path.read_text())
    return []


def save_records(grid_path: Path, records: list):
    tmp = grid_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(records, indent=1))
    os.replace(tmp, grid_path)


def fit_cell(tile_seed, xs, xc, cfg, args):
    """One untiled fit: unit weights, tile-local data-scale priors."""
    sp = cavi.default_niw_prior(xs, cfg.T, cfg.kappa0, cfg.nu0_offset)
    if args.prior_scale != 1.0:
        sp = sp._replace(Psi=sp.Psi * args.prior_scale ** 2)
    cp = cavi.default_niw_prior(xc, cfg.T, cfg.kappa0, cfg.nu0_offset)
    w = jnp.ones(xs.shape[0])
    if xs.shape[0] > args.svi_threshold:
        return rtf.fit_tile_svi(tile_seed, xs, xc, w, cfg, sp, cp, args)
    return rtf.fit_tile_cavi(tile_seed, xs, xc, w, cfg, sp, cp)


def main():
    args = parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    paths = sorted(glob.glob(str(Path(args.inputs).expanduser())), key=natural_key)
    if not paths:
        raise SystemExit(f"--inputs {args.inputs!r} matched no files")

    grid_path = Path(f"{args.out_prefix}_grid.json")
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(grid_path)
    done = {(r["tile"], r["alpha"], r.get("prior_scale", 1.0), r["seed"]) for r in records}
    print(f"[grid] {len(paths)} tiles x {len(alphas)} alphas x {len(seeds)} seeds; "
          f"{len(done)} cells already recorded in {grid_path}", flush=True)

    for path_str in paths:
        path = Path(path_str)
        tile = path.stem
        todo = [(a, s) for a in alphas for s in seeds
                if (tile, a, args.prior_scale, s) not in done]
        if not todo:
            print(f"[{tile}] all cells recorded, skipping", flush=True)
            continue

        t0 = time.perf_counter()
        subsample = int(args.subsample) if args.subsample else None
        s_raw, c, extras = io_aerial.load_ply(path, subsample=subsample,
                                              rng=args.proxy_seed)
        proxies = tile_proxies(path, s_raw, extras, args)
        s, origin = io_aerial.center_scene(s_raw)
        del s_raw, extras
        n = s.shape[0]
        xs = jnp.asarray(s)
        xc = jnp.asarray(c)
        t_load = time.perf_counter() - t0
        print(f"[{tile}] n_raw={proxies['n_raw']} n_used={n} "
              f"class_entropy={proxies['class_entropy']:.3f} "
              f"n_classes={proxies['n_classes']} "
              f"density={proxies['density_per_m2']:.1f}/m^2 "
              f"surf_var={proxies['surface_variation']:.4f} "
              f"z_range={proxies['z_range']:.1f}m "
              f"({t_load:.1f}s load+proxies)", flush=True)

        for alpha, seed in todo:
            cfg = cavi.Config(weight_prior="dp", T=args.truncation, alpha=alpha,
                              max_iters=args.max_iters, tol=args.tol)
            t0 = time.perf_counter()
            state, counts, history, meta = fit_cell(seed, xs, xc, cfg, args)
            seconds = time.perf_counter() - t0
            dips = sum(
                1 for i in range(1, len(history))
                if history[i] < history[i - 1] - 1e-9 * abs(history[i - 1])
            )
            khat_count = int((counts > args.n_min).sum())
            khat_entropy = prune.entropy_effective_k(state, cfg)
            records.append(dict(
                tile=tile,
                input=str(path),
                alpha=alpha,
                prior_scale=args.prior_scale,
                seed=seed,
                n_used=n,
                subsample=subsample,
                truncation=args.truncation,
                n_min=args.n_min,
                origin=origin.tolist(),
                khat_count=khat_count,
                khat_entropy=khat_entropy,
                elbo_tail=history[-5:],
                elbo_dips=dips,
                seconds=seconds,
                proxies=proxies,
                versions=dict(jax=jax.__version__, numpy=np.__version__),
                **meta,
            ))
            save_records(grid_path, records)
            done.add((tile, alpha, seed))
            print(f"[{tile}] alpha={alpha} seed={seed} path={meta['path']} "
                  f"K-hat={khat_count} (entropy {khat_entropy:.1f}) "
                  f"dips={dips} {seconds:.1f}s", flush=True)

    print(f"[done] {len(records)} records in {grid_path}", flush=True)


if __name__ == "__main__":
    main()

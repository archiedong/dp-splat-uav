"""Spatial complexity map: per-tile pre-merge K-hat over the source point density.

Consumes the outputs of experiments/run_tiled_fit.py (step 7): the <name>_model.npz
tile geometry (core rectangles in centered scene meters, saved origin) and the
per-tile pre-merge K-hat diagnostics. Each tile core is filled by its count-estimator
K-hat (soft-count threshold, brief section 3.7) on a colorblind-safe sequential ramp
and annotated with both estimators -- the count K-hat and the entropy (perplexity)
K-hat -- so exact values survive any color reproduction. Underneath, a light gray
hexbin of a subsample of the source cloud shows the scene structure the complexity
map should track (built-up tiles high, open-terrain tiles low).

Coordinates: the model stores tile geometry in centered scene meters together with
the centering origin, so the source cloud is shifted by that saved origin (never
re-centered from the subsample) to align exactly.

Output: <out>.png at 300 dpi and <out>.pdf (vector), same basename.

Run:
  ~/.venvs/dp-splat/bin/python experiments/fig_khat_map.py \
      --record experiments/out/full_val_record.json \
      --model experiments/out/full_val_model.npz \
      --out figures/full_val_khat_map.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dp_splat_uav import io_aerial


def parse_args():
    p = argparse.ArgumentParser(description="Per-tile K-hat spatial complexity map")
    p.add_argument("--record", required=True, help="<name>_record.json from run_tiled_fit")
    p.add_argument("--model", required=True, help="<name>_model.npz from run_tiled_fit")
    p.add_argument("--out", required=True,
                   help="output path; .png (300 dpi) and .pdf are written with this basename")
    p.add_argument("--input", default=None,
                   help="source point cloud for the density layer "
                        "(default: the input path recorded in the fit config)")
    p.add_argument("--subsample", type=int, default=300_000,
                   help="points drawn for the density hexbin")
    p.add_argument("--seed", type=int, default=0, help="subsample seed (rendering only)")
    p.add_argument("--gridsize", type=int, default=180, help="hexbin resolution")
    return p.parse_args()


def load_density_points(args, record, origin):
    """Ground-plane (x, y) subsample of the source cloud, in centered scene meters."""
    path = Path(args.input or record["config"]["input"]).expanduser()
    suffix = path.suffix.lower()
    if suffix in (".las", ".laz"):
        s, _, _ = io_aerial.load_laz(path, subsample=args.subsample, rng=args.seed)
    elif suffix == ".ply":
        s, _, _ = io_aerial.load_ply(path, subsample=args.subsample, rng=args.seed)
    else:
        raise ValueError(f"unsupported point-cloud format {suffix!r}")
    return (s - origin)[:, :2]


def main():
    args = parse_args()
    record = json.loads(Path(args.record).read_text())
    model = np.load(args.model)

    core_lo = model["core_lo"]  # (T, 2), centered scene meters
    core_hi = model["core_hi"]
    khat = model["tile_khat_count"].astype(int)
    khat_h = model["tile_khat_entropy"]
    origin = model["origin"]
    xy = load_density_points(args, record, origin)

    # Sequential ramp for a small integer magnitude: ColorBrewer YlGnBu
    # (colorblind-safe), one bin per integer K-hat so the colorbar reads exactly.
    cmap = plt.get_cmap("YlGnBu")
    levels = np.arange(khat.min() - 0.5, khat.max() + 1.5)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    fill_alpha = 0.55

    extent = np.concatenate([model["bbox_lo"], model["bbox_hi"]])[[0, 2, 1, 3]]
    span = (extent[1] - extent[0], extent[3] - extent[2])
    fig_w = 7.0
    fig_h = fig_w * span[1] / span[0] + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Base layer: light gray log-density of the scene points.
    grey = mcolors.LinearSegmentedColormap.from_list("light_grey", ["#ffffff", "#595959"])
    ax.hexbin(xy[:, 0], xy[:, 1], gridsize=args.gridsize, cmap=grey,
              bins="log", mincnt=1, linewidths=0.0, zorder=1)

    # Complexity layer: translucent core rectangles colored by pre-merge K-hat.
    for t in range(core_lo.shape[0]):
        w, h = core_hi[t] - core_lo[t]
        face = cmap(norm(khat[t]))
        ax.add_patch(Rectangle(core_lo[t], w, h, facecolor=face, alpha=fill_alpha,
                               edgecolor="none", zorder=2))
        ax.add_patch(Rectangle(core_lo[t], w, h, facecolor="none",
                               edgecolor="black", linewidth=0.8, zorder=3))
        # Ink color from the on-page tile luminance: the fill is blended over the
        # density layer, which sits near mid-gray wherever there is structure.
        blended = fill_alpha * np.asarray(face[:3]) + (1.0 - fill_alpha) * 0.60
        lum = float(np.dot(blended, [0.299, 0.587, 0.114]))
        ink = "black" if lum > 0.45 else "white"
        cx, cy = core_lo[t] + 0.5 * np.array([w, h])
        ax.text(cx, cy, f"$\\hat{{K}} = {khat[t]}$\n$\\hat{{K}}_H = {khat_h[t]:.1f}$",
                ha="center", va="center", color=ink, fontsize=9, zorder=4,
                linespacing=1.4)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.tick_params(labelsize=9)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02,
                        ticks=np.arange(khat.min(), khat.max() + 1))
    cbar.set_label("pre-merge $\\hat{K}$ per tile", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    base = Path(args.out).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[fig] wrote {base.with_suffix('.png')} (300 dpi) and {base.with_suffix('.pdf')}; "
          f"tiles {core_lo.shape[0]}, K-hat range [{khat.min()}, {khat.max()}], "
          f"{xy.shape[0]} density points")


if __name__ == "__main__":
    main()

"""K-hat vs scene-complexity proxies across STPLS3D tiles.

Consumes the <out-prefix>_grid.json written by stpls3d_khat_grid.py: one panel
per complexity proxy, K-hat on the y axis, all (tile, alpha, seed) cells as
points colored by alpha. Each panel is annotated with the Spearman rank
correlation pooled over all cells (rank-based, so monotone but nonlinear
relationships still register), and a per-alpha least-squares line is drawn
where enough points exist -- the DP concentration shifts the K-hat level, so
the within-alpha trend is the honest read of the complexity relationship.

Seeds enter as separate points; with the default two seeds per cell this
slightly double-counts tiles in the pooled rho, which is acceptable for a
screening figure (per-alpha annotations are the per-condition check).

Output: <out>.png at 300 dpi and <out>.pdf (vector), same basename.

Run:
  ~/.venvs/dp-splat/bin/python experiments/fig_khat_vs_complexity.py \
      --grid experiments/out/stpls3d_synth_grid.json \
      --out figures/stpls3d_khat_vs_complexity.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Proxy key in each record's "proxies" dict -> axis label.
PROXIES = [
    ("class_entropy", "semantic class entropy (nats)"),
    ("n_classes", "distinct semantic classes"),
    ("density_per_m2", "point density (points / m$^2$)"),
    ("surface_variation", "surface variation $\\lambda_1 / \\sum_i \\lambda_i$"),
    ("z_range", "z range (m)"),
]

# Okabe-Ito, colorblind-safe; alphas are assigned in sorted order.
ALPHA_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Scatter/regression panels: K-hat vs complexity proxies")
    p.add_argument("--grid", required=True,
                   help="<out-prefix>_grid.json from stpls3d_khat_grid.py")
    p.add_argument("--out", required=True,
                   help="output path; .png (300 dpi) and .pdf share this basename")
    p.add_argument("--khat-key", default="khat_count",
                   choices=["khat_count", "khat_entropy"],
                   help="which K-hat estimator to plot")
    return p.parse_args()


def spearman_label(x, y):
    """Pooled Spearman rho annotation; degenerate inputs annotate as n/a."""
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return f"$\\rho$ = n/a (n = {len(x)})"
    rho, pval = stats.spearmanr(x, y)
    return f"$\\rho$ = {rho:.2f} (p = {pval:.2g}, n = {len(x)})"


def main():
    args = parse_args()
    records = json.loads(Path(args.grid).read_text())
    if not records:
        raise SystemExit(f"{args.grid}: no records")

    alphas = sorted({r["alpha"] for r in records})
    colors = {a: ALPHA_COLORS[i % len(ALPHA_COLORS)] for i, a in enumerate(alphas)}
    khat = np.array([r[args.khat_key] for r in records], dtype=float)
    rec_alpha = np.array([r["alpha"] for r in records])
    ylabel = ("$\\hat{K}$ (count estimator)" if args.khat_key == "khat_count"
              else "$\\hat{K}$ (entropy estimator)")

    n_panels = len(PROXIES)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                             squeeze=False)

    for i, (key, label) in enumerate(PROXIES):
        ax = axes[i // ncols][i % ncols]
        x = np.array([r["proxies"][key] for r in records], dtype=float)
        for a in alphas:
            sel = rec_alpha == a
            ax.scatter(x[sel], khat[sel], s=28, alpha=0.8, linewidths=0.4,
                       edgecolors="white", color=colors[a],
                       label=f"$\\alpha$ = {a:g}", zorder=3)
            # Per-alpha least-squares line where a slope is identifiable.
            if sel.sum() >= 3 and np.unique(x[sel]).size >= 2:
                b, c = np.polyfit(x[sel], khat[sel], 1)
                xg = np.linspace(x[sel].min(), x[sel].max(), 32)
                ax.plot(xg, b * xg + c, color=colors[a], linewidth=1.2,
                        alpha=0.7, zorder=2)
        ax.annotate(spearman_label(x, khat), xy=(0.03, 0.97),
                    xycoords="axes fraction", ha="left", va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="0.7", alpha=0.85))
        ax.set_xlabel(label, fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.grid(True, linewidth=0.4, alpha=0.4, zorder=0)

    # Shared alpha legend in the spare panel slot (or below if the grid is full).
    handles, labels = axes[0][0].get_legend_handles_labels()
    spare = [axes[i // ncols][i % ncols] for i in range(n_panels, nrows * ncols)]
    for ax in spare:
        ax.axis("off")
    if spare:
        spare[0].legend(handles, labels, loc="center", fontsize=10,
                        title="concentration", frameon=False)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=len(alphas),
                   fontsize=9, frameon=False)

    n_tiles = len({r["tile"] for r in records})
    fig.suptitle(f"STPLS3D: posterior complexity vs scene-complexity proxies "
                 f"({n_tiles} tiles, {len(records)} fits)", fontsize=11)
    fig.tight_layout(rect=(0, 0.02 if not spare else 0, 1, 0.96))

    base = Path(args.out).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[fig] wrote {base.with_suffix('.png')} (300 dpi) and "
          f"{base.with_suffix('.pdf')}; {len(records)} records, "
          f"{n_tiles} tiles, alphas {alphas}")


if __name__ == "__main__":
    main()

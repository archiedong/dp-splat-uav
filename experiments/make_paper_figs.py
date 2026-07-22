"""Camera-ready paper figures, regenerated purely from experiments/out artifacts.

Produces the three figures embedded in paper/figs/, at IEEE column width
(252 pt), TrueType (Type 42) fonts for PDF eXpress, serif text matched to
IEEEtran, no in-image titles, and >= 7 pt effective type at print size:

  fig 2  mar18_rich_sparsification.pdf   <- out/mar18_rich_sparsification.json
  fig 3  change_nov18_mar19.pdf          <- out/change_nov18_mar19_cells.npz
  fig 4  stpls3d_khat_vs_complexity.pdf  <- out/stpls3d_full_grid.json

The change figure carries three panels (|d|, per-cell LoD95, flag map) as the
caption promises; the K-hat figure annotates the alpha=1 arm's Spearman rho
(the values quoted in Sec. VI-D), not the pooled-arm rho of the exploratory
figure. No raw dataset access; every number is read from the artifacts.

Run:
  ~/.venvs/dp-splat/bin/python experiments/make_paper_figs.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42,          # TrueType, not Type 3 (IEEE PDF eXpress)
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage, stats

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiments" / "out"
FIGS = REPO / "figures"
PAPER_FIGS = REPO / "paper" / "figs"

COLW = 252.0 / 72.27  # IEEEtran \columnwidth in inches

# Okabe-Ito (colorblind-safe), fixed assignment.
C_BLUE, C_VERM, C_ORANGE, C_SKY, C_GREEN = (
    "#0072B2", "#D55E00", "#E69F00", "#56B4E9", "#009E73")


def save(fig, stem):
    for d in (FIGS, PAPER_FIGS):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.pdf")
        fig.savefig(d / f"{stem}.png", dpi=400)
    plt.close(fig)
    print(f"[fig] wrote {stem}.pdf/.png to figures/ and paper/figs/")


# ---------------------------------------------------------------------------
# Fig. 2 — sparsification (March 2018 production hold-out)
# ---------------------------------------------------------------------------

def fig_sparsification():
    d = json.loads((OUT / "mar18_rich_sparsification.json").read_text())
    fr = np.asarray(d["fractions"], float)
    var = d["u_variants"]["sqrt_mean_marginal_var"]["global_"]
    nlp = d["u_variants"]["neg_log_predictive"]["global_"]
    dens = d["u_variants"]["local_density"]["global_"]

    fig, ax = plt.subplots(figsize=(COLW, 2.28), layout="constrained")
    ax.plot(fr, var["curve_oracle"], color="black", ls="--", lw=1.0,
            zorder=5, label="oracle (sort by $|e|$)")
    ax.plot(fr, var["curve_random"], color="0.55", ls=":", lw=1.0,
            label="random")
    ax.plot(fr, var["curve_u"], color=C_SKY, lw=1.4,
            label=(r"pooled $\sqrt{\overline{\mathrm{Var}}}$"
                   f"  (AUSE {var['ause']:.2f})"))
    ax.plot(fr, nlp["curve_u"], color=C_VERM, lw=1.4,
            label=(r"predictive $-\log \hat p$"
                   f"  (AUSE {nlp['ause']:.2f})"))
    ax.plot(fr, dens["curve_u"], color=C_GREEN, lw=1.4,
            label=("model-free local density"
                   f"  (AUSE {dens['ause']:.2f})"))
    ax.set_xlabel("fraction removed (by decreasing $u$)")
    ax.set_ylabel(r"normalized mean $|e|$ retained")
    ax.set_xlim(0.0, 0.98)
    # AURG values live in Table II and the Sec. V-C prose; AUSE-only labels
    # keep the legend narrow enough for the free band between the pooled
    # and predictive curves (it must not occlude the oracle curve).
    ax.legend(loc="upper right", bbox_to_anchor=(0.985, 0.82),
              framealpha=1.0, borderpad=0.4, handlelength=1.6)
    ax.grid(alpha=0.25, lw=0.4)
    save(fig, "mar18_rich_sparsification")


# ---------------------------------------------------------------------------
# Fig. 3 — change detection maps (Nov 2018 vs Mar 2019)
# ---------------------------------------------------------------------------

VEG_CLASSES = (0, 6, 7)
IMPERVIOUS = 1


def fig_change():
    z = np.load(OUT / "change_nov18_mar19_cells.npz")
    nx, ny = (int(v) for v in z["shape"])
    cell = float(z["cell"])
    extent = [0.0, nx * cell, 0.0, ny * cell]
    valid, flagged = z["valid"], z["flagged"]
    dom_a, dom_b = z["dom_a"], z["dom_b"]
    pur_a, pur_b = z["purity_a"], z["purity_b"]

    veg = np.isin(dom_a, VEG_CLASSES) | np.isin(dom_b, VEG_CLASSES)
    elig = valid & (pur_a > 0.6) & (pur_b > 0.6) & ~veg
    pos = dom_a != dom_b

    abs_d = np.abs(np.where(valid, z["d"], np.nan)).reshape(ny, nx)
    lod = np.where(valid, z["lod"], np.nan).reshape(ny, nx)
    vmax_d = float(np.nanquantile(abs_d, 0.99))
    vmax_l = float(np.nanquantile(lod, 0.99))

    cat = np.zeros(nx * ny, dtype=np.int8)
    cat[valid & ~elig] = 1
    cat[elig & ~pos & ~flagged] = 2
    cat[elig & ~pos & flagged] = 3
    cat[elig & pos & flagged] = 4
    cat[elig & pos & ~flagged] = 5
    # Isolated 1 m cells of the rare categories (3-5) are sub-half-point
    # specks at print size: dilate each by one cell, expanding only into
    # background categories (0-2) so rare categories never erase each other.
    # Disclosed in the figure caption.
    cat2d = cat.reshape(ny, nx)
    base = cat2d.copy()
    for c in (3, 5, 4):
        grow = ndimage.binary_dilation(base == c)
        cat2d[grow & (cat2d <= 2)] = c
    cat = cat2d.reshape(-1)
    cat_colors = ["white", "#dcdcdc", "#bcd8ec", C_ORANGE, C_VERM, C_BLUE]
    cmap_cat = ListedColormap(cat_colors)
    norm_cat = BoundaryNorm(np.arange(7) - 0.5, cmap_cat.N)

    fig = plt.figure(figsize=(COLW, 3.30), layout="constrained")
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.045, 0.10])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    cmap_d = matplotlib.colormaps["viridis"].copy()
    cmap_d.set_bad("white")
    im_d = axes[0].imshow(abs_d, origin="lower", extent=extent, cmap=cmap_d,
                          vmin=0.0, vmax=vmax_d, interpolation="nearest")
    axes[0].set_title(r"$|d|$ (m)")
    im_l = axes[1].imshow(lod, origin="lower", extent=extent, cmap=cmap_d,
                          vmin=0.0, vmax=vmax_l, interpolation="nearest")
    axes[1].set_title(r"$\mathrm{LoD}_{95}$ (m)")
    axes[2].imshow(cat.reshape(ny, nx), origin="lower", extent=extent,
                   cmap=cmap_cat, norm=norm_cat, interpolation="nearest")
    axes[2].set_title(r"$|d|>\mathrm{LoD}_{95}$ vs. proxy")

    for im, col in ((im_d, 0), (im_l, 1)):
        cax = fig.add_subplot(gs[1, col])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        cax.tick_params(labelsize=7, pad=1.5, length=2)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.6", lw=0.4) for c in
               (cat_colors[4], cat_colors[5], cat_colors[3], cat_colors[2],
                cat_colors[1])]
    labels = ["transition, flagged", "transition, missed", "stable, flagged",
              "stable, not flagged", "ineligible"]
    lax = fig.add_subplot(gs[2, :])
    lax.axis("off")
    lax.legend(handles, labels, loc="center", ncol=3, fontsize=7,
               frameon=False, handlelength=1.0, handleheight=0.8,
               labelspacing=0.3, columnspacing=0.9, borderaxespad=0.0)

    for i, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.set_xticks([0, 150, 300])
        if i == 0:
            ax.set_ylabel("northing (m, local)", fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("easting (m)", fontsize=7, labelpad=1)
    save(fig, "change_nov18_mar19")


# ---------------------------------------------------------------------------
# Fig. 4 — K-hat vs scene-content proxies (STPLS3D matched-N grid)
# ---------------------------------------------------------------------------

PROXIES = [
    ("class_entropy", "class entropy (nats)"),
    ("surface_variation", r"surface variation"),
    ("density_per_m2", r"density (pts/m$^2$)"),
]
ALPHA_COLORS = {0.1: C_SKY, 1.0: C_BLUE, 10.0: C_ORANGE}


def fig_khat():
    recs = json.loads((OUT / "stpls3d_full_grid.json").read_text())
    alphas = sorted({r["alpha"] for r in recs})
    khat = np.array([r["khat_count"] for r in recs], float)
    rec_alpha = np.array([r["alpha"] for r in recs], float)
    a1 = rec_alpha == 1.0

    fig, axgrid = plt.subplots(2, 2, figsize=(COLW, 2.85),
                               layout="constrained")
    axes = axgrid.ravel()
    for i, (key, label) in enumerate(PROXIES):
        ax = axes[i]
        x = np.array([r["proxies"][key] for r in recs], float)
        for a in alphas:
            sel = rec_alpha == a
            ax.scatter(x[sel], khat[sel], s=7, alpha=0.8, linewidths=0.25,
                       edgecolors="white", color=ALPHA_COLORS[a],
                       label=rf"$\alpha={a:g}$", zorder=3)
        b, c = np.polyfit(x[a1], khat[a1], 1)
        xg = np.linspace(x[a1].min(), x[a1].max(), 16)
        ax.plot(xg, b * xg + c, color=C_BLUE, lw=1.0, alpha=0.7, zorder=2)
        rho, p = stats.spearmanr(x[a1], khat[a1])
        ax.annotate(rf"$\rho_{{\alpha=1}}={rho:.2f}$" + f" (p={p:.2f})",
                    xy=(0.04, 0.47), xycoords="axes fraction", ha="left",
                    va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor="0.7", alpha=0.9))
        ax.set_xlabel(label, fontsize=7, labelpad=1.5)
        if i % 2 == 0:
            ax.set_ylabel(r"$\hat{K}$", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yscale("log")
        ax.set_yticks([4, 8, 16, 32, 64])
        ax.set_yticklabels(["4", "8", "16", "32", "64"])
        # Headroom above the truncation value 64 so the alpha=10 arm's
        # pinned row separates from the top spine (the ceiling is visible).
        ax.set_ylim(2.5, 90)
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, lw=0.3, alpha=0.35, zorder=0)

    ax = axes[3]
    ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", frameon=False,
              title="concentration", title_fontsize=7.5, fontsize=7.5,
              markerscale=1.8, labelspacing=0.4,
              bbox_to_anchor=(0.02, 1.02))
    ax.annotate("matched $N=3{\\times}10^{6}$;\n"
                "$\\rho$: $\\alpha{=}1$ arm ($n{=}58$);\n"
                "$\\alpha{=}10$ pinned at $T{=}64$",
                xy=(0.02, 0.02), xycoords="axes fraction", ha="left",
                va="bottom", fontsize=7, color="0.25")
    save(fig, "stpls3d_khat_vs_complexity")


if __name__ == "__main__":
    fig_sparsification()
    fig_change()
    fig_khat()

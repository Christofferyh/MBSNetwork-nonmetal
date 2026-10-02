"""Threshold-derivation figure for the Tier 1 dataset: mode separation
(Ashman's D) per RMSD bin (panel A) alongside the full RMSD density with the
chosen threshold marked (panel B).

Panel A's dot-per-bin style and Panel B's smooth shaded density curve
deliberately match the reference thesis's own Fig 3A/3B, rather than the
bar-chart style used in this repo's earlier draft of this figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from config import config
from network.tier1_threshold import MIN_SEPARATION

COLOR = "#B4D6E3"

# The pooled sparse-tail bin plus nine 0.1-wide bins from 1.1 to 2.0; the
# per-bin Ashman's D values and component counts are loaded from
# network.tier1_threshold's saved multi-seed output (one row per seed). Per
# bin, seeds with an accepted 2-component fit are plotted as the mean D with
# min-max whiskers; D is undefined for seeds where the bin was found
# unimodal, so those are marked on the axis rather than given a value.
RANGES = [
    (0.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.4), (1.4, 1.5),
    (1.5, 1.6), (1.6, 1.7), (1.7, 1.8), (1.8, 1.9), (1.9, 2.0),
]


def plot_threshold_derivation(
    rmsd: np.ndarray,
    n_components_by_seed: np.ndarray,
    ashman_d_by_seed: np.ndarray,
    threshold: float = 1.1,
) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.0, "pdf.fonttype": 42,
    })

    bin_labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in RANGES]
    bin_labels[0] = "0.0-1.1\n(pooled)"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9, 4))

    x = np.arange(len(RANGES))
    n_seeds = n_components_by_seed.shape[0]
    fitted = n_components_by_seed == 2
    for i in x:
        values = ashman_d_by_seed[fitted[:, i], i]
        n_fitted = len(values)
        mixed = 0 < n_fitted < n_seeds
        if n_fitted:
            mean = values.mean()
            axA.errorbar(
                i, mean, yerr=[[mean - values.min()], [values.max() - mean]], fmt="none",
                ecolor="black", elinewidth=0.8, capsize=3, zorder=2,
            )
            axA.scatter(i, mean, color=COLOR, edgecolor="black", linewidth=0.8, s=60, zorder=3)
            if mixed:
                axA.text(i + 0.22, mean, f"{n_fitted}/{n_seeds}", fontsize=7, va="center")
        if n_fitted < n_seeds:
            axA.scatter(i, 0, marker="x", color="grey", linewidth=1.0, s=25, zorder=3, clip_on=False)
            if mixed:
                axA.text(i - 0.15, 0.1, f"{n_seeds - n_fitted}/{n_seeds}", fontsize=7, ha="right", va="center")

    axA.axhline(MIN_SEPARATION, color="black", linestyle="--", linewidth=1.0, zorder=1)
    axA.text(len(RANGES) - 0.5, MIN_SEPARATION + 0.06, f"D = {MIN_SEPARATION:g} (separation cut-off)",
             fontsize=8, ha="right", va="bottom")

    axA.scatter([], [], color=COLOR, edgecolor="black", linewidth=0.8, s=60,
                label="Two-component fit\n(mean, min–max over seeds)")
    axA.scatter([], [], marker="x", color="grey", linewidth=1.0, s=25,
                label="Unimodal (D undefined)")
    axA.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=8, frameon=False)
    axA.set_xticks(x)
    axA.set_xticklabels(bin_labels, rotation=90, fontsize=8)
    axA.set_xlabel("RMSD range (Å)")
    axA.set_ylabel("Ashman's D")
    axA.set_ylim(0, 3.6)
    axA.text(-0.28, 1.05, "A", transform=axA.transAxes, fontsize=16, fontweight="bold")

    kde = gaussian_kde(rmsd)
    x_grid = np.linspace(0, rmsd.max(), 500)
    density = kde(x_grid)
    axB.plot(x_grid, density, color=COLOR, linewidth=2.0, zorder=3)
    axB.fill_between(x_grid, density, color=COLOR, alpha=0.35, zorder=2)
    axB.axvline(threshold, color="black", linestyle="--", linewidth=1.5, zorder=4)
    axB.text(threshold, max(density) * 1.05, rf"  $\tau$ = {threshold} Å", fontsize=10)
    axB.set_xlabel("RMSD (Å)")
    axB.set_ylabel("Probability density")
    axB.set_xlim(0, rmsd.max())
    axB.set_ylim(0, max(density) * 1.15)
    axB.text(-0.18, 1.05, "B", transform=axB.transAxes, fontsize=16, fontweight="bold")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    alignments = np.load(
        config.directory.alignments / "tier1_all_to_all_redundancy_reduced.npy"
    )
    rmsd = alignments[:, 2]
    summary = np.load(
        config.directory.analysis / "tier1_bimodal_overlap_redundancy_reduced_seeds.npz"
    )
    n_components_by_seed = summary["n_components"]
    ashman_d_by_seed = summary["ashman_d"]
    assert ashman_d_by_seed.shape[1] == len(RANGES)
    print(
        f"{len(rmsd)} pairs; seeds {summary['seeds'].tolist()}; "
        f"selected thresholds {summary['thresholds'].tolist()}"
    )

    fig = plot_threshold_derivation(rmsd, n_components_by_seed, ashman_d_by_seed)
    out_path = config.directory.figures / "tier1_threshold_derivation.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_name("tier1_threshold_derivation_preview.png"), dpi=200)
    print(f"Saved {out_path}")

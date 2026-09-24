"""Threshold-derivation figure for the Tier 1 dataset: bimodal overlap per
RMSD bin (panel A) alongside the full RMSD density with the chosen
threshold marked (panel B).

Panel A's dot-per-bin style and Panel B's smooth shaded density curve
deliberately match the reference thesis's own Fig 3A/3B, rather than the
bar-chart style used in this repo's earlier draft of this figure.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from config import config

COLOR = "#B4D6E3"

# The pooled sparse-tail bin plus nine 0.1-wide bins from 1.1 to 2.0; see
# network.tier1_threshold for how these overlap values were computed.
RANGES = [
    (0.0, 1.1), (1.1, 1.2), (1.2, 1.3), (1.3, 1.4), (1.4, 1.5),
    (1.5, 1.6), (1.6, 1.7), (1.7, 1.8), (1.8, 1.9), (1.9, 2.0),
]
OVERLAPS = [0.1051, 0.5286, 0.5819, 0.4744, 0.5610, 0.5741, 0.6636, 0.6209, 0.6777, 0.6384]


def plot_threshold_derivation(rmsd: np.ndarray, threshold: float = 1.1) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.0, "pdf.fonttype": 42,
    })

    bin_labels = [f"{lo:.1f}-{hi:.1f}" for lo, hi in RANGES]
    bin_labels[0] = "0.0-1.1\n(pooled)"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9, 4))

    x = np.arange(len(RANGES))
    axA.scatter(x, OVERLAPS, color=COLOR, edgecolor="black", linewidth=0.8, s=60, zorder=3)
    axA.set_xticks(x)
    axA.set_xticklabels(bin_labels, rotation=90, fontsize=8)
    axA.set_xlabel("RMSD range (Å)")
    axA.set_ylabel(r"$\delta$")
    axA.set_ylim(0, max(OVERLAPS) * 1.15)
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
    alignments = np.load(config.directory.alignments / "tier1_all_to_all.npy")
    rmsd = alignments[:, 2]

    fig = plot_threshold_derivation(rmsd)
    out_path = config.directory.figures / "tier1_threshold_derivation.pdf"
    fig.savefig(out_path)
    print(f"Saved {out_path}")

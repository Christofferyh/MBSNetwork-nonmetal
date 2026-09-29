"""Component-size distribution figure for the final Tier 1 network
(redundancy-reduced, topology-guided recovery applied, threshold 1.1):
every connected component's size, sorted descending, log-scale y-axis
given how skewed this is (one giant component, then a long thin tail).

Only covers nodes that actually appear in the built graph -- sites with
zero edges at that threshold are never added as nodes at all
(build_network() only adds edge endpoints), so they aren't represented
here as size-1 components; see tier1_network.py for that distinction.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from config import config
from network.network import read_network_json

COLOR = "#B4D6E3"


def plot_component_sizes(sizes: list[int]) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.0, "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(range(1, len(sizes) + 1), sizes, color=COLOR, edgecolor="black", linewidth=0.8, s=60, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Component rank (descending size)")
    ax.set_ylabel("Component size (log scale)")
    ax.set_xlim(0, len(sizes) + 1)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    network = read_network_json("Tier1NetworkRecovered")
    print(f"Network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")

    components = sorted(nx.connected_components(network), key=len, reverse=True)
    sizes = [len(c) for c in components]
    print(f"Total connected components: {len(sizes)}")
    print(f"Top 5 sizes: {sizes[:5]}")
    print(f"Singletons (size 1) among these: {sum(1 for s in sizes if s == 1)}")

    fig = plot_component_sizes(sizes)
    out_path = config.directory.figures / "tier1_component_size_distribution.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_name("tier1_component_size_distribution_preview.png"), dpi=200)
    print(f"Saved {out_path}")

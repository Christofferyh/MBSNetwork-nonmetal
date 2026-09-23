"""Tier 1 network construction: build a similarity network among stored
ActiveSite rows.

Reuses PairwiseAligner and build_network() from the existing codebase
unchanged. A fresh AlignmentConfig() must be used here rather than one
already carrying manually-attached Open3D objects (as align_point_clouds()
needs for single-pair use): PairwiseAligner builds those objects itself,
independently, inside each worker process it spawns, and a config that
already has them attached fails to pickle when handed to a worker.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from alignment.alignment import PairwiseAligner
from config import AlignmentConfig, config
from database.datamodel.models import ActiveSite
from network.network import build_network, write_network_json

logger = logging.getLogger(__name__)


def build_tier1_network(
    sites: list[ActiveSite],
    alignments: np.ndarray | None = None,
    threshold: float | None = None,
) -> tuple[nx.Graph, np.ndarray]:
    """Build a weighted similarity network from ActiveSite point cloud
    alignments.

    If `alignments` is given (e.g. a precomputed all-to-all array already on
    disk), it's used directly instead of recomputing every pair with
    PairwiseAligner, which is wasteful once that full alignment already
    exists; otherwise every pair among `sites` is aligned fresh. If
    `threshold` is given, only pairs with rmsd <= threshold become edges
    (and only sites appearing in at least one such pair become nodes);
    otherwise every pair becomes an edge.
    """
    if alignments is None:
        fresh_alignment_config = AlignmentConfig()
        aligner = PairwiseAligner(sites, fresh_alignment_config)
        alignments = aligner.align()

    if threshold is None:
        edge_bunch = [(int(r[0]), int(r[1]), float(r[2])) for r in alignments]
    else:
        edge_bunch = [
            (int(r[0]), int(r[1]), float(r[2]))
            for r in alignments
            if r[2] <= threshold
        ]
    network = build_network(edge_bunch)

    for site in sites:
        if network.has_node(site.id):
            network.nodes[site.id].update(
                pdb_id=site.pdb_id,
                chain=site.chain,
                ec_numbers=site.ec_numbers,
                confidence_tier=site.confidence_tier,
                source=site.source,
                mcsa_id=site.mcsa_id,
            )

    return network, alignments


if __name__ == "__main__":
    from sqlalchemy import select

    from database import Session

    with Session() as session:
        all_sites = session.execute(select(ActiveSite)).scalars().all()

    alignments_path = config.directory.alignments / "tier1_all_to_all.npy"
    alignments = np.load(alignments_path)
    print(
        f"Building network from {len(all_sites)} stored Tier 1 sites using "
        f"the precomputed all-to-all alignment ({len(alignments)} pairs)..."
    )

    old_threshold = 0.8
    threshold = 1.1

    old_network, _ = build_tier1_network(
        all_sites, alignments=alignments, threshold=old_threshold
    )
    print(f"Edges at threshold <= {old_threshold}: {old_network.number_of_edges()}")

    network, _ = build_tier1_network(all_sites, alignments=alignments, threshold=threshold)
    print(f"Edges at threshold <= {threshold}: {network.number_of_edges()}")
    print(
        f"Nodes with >=1 edge at threshold <= {threshold}: "
        f"{network.number_of_nodes()}"
    )

    name = "Tier1Network"
    network_dir = config.directory.networks / name
    network_dir.mkdir(parents=True, exist_ok=True)
    write_network_json(network, network_dir)
    print(f"Saved network to {network_dir}")

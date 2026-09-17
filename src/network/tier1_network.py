"""Tier 1 network construction: compute all pairwise alignments among
stored ActiveSite rows and build a similarity network from the result.

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

from alignment.alignment import PairwiseAligner
from config import AlignmentConfig
from database.datamodel.models import ActiveSite
from network.network import build_network

logger = logging.getLogger(__name__)


def build_tier1_network(sites: list[ActiveSite]) -> tuple[nx.Graph, "np.ndarray"]:
    """Align every pair of the given sites and build a weighted network
    from the result. Returns both the graph and the raw alignment array,
    since the array is needed separately for threshold analysis.
    """
    fresh_alignment_config = AlignmentConfig()
    aligner = PairwiseAligner(sites, fresh_alignment_config)
    alignments = aligner.align()

    edge_bunch = [(int(r[0]), int(r[1]), float(r[2])) for r in alignments]
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

    print(f"Building network from {len(all_sites)} stored Tier 1 sites...")
    network, alignments = build_tier1_network(all_sites)
    print(f"{network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
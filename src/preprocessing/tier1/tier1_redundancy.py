"""Tier 1 redundancy reduction: collapse near-duplicate ActiveSite rows
into one representative per cluster, adapted from the redundancy-
reduction method already used for the metal-binding-site dataset (see
preprocessing/dataset.py's representative/constituent convention) --
but grouped by UniProt accession only, since Tier 1 sites have no metal
ligand to group by.

Algorithm, per UniProt-accession group:
  - 1 site: trivially its own representative, no comparison needed.
  - 2+ sites: build a within-group graph (nodes = the group's sites,
    edge between two sites if their real RMSD, looked up from the
    existing all-to-all alignment array, is below `threshold`). A
    group can contain more than one connected component (e.g. one
    UniProt accession's sites splitting into two unrelated near-
    duplicate clusters plus some singles); each is reduced
    independently. Within a component: repeatedly pick the node with
    the smallest mean RMSD to its *current* neighbors as the
    representative, remove it and its neighbors, and repeat until no
    edges remain. Any node left with no edges at all -- whether it had
    none from the start, or lost them all during pruning -- becomes
    its own representative too.

A single-node component (no edges) falls out of this same loop
naturally (the while-loop body never runs), so group-of-1 and isolated
members don't need special-casing.

This module is a pure, in-memory computation over the cached UniProt
mapping (data/analysis/pdb_uniprot_map.json) and the existing
tier1_all_to_all.npy alignment array. It doesn't write anything back to
the database or touch downstream figures/tables -- see
tier1_module_h_case_study.py or the earlier RCSB fetch scripts for how
to regenerate the UniProt cache if it's missing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from sqlalchemy import select

from config import config
from database import Session
from database.datamodel.models import ActiveSite

DEFAULT_THRESHOLD = 0.5
UNIPROT_CACHE_PATH = config.directory.analysis / "pdb_uniprot_map.json"


@dataclass
class RedundancyResult:
    site_to_representative: dict[int, int]
    representatives: set[int] = field(init=False)

    def __post_init__(self) -> None:
        self.representatives = set(self.site_to_representative.values())

    def constituents_of(self, representative: int) -> list[int]:
        return sorted(
            site
            for site, rep in self.site_to_representative.items()
            if rep == representative
        )


def load_uniprot_map(path=UNIPROT_CACHE_PATH) -> dict[str, dict[str, list[str]]]:
    with open(path) as f:
        return json.load(f)


def group_sites_by_uniprot(
    site_pdb_chain: dict[int, tuple[str, str]],
    uniprot_map: dict[str, dict[str, list[str]]],
) -> dict[tuple[str, ...], list[int]]:
    """Group site ids by resolved UniProt accession (as a sorted tuple,
    to handle the rare multi-accession case). A site with no resolved
    accession is put in its own singleton group keyed by its site id, so
    it still ends up as a trivial representative downstream.
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for site_id, (pdb_id, chain) in site_pdb_chain.items():
        accessions = tuple(sorted(uniprot_map.get(pdb_id, {}).get(chain, [])))
        key = accessions if accessions else ("__unresolved__", site_id)
        groups[key].append(site_id)
    return groups


def reduce_component(subgraph: nx.Graph) -> dict[int, int]:
    """Reduce one connected component to representative(s), returning a
    site_id -> representative_id map covering every node in `subgraph`.
    """
    working = subgraph.copy()
    assignment: dict[int, int] = {}

    while working.number_of_edges() > 0:
        candidates = [n for n in working.nodes if working.degree(n) > 0]
        means = {
            n: float(
                np.mean([working.edges[n, nb]["rmsd"] for nb in working.neighbors(n)])
            )
            for n in candidates
        }
        representative = min(candidates, key=lambda n: (means[n], n))
        neighbors = list(working.neighbors(representative))

        assignment[representative] = representative
        for neighbor in neighbors:
            assignment[neighbor] = representative

        working.remove_node(representative)
        working.remove_nodes_from(neighbors)

    # Anything left has no edges at all (isolated from the start, or
    # pruned down to isolation) -- each becomes its own representative.
    for node in working.nodes:
        assignment[node] = node

    return assignment


def build_rmsd_lookup(alignments: np.ndarray) -> dict[tuple[int, int], float]:
    lookup: dict[tuple[int, int], float] = {}
    for src, tgt, rmsd in alignments:
        key = (int(min(src, tgt)), int(max(src, tgt)))
        lookup[key] = float(rmsd)
    return lookup


def reduce_group(
    site_ids: list[int],
    rmsd_lookup: dict[tuple[int, int], float],
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[int, int]:
    graph = nx.Graph()
    graph.add_nodes_from(site_ids)
    for i, a in enumerate(site_ids):
        for b in site_ids[i + 1 :]:
            key = (min(a, b), max(a, b))
            rmsd = rmsd_lookup.get(key)
            if rmsd is not None and rmsd < threshold:
                graph.add_edge(a, b, rmsd=rmsd)

    assignment: dict[int, int] = {}
    for component_nodes in nx.connected_components(graph):
        component = graph.subgraph(component_nodes)
        assignment.update(reduce_component(component))
    return assignment


def compute_representatives(
    sites: list[ActiveSite],
    alignments: np.ndarray,
    uniprot_map: dict[str, dict[str, list[str]]],
    threshold: float = DEFAULT_THRESHOLD,
) -> RedundancyResult:
    site_pdb_chain = {site.id: (site.pdb_id, site.chain) for site in sites}
    groups = group_sites_by_uniprot(site_pdb_chain, uniprot_map)
    rmsd_lookup = build_rmsd_lookup(alignments)

    site_to_representative: dict[int, int] = {}
    for site_ids in groups.values():
        site_to_representative.update(reduce_group(site_ids, rmsd_lookup, threshold))

    return RedundancyResult(site_to_representative=site_to_representative)


if __name__ == "__main__":
    with Session() as session:
        all_sites = session.execute(select(ActiveSite)).scalars().all()

    alignments = np.load(config.directory.alignments / "tier1_all_to_all.npy")
    uniprot_map = load_uniprot_map()

    result = compute_representatives(all_sites, alignments, uniprot_map)

    print(f"Total sites: {len(all_sites)}")
    print(f"Representatives kept: {len(result.representatives)}")
    print(f"Sites removed (collapsed into a representative): {len(all_sites) - len(result.representatives)}")

    site_pdb_chain = {site.id: (site.pdb_id, site.chain) for site in all_sites}
    groups = group_sites_by_uniprot(site_pdb_chain, load_uniprot_map())
    multi_site_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"UniProt groups with >=2 sites: {len(multi_site_groups)}")

    multi_rep_groups = 0
    for accession, site_ids in multi_site_groups.items():
        reps_in_group = {result.site_to_representative[s] for s in site_ids}
        if len(reps_in_group) > 1:
            multi_rep_groups += 1
    print(f"Groups contributing more than one representative: {multi_rep_groups}")

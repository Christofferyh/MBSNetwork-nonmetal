"""Module H case study: the Asp/Gly/His/Ser catalytic-motif sites.

These 5 sites were identified via a broader, one-off exploratory
analysis (residue-identity signatures across every site in the Tier 1
network's largest connected component, cross-tabulated against Leiden
module assignment) that is not itself part of the permanent pipeline.
This script keeps only the durable, reproducible part: given the 5
site ids that analysis found, verify their residue signature directly
against the downloaded structure files (not a hardcoded string) and
look up each one's UniProt accession, to confirm they are genuinely
distinct proteins independently converging on the same catalytic motif
and the same geometric module.
"""

from __future__ import annotations

import asyncio
import gzip
import shutil
import tempfile

from Bio.PDB import FastMMCIFParser
from sqlalchemy import select

from config import config
from database import Session
from database.datamodel.models import ActiveSite
from preprocessing.active_site import find_chain
from preprocessing.api import get_pdb_api

# Identified by the exploratory residue-signature / module cross-tabulation
# analysis: the 5 sites in the largest component's module H whose curated
# M-CSA residues spell out ASP/GLY/HIS/SER (or its 5-residue variant).
MODULE_H_SITE_IDS = [801, 819, 467, 641, 642]


def get_residue_signature(site: ActiveSite) -> tuple[str, ...]:
    cif_path = config.directory.structures / f"{site.pdb_id}_assembly{site.assembly}.cif.gz"
    parser = FastMMCIFParser(QUIET=True)
    with tempfile.NamedTemporaryFile() as tf:
        with gzip.open(cif_path) as zf:
            shutil.copyfileobj(zf, tf)
        tf.flush()
        structure = parser.get_structure(site.pdb_id, tf.name)
    model = structure[0]
    chain = find_chain(model, site.chain)
    resnames = [chain[resnum].get_resname() for resnum in site.residues_found]
    return tuple(sorted(resnames))


async def get_uniprot(pdb_id: str, chain_name: str) -> str:
    entry = await get_pdb_api((pdb_id,), "entry")
    entity_ids = (entry or {}).get("rcsb_entry_container_identifiers", {}).get(
        "polymer_entity_ids"
    ) or []
    for entity_id in entity_ids:
        pe = await get_pdb_api((pdb_id, entity_id), "peptide")
        if pe is None:
            continue
        strand_ids = (pe.get("entity_poly", {}) or {}).get("pdbx_strand_id") or ""
        chains = [c.strip() for c in strand_ids.split(",") if c.strip()]
        if chain_name in chains:
            uniprot_ids = (
                pe.get("rcsb_polymer_entity_container_identifiers", {}) or {}
            ).get("uniprot_ids") or []
            return ",".join(uniprot_ids) if uniprot_ids else "(none)"
    return "(not found)"


async def main() -> None:
    with Session() as session:
        sites = (
            session.execute(
                select(ActiveSite).where(ActiveSite.id.in_(MODULE_H_SITE_IDS))
            )
            .scalars()
            .all()
        )
    sites_by_id = {s.id: s for s in sites}
    sites_in_order = [sites_by_id[sid] for sid in MODULE_H_SITE_IDS]

    rows = []
    for site in sites_in_order:
        signature = get_residue_signature(site)
        uniprot = await get_uniprot(site.pdb_id, site.chain)
        rows.append((site.pdb_id, site.mcsa_id, uniprot, ", ".join(signature)))

    print("| pdb_id | mcsa_id | UniProt accession | Residue signature |")
    print("|---|---|---|---|")
    for pdb_id, mcsa_id, uniprot, signature in rows:
        print(f"| {pdb_id} | {mcsa_id} | {uniprot} | {signature} |")

    distinct_uniprot = {r[2] for r in rows}
    print()
    print(
        f"{len(rows)} sites, {len(distinct_uniprot)} distinct UniProt accessions "
        f"-> {'genuinely distinct proteins' if len(distinct_uniprot) == len(rows) else 'NOT all distinct'}"
    )


if __name__ == "__main__":
    asyncio.run(main())

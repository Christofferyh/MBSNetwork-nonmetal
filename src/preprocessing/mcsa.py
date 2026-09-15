"""Tier 1 retrieval: catalytic active-site residues from M-CSA.

M-CSA (Mechanism and Catalytic Site Atlas, EBI/Thornton group) gives
residue-level annotations for each of its entries: which PDB structure, which
chain, and which residue numbers form the catalytic site. This is metadata
only, it does not include atomic 3D coordinates. The actual structure files
still need to be downloaded separately from RCSB PDB, using the existing
preprocessing.util.download_cif().

API docs: https://www.ebi.ac.uk/thornton-srv/m-csa/download/
No authentication required.

Note: M-CSA's documented query-string filters (e.g. ?entries.mcsa_ids=1,2,3)
are not reliable in practice, so this module always does a full paginated
scan of the entries endpoint rather than trying to filter server-side.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

from preprocessing.api import api_request

logger = logging.getLogger(__name__)

MCSA_ENTRIES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/"
PAGE_SIZE = 100


@dataclass
class MCSACandidateSite:
    """One Tier 1 candidate active site: a specific chain of a specific PDB
    structure, with the catalytic residue numbers M-CSA has annotated there.
    """

    mcsa_id: int
    enzyme_name: str
    ec_numbers: list[str]
    pdb_id: str
    chain: str
    assembly: int
    residue_numbers: list[int]


async def fetch_all_mcsa_entries(max_pages: int | None = None) -> list[dict]:
    """Page through the M-CSA entries endpoint and return the raw entry dicts.

    Use max_pages to limit the scan while testing (each page is ~100 entries,
    ~1000 entries total as of writing, so ~10 pages for a full run).
    """
    entries: list[dict] = []
    page = 1
    while True:
        url = f"{MCSA_ENTRIES_URL}?format=json&page={page}&page_size={PAGE_SIZE}"
        payload = await api_request(url)
        if payload is None:
            logger.error(f"M-CSA request failed on page {page}, stopping scan.")
            break

        results = payload.get("results") or []
        if not results:
            break

        entries.extend(results)
        logger.info(
            f"Fetched M-CSA page {page} ({len(results)} entries, "
            f"{len(entries)}/{payload.get('count', '?')} so far)"
        )

        if not payload.get("next") or (max_pages and page >= max_pages):
            break
        page += 1

    return entries


def extract_candidate_sites(entry: dict) -> list[MCSACandidateSite]:
    """Reconstruct one candidate site per (PDB entry, chain) referenced by
    this M-CSA entry, with the full set of catalytic residue numbers for that
    structure.

    Each catalytic residue in M-CSA can be mapped onto multiple homologous PDB
    structures: entry["residues"][i]["residue_chains"] is a list, not a single
    chain. So residues from all of an entry's catalytic positions are grouped
    by which (pdb_id, chain) they actually landed on, to reconstruct the full
    active site for each real structure.

    Important: this uses "auth_resid", not "resid". auth_resid is the
    author-assigned residue number as it appears in the actual PDB/mmCIF file
    (Biopython's residue.id numbering matches this), while "resid" is M-CSA's
    own internal/UniProt-based numbering and will not line up with the
    downloaded structure. Verified against a live example (M-CSA entry 2,
    Class A beta-lactamase): auth_resid correctly reproduces the well-known
    Ser70/Lys73/Ser130/Glu166/Lys234 catalytic site, resid does not.

    Also carries through "assembly" (an integer, e.g. 1), since
    preprocessing.util.download_cif() needs both the PDB ID and the assembly
    number, not the PDB ID alone. This assumes a given (pdb_id, chain) maps to
    a single assembly, true in every case checked so far; if a residue omits
    the field, the assembly already recorded for that structure is kept, and
    1 (RCSB's near-universal default first assembly) is used as a last resort.
    """
    sites_by_structure: dict[tuple[str, str], set[int]] = defaultdict(set)
    assembly_by_structure: dict[tuple[str, str], int] = {}

    for residue in entry.get("residues") or []:
        for chain_instance in residue.get("residue_chains") or []:
            pdb_id = chain_instance.get("pdb_id")
            chain_name = chain_instance.get("chain_name")
            auth_resid = chain_instance.get("auth_resid")
            assembly = chain_instance.get("assembly")
            if pdb_id and chain_name and auth_resid is not None:
                key = (pdb_id.lower(), chain_name)
                sites_by_structure[key].add(auth_resid)
                if assembly is not None:
                    assembly_by_structure[key] = assembly

    return [
        MCSACandidateSite(
            mcsa_id=entry["mcsa_id"],
            enzyme_name=entry.get("enzyme_name", ""),
            ec_numbers=entry.get("all_ecs") or [],
            pdb_id=pdb_id,
            chain=chain_name,
            assembly=assembly_by_structure.get((pdb_id, chain_name), 1),
            residue_numbers=sorted(residue_numbers),
        )
        for (pdb_id, chain_name), residue_numbers in sites_by_structure.items()
    ]


async def retrieve_mcsa_candidates(
    max_pages: int | None = None,
) -> list[MCSACandidateSite]:
    """Full Tier 1 retrieval: fetch M-CSA entries and flatten into one
    candidate site per (PDB structure, chain).
    """
    entries = await fetch_all_mcsa_entries(max_pages=max_pages)
    candidates: list[MCSACandidateSite] = []
    for entry in entries:
        candidates.extend(extract_candidate_sites(entry))

    logger.info(
        f"Extracted {len(candidates)} candidate sites from {len(entries)} M-CSA entries."
    )
    return candidates


if __name__ == "__main__":
    import asyncio

    async def _main() -> None:
        # Small page limit for a quick smoke test; drop max_pages for the
        # real full run.
        candidates = await retrieve_mcsa_candidates(max_pages=1)
        print(f"Retrieved {len(candidates)} candidate sites from the first page of entries.\n")
        for site in candidates[:5]:
            print(
                f"M-CSA {site.mcsa_id} ({site.enzyme_name}, EC {site.ec_numbers}): "
                f"{site.pdb_id} chain {site.chain}, "
                f"{len(site.residue_numbers)} residues {site.residue_numbers}"
            )

    asyncio.run(_main())
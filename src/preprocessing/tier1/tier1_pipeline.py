"""Tier 1 pipeline entry point: download the real 3D structures backing the
M-CSA candidate sites.

Deliberately thin: preprocessing.mcsa gives candidate sites (metadata only),
and preprocessing.util.download_cif() already handles fetching a specific PDB
assembly, with no knowledge of what kind of active site is being studied.
This module just connects the two, downloading exactly one CIF file per
unique (pdb_id, assembly) pair referenced by the candidates, even when
several chains or several M-CSA entries share the same structure.
"""

from __future__ import annotations

import asyncio
import logging

from tqdm.asyncio import tqdm_asyncio

from preprocessing.tier1.mcsa import MCSACandidateSite, retrieve_mcsa_candidates
from preprocessing.util import download_cif

logger = logging.getLogger(__name__)


async def download_structures_for_candidates(
    candidates: list[MCSACandidateSite],
) -> None:
    """Download one CIF file per unique (pdb_id, assembly) pair referenced by
    the given candidate sites. Already-downloaded files are skipped
    (download_cif() checks this itself), so this is safe to rerun.
    """
    unique_structures = {(c.pdb_id, c.assembly) for c in candidates}
    logger.info(
        f"{len(candidates)} candidate sites reference "
        f"{len(unique_structures)} unique structures."
    )

    async def download_with_progress(
        pdb_id: str, assembly: int, pbar: tqdm_asyncio
    ) -> None:
        result = await download_cif(pdb_id, str(assembly))
        if result is None:
            logger.warning(f"Could not download {pdb_id} assembly {assembly}.")
        pbar.update(1)

    with tqdm_asyncio(total=len(unique_structures)) as pbar:
        tasks = [
            download_with_progress(pdb_id, assembly, pbar)
            for pdb_id, assembly in unique_structures
        ]
        await asyncio.gather(*tasks)


if __name__ == "__main__":

    async def _main() -> None:
        # Small page limit for a quick smoke test; drop max_pages for the
        # real full run.
        candidates = await retrieve_mcsa_candidates(max_pages=1)
        await download_structures_for_candidates(candidates)
        print("Done, check data/structures/ for the downloaded .cif.gz files.")

    asyncio.run(_main())
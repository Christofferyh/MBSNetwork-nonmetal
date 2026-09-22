"""Full Tier 1 ingestion: retrieve every M-CSA entry, download structures,
localize, filter, and store. Safe to re-run: every stage (download_cif,
store_active_site) already skips work that's already done, so an
interrupted run can just be started again.
"""

from __future__ import annotations

import asyncio
import logging

from preprocessing.active_site import build_active_site_point_cloud
from preprocessing.tier1.mcsa import retrieve_mcsa_candidates
from preprocessing.tier1.tier1_filters import (
    fetch_resolution,
    passes_atom_count_filter,
    passes_resolution_filter,
)
from preprocessing.tier1.tier1_pipeline import download_structures_for_candidates
from preprocessing.tier1.tier1_store import store_active_site

logger = logging.getLogger(__name__)


async def ingest_all() -> None:
    print("Retrieving all M-CSA entries...")
    candidates = await retrieve_mcsa_candidates()  # no max_pages: the full set
    print(f"{len(candidates)} candidate sites found.")

    print("Downloading structures...")
    await download_structures_for_candidates(candidates)

    stored, skipped_localization, skipped_filters = 0, 0, 0
    for i, site in enumerate(candidates):
        cloud = build_active_site_point_cloud(site)
        if cloud is None:
            skipped_localization += 1
            continue

        resolution = await fetch_resolution(site.pdb_id)
        if not (
            passes_resolution_filter(resolution) and passes_atom_count_filter(cloud)
        ):
            skipped_filters += 1
            continue

        store_active_site(cloud, confidence_tier=1, source="mcsa")
        stored += 1

        if (i + 1) % 50 == 0:
            print(f"...{i + 1}/{len(candidates)} processed ({stored} stored)")

    print(
        f"\nDone: {stored} stored, {skipped_localization} skipped "
        f"(localization), {skipped_filters} skipped (filters)."
    )


if __name__ == "__main__":
    asyncio.run(ingest_all())
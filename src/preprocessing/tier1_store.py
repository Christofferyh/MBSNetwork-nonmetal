"""Tier 1 storage: write localized active sites to the database.

Deliberately kept separate from database/datamodel/models.py (which only
defines the schema) and preprocessing/active_site.py (which only does the
geometry): this is the one place that actually talks to the database for
Tier 1 sites, using the same database.Session the rest of the codebase uses.
"""

from __future__ import annotations

import logging

from sqlalchemy import select

from database import Session
from database.datamodel.models import ActiveSite
from preprocessing.active_site import ActiveSitePointCloud

logger = logging.getLogger(__name__)


def store_active_site(
    cloud: ActiveSitePointCloud, confidence_tier: int, source: str
) -> ActiveSite:
    """Insert one localized site into the active_site table.

    Safe to call repeatedly: if a row with the same (pdb_id, chain,
    assembly, source) already exists, matching the table's unique
    constraint, the existing row is returned instead of inserting a
    duplicate, so re-running the pipeline never creates copies.
    """
    with Session() as session:
        existing = session.execute(
            select(ActiveSite).where(
                ActiveSite.pdb_id == cloud.pdb_id,
                ActiveSite.chain == cloud.chain,
                ActiveSite.assembly == cloud.assembly,
                ActiveSite.source == source,
            )
        ).scalar_one_or_none()
        if existing is not None:
            logger.info(
                f"Active site for {cloud.pdb_id} chain {cloud.chain} "
                f"already stored, skipping."
            )
            return existing

        row = ActiveSite(
            pdb_id=cloud.pdb_id,
            chain=cloud.chain,
            assembly=cloud.assembly,
            confidence_tier=confidence_tier,
            source=source,
            mcsa_id=cloud.mcsa_id,
            ec_numbers=cloud.ec_numbers,
            residues_found=cloud.residues_found,
            residues_missing=cloud.residues_missing,
            # .tolist() matters here, not just style: these are numpy arrays
            # from active_site.py's point-cloud math, and psycopg's ARRAY
            # binding expects plain Python lists, not numpy ndarrays.
            x_coords=cloud.x_coords.tolist(),
            y_coords=cloud.y_coords.tolist(),
            z_coords=cloud.z_coords.tolist(),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row


if __name__ == "__main__":
    import asyncio

    from preprocessing.active_site import build_active_site_point_cloud
    from preprocessing.mcsa import retrieve_mcsa_candidates

    async def _main() -> None:
        candidates = await retrieve_mcsa_candidates(max_pages=1)
        stored = 0
        skipped = 0
        for site in candidates[:10]:
            cloud = build_active_site_point_cloud(site)
            if cloud is None:
                skipped += 1
                continue
            row = store_active_site(cloud, confidence_tier=1, source="mcsa")
            print(f"Stored active_site id={row.id} ({row.pdb_id} chain {row.chain})")
            stored += 1
        print(f"\nDone: {stored} stored, {skipped} skipped (failed localization).")

    asyncio.run(_main())
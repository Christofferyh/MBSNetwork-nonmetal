"""Tier 1 quality filters: the same two checks the existing metal pipeline
applies in preprocessing/dataset.py's process_entry() (resolution, minimum
atom count), adapted for Tier 1 sites.

Tier 1 has no Entry table with resolution already stored, that table is
only populated by the existing metal-specific retrieval pipeline, which
Tier 1 bypasses entirely. Resolution is instead fetched directly per PDB ID
from RCSB's Data API, the same field (rcsb_entry_info.resolution_combined)
the metal pipeline already reads during its own retrieval, just via a
single-entry lookup endpoint suited to a PDB ID already in hand, rather
than the bulk search API used when discovering new entries.
"""

from __future__ import annotations

import logging
import statistics
from typing import Iterable

from config import config
from preprocessing.active_site import ActiveSitePointCloud
from preprocessing.api import api_request

logger = logging.getLogger(__name__)

RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/"


async def fetch_resolution(pdb_id: str) -> float | None:
    """Fetch a PDB entry's combined resolution from RCSB's Data API.

    Returns None if the entry has no reported resolution, or the lookup
    fails, treated conservatively as "does not pass" by
    passes_resolution_filter() below, rather than silently let through.
    """
    payload = await api_request(f"{RCSB_ENTRY_URL}{pdb_id.lower()}")
    if payload is None:
        return None

    resolution = payload.get("rcsb_entry_info", {}).get("resolution_combined")
    if not resolution:
        return None
    if isinstance(resolution, Iterable) and not isinstance(resolution, (str, bytes)):
        return statistics.mean(resolution)
    return resolution


def passes_resolution_filter(
    resolution: float | None, max_resolution: float | None = None
) -> bool:
    """Mirrors the metal pipeline's `entry.resolution > config.resolution`
    check: lower resolution values mean sharper crystallography, so this
    passes only when the reported resolution is at or below the threshold.
    A missing resolution (None) fails conservatively, it is not assumed
    acceptable just because it is unknown.
    """
    threshold = max_resolution if max_resolution is not None else config.dataset.resolution
    return resolution is not None and resolution <= threshold


def passes_atom_count_filter(
    cloud: ActiveSitePointCloud, min_atoms: int | None = None
) -> bool:
    """Mirrors the metal pipeline's `n_atoms < config.atoms` check, but
    applied to the finished point cloud rather than the raw structure.

    A cloud can come back smaller than requested when active_site.py falls
    back to "use fewer atoms" for an unusually small structure. Rather than
    silently keep such a site, it is excluded here, since the alignment
    step assumes a consistent point-cloud size across every site being
    compared, a smaller cloud would be an unfair comparison, not a
    legitimate small site.
    """
    threshold = min_atoms if min_atoms is not None else config.dataset.atoms
    return len(cloud.x_coords) >= threshold
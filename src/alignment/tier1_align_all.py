"""Full Tier 1 all-to-all alignment: every pairwise RMSD among every
currently-stored ActiveSite. Meant to run as a Slurm batch job, not
interactively, given the runtime at full scale (~8 hours measured).

Saves the raw (source_id, target_id, rmsd) array to disk, the same shape
alignment_files elsewhere in this codebase expect, so downstream network
and threshold analysis can load it directly.
"""

from __future__ import annotations

import time

import numpy as np
from sqlalchemy import select

from alignment.alignment import PairwiseAligner
from config import AlignmentConfig, config
from database import Session
from database.datamodel.models import ActiveSite

if __name__ == "__main__":
    with Session() as session:
        all_sites = session.execute(select(ActiveSite)).scalars().all()
    print(f"Aligning all {len(all_sites)} stored Tier 1 sites "
          f"({len(all_sites) * (len(all_sites) - 1) // 2} pairs)...")

    start = time.time()
    aligner = PairwiseAligner(all_sites, AlignmentConfig())
    alignments = aligner.align()
    elapsed = time.time() - start
    print(f"Done in {elapsed / 3600:.2f} hours.")

    out_path = config.directory.alignments / "tier1_all_to_all.npy"
    np.save(out_path, alignments)
    print(f"Saved {len(alignments)} alignments to {out_path}")
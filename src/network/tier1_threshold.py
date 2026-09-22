from __future__ import annotations

import numpy as np
from alignment.alignment import PairwiseAligner
from config import AlignmentConfig, config
from database import Session
from database.datamodel.models import ActiveSite
from network.threshold import compute_bimodal_overlap_gmm
from sqlalchemy import select


def bimodal_overlap(ranges: list[tuple[float, float]], n_pairs: int) -> np.ndarray:
    """Compute mode overlap between modes of bimodal RMSD distributions for a given
    number of ActiveSite point cloud pairs within specified ranges.

    Adapted from network.threshold.bimodal_overlap for the Tier 1 dataset, whose
    RMSD distribution is sparse and near-empty below ~1.1 (peak is at 1.3-1.4,
    strongly unimodal) rather than bimodal across 0.0-1.2 like the metal data
    that function was designed around. Two changes from the original:

    - `ranges[0]` is expected to be one wide pooled bin covering the sparse
      low-RMSD region (there are nowhere near `n_pairs` real pairs available
      there), rather than ten narrow, near-empty 0.1-wide bins. Every range
      still only samples as many pairs as actually exist in it; `n_pairs` is
      a cap, not a guarantee.
    - Each range's real sampled count is tracked explicitly (`range_counts`)
      and used to slice the re-aligned RMSD array, instead of assuming every
      range filled to exactly `n_pairs` and slicing by `i * n_pairs`. The
      original's fixed-stride slicing silently misattributes RMSD values
      across ranges the moment any range comes up short, which the pooled
      low-RMSD range does immediately.

    Additionally, before sampling into the pooled low-RMSD range (`ranges[0]`),
    candidate pairs are checked for a specific artifact: two ActiveSite rows
    from the same PDB entry catalogued under different M-CSA ids. Those land
    at near-zero RMSD because they're (near-)literally the same structure, not
    because they're a genuine similar-active-site match, so they're excluded
    from that range's sample and reported rather than silently counted.
    """
    alignments_path = config.directory.alignments / "tier1_all_to_all.npy"
    alignments = np.load(alignments_path)

    # Preload (pdb_id, mcsa_id) for every stored site so candidates for the
    # pooled low-RMSD range can be checked for the same-PDB-different-M-CSA
    # artifact. Only 3 columns, not the full ActiveSite row with coordinates.
    with Session() as session:
        site_meta = {
            row.id: (row.pdb_id, row.mcsa_id)
            for row in session.execute(
                select(ActiveSite.id, ActiveSite.pdb_id, ActiveSite.mcsa_id)
            )
        }

    pooled_low, pooled_high = ranges[0]
    flagged_duplicates: list[tuple[int, int, str, int | None, int | None]] = []

    pairs_by_range: dict[float, list[tuple[int, int]]] = {low: [] for low, _ in ranges}
    all_site_ids = set()

    # Iterate over alignments until we have n_pairs for every range.
    indices = np.arange(len(alignments))
    np.random.shuffle(indices)

    for i in indices:
        source_id, target_id, rmsd = alignments[i]
        if all(len(pairs_by_range[low]) >= n_pairs for low, _ in ranges):
            break

        for low, high in ranges:
            if not (low <= rmsd < high) or len(pairs_by_range[low]) >= n_pairs:
                continue

            if (low, high) == (pooled_low, pooled_high):
                src_pdb, src_mcsa = site_meta[source_id]
                tgt_pdb, tgt_mcsa = site_meta[target_id]
                if src_pdb == tgt_pdb and src_mcsa != tgt_mcsa:
                    flagged_duplicates.append(
                        (source_id, target_id, src_pdb, src_mcsa, tgt_mcsa)
                    )
                    break

            pairs_by_range[low].append((source_id, target_id))
            all_site_ids.update([source_id, target_id])
            break

    if flagged_duplicates:
        print(
            f"Excluded {len(flagged_duplicates)} pair(s) from range "
            f"({pooled_low}, {pooled_high}): same PDB entry under different "
            "M-CSA ids, not a genuine similar-site signal."
        )
        for src_id, tgt_id, pdb_id, src_mcsa, tgt_mcsa in flagged_duplicates:
            print(
                f"  site {src_id} (mcsa {src_mcsa}) vs site {tgt_id} "
                f"(mcsa {tgt_mcsa}), both pdb_id={pdb_id}"
            )

    # Deduplicate and index ActiveSite IDs.
    ordered_site_ids = sorted(all_site_ids)
    site_id_to_idx = {site_id: i for i, site_id in enumerate(ordered_site_ids)}

    # Re-encode all pairs as index pairs into ordered_site_ids, tracking each
    # range's real pair count explicitly since ranges are not guaranteed to
    # fill to n_pairs (see docstring).
    ordered_pairs = []
    range_counts = []
    for low, _ in ranges:
        pairs = pairs_by_range[low]
        range_counts.append(len(pairs))
        for src_id, tgt_id in pairs:
            src_idx = site_id_to_idx[src_id]
            tgt_idx = site_id_to_idx[tgt_id]
            ordered_pairs.append((src_idx, tgt_idx))

    alignments = compute_rmsd_distribution(ordered_site_ids, ordered_pairs)

    # Compute GMM overlap per range, slicing by each range's real count
    # rather than assuming every range filled to n_pairs.
    mode_overlaps = np.zeros(len(ranges))
    start = 0
    for i, count in enumerate(range_counts):
        end = start + count
        rmsd_values = alignments[start:end, 2]
        mode_overlaps[i] = compute_bimodal_overlap_gmm(rmsd_values)
        start = end

    path = config.directory.analysis / "tier1_bimodal_overlap.npy"
    with path.open("wb") as f:
        np.save(f, mode_overlaps)

    return mode_overlaps


def compute_rmsd_distribution(
    site_ids: list[int], pairs: list[tuple[int, int]]
) -> np.ndarray:
    """Calculate RMSD distributions for pairs of ActiveSite point clouds."""
    with Session() as session:
        sites = (
            session.execute(select(ActiveSite).where(ActiveSite.id.in_(site_ids)))
            .scalars()
            .all()
        )

    # Align all pairs of ActiveSite point clouds. A fresh AlignmentConfig()
    # is used here rather than the global config.alignment, which may already
    # carry manually-attached Open3D objects: PairwiseAligner builds those
    # itself, independently, inside each worker process it spawns, and a
    # config that already has them attached fails to pickle when handed to
    # a worker (see network.tier1_network.build_tier1_network).
    site_dict = {site.id: site for site in sites}
    sorted_sites = [site_dict[site_id] for site_id in site_ids]
    aligner = PairwiseAligner(sorted_sites, AlignmentConfig(), pair_indices=pairs)
    alignments = aligner.align()

    return alignments


if __name__ == "__main__":
    ranges = [
        (0.0, 1.1),  # pooled sparse tail: ~400 pairs total, capped not fixed
        (1.1, 1.2),
        (1.2, 1.3),
        (1.3, 1.4),
        (1.4, 1.5),
        (1.5, 1.6),
        (1.6, 1.7),
        (1.7, 1.8),
        (1.8, 1.9),
        (1.9, 2.0),
    ]
    n_pairs = 1000
    gmm_overlap_areas = bimodal_overlap(ranges, n_pairs)

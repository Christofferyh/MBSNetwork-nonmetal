from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from alignment.alignment import PairwiseAligner
from config import AlignmentConfig, config
from database import Session
from database.datamodel.models import ActiveSite
from network.threshold import compute_bimodal_overlap_gmm
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
from sqlalchemy import select

# Fixed top-level seeds, one full analysis per seed, so the reported overlaps
# carry their real run-to-run spread. Each seed drives a single generator that
# both shuffles the pair sample and draws one seed per re-alignment task, so
# results don't depend on N_JOBS or how pairs are split across workers.
SEEDS = (1, 2, 3, 4, 5)

# A 2-component fit is only accepted if BIC prefers it over 1 component and
# each component holds at least this fraction of the bin, so a handful of
# outliers can't be handed a component of their own.
MIN_COMPONENT_WEIGHT = 0.05

# Ashman's D above which an accepted 2-component fit counts as two separated
# populations rather than one skewed or heavy-tailed peak (Ashman, Bird &
# Zepf 1994, AJ 108:2348). BIC alone also prefers 2 components for peaks that
# are merely non-Gaussian, e.g. two components with near-identical means.
MIN_SEPARATION = 2.0


@dataclass
class BinFit:
    """Mixture-model summary of one RMSD bin's re-aligned values. The
    2-component parameters are sorted by mean and kept for inspection even
    when that fit is rejected."""

    n_components: int
    delta_bic: float  # BIC(2 components) - BIC(1 component); negative favours 2
    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray
    ashman_d: float
    overlap: float  # GMM mode overlap (delta); 1.0 for a unimodal bin

    @property
    def separated(self) -> bool:
        return self.n_components == 2 and self.ashman_d > MIN_SEPARATION


def fit_bin(rmsd_values: np.ndarray) -> BinFit:
    x = rmsd_values.reshape(-1, 1)
    one = GaussianMixture(n_components=1, covariance_type="full", random_state=42).fit(x)
    # Same fit as compute_bimodal_overlap_gmm, which is reused for delta below.
    two = GaussianMixture(n_components=2, covariance_type="full", random_state=42).fit(x)

    order = np.argsort(two.means_.ravel())
    weights = two.weights_[order]
    means = two.means_.ravel()[order]
    sds = np.sqrt(two.covariances_.ravel()[order])
    ashman_d = float(np.sqrt(2) * (means[1] - means[0]) / np.sqrt(sds[0] ** 2 + sds[1] ** 2))
    delta_bic = float(two.bic(x) - one.bic(x))

    n_components = 2 if delta_bic < 0 and weights.min() >= MIN_COMPONENT_WEIGHT else 1
    overlap = compute_bimodal_overlap_gmm(rmsd_values) if n_components == 2 else 1.0
    return BinFit(n_components, delta_bic, weights, means, sds, ashman_d, overlap)


def select_threshold(ranges: list[tuple[float, float]], fits: list[BinFit]) -> float | None:
    """Threshold-selection rule: scanning bins upward from the lowest, the
    transition is the first bin that is not two separated populations; the
    threshold is the upper edge of the bin preceding it. None if even the
    lowest bin isn't separated."""
    threshold = None
    for (_, high), fit in zip(ranges, fits):
        if not fit.separated:
            break
        threshold = high
    return threshold


def bimodal_overlap(
    ranges: list[tuple[float, float]],
    n_pairs: int,
    seed: int,
    alignments_file: str = "tier1_all_to_all.npy",
    out_file: str = "tier1_bimodal_overlap.npz",
) -> list[BinFit]:
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
    alignments_path = config.directory.alignments / alignments_file
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

    pairs_by_range: dict[float, list[tuple[int, int, float]]] = {low: [] for low, _ in ranges}
    all_site_ids = set()

    # Iterate over alignments until we have n_pairs for every range.
    rng = np.random.default_rng(seed)
    indices = np.arange(len(alignments))
    rng.shuffle(indices)

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

            pairs_by_range[low].append((source_id, target_id, rmsd))
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
    original_rmsds = []
    range_counts = []
    for low, _ in ranges:
        pairs = pairs_by_range[low]
        range_counts.append(len(pairs))
        for src_id, tgt_id, rmsd in pairs:
            src_idx = site_id_to_idx[src_id]
            tgt_idx = site_id_to_idx[tgt_id]
            ordered_pairs.append((src_idx, tgt_idx))
            original_rmsds.append(rmsd)

    # One seed per re-alignment task, drawn from the same seeded generator.
    pair_seeds = rng.integers(0, 2**32, size=len(ordered_pairs), dtype=np.uint32).tolist()
    alignments = compute_rmsd_distribution(ordered_site_ids, ordered_pairs, pair_seeds)

    # Fit each range, slicing by its real count rather than assuming every
    # range filled to n_pairs.
    fits = []
    start = 0
    for count in range_counts:
        end = start + count
        fits.append(fit_bin(alignments[start:end, 2]))
        start = end

    for (low, high), count, fit in zip(ranges, range_counts, fits):
        print(
            f"  ({low}, {high}): {count} pairs, {fit.n_components} component(s), "
            f"dBIC {fit.delta_bic:+.1f}, min weight {fit.weights.min():.3f}, "
            f"D {fit.ashman_d:.2f}, overlap {fit.overlap:.4f}, separated {fit.separated}"
        )

    np.savez(
        config.directory.analysis / out_file,
        source_id=alignments[:, 0],
        target_id=alignments[:, 1],
        original_rmsd=np.array(original_rmsds),
        realigned_rmsd=alignments[:, 2],
        bin_index=np.repeat(np.arange(len(ranges)), range_counts),
        overlaps=np.array([f.overlap for f in fits]),
        n_components=np.array([f.n_components for f in fits]),
        delta_bic=np.array([f.delta_bic for f in fits]),
        weights=np.array([f.weights for f in fits]),
        means=np.array([f.means for f in fits]),
        sds=np.array([f.sds for f in fits]),
        ashman_d=np.array([f.ashman_d for f in fits]),
        separated=np.array([f.separated for f in fits]),
    )

    return fits


def compute_rmsd_distribution(
    site_ids: list[int], pairs: list[tuple[int, int]], pair_seeds: list[int]
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
    aligner = PairwiseAligner(
        sorted_sites, AlignmentConfig(), pair_indices=pairs, pair_seeds=pair_seeds
    )
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
    per_seed = []
    thresholds = []
    for seed in SEEDS:
        print(f"Seed {seed}:")
        fits = bimodal_overlap(
            ranges,
            n_pairs,
            seed,
            alignments_file="tier1_all_to_all_redundancy_reduced.npy",
            out_file=f"tier1_bimodal_overlap_redundancy_reduced_seed{seed}.npz",
        )
        per_seed.append(fits)
        thresholds.append(select_threshold(ranges, fits))
        print(f"  selected threshold: {thresholds[-1]}")

    # Stacked (n_seeds, n_ranges) summary, read by network.tier1_threshold_figure.
    np.savez(
        config.directory.analysis / "tier1_bimodal_overlap_redundancy_reduced_seeds.npz",
        seeds=np.array(SEEDS),
        overlaps=np.array([[f.overlap for f in fits] for fits in per_seed]),
        n_components=np.array([[f.n_components for f in fits] for fits in per_seed]),
        ashman_d=np.array([[f.ashman_d for f in fits] for fits in per_seed]),
        separated=np.array([[f.separated for f in fits] for fits in per_seed]),
        thresholds=np.array([np.nan if t is None else t for t in thresholds]),
    )

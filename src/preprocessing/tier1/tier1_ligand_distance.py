"""Tier 1 ligand support: how close a real, non-additive ligand sits to
each ActiveSite's residue-centroid anchor.

Distances are always ligand-centroid-to-anchor-centroid (mean atom position
of the ligand residue vs. the same residue-centroid anchor already computed
in preprocessing.active_site.get_active_site_centroid), same style as the
anchor itself, never a nearest-single-atom measurement.

Candidate HETATM residues are restricted to the anchor's own chain plus its
symmetry-suffixed copies (the same tolerant "-N" suffix matching find_chain()
already uses in active_site.py), not searched across the whole model. An
earlier unscoped version (searching every chain in the model) let a ligand
bound to a *different* subunit get credited to this site's own active site
whenever it happened to be geometrically closer than anything actually in
this site's own chain -- a real problem for homo-oligomers with
heterogeneous per-subunit occupancy, since it can only ever make the
reported distance look smaller than it should (more candidates to search
over can never increase a minimum), silently contaminating the near end of
the distribution. Scoping to the anchor's own chain removes that.

Two exclusion lists are applied on top of water (HOH), neither of which is
"real vs. not a ligand" itself, just clearly non-ligand contamination:
  - CRYSTALLIZATION_ADDITIVES: near-universal buffer/cryoprotectant/counter-
    ion chemistry (sulfate, glycerol, PEG fragments, halide soak ions, common
    buffers, etc.), essentially never catalytically relevant. Deliberately
    NOT included: CIT, SIN, MLI, GLY, BEZ and similar -- all also genuine
    substrates/products/inhibitors for real enzyme families, so excluding
    them would be exactly the premature "real vs. not" judgment call this
    module avoids making.
  - MODIFIED_RESIDUE_ARTIFACTS: currently just MSE (selenomethionine), a
    normal Met residue substituted for X-ray phasing, covalently part of the
    protein backbone, not a ligand at all. Deliberately NOT extended to other
    modified-amino-acid HETATM codes (PTR, SEP, TPO, LLP, TPQ, ...): several
    of those (e.g. LLP = PLP-lysine Schiff base, TPQ = topaquinone) are
    themselves the catalytic cofactor, so a blanket "is it a modified amino
    acid" filter would hide real signal instead of removing noise.

KNOWN_COFACTORS is a fixed list of common, unambiguous catalytic
cofactors/metals, used only to split the reported distribution for
inspection, not to decide qualification.

QUALIFYING_DISTANCE_THRESHOLD (7 A) is the project brief's own suggested
calibration radius, already consistent with the N=65/7 A point-cloud
calibration used elsewhere: a site is "ligand-supported" if it has a
qualifying own-chain ligand within this distance of its anchor.
"""

from __future__ import annotations

import gzip
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
from Bio.PDB import FastMMCIFParser
from Bio.PDB.Model import Model

from config import config
from database.datamodel.models import ActiveSite
from preprocessing.active_site import get_active_site_centroid

CRYSTALLIZATION_ADDITIVES = {
    "SO4", "PO4", "GOL", "EDO", "ACT", "PEG",
    "FMT", "TRS", "BME", "DTT", "DMS", "MPD", "1PE", "PGE", "PG4",
    "IPA", "MRD", "IOD", "BR", "EPE", "MES",
    "NA", "K", "CL",
}
MODIFIED_RESIDUE_ARTIFACTS = {"MSE"}
EXCLUDED_CODES = CRYSTALLIZATION_ADDITIVES | MODIFIED_RESIDUE_ARTIFACTS | {"HOH"}

KNOWN_COFACTORS = {
    "NAD", "NAP", "FAD", "FMN", "PLP", "HEM", "ATP", "ADP", "TPP",
    "SF4", "FES", "MG", "ZN", "CA", "MN", "NI", "FE",
}

QUALIFYING_DISTANCE_THRESHOLD = 7.0


@dataclass
class LigandDistanceResult:
    site_id: int
    pdb_id: str
    status: str  # "ok", "no_hetatm", "no_qualifying_own_chain", "anchor_failed"
    min_distance: float | None
    closest_code: str | None
    group: str | None  # "Known cofactor" or "Other", only set when status == "ok"


def load_model(pdb_id: str, assembly: int) -> Model:
    """Parse a downloaded assembly CIF file's first model, heteroatoms
    included (unlike active_site.py's load_structure_from_file, which
    strips them for point-cloud extraction)."""
    cif_path = config.directory.structures / f"{pdb_id}_assembly{assembly}.cif.gz"
    parser = FastMMCIFParser(QUIET=True)
    with tempfile.NamedTemporaryFile() as tf:
        with gzip.open(cif_path) as zf:
            shutil.copyfileobj(zf, tf)
        tf.flush()
        structure = parser.get_structure(pdb_id, tf.name)
    return structure[0]


def matching_chain_ids(model: Model, chain_name: str) -> set[str]:
    """Every chain belonging to the same protein copy as chain_name: an
    exact match, plus any chain whose ID matches once a "-N" suffix is
    stripped (RCSB's symmetry-generated duplicate chains in assembly
    files). Mirrors find_chain()'s tolerance in active_site.py, but
    collects every matching chain instead of just the first."""
    return {
        chain.id
        for chain in model
        if chain.id == chain_name or chain.id.split("-")[0] == chain_name
    }


def compute_ligand_distance(site: ActiveSite, model: Model | None = None) -> LigandDistanceResult:
    """The closest qualifying (non-water, non-additive, own-chain) ligand
    to `site`'s residue-centroid anchor, as a centroid-to-centroid distance.

    `model` can be passed in to reuse an already-parsed structure across
    multiple ActiveSite rows sharing the same (pdb_id, assembly) file.
    """
    if model is None:
        model = load_model(site.pdb_id, site.assembly)

    anchor, _found, _missing = get_active_site_centroid(
        model, site.chain, site.residues_found
    )
    if anchor is None:
        return LigandDistanceResult(site.id, site.pdb_id, "anchor_failed", None, None, None)

    own_chain_ids = matching_chain_ids(model, site.chain)
    het_residues = [r for r in model.get_residues() if r.id[0] != " "]

    saw_any_hetatm = False
    candidates: list[tuple[float, str]] = []
    for res in het_residues:
        resname = res.get_resname()
        if resname == "HOH":
            continue
        saw_any_hetatm = True
        if resname in EXCLUDED_CODES or res.get_parent().id not in own_chain_ids:
            continue
        atom_coords = [a.get_coord() for a in res if a.element != "H"]
        if not atom_coords:
            continue
        centroid = np.mean(atom_coords, axis=0)
        candidates.append((float(np.linalg.norm(centroid - anchor)), resname))

    if not saw_any_hetatm:
        return LigandDistanceResult(site.id, site.pdb_id, "no_hetatm", None, None, None)
    if not candidates:
        return LigandDistanceResult(
            site.id, site.pdb_id, "no_qualifying_own_chain", None, None, None
        )

    min_distance, closest_code = min(candidates, key=lambda c: c[0])
    group = "Known cofactor" if closest_code in KNOWN_COFACTORS else "Other"
    return LigandDistanceResult(site.id, site.pdb_id, "ok", min_distance, closest_code, group)


def compute_ligand_distances(sites: list[ActiveSite]) -> list[LigandDistanceResult]:
    """compute_ligand_distance() for every site, caching parsed structures
    by (pdb_id, assembly) since multiple sites can share the same file."""
    model_cache: dict[tuple[str, int], Model] = {}
    results = []
    for site in sites:
        key = (site.pdb_id, site.assembly)
        if key not in model_cache:
            model_cache[key] = load_model(site.pdb_id, site.assembly)
        results.append(compute_ligand_distance(site, model_cache[key]))
    return results


if __name__ == "__main__":
    from collections import Counter

    from sqlalchemy import select

    from database import Session

    with Session() as session:
        all_sites = session.execute(select(ActiveSite)).scalars().all()

    print(f"Computing ligand distances for {len(all_sites)} Tier 1 sites...")
    results = compute_ligand_distances(all_sites)

    status_counts = Counter(r.status for r in results)
    print(f"Status breakdown: {dict(status_counts)}")

    ok_results = [r for r in results if r.status == "ok"]
    within = [r for r in ok_results if r.min_distance <= QUALIFYING_DISTANCE_THRESHOLD]
    cofactor_within = [r for r in within if r.group == "Known cofactor"]
    other_within = [r for r in within if r.group == "Other"]

    print()
    print(
        f"Ligand-supported sites (own-chain, non-additive ligand within "
        f"{QUALIFYING_DISTANCE_THRESHOLD} A): {len(within)} of {len(all_sites)} "
        f"({100 * len(within) / len(all_sites):.1f}%)"
    )
    print(f"  Known cofactor: {len(cofactor_within)}")
    print(f"  Other:          {len(other_within)}")

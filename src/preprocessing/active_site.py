"""Tier 1 localization: build a comparable point cloud for each M-CSA
candidate site, without any metal atom to anchor on.

Mirrors the existing structure.py pipeline (build_structure -> get_mbs), but:
  - loads a structure directly from a PDB ID + assembly number, since there
    is no database Assembly object yet for Tier 1 sites (that's Step 5's job);
  - finds the anchor point by averaging the coordinates of M-CSA's known
    catalytic residues, instead of finding a single ligand/metal atom.

The actual nearest-neighbor extraction reuses the same math get_mbs() already
uses in structure.py, just applied to a different kind of anchor point.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
from Bio.PDB import FastMMCIFParser, Structure
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from config import config
from preprocessing.mcsa import MCSACandidateSite
from preprocessing.structure import (
    get_atom_coordinates,
    get_non_protein_chains,
    remove_residues,
)

logger = logging.getLogger(__name__)


@dataclass
class ActiveSitePointCloud:
    """A Tier 1 site's point cloud, in the same shape MBSPointCloud.from_mbs()
    already knows how to read (x_coords/y_coords/z_coords), plus the metadata
    needed to eventually store it in Step 5.
    """

    mcsa_id: int
    pdb_id: str
    chain: str
    assembly: int
    ec_numbers: list[str]
    residues_found: list[int]
    residues_missing: list[int]
    x_coords: np.ndarray
    y_coords: np.ndarray
    z_coords: np.ndarray


def load_structure_from_file(pdb_id: str, assembly: int) -> Structure.Structure:
    """Parse a downloaded assembly CIF file directly, with no database
    Assembly object required.

    Strips non-protein chains (DNA/RNA) and every heteroatom (waters, buffer
    molecules, crystallization additives): Tier 1 sites are defined purely by
    amino acid residues, not a bound ligand, so there is no equivalent to
    build_structure()'s "keep the metal ligand" exception.
    """
    cif_file = (
        config.directory.structures / f"{pdb_id.lower()}_assembly{assembly}.cif.gz"
    )
    parser = FastMMCIFParser(QUIET=True)

    with tempfile.NamedTemporaryFile() as temp_file:
        with gzip.open(cif_file) as zip_file:
            shutil.copyfileobj(zip_file, temp_file)  # type: ignore[misc]
        temp_file.flush()
        structure = parser.get_structure(pdb_id, temp_file.name)
        mmcif_dict = MMCIF2Dict(temp_file.name)

    non_protein_chains = get_non_protein_chains(mmcif_dict)

    residues_to_remove = set()
    for residue in structure.get_residues():
        chain = residue.get_parent()
        is_het = residue.id[0] != " "
        if chain.id in non_protein_chains or is_het:
            residues_to_remove.add(residue.id)

    remove_residues(structure, residues_to_remove)
    return structure


def find_chain(model, chain_name: str):
    """Find a chain matching chain_name, tolerating the numeric suffixes RCSB
    sometimes adds to symmetry-generated copies in assembly files (e.g. a
    homodimer's second copy of chain "A" appearing as "A-2"). Prefers an
    exact match; otherwise falls back to the first chain whose ID matches
    once any "-N" suffix is stripped.

    This mirrors an existing, documented pattern already used elsewhere in
    this codebase for the same reason: see get_mbs()'s dominant_chain_id
    handling in structure.py, and its comment on repeated chain IDs with
    numerical suffixes.
    """
    if chain_name in model:
        return model[chain_name]
    for chain in model:
        if chain.id.split("-")[0] == chain_name:
            return chain
    return None


def get_active_site_centroid(
    model, chain_name: str, residue_numbers: list[int]
) -> tuple[np.ndarray | None, list[int], list[int]]:
    """Find the given residues in the given chain and return the centroid of
    their atoms, plus which residue numbers were actually found vs missing.

    Real structures don't always have every residue resolved (disordered
    loops, missing density), so this is expected to sometimes only find a
    subset. Returns (None, [], residue_numbers) if none were found at all,
    signalling the caller to skip this candidate entirely.
    """
    chain = find_chain(model, chain_name)
    if chain is None:
        logger.warning(f"Chain {chain_name} not found in structure.")
        return None, [], list(residue_numbers)

    found: list[int] = []
    missing: list[int] = []
    atom_coords: list[np.ndarray] = []

    for resnum in residue_numbers:
        try:
            residue = chain[resnum]
        except KeyError:
            missing.append(resnum)
            continue
        found.append(resnum)
        atom_coords.extend(
            atom.get_coord() for atom in residue if atom.element != "H"
        )

    if not atom_coords:
        return None, found, missing

    centroid = np.mean(atom_coords, axis=0)
    return centroid, found, missing


def build_active_site_point_cloud(
    site: MCSACandidateSite, n_atoms: int = 65
) -> ActiveSitePointCloud | None:
    """Full Tier 1 localization for one candidate site: load its structure,
    find the anchor (residue centroid), and extract the N nearest atoms
    around it, using the same nearest-neighbor math get_mbs() already uses.

    Returns None if the structure can't be loaded, the chain can't be found,
    or none of the annotated residues exist in the resolved structure.
    """
    try:
        structure = load_structure_from_file(site.pdb_id, site.assembly)
    except Exception as e:
        # Real-world CIF files, especially assembly files with
        # symmetry-generated duplicate chains, sometimes fail to parse.
        # This is a known, documented Biopython limitation (not unique to
        # this codebase), and the existing metal pipeline already handles
        # it the same way in preprocessing/dataset.py's process_entry():
        # log it and skip this one site, rather than let one bad structure
        # crash a run of many.
        logger.error(f"Could not load structure for {site.pdb_id}: {e}")
        return None

    model = structure[0]
    centroid, found, missing = get_active_site_centroid(
        model, site.chain, site.residue_numbers
    )
    if centroid is None:
        logger.warning(
            f"M-CSA {site.mcsa_id} ({site.pdb_id} chain {site.chain}): "
            f"none of the annotated residues {site.residue_numbers} were found."
        )
        return None

    if missing:
        logger.info(
            f"M-CSA {site.mcsa_id} ({site.pdb_id} chain {site.chain}): "
            f"residues {missing} not found in the resolved structure, "
            f"using the remaining {found}."
        )

    points, _ = get_atom_coordinates(model)
    if len(points) < n_atoms:
        logger.warning(
            f"M-CSA {site.mcsa_id} ({site.pdb_id}): only {len(points)} atoms "
            f"available, fewer than the requested {n_atoms}."
        )
        n_atoms = len(points)

    distances = np.linalg.norm(points - centroid, axis=1)
    indices = np.argpartition(distances, n_atoms)[:n_atoms]
    site_points = points[indices]

    return ActiveSitePointCloud(
        mcsa_id=site.mcsa_id,
        pdb_id=site.pdb_id,
        chain=site.chain,
        assembly=site.assembly,
        ec_numbers=site.ec_numbers,
        residues_found=found,
        residues_missing=missing,
        x_coords=site_points[:, 0],
        y_coords=site_points[:, 1],
        z_coords=site_points[:, 2],
    )
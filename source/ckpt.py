####################################################################################################

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import ase
import ase.io

####################################################################################################

def format_log_preamble(ns: argparse.Namespace) -> str:
    """Format argparse Namespace into a comment preamble for log files."""
    lines = ["# Parameters"]
    for key in sorted(vars(ns)):
        val = getattr(ns, key)
        lines.append(f"# {key} = {val}")
    return "\n".join(lines) + "\n"

####################################################################################################

def load_structure_from_json(json_path: str | Path) -> Tuple[ase.Atoms, Dict[str, Any]]:
    """
    Load a structure and related arrays from a JSON file into ASE and NumPy objects.

    Args:
        json_path (str | Path): Path to the JSON file containing structure data.

    Returns:
        Tuple[ase.Atoms, Dict[str, Any]]:
            - atoms: ASE Atoms object with atomic symbols, positions, cell, and PBC.
            - info: Dictionary containing:
                {
                    "supercell_size": np.ndarray,
                    "density": float,
                    "O_neighbors": np.ndarray,
                    "H_to_OO_pairs": np.ndarray,
                    "state_hydrogens": np.ndarray,
                }

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        KeyError: If required keys are missing in the JSON data.
        ValueError: If array shapes or data types are inconsistent.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    required_keys = [
        "atoms", 
        "supercell_size", 
        "density", 
        "O_neighbors", 
        "H_to_OO_pairs",
        "state_hydrogens",
        "atomcoords_O",
        "H2_candidates",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in JSON file {json_path}")

    atoms_info = data["atoms"]
    try:
        atoms = ase.Atoms(
            symbols=atoms_info["symbols"],
            positions=np.array(atoms_info["positions"], dtype=float),
            cell=np.array(atoms_info["cell"], dtype=float),
            pbc=np.array(atoms_info["pbc"], dtype=bool),
        )
    except Exception as e:
        raise ValueError(f"Failed to construct ASE Atoms: {e}")

    data = {
        "supercell_size": np.array(data["supercell_size"], dtype=int),
        "density": float(data["density"]),
        "O_neighbors": np.array(data["O_neighbors"], dtype=int),
        "H_to_OO_pairs": np.array(data["H_to_OO_pairs"], dtype=int),
        "state_hydrogens": np.array(data["state_hydrogens"], dtype=int),
        "atomcoords_O": np.array(data["atomcoords_O"], dtype=float),
        "H2_candidates": np.array(data["H2_candidates"], dtype=float),
    }

    return atoms, data


####################################################################################################

def save_structure_to_vasp(
    json_path: str | Path,
    vasp_path: str | Path,
    vasp_format: str = "vasp",
) -> None:
    """
    Convert a JSON structure file to VASP POSCAR format.

    Args:
        json_path (str | Path): Path to the input JSON file containing structure data.
        vasp_path (str | Path): Path to the output VASP file (e.g., POSCAR or CONTCAR).
        vasp_format (str): VASP format variant. Options:
            - "vasp": Standard VASP5 format (default)
            - "vasp-xdatcar": XDATCAR format for trajectories

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        IOError: If writing the VASP file fails.

    Example:
        >>> save_structure_to_vasp("structure.json", "POSCAR")
        >>> save_structure_to_vasp("structure.json", "CONTCAR", vasp_format="vasp")
    """
    json_path = Path(json_path)
    vasp_path = Path(vasp_path)

    # Load structure from JSON
    atoms, _ = load_structure_from_json(json_path)

    # Ensure output directory exists
    vasp_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to VASP format
    try:
        ase.io.write(
            filename=str(vasp_path),
            images=atoms,
            format=vasp_format,
            direct=True,  # Use direct (fractional) coordinates
            vasp5=True,   # Use VASP5 format with element symbols
        )
    except Exception as e:
        raise IOError(f"Failed to write VASP file {vasp_path}: {e}")


####################################################################################################
####################################################################################################
# Helpers for saving
####################################################################################################
def resolve_save_paths(base_path: Path) -> Tuple[Path, Path, Path]:
    """
    Normalize a user-provided base path to three output files:
        - log_txt_path   : <stub>.txt
        - snap_json_path : <stub>.json
        - traj_xyz_path  : <stub>.xyz

    Rules:
      - If base_path ends with '.json' / '.txt' / '.xyz', we strip this suffix
        to form the stub.
      - Otherwise we treat the *full* filename (including things like '.5')
        as the stub name, and only append extensions.
    """
    base_path = Path(base_path)

    # Case 1: user gave a "real" extension we recognize -> strip it
    if base_path.suffix.lower() in {".json", ".txt", ".xyz"}:
        parent = base_path.parent
        stub_str = base_path.stem        # e.g. 'sc_322_..._T_42.5'
    else:
        # Case 2: no known extension -> keep full name as stub
        parent = base_path.parent
        stub_str = base_path.name        # e.g. 'sc_322_..._T_42.5'

    # ensure directory exists
    parent.mkdir(parents=True, exist_ok=True)

    # build paths explicitly, avoiding Path.with_suffix on '.5'
    log_txt_path   = parent / f"{stub_str}.txt"
    snap_json_path = parent / f"{stub_str}.json"
    traj_xyz_path  = parent / f"{stub_str}.xyz"

    return log_txt_path, snap_json_path, traj_xyz_path

####################################################################################################
####################################################################################################
def save_snapshot_json(
    *,
    json_path: Path,
    atoms: ase.Atoms,
    supercell_size: np.ndarray,
    density: float,
    O_neighbors: np.ndarray,
    H_to_OO_pairs: np.ndarray,
    state_hydrogens: np.ndarray,
    atomcoords_O: np.ndarray,
    H2_candidates: np.ndarray,
) -> None:
    """
    Save (overwrite) the current simulation state into a JSON file.
    Final result always reflects the latest MC status.
    """
    # Update atom positions using the latest state of hydrogens
    atoms_info = {
        "symbols": atoms.get_chemical_symbols(),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
    }

    data = {
        "atoms": atoms_info,
        "supercell_size": supercell_size.tolist(),
        "density": float(density),
        "O_neighbors": O_neighbors.tolist(),
        "H_to_OO_pairs": H_to_OO_pairs.tolist(),
        "state_hydrogens": state_hydrogens.tolist(),
        "atomcoords_O": atomcoords_O.tolist(),
        "H2_candidates": H2_candidates.tolist(),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
####################################################################################################
def auto_rename_log_file(log_path: Path) -> Path:
    """Automatically rename the log file if a file with the same name already exists.

    This avoids overwriting existing logs by appending '_reset_N' before the extension.

    Args:
        log_path (Path): Intended log file path (e.g., 'run.txt').

    Returns:
        Path: Final usable log path (possibly renamed as 'run_reset_1.txt', etc.)

    Raises:
        ValueError: If log_path points to a directory.
    """
    if log_path.exists():
        if log_path.is_dir():
            raise ValueError(f"log_path refers to a directory: {log_path}")
        stem = log_path.stem
        suffix = log_path.suffix
        parent = log_path.parent
        idx = 1
        while True:
            candidate = parent / f"{stem}_reset_{idx}{suffix}"
            if not candidate.exists():
                log_path = candidate
                break
            idx += 1
    return log_path

####################################################################################################
### for all situation
def append_block_summary_line(
    *,
    log_path: Path,
    block_idx: int,
    num_blocks: int,
    time_s: float,
    energy_eV: float,
    # Loop stats
    loop_accepts_block: int,
    loop_attempts_block: int,
    loop_accepts_total: int,
    loop_attempts_total: int,
    # MALA stats
    mala_accepts_block: int,
    mala_attempts_block: int,
    mala_accepts_total: int,
    mala_attempts_total: int,
    # Cell stats
    cell_accepts_block: int,
    cell_attempts_block: int,
    cell_accepts_total: int,
    cell_attempts_total: int,
    # Dielectric
    moment_vec: np.ndarray,
    mu2_mean: float,
    # System info
    num_molecule: int,
    temperature_K: float,
    cell_lengths: np.ndarray,  # (a, b, c) in Angstrom
    pressure_GPa: float,  # Total pressure in GPa
    stress_xx: float,  # Stress component σ_xx (eV/Å^3)
    stress_yy: float,  # Stress component σ_yy (eV/Å^3)
    stress_zz: float,  # Stress component σ_zz (eV/Å^3)
    # Optional discrete-state bitstring
    state_bitstring: str = "",
    # Optional preamble text written before header on first write
    preamble_text: str = "",
) -> None:
    """
    Append per-block summary (loop + MALA + cell) to a CSV-like .txt log.
    Auto-creates header on first write; optionally prepends a preamble block.
    """
    log_path = Path(log_path)
    is_new = not log_path.exists()

    header = (
        "block,num_blocks,time,energy,"
        "loop_accepts_block,loop_attempts_block,loop_accepts_total,loop_attempts_total,"
        "mala_accepts_block,mala_attempts_block,mala_accepts_total,mala_attempts_total,"
        "cell_accepts_block,cell_attempts_block,cell_accepts_total,cell_attempts_total,"
        "mu2_mean,"
        "mx,my,mz,"
        "cell_a,cell_b,cell_c,"
        "pressure_GPa,stress_xx,stress_yy,stress_zz,"
        "num_molecule,temperature_K,state\n"
    )

    line = (
        f"{block_idx:10d},{num_blocks:10d},"
        f"{time_s:10.3f},{energy_eV:20.12f},"
        f"{loop_accepts_block:6d},{loop_attempts_block:6d},{loop_accepts_total:12d},{loop_attempts_total:12d},"
        f"{mala_accepts_block:6d},{mala_attempts_block:6d},{mala_accepts_total:12d},{mala_attempts_total:12d},"
        f"{cell_accepts_block:6d},{cell_attempts_block:6d},{cell_accepts_total:12d},{cell_attempts_total:12d},"
        f" {mu2_mean:.6e},"
        f" {moment_vec[0]: .6e}, {moment_vec[1]: .6e}, {moment_vec[2]: .6e},"
        f" {cell_lengths[0]:12.6f}, {cell_lengths[1]:12.6f}, {cell_lengths[2]:12.6f},"
        f" {pressure_GPa:12.6f}, {stress_xx: .6e}, {stress_yy: .6e}, {stress_zz: .6e},"
        f" {num_molecule:4d}, {temperature_K:8.3f},  {state_bitstring}\n"
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        if is_new:
            if preamble_text:
                if not preamble_text.endswith("\n"):
                    preamble_text += "\n"
                f.write(preamble_text)
            f.write(header)
        f.write(line)


####################################################################################################
####################################################################################################

def append_xyz_snapshot(
    *,
    atoms: ase.Atoms,
    xyz_path: Path,
) -> None:
    """
    Append one ASE Atoms snapshot to a multi-frame XYZ file.

    This function appends a new frame (atoms positions, cell, etc.)
    to the specified .xyz file. If the file does not exist, it will
    be created automatically.

    Args:
        atoms: ASE Atoms object (already ordered O first, H after).
        xyz_path: Output XYZ file path.
    """
    xyz_path.parent.mkdir(parents=True, exist_ok=True)

    ase.io.write(
        filename=str(xyz_path),
        images=atoms,
        format="extxyz",     # <--- key difference!
        append=True,
    )
    

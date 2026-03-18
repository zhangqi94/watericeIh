import warnings
import numpy as np
from ase.neighborlist import neighbor_list
from typing import Callable, Any, Optional, Tuple, Dict, List

####################################################################################################
# --------------------------------- small utils ---------------------------------
try:
    from tools import mic_vec, wrap_pos, normalize_vec
    from createcrystal import classify_h_by_oxygen
except Exception:
    from source.tools import mic_vec, wrap_pos, normalize_vec
    from source.createcrystal import classify_h_by_oxygen

####################################################################################################
# ------------------------------ core building blocks ------------------------------
####################################################################################################
def compute_OH_bond_lengths_angles(
    coords: np.ndarray,
    H_to_OO_pairs: np.ndarray,
    state_hydrogens: np.ndarray,
    box_lengths: float | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute O–H bond lengths and H–O–H angles.

    Parameters
    ----------
    atomcoords : (n_O + n_H, 3) array
        Cartesian coordinates of all atoms, with O atoms first and H atoms after.
    H_to_OO_pairs : (n_H, 3) array
        Each row is [H_global_index, O1_index, O2_index].
    state_hydrogens : (n_H,) array
        For each hydrogen: 0 → bonded to O1, 1 → bonded to O2.
    box_lengths : float or (3,) array
        Box lengths along x, y, z (orthorhombic PBC).

    Returns
    -------
    d_OH : (n_O, 2) array
        Two O–H bond lengths (Å) for each oxygen. NaN if fewer than 2 H present.
    angle_deg : (n_O,) array
        H–O–H angle for each oxygen in degrees. NaN if fewer than 2 H present.
    """
    coords = np.asarray(coords, dtype=float)
    H_to_OO_pairs = np.asarray(H_to_OO_pairs, dtype=int)
    state_hydrogens = np.asarray(state_hydrogens, dtype=int)

    n_total = coords.shape[0]
    n_H = H_to_OO_pairs.shape[0]
    n_O = n_total - n_H

    o1_idx = H_to_OO_pairs[:, 1]
    o2_idx = H_to_OO_pairs[:, 2]

    # Split O and H coordinates
    atomcoords_O = coords[:n_O]
    atomcoords_H = coords[n_O:]

    # Select bonded oxygen for each H
    owner_O = np.where(state_hydrogens == 0, o1_idx, o2_idx)

    # Collect up to two hydrogens per oxygen
    H2_pos = np.full((n_O, 2, 3), np.nan, dtype=float)
    count = np.zeros(n_O, dtype=int)

    for h_idx in range(n_H):
        o_idx = owner_O[h_idx]
        # guard for unlikely overflow (>2 H per O)
        if count[o_idx] < 2:
            H2_pos[o_idx, count[o_idx]] = atomcoords_H[h_idx]
            count[o_idx] += 1

    # Allocate outputs
    distance_OH = np.full((n_O, 2), np.nan, dtype=float)
    angle_HOH = np.full(n_O, np.nan, dtype=float)

    # Compute for oxygens that have exactly 2 bonded H
    valid = count == 2
    if np.any(valid):
        O_valid = atomcoords_O[valid]
        H1 = H2_pos[valid, 0]
        H2 = H2_pos[valid, 1]

        # --- use mic_vec here ---
        v1 = mic_vec(H1 - O_valid, box_lengths)
        v2 = mic_vec(H2 - O_valid, box_lengths)

        d1 = np.linalg.norm(v1, axis=1)
        d2 = np.linalg.norm(v2, axis=1)
        distance_OH[valid, 0] = d1
        distance_OH[valid, 1] = d2

        denom = np.clip(d1 * d2, 1e-15, None)
        cosang = np.sum(v1 * v2, axis=1) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        angle_HOH[valid] = np.degrees(np.arccos(cosang))

    return distance_OH, angle_HOH

####################################################################################################
# ------------------------------ core building blocks ------------------------------
####################################################################################################
def two_nearest_H_per_O(
    atoms, 
    cutoff: float = 1.2, 
    strict: bool = True
) -> np.ndarray:
    """
    For each O, return its two nearest distinct H indices within a fixed cutoff (Å).
    PBC handled by ASE. De-duplicates multiple images/directions.

    Returns:
        (n_O, 3) int array: rows [O_index, H1_index, H2_index] (H1/H2 sorted by distance).
    """
    symbols = np.array(atoms.get_chemical_symbols())
    O_idx = np.where(symbols == 'O')[0]
    H_idx = np.where(symbols == 'H')[0]
    if len(O_idx) == 0 or len(H_idx) == 0:
        raise ValueError("Need both O and H in atoms.")

    # Get neighbors once (PBC-aware)
    ii, jj, dd = neighbor_list('ijd', atoms, cutoff)

    # Keep only O->H direction
    mask = (symbols[ii] == 'O') & (symbols[jj] == 'H')
    ii, jj, dd = ii[mask], jj[mask], dd[mask]

    # Build per-O map to unique H with minimal distance (deduplicate images)
    # o2h[o] = dict {h: d_min}
    o2h = {int(o): {} for o in O_idx}
    for o, h, d in zip(ii, jj, dd):
        o = int(o); h = int(h)
        dm = o2h[o].get(h)
        if dm is None or d < dm:
            o2h[o][h] = float(d)

    rows = []
    for o in O_idx:
        # Sort H neighbors by distance
        items = sorted(o2h[o].items(), key=lambda kv: kv[1])
        n = len(items)

        if strict and n < 2:
            raise RuntimeError(f"O {o} has only {n} H within {cutoff} Å (expected 2).")

        # If strict=False, warn when the O does not have exactly two H
        if (not strict) and (n != 2):
            warnings.warn(
                f"[two_nearest_H_per_O] O {o} has {n} H < {cutoff} Å (expected 2).",
                RuntimeWarning
            )

        h1 = int(items[0][0]) if n >= 1 else -1
        h2 = int(items[1][0]) if n >= 2 else -1
        rows.append([int(o), h1, h2])

    return np.array(rows, dtype=int)

####################################################################################################
def check_hydrogen_consistency(
    atoms: Any,
    H_to_OO_pairs: np.ndarray,
    state_curr: np.ndarray,
    cutoff: float = 1.2,
) -> int:
    """
    Check if each oxygen has the same two hydrogens by geometry and by state.

    Args:
        atoms: ASE Atoms object.
        H_to_OO_pairs: (N_H, 3) array, rows [H_index, O1, O2].
        state_curr: (N_H,) array of hydrogen states (0 = closer to O1, 1 = closer to O2).
        cutoff: Distance cutoff (A) for geometric O-H detection.

    Returns:
        Number of oxygens where geometry and state assignments differ.
    """

    h_to_oo_list = two_nearest_H_per_O(atoms, cutoff=cutoff, strict=True)
    O_map_curr = classify_h_by_oxygen(H_to_OO_pairs, state_curr)

    mismatch = 0
    for O, h1, h2 in h_to_oo_list:
        near_geom = {int(h1), int(h2)}
        near_state = set(O_map_curr[int(O)][0])
        ok = (near_geom == near_state)
        if not ok:
            mismatch += 1
        # print(f"{O:2d} | {sorted(near_geom)} | {sorted(near_state)} | {'OK' if ok else 'DIFF'}")

    if mismatch == 0:
        print("[OK] All oxygens have consistent hydrogens (geometry matches state).")
    else:
        # print(f"\n{mismatch} O atoms differ: geometry and state are inconsistent.")
        raise ValueError(f"Hydrogen-O-H pairs do not match: {mismatch} mismatches.")
    
    return mismatch


####################################################################################################
# --------------------------- Test Section ---------------------------

if __name__ == '__main__':
    import ase
    import ase.io
    from ckpt import load_structure_from_json

    # ----------------------------------------------------------------------------------------------
    # Load initial structure from JSON (contains both atom data and hydrogen bond topology)
    # ----------------------------------------------------------------------------------------------
    initial_structure_file = "structure/initstru/supercell_211_n_16.json"
    init_stru = "structure/initstru/test_n16.vasp"
    
    initial_structure_file = "structure/initstru/supercell_212_n_32.json"
    init_stru = "structure/initstru/test_n32.vasp"
    
    atoms, data = load_structure_from_json(initial_structure_file)

    # Extract topology and hydrogen states
    H_to_OO_pairs = data['H_to_OO_pairs']
    state_hydrogens = data['state_hydrogens']

    # Get simulation box and atomic coordinates
    box_lengths = np.diag(atoms.get_cell())
    coords = atoms.get_positions()

    # ----------------------------------------------------------------------------------------------
    # Compute bond lengths and angles for the loaded structure
    # ----------------------------------------------------------------------------------------------
    distance_OH, angle_HOH = compute_OH_bond_lengths_angles(
        coords=coords,
        H_to_OO_pairs=H_to_OO_pairs,
        state_hydrogens=state_hydrogens,
        box_lengths=box_lengths,
    )

    print("\n=== From JSON structure ===")
    print("O–H bond lengths (Å):\n", np.round(distance_OH, 4))
    print("H–O–H angles (deg):\n", np.round(angle_HOH, 3))

    # ----------------------------------------------------------------------------------------------
    # Now test with a geometry read directly from a VASP .vasp file
    # ----------------------------------------------------------------------------------------------

    atoms_ase = ase.io.read(init_stru)
    coords = atoms_ase.get_positions()

    # Replace positions in the ASE object to test the same topology on new coordinates
    atoms.set_positions(coords)

    distance_OH, angle_HOH = compute_OH_bond_lengths_angles(
        coords=coords,
        H_to_OO_pairs=H_to_OO_pairs,
        state_hydrogens=state_hydrogens,
        box_lengths=box_lengths,
    )

    print("\n=== From VASP structure ===")
    print("O–H bond lengths (Å):\n", np.round(distance_OH, 4))
    print("H–O–H angles (deg):\n", np.round(angle_HOH, 3))
    
    # ----------------------------------------------------------------------------------------------
    # Now test with a geometry read directly from a VASP .vasp file
    # ----------------------------------------------------------------------------------------------
    h_to_oo = two_nearest_H_per_O(atoms, cutoff=1.2, strict=True)
    print(f"\n[O–(H,H) topology] cutoff = 1.2 Å  (O-index, H1, H2)")
    for o, h1, h2 in h_to_oo:
        d1 = atoms.get_distance(int(o), int(h1), mic=True)
        d2 = atoms.get_distance(int(o), int(h2), mic=True)
        print(f"O {o:3d} -> H {h1:3d} (d={d1:5.3f} Å), H {h2:3d} (d={d2:5.3f} Å)")
        
    


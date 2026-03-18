import numpy as np
from pathlib import Path
import ase
import ase.io
import ase.neighborlist
import json
from typing import Callable, Any, Optional, Tuple, Dict, List

try:
    import units
    from tools import mic_vec, wrap_pos, normalize_vec
except Exception:
    from source import units
    from source.tools import mic_vec, wrap_pos, normalize_vec

####################################################################################################
####################################################################################################
# ------------------------------ core building blocks ------------------------------
####################################################################################################
def make_supercell(
    atoms: ase.Atoms,
    supercell_size: tuple[int, int, int] | list[int] = [2, 1, 1],
    target_density: float = 992.0,
) -> ase.Atoms:
    """
    Repeat a molecular cell into a supercell, enable PBC, reorder atoms (O first, H later),
    and optionally rescale the cell to reach a target density.

    Args:
        atoms (ase.Atoms):
            Input structure representing H₂O molecules.
        supercell_size (tuple[int, int, int] | list[int]):
            Replication factors (nx, ny, nz); must be length-3.
        target_density (float):
            Target density in kg/m³. If <= 0, no rescaling is applied.

    Returns:
        ase.Atoms:
            Supercell with periodic boundary conditions enabled, O-first ordering,
            and optionally rescaled to match the target density. All positions
            are wrapped back into [0, L) along each axis.
    """
    
    # 0) validate inputs
    if len(supercell_size) != 3:
        raise ValueError("supercell_size must be length-3, e.g. [2,2,1].")

    # 1) repeat + PBC
    atoms = atoms.repeat(supercell_size)
    atoms.set_pbc(True)
    
    # 2) reorder: O first, H later
    Z = np.array(atoms.get_chemical_symbols())
    O_idx = np.where(Z == "O")[0]
    H_idx = np.where(Z == "H")[0]
    atoms = atoms[np.concatenate([O_idx, H_idx])]

    # 3) optional density rescaling (orthorhombic assumption)
    if (target_density is not None) and (target_density > 0.0):
        # Ensure orthorhombic: off-diagonals ~ 0
        cell = np.asarray(atoms.get_cell(), dtype=float)
        offdiag = np.abs(cell - np.diag(np.diag(cell)))
        if np.any(offdiag > 1e-10):
            raise ValueError("Non-orthorhombic cell detected; this function assumes orthorhombic.")
        num_H2O = len(atoms) // 3
        mass_H2O_amu = units.MASS_H2O  # amu per molecule
        mass_total_kg = mass_H2O_amu * num_H2O * units.AMU_TO_KG
        
        # Volume via diagonal lengths (Å^3 -> m^3)
        box_lengths = atoms.cell.lengths()              # (3,)
        volume_ang3 = float(np.prod(box_lengths))
        volume_m3 = volume_ang3 * 1e-30

        density_old = mass_total_kg / volume_m3
        scale_factor = (density_old / float(target_density)) ** (1.0 / 3.0)
        atoms.set_cell(atoms.cell * scale_factor, scale_atoms=True)
        
    # 4) wrap into box (use lengths, not full 3x3)
    atoms.set_positions(wrap_pos(atoms.get_positions(), atoms.cell.lengths()))
    return atoms

####################################################################################################
def oxygen_graph(
    atoms: ase.Atoms, 
    cutoff: float = 1.5, 
    verbose: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an O–O neighbor graph and return both neighbor lists and oxygen coordinates.

    Args:
        atoms (ase.Atoms):
            Input structure containing O and H atoms.
        cutoff (float):
            Cutoff distance (Å) for considering O–O neighbors.
        verbose (int):
            If >0, print neighbor information and summary.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - **O_neighbors**: Array of shape (n_O, ≤5), each row [O, n1, n2, n3, n4, ...],
              listing the central O atom and its neighboring O indices.
            - **atomcoords_O**: Array (n_O, 3) containing oxygen coordinates ordered
              by global O indices.
    """
    # 0) validate inputs (require at least one O)
    Z = np.array(atoms.get_chemical_symbols())
    O_idx = np.where(Z == "O")[0]
    nO = len(O_idx)

    # 1) build O-only NeighborList
    atoms_O = atoms[O_idx]
    nl = ase.neighborlist.NeighborList([cutoff] * nO, self_interaction=False, bothways=True)
    nl.update(atoms_O)

    # 2) collect neighbors per O and accumulate unique O–O pairs
    neigh_rows = []
    pair_set = set()
    for i in range(nO):
        idxs, _ = nl.get_neighbors(i)
        g_center = int(O_idx[i])
        g_neighs = [int(O_idx[j]) for j in idxs]
        g_neighs = sorted(g_neighs)  # determinism
        neigh_rows.append([g_center] + g_neighs)
        for j in g_neighs:
            pair_set.add(tuple(sorted((g_center, j))))

    # 3) pack outputs (optionally print summary)
    O_neighbors = np.array(neigh_rows, dtype=int)
    # OO_pairs = np.array(sorted(pair_set), dtype=int)
    atomcoords_O = atoms.positions[O_idx]

    # 4) Validate that each O has exactly 4 neighbors
    bad_rows = [(row[0], len(row) - 1) for row in O_neighbors if len(row) != 5]
    if bad_rows:
        msg = "\n".join([f"O {oid:3d} -> {nnb} neighbors" for oid, nnb in bad_rows])
        raise ValueError(
            f"Neighbor validation failed: expected 4 O–O neighbors per O (cutoff={cutoff:.2f} Å).\n{msg}"
        )

    # 5) Optionally print neighbor table
    if verbose:
        print(f"O-O neighbor list (cutoff = {cutoff:.2f} Å):")
        for row in O_neighbors:
            print(f"O {row[0]:2d}: neighbors -> {row[1:]}")
        # print(f"Found {len(OO_pairs)} unique O-O pairs.")

    return O_neighbors, atomcoords_O

####################################################################################################
def assign_hydrogens(
    atoms: ase.Atoms, 
    verbose: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign each hydrogen atom to its two nearest oxygens (O1, O2) using the minimum-image convention.
    Define a binary state: 0 if the H is closer to O1, else 1.

    Args:
        atoms (ase.Atoms):
            Input structure containing both O and H atoms.
        verbose (int):
            If >0, print per-H assignment and state summary.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - **H_to_OO**: (N_H, 3) integer array, where each row is [H, O1, O2] with O1 < O2.
            - **states**:  (N_H,) integer array in {0, 1}, indicating which O the H is closer to.
    """
    # 0) indices & cached arrays
    Z = np.array(atoms.get_chemical_symbols())
    O_idx = np.where(Z == "O")[0]
    H_idx = np.where(Z == "H")[0]
    pos = atoms.positions
    O_pos = pos[O_idx]
    L = atoms.cell.lengths()  # (3,) for MIC

    # 1) loop over H atoms and find two nearest O under MIC
    H_to_OO, states = [], []
    for h in H_idx:
        # h -> all O MIC vectors and distances
        dvec = mic_vec(O_pos - pos[h], L)  # (nO,3)
        d = np.linalg.norm(dvec, axis=1)
        two = np.argsort(d)[:2]
        o1, o2 = int(O_idx[two[0]]), int(O_idx[two[1]])
        O1, O2 = sorted((o1, o2))

        # 2) state by which O is closer (use ASE's MIC distance for robustness)
        d1 = atoms.get_distance(h, O1, mic=True)
        d2 = atoms.get_distance(h, O2, mic=True)
        state = 0 if d1 <= d2 else 1

        H_to_OO.append([int(h), O1, O2])
        states.append(int(state))

    # 3) pack arrays (sorted by global H index for determinism)
    H_to_OO_arr = np.array(sorted(H_to_OO, key=lambda x: x[0]), dtype=int)
    states_arr = np.array(states, dtype=int)

    if verbose:
        print("Hydrogen → O-O pair assignments:")
        for (h, o1, o2), s in zip(H_to_OO_arr, states_arr):
            print(f"H {h:3d} → O({o1},{o2}), closer to O{s}")

    return H_to_OO_arr, states_arr

####################################################################################################
def make_H_positions(atomcoords_O: np.ndarray,
                     box_lengths: np.ndarray,
                     H_to_OO_pairs: np.ndarray,
                     bond_length: float = 1.0
) -> np.ndarray:
    """
    Generate two candidate hydrogen positions for each O–O pair at ±bond_length
    along the O1–O2 axis (minimum-image convention).

    Args:
        atomcoords_O (np.ndarray):
            Array of oxygen coordinates, shape (n_O, 3).
        box_lengths (np.ndarray):
            Box lengths along x, y, z (Å); assumed orthorhombic.
        H_to_OO_pairs (np.ndarray):
            Integer array (N_H, 3) with rows [H, O1, O2].
        bond_length (float):
            Displacement distance (Å) along the O–O bond direction.

    Returns:
        np.ndarray:
            Hydrogen candidate coordinates of shape (N_H, 2, 3):
            for each H, positions near O1 and O2.
    """
    # 0) build diagonal lattice and inverse (orthorhombic)
    lattice = np.diag(box_lengths)
    invlat = np.linalg.inv(lattice)

    # 1) allocate output
    H2 = np.empty((len(H_to_OO_pairs), 2, 3), dtype=float)

    # 2) for each (O1,O2), place candidates along MIC O1->O2 axis
    for i, (_, o1, o2) in enumerate(H_to_OO_pairs):
        r1 = atomcoords_O[o1]
        r2 = atomcoords_O[o2]
        dv = r2 - r1
        f = dv @ invlat
        f -= np.round(f)
        dv_mic = f @ lattice
        e = normalize_vec(dv_mic)
        H2[i, 0] = r1 + bond_length * e  # near O1
        H2[i, 1] = r2 - bond_length * e  # near O2

    # 3) wrap (vectorized mod by lengths)
    H2 = np.mod(H2, box_lengths)
    return H2

####################################################################################################
def validate_ice_rule(
    H_to_OO_pairs: np.ndarray, 
    state_hydrogens: np.ndarray
) -> None:
    """
    Validate the local "ice rule" connectivity constraints:
      (i) every H atom must appear exactly once,
      (ii) each O must have exactly two nearby hydrogens (based on states).

    Args:
        H_to_OO_pairs (np.ndarray):
            Integer array (N_H, 3), rows [H, O1, O2].
        state_hydrogens (np.ndarray):
            Integer array (N_H,) with values {0, 1}, selecting which O each H is assigned to.

    Raises:
        ValueError:
            - If any H appears more than once.
            - If any O has an incorrect hydrogen count (≠ 2).
    """
    # 0) unique H
    H_ids, counts = np.unique(H_to_OO_pairs[:, 0], return_counts=True)
    dup = H_ids[counts > 1]
    if dup.size:
        raise ValueError(f"Duplicate H assignments: {dup.tolist()}")

    # 1) count H near each O per state
    counts_O: dict[int, int] = {}
    for (h, o1, o2), s in zip(H_to_OO_pairs, state_hydrogens):
        o = int(o1) if int(s) == 0 else int(o2)
        counts_O[o] = counts_O.get(o, 0) + 1

    # 2) check each involved O has exactly 2 H
    all_O_involved = np.unique(H_to_OO_pairs[:, 1:3].reshape(-1))
    bad = [(int(o), int(counts_O.get(int(o), 0))) for o in all_O_involved if counts_O.get(int(o), 0) != 2]
    if bad:
        msg = "\n".join([f"O {o:3d} -> {c} H atoms" for o, c in bad])
        raise ValueError("Some O atoms have abnormal H counts:\n" + msg)

####################################################################################################

def classify_h_by_oxygen(
    H_to_OO_pairs: np.ndarray,
    state_hydrogens: np.ndarray,
) -> dict[int, tuple[list[int], list[int]]]:
    """For each oxygen O, classify its 4 connected hydrogens into near vs far.

    Conventions:
      - Each row of H_to_OO_pairs is [H, O1, O2].
      - state==0 → H is near O1 and far from O2.
      - state==1 → H is near O2 and far from O1.

    Returns:
        Dict mapping O -> (near_H_list, far_H_list).
        Each O must have exactly 4 hydrogens: 2 near and 2 far.
    """
    # --- flatten all O–H links ---
    H = H_to_OO_pairs[:, 0]
    O1 = H_to_OO_pairs[:, 1]
    O2 = H_to_OO_pairs[:, 2]
    s = state_hydrogens

    # For each bond, decide which O is near/far
    near_O = np.where(s == 0, O1, O2)
    far_O  = np.where(s == 0, O2, O1)

    # Each O appears 4 times in near_O or far_O
    # Build a flat list of (O,H,is_near)
    O_all = np.concatenate([near_O, far_O])
    H_all = np.concatenate([H, H])
    is_near = np.concatenate([np.ones_like(H), np.zeros_like(H)])

    # Sort/group by O
    sort_idx = np.argsort(O_all)
    O_sorted = O_all[sort_idx]
    H_sorted = H_all[sort_idx]
    near_sorted = is_near[sort_idx]

    # Split by unique O
    unique_O, counts = np.unique(O_sorted, return_counts=True)
    splits = np.split(np.arange(len(O_sorted)), np.cumsum(counts)[:-1])

    result = {}
    for O, idxs in zip(unique_O, splits):
        Hs = H_sorted[idxs]
        near_mask = near_sorted[idxs].astype(bool)
        near = Hs[near_mask].tolist()
        far  = Hs[~near_mask].tolist()

        if len(near) != 2 or len(far) != 2:
            raise ValueError(f"O {int(O)}: expected 2 near, 2 far (got {len(near)}, {len(far)}).")
        result[int(O)] = (near, far)

    return result

####################################################################################################
####################################################################################################
####################################################################################################
# ----------------------------------------- main -----------------------------------------
if __name__ == "__main__":

    # ---- user parameters ----
    stru_file = Path("structure/initstru/stru_iceXI_ideal.vasp")

    # List of supercell sizes to generate (customize as needed)
    supercell_sizes = [
        (2, 1, 1),
        (3, 1, 1),
        (2, 1, 2),
        (2, 2, 2),
        (3, 2, 2),
        (4, 2, 2),
        (4, 2, 3),
        (4, 3, 3),
        (5, 3, 3),
        (6, 4, 4),
        (7, 4, 4),
        (8, 4, 4),
    ]

    # You can provide a single density value (broadcasted to all supercells),
    # or one density per supercell.
    # density = 900.0
    # density = 920.0
    density = 933.0
    # density = 945.0
    # density = 965.0
    # density = 992.0

    # Output directory
    out_dir = Path("structure/initstru")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- main loop ----
    for supercell_size in supercell_sizes:
        # Output file naming pattern: sc_{ijk}_n_{8*np.prod(supercell_size)}_rho_{int(density)}.vasp
        stub = (
            f"sc_{''.join(map(str, supercell_size))}"
            f"_n_{8 * int(np.prod(supercell_size))}"
            # f"_rho_{int(density)}"
        )
        out_vasp = out_dir / f"{stub}.vasp"
        out_json = out_vasp.with_suffix(".json")

        print(f"\n=== Building supercell {supercell_size}, density={density:.3f} kg/m^3 ===")

        # 1) Build supercell
        atoms = ase.io.read(stru_file)
        atoms = make_supercell(atoms, supercell_size=supercell_size, target_density=float(density))
        print(f"Supercell ready -> {out_vasp.name}")

        # 2) Build O–O neighbor graph
        O_neighbors, atomcoords_O = oxygen_graph(atoms, cutoff=1.5, verbose=1)

        # 3) Assign hydrogens to O–O pairs and determine state (0 or 1)
        H_to_OO_pairs, state_hydrogens = assign_hydrogens(atoms, verbose=1)

        # 4) Validate ice rules
        validate_ice_rule(H_to_OO_pairs, state_hydrogens)
        print("Validation passed: each OO has 1 H; each O has exactly 2 nearby H.")

        # Optional: preview first few O atoms’ H mapping
        try:
            O_map = classify_h_by_oxygen(H_to_OO_pairs, state_hydrogens)
            preview = min(3, len(O_map))
            for O, (near, far) in list(sorted(O_map.items()))[:preview]:
                print(f"O {O}: near={sorted(near)}, far={sorted(far)}")
        except Exception:
            pass  # Classification is for preview only; ignore errors

        # 5) Place H coordinates (two candidates) and pick according to state
        box_lengths = atoms.cell.lengths()
        H2_candidates = make_H_positions(
            atomcoords_O, box_lengths, H_to_OO_pairs, bond_length=1.0
        )
        H_selected = H2_candidates[np.arange(len(state_hydrogens)), state_hydrogens]

        # Update atomic positions (O first, followed by selected H)
        coords = np.concatenate([atomcoords_O, H_selected], axis=0)
        atoms.set_positions(coords)

        # 6) Save supercell to VASP file
        ase.io.write(out_vasp, atoms, format="vasp")
        print(f"[Write] VASP -> {out_vasp}")

        # 7) Pack metadata and save to JSON
        atoms_info = {
            "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions().tolist(),
            "cell": atoms.cell.array.tolist(),
            "pbc": atoms.pbc.tolist(),
        }
        data = {
            "atoms": atoms_info,
            "supercell_size": list(supercell_size),
            "density": float(density),
            "O_neighbors": O_neighbors.tolist(),
            "H_to_OO_pairs": H_to_OO_pairs.tolist(),
            "state_hydrogens": state_hydrogens.tolist(),
            "atomcoords_O": atomcoords_O.tolist(),
            "H2_candidates": H2_candidates.tolist(),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[Write] JSON  -> {out_json}")



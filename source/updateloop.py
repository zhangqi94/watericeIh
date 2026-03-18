import warnings
import numpy as np
import ase
import ase.io
from typing import Callable, Any, Optional, Tuple, Dict, List

try:
    from crystaltools import two_nearest_H_per_O, compute_OH_bond_lengths_angles
    from rotation import get_loop_O_H_pairs_from_maps, update_loop_H_coordinates
except Exception:
    from source.crystaltools import two_nearest_H_per_O
    from source.rotation import get_loop_O_H_pairs_from_maps, update_loop_H_coordinates

####################################################################################################
## useful functions for working with state arrays and bitstrings
def state_to_bitstring(state: np.ndarray) -> str:
    """Convert a 0/1 integer array to a bitstring, preserving leading zeros.
    """
    return "".join("1" if int(x) == 1 else "0" for x in state)

def bitstring_to_hexstr(bitstr: str) -> str:
    # left-pad to multiple of 4 bits
    pad = (-len(bitstr)) % 4
    bitstr = bitstr.zfill(len(bitstr) + pad)
    return ''.join(f"{int(bitstr[i:i+4], 2):X}" for i in range(0, len(bitstr), 4))

def hexstr_to_bitstring(hexstr: str) -> str:
    # each hex digit back to 4 bits (keeps leading zeros)
    return ''.join(f"{int(ch, 16):04b}" for ch in hexstr)


####################################################################################################
def build_bond_map(H_to_OO_pairs: np.ndarray) -> dict[Tuple[int, int], int]:
    """Build a lookup table mapping each directed O–O pair to its associated hydrogen index."""
    bond_map = {}
    for h_idx, (_, o1, o2) in enumerate(H_to_OO_pairs):
        o1 = int(o1); o2 = int(o2)
        bond_map[(o1, o2)] = h_idx
        bond_map[(o2, o1)] = h_idx
    return bond_map

####################################################################################################
def short_loop_update(
    O_neighbors: np.ndarray,
    H_to_OO_pairs: np.ndarray,
    state_hydrogens: np.ndarray,
    start_O: Optional[int] = None,
    bond_map: Optional[Dict[Tuple[int,int],int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a short-loop (closed cycle) update on the hydrogen-bond network.

    Starting from an oxygen, follow the directed O–O graph defined by current
    hydrogen orientations until a loop is detected. The hydrogen orientations
    along that loop are then flipped.

    Args:
        O_neighbors (np.ndarray):
            (n_O, 5) array: [O, n1, n2, n3, n4] listing the four O neighbors of each O.
        H_to_OO_pairs (np.ndarray):
            (n_H, 3) array: [H_index, O1, O2] giving the two oxygens bridged by each H.
        state_hydrogens (np.ndarray):
            (n_H,) array of 0/1 indicating orientation (closer to O1 or O2).
        start_O (int, optional):
            If given, loop search begins from this oxygen; otherwise random.

        bond_map (dict, optional):
            Precomputed {(O1,O2): H_index}. If None, built internally.

    Returns:
        new_state (np.ndarray): Updated hydrogen orientation array.
        loop_O (np.ndarray): O-indices forming the detected loop (simple cycle).
        flipped_H_indices (np.ndarray): Hydrogen indices flipped along the loop.
    """
    
    # Build bond lookup map
    if bond_map is None:
        bond_map = build_bond_map(H_to_OO_pairs)
    
    new_state = state_hydrogens.copy()
    flipped_H_indices = []
    loop_O = []

    # Choose starting O
    if start_O is None:
        current_O = O_neighbors[np.random.randint(len(O_neighbors)), 0]
    else:
        current_O = start_O

    visited_O = [current_O]
    visited_bonds = []

    while True:
        outgoing = []
        neighbors = O_neighbors[O_neighbors[:, 0] == current_O, 1:].flatten()
        for neigh in neighbors:
            bond_idx = bond_map.get((current_O, neigh))
            if bond_idx is None:
                continue
            direction = new_state[bond_idx]
            O1, O2 = H_to_OO_pairs[bond_idx][1], H_to_OO_pairs[bond_idx][2]

            # outgoing hydrogen direction
            if (current_O == O1 and direction == 0) or (current_O == O2 and direction == 1):
                outgoing.append((neigh, bond_idx))

        if not outgoing:
            break

        next_O, bond_idx = outgoing[np.random.randint(len(outgoing))]
        visited_bonds.append(bond_idx)

        if next_O in visited_O:
            # Loop closed
            loop_start = visited_O.index(next_O)
            loop_bonds = visited_bonds[loop_start:]
            for bidx in loop_bonds:
                new_state[bidx] = 1 - new_state[bidx]
            flipped_H_indices = loop_bonds[:]

            # Construct loop_O (closed cycle, O...O back to start)
            loop_path = visited_O[loop_start:]
            loop_O = loop_path[:]
            break

        visited_O.append(next_O)
        current_O = next_O

    return (
        np.array(new_state, dtype=int),
        np.array(loop_O, dtype=int),
        np.array(flipped_H_indices, dtype=int),
    )

####################################################################################################
def make_metropolis_loop_update_functions(
    O_neighbors: np.ndarray,
    H_to_OO_pairs: np.ndarray,
    mace_inference: Callable,
    create_neighborlist_device: str = "gpu"
) -> Callable:
    """Create a Metropolis loop-update function that proposes a short-loop flip and evaluates ΔE.

    Args:
        O_neighbors (np.ndarray):
            Integer array of shape (n_O, 5). Each row: [O, n1, n2, n3, n4], giving the
            four O neighbors of oxygen O. The first column must be the O index itself.
        H_to_OO_pairs (np.ndarray):
            Integer array of shape (n_H, 3). Each row: [H_index, O1, O2], where O1 and O2
            are the two oxygens bridged by hydrogen H_index. The row order aligns with the
            state array indices.
        mace_inference (Callable):
            A function that computes energy (and optionally forces/stress) for the given
            `atoms` at provided Cartesian coordinates. Expected minimal signature:
                energy_eV, forces, stress = mace_inference(atoms, compute_force=False)
            Only `energy_eV` is required by this updater.

            If your inference engine expects a different signature, wrap it with a small
            adapter to match this one.
        create_neighborlist_device (str):
            Device to use for creating neighborlist ("gpu" or "cpu"). Default: "gpu".

    Returns:
        Callable:
            The function
                metropolis_loop_update(
                    state_hydrogens, start_O, potential_energy,
                    temperature_in_eV, atoms,
                ) -> (state_new, energy_new, coords_new, atoms, accepted)
            which performs one Metropolis proposal and returns the accepted state or the
            original state if rejected.
    """

    # -------------------------- validation --------------------------
    if not isinstance(O_neighbors, np.ndarray) or O_neighbors.ndim != 2 or O_neighbors.shape[1] != 5:
        raise ValueError("O_neighbors must be an int array of shape (n_O, 5): [O, n1, n2, n3, n4].")
    if not isinstance(H_to_OO_pairs, np.ndarray) or H_to_OO_pairs.ndim != 2 or H_to_OO_pairs.shape[1] != 3:
        raise ValueError("H_to_OO_pairs must be an int array of shape (n_H, 3): [H_index, O1, O2].")

    # Build bond lookup once; reused by the inner updater
    bond_map: Dict[Tuple[int, int], int] = build_bond_map(H_to_OO_pairs)

    #===============================================================================================
    # -------------------------- update function --------------------------
    def metropolis_loop_update(
        state_hydrogens: np.ndarray,
        start_O: Optional[int],
        potential_energy: float,
        temperature_in_eV: float,
        atoms: ase.Atoms,
    ) -> Tuple[np.ndarray, float, ase.Atoms, bool]:
        """Propose one short-loop flip and accept/reject with the Metropolis criterion.

        Args:
            state_hydrogens (np.ndarray):
                Integer array of shape (n_H,) with values in {0,1}. Orientation of each
                H with respect to its (O1,O2) pair in `H_to_OO_pairs`.
            start_O (Optional[int]):
                If provided, the loop search starts from this oxygen index; otherwise
                a random start O is chosen internally by `short_loop_update`.
            potential_energy (float):
                Current potential energy (in eV) of the configuration.
            temperature_in_eV (float):
                Temperature in eV units (k_B * T). Must be > 0. Example: T[K] * k_B[eV/K].
            atoms (ase.Atoms):
                ASE Atoms for the system. Positions and cell are read from this object.

        Returns:
            Tuple[np.ndarray, float, ase.Atoms, bool]:
                - state_new: (n_H,) int array, either the proposed (if accepted) or the original.
                - energy_new: float energy (eV), either the new (if accepted) or the original.
                - atoms: ase.Atoms object with updated positions.
                - accepted: bool indicating whether the proposal was accepted.
        """
        # Read current coordinates and box lengths from atoms
        coords = np.asarray(atoms.get_positions(), dtype=float)
        box_lengths = np.asarray(atoms.cell.lengths(), dtype=float)
        # (1) Propose a new discrete hydrogen configuration via short-loop update
        state_hydrogens_new, loop_O, _ = short_loop_update(
            O_neighbors=O_neighbors,
            H_to_OO_pairs=H_to_OO_pairs,
            state_hydrogens=state_hydrogens,
            start_O=start_O,
            bond_map=bond_map,
        )

        # Edge case: if no loop was found, reject the proposal gracefully
        if loop_O.size == 0:
            warnings.warn(
                "[Metropolis-Loop] No closed hydrogen-bond loop found — proposal skipped.",
                category=UserWarning,
                stacklevel=2,
            )
            return state_hydrogens, potential_energy, atoms, False

        # (2) Build consistent new coordinates for H atoms along the loop
        #     by rotating within each molecule to preserve geometry.
        h2o_idx_move = get_loop_O_H_pairs_from_maps(
            H_to_OO_pairs=H_to_OO_pairs,
            loop_O=loop_O,
            state_before=state_hydrogens,
            state_after=state_hydrogens_new,
        )

        coords_new = update_loop_H_coordinates(
            coords=coords,
            loop_O=loop_O,
            h2o_idx_move=h2o_idx_move,
            box_lengths=box_lengths,
            magnitude="obtuse",  # typically ~120° rotation around O–O midpoint
        )

        # (3) Evaluate energy of the proposed configuration (no side-effects on atoms)
        atoms_prop = atoms.copy()
        atoms_prop.set_positions(coords_new)
        potential_energy_new, _, _ = mace_inference(atoms_prop, compute_force=False, create_neighborlist_device=create_neighborlist_device)

        # (4) Metropolis criterion: accept with probability min(1, exp(-ΔE / kT))
        dE = float(potential_energy_new - potential_energy)
        accepted = bool(np.log(np.random.rand()) < -dE / float(temperature_in_eV))

        # (5) Commit or revert
        if accepted:
            atoms.set_positions(coords_new)
            return state_hydrogens_new, potential_energy_new, atoms, True
        else:
            atoms.set_positions(coords)
            return state_hydrogens, potential_energy, atoms, False


    #===============================================================================================
    # -------------------------- force loop update function --------------------------
    def perform_loop_flip(
        state_curr: np.ndarray,
        atoms: ase.Atoms,
        start_O: Optional[int] = None,
    ):
        """
        Apply one loop flip (if found) and update H coordinates. No energy/force used.

        Args:
            state_curr: (n_H,) int array in {0,1}.
            atoms: ASE Atoms (positions and cell are read from this object, positions updated in-place if a loop is applied).
            start_O: optional O index to seed the loop search.

        Returns:
            state_new, atoms
        """
        # Read current coordinates and box lengths from atoms
        coords_curr = np.asarray(atoms.get_positions(), dtype=float)
        box_lengths = np.asarray(atoms.cell.lengths(), dtype=float)
        # 1) Discrete proposal (loop flip)
        state_new, loop_O, flipped_H = short_loop_update(
            O_neighbors=O_neighbors,
            H_to_OO_pairs=H_to_OO_pairs,
            state_hydrogens=state_curr,
            start_O=start_O,
            bond_map=bond_map,
        )

        # 2) Build mapping for which H moves with which O and update geometry
        h2o_idx_move = get_loop_O_H_pairs_from_maps(
            H_to_OO_pairs=H_to_OO_pairs,
            loop_O=loop_O,
            state_before=state_curr,
            state_after=state_new,
        )

        coords_new = update_loop_H_coordinates(
            coords=coords_curr,
            loop_O=loop_O,
            h2o_idx_move=h2o_idx_move,
            box_lengths=box_lengths,
        )

        # 4) Commit coordinates to atoms
        atoms.set_positions(coords_new)
        return state_new, atoms

    #===============================================================================================
    # -------------------------- only loop update function --------------------------
    def metropolis_only_loop_update(
        state_hydrogens: np.ndarray,
        start_O: Optional[int],
        potential_energy: float,
        temperature_in_eV: float,
        atoms: ase.Atoms,
        atomcoords_O: np.ndarray,
        H2_candidates: np.ndarray,
    ) -> Tuple[np.ndarray, float, ase.Atoms, bool]:
        """Propose one short-loop flip and accept/reject with the Metropolis criterion.
        """
        H_selected = H2_candidates[np.arange(len(state_hydrogens)), state_hydrogens]
        coords = np.concatenate([atomcoords_O, H_selected], axis=0)
        atoms.set_positions(coords)
        
        # (1) Propose a new discrete hydrogen configuration via short-loop update
        state_hydrogens_new, loop_O, _ = short_loop_update(
            O_neighbors=O_neighbors,
            H_to_OO_pairs=H_to_OO_pairs,
            state_hydrogens=state_hydrogens,
            start_O=start_O,
            bond_map=bond_map,
        )

        # Edge case: if no loop was found, reject the proposal gracefully
        if loop_O.size == 0:
            warnings.warn(
                "[Metropolis-Loop] No closed hydrogen-bond loop found — proposal skipped.",
                category=UserWarning,
                stacklevel=2,
            )
            return state_hydrogens, potential_energy, atoms, False

        # (2) Build consistent new coordinates for H atoms along the loop
        H_selected = H2_candidates[np.arange(len(state_hydrogens_new)), state_hydrogens_new]
        coords_new = np.concatenate([atomcoords_O, H_selected], axis=0)

        # (3) Evaluate energy of the proposed configuration (no side-effects on atoms)
        atoms_prop = atoms.copy()
        atoms_prop.set_positions(coords_new)
        potential_energy_new, _, _ = mace_inference(atoms_prop, compute_force=False, create_neighborlist_device=create_neighborlist_device)

        # (4) Metropolis criterion: accept with probability min(1, exp(-ΔE / kT))
        dE = float(potential_energy_new - potential_energy)
        accepted = bool(np.log(np.random.rand()) < -dE / float(temperature_in_eV))

        # (5) Commit or revert
        if accepted:
            atoms.set_positions(coords_new)
            return state_hydrogens_new, potential_energy_new, atoms, True
        else:
            atoms.set_positions(coords)
            return state_hydrogens, potential_energy, atoms, False


    return metropolis_loop_update, perform_loop_flip, metropolis_only_loop_update
    
    
####################################################################################################
####################################################################################################
####################################################################################################
if __name__ == "__main__":

    import time
    from datetime import datetime
    from ckpt import load_structure_from_json
    from createcrystal import classify_h_by_oxygen
    from potentialmace_cueq import initialize_mace_model
    import units

    # ==============================================================================================
    # User parameters
    # ==============================================================================================
    initial_structure_file = "structure/initstru/sc_222_n_64.json"

    # ==============================================================================================
    # Load structure
    # ==============================================================================================
    atoms, data = load_structure_from_json(initial_structure_file)

    coords = atoms.get_positions()
    O_neighbors = data["O_neighbors"]
    H_to_OO_pairs = data["H_to_OO_pairs"]
    state_hydrogens = data["state_hydrogens"]
    num_O = atoms.get_chemical_symbols().count("O")
    atomcoords_O = data["atomcoords_O"]
    H2_candidates = data["H2_candidates"]
    box_lengths = atoms.cell.lengths()  
    
    print(f"[LOAD] Loaded structure: {num_O} O atoms, {len(state_hydrogens)} H atoms.")
    print(f"[LOAD] Box lengths (Angstrom): {box_lengths}")
    coords = atoms.get_positions()

    # Select test mode
    # test_mode = "short_loop_update"
    # test_mode = "metropolis_loop_update"
    test_mode = "metropolis_only_loop"
    
    # ==============================================================================================
    # Test mode: short_loop_update
    # ==============================================================================================
    if test_mode == "short_loop_update":
        # Current (before) state bitstring — preserves leading zeros
        state_before = state_hydrogens.copy()
        print("[STATE before]", state_to_bitstring(state_before))
        O_map = classify_h_by_oxygen(H_to_OO_pairs, state_before)
        for O, (near, far) in sorted(O_map.items()):
            print(f"O {O}: near={sorted(near)}, far={sorted(far)}")
        
        # Run one short-loop update
        state_after, loop_O, flipped_H_indices = short_loop_update(
            O_neighbors, H_to_OO_pairs, state_hydrogens, start_O=0
        )

        # After state bitstring — preserves leading zeros
        print("[STATE after ]", state_to_bitstring(state_after))
        O_map = classify_h_by_oxygen(H_to_OO_pairs, state_after)
        for O, (near, far) in sorted(O_map.items()):
            print(f"O {O}: near={sorted(near)}, far={sorted(far)}")
        
        print("[FLIPPED H ]", flipped_H_indices.tolist())
        print("[LOOP    O ]", loop_O.tolist())

        # Benchmark: without bond_map
        state_curr = state_hydrogens.copy()
        t1 = time.perf_counter()
        for _ in range(100):
            state_curr, loop_O, flipped_H_indices = short_loop_update(
                O_neighbors, H_to_OO_pairs, state_curr, start_O=0
            )
        t2 = time.perf_counter()
        print(f"Without bond_map time elapsed: {t2-t1:.6f} s")

        # Benchmark: with bond_map
        bond_map = build_bond_map(H_to_OO_pairs)
        state_curr = state_hydrogens.copy()
        t1 = time.perf_counter()
        for _ in range(100):
            state_curr, loop_O, flipped_H_indices = short_loop_update(
                O_neighbors, H_to_OO_pairs, state_curr, start_O=0, bond_map=bond_map
            )
        t2 = time.perf_counter()
        print(f"With bond_map time elapsed: {t2-t1:.6f} s")

        # Full loop update with coordinate updates
        print("=" * 80)
        bond_map = build_bond_map(H_to_OO_pairs)
        state_curr = state_hydrogens.copy()
        coords_curr = coords.copy()
        h_to_oo_list = two_nearest_H_per_O(atoms, cutoff=1.2, strict=True)
        
        t1 = time.perf_counter()
        for _ in range(10):
            state_new, loop_O, _ = short_loop_update(
                O_neighbors, H_to_OO_pairs, state_curr, start_O=None, bond_map=bond_map
            )
            h2o_idx_move = get_loop_O_H_pairs_from_maps(
                H_to_OO_pairs, 
                loop_O, 
                state_curr, 
                state_new
            )
            coords_updated = update_loop_H_coordinates(
                coords_curr,
                loop_O,
                h2o_idx_move,
                box_lengths,
                magnitude="obtuse",  # typically ~120°
            )
            
            state_curr = state_new.copy()
            coords_curr = coords_updated.copy()
            atoms.set_positions(coords_curr)
            
        t2 = time.perf_counter()
        print(f"With bond_map time elapsed: {t2-t1:.6f} s")

        # ----------------------------------------------------------------------------------------------
        # (a) Geometric nearest-neighbor method
        h_to_oo_list = two_nearest_H_per_O(atoms, cutoff=1.2, strict=True)

        # (b) Covalent-bond classification from state_after
        O_map_curr = classify_h_by_oxygen(H_to_OO_pairs, state_curr)

        print("O  |  nearest-two-H   |  covalent-near-from-state  | match?")
        print("-------------------------------------------------------------")

        mismatch = 0
        for O, h1, h2 in h_to_oo_list:
            near_geom = {int(h1), int(h2)}
            near_state = set(O_map_curr[int(O)][0])
            ok = (near_geom == near_state)
            if not ok:
                mismatch += 1
            print(f"{O:2d} | {sorted(near_geom)}   | {sorted(near_state)}   | {'OK' if ok else 'DIFF'}")

        if mismatch == 0:
            print("\n  All O atoms have consistent hydrogens: geometry matches state_after.")
        else:
            print(f"\n  {mismatch} O atoms differ: geometry and state_after are inconsistent.")
        
        # import ase.io
        # ase.io.write("update_loop.vasp", atoms, format="vasp")

    # ==============================================================================================
    # Test mode: metropolis_loop_update
    # ==============================================================================================
    elif test_mode == "metropolis_loop_update":
        
        # # Compute bond lengths and angles for the loaded structure
        # distance_OH, angle_HOH = compute_OH_bond_lengths_angles(
        #     coords=coords,
        #     H_to_OO_pairs=H_to_OO_pairs,
        #     state_hydrogens=state_hydrogens,
        #     box_lengths=box_lengths,
        # )

        # print("\n=== From vasp structure ===")
        # print("O–H bond lengths (Å):\n", np.round(distance_OH, 4))
        # print("H–O–H angles (deg):\n", np.round(angle_HOH, 3))
            
        # Prepare MACE model
        mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel251125/mace_iceIh_128x0e128x1o_r4.5_float32_seed144_cueq.model"
        mace_dtype = "float32"
        mace_device = "cuda"

        mace_inference = initialize_mace_model(
            mace_model_path,
            mace_dtype,
            mace_device,
        )

        # Build the Metropolis short-loop update function
        metropolis_loop_update, perform_loop_flip, _ = make_metropolis_loop_update_functions(
            O_neighbors=O_neighbors,
            H_to_OO_pairs=H_to_OO_pairs,
            mace_inference=mace_inference,
        )

        # Monte Carlo loop test
        state_curr = state_hydrogens.copy()

        num_steps = 1000
        temperature_K = 150.0
        temperature_in_eV = temperature_K * units.K_B_EV_PER_K

        accepts = 0
        attempts = 0

        # Force perform loop flip
        # for step in range(1, 20):
        #     state_curr, atoms = perform_loop_flip(state_curr, atoms)
        #     print(
        #         f"[FORCE] Step {step:5d}  "
        #         f"state = {bitstring_to_hexstr(state_to_bitstring(state_curr))}",
        #         flush=True,
        #     )

        # Compute initial potential energy
        energy_curr, _, _ = mace_inference(atoms, compute_force=False)
        print(f"[INIT] Energy = {energy_curr:.12f} eV")
        energy_traj = [energy_curr]             
        
        t0 = time.time()

        for step in range(1, num_steps + 1):
            # Randomly select a starting oxygen
            start_O = int(O_neighbors[np.random.randint(len(O_neighbors)), 0])

            # Propose and (internally) accept/reject; always returns a valid current state
            state_curr, energy_curr, atoms, accepted = metropolis_loop_update(
                state_hydrogens=state_curr,
                start_O=start_O,
                potential_energy=energy_curr,
                temperature_in_eV=temperature_in_eV,
                atoms=atoms,
            )

            attempts += 1
            accepts += int(accepted)
            energy_traj.append(energy_curr)

            # Progress print
            if (step % 1 == 0) or (step == num_steps):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                acc_rate = accepts / attempts if attempts > 0 else float("nan")
                print(
                    f"[{ts}] Step {step:5d}/{num_steps:5d}  "
                    f"E = {energy_curr: .12f} eV  "
                    f"acc_rate = {acc_rate:.3f} ({accepts}/{attempts})  "
                    f"state = {bitstring_to_hexstr(state_to_bitstring(state_curr))}",
                    flush=True,
                )

        t1 = time.time()
        print("\n[MC Summary]")
        print(f"  Steps attempted : {attempts}")
        print(f"  Steps accepted  : {accepts}")
        print(f"  Acceptance rate : {accepts/attempts:.3f}")
        print(f"  Final energy    : {energy_curr:.12f} eV")
        print(f"  Temperature     : {temperature_K:.1f} K  (kT = {temperature_in_eV:.6f} eV)")
        print(f"  Total time      : {t1 - t0:.2f} s")

        print("MC loop finished, structure updated in `atoms`.")

        # Compute bond lengths and angles for the final structure
        coords_curr = atoms.get_positions()
        box_lengths_curr = atoms.cell.lengths()
        distance_OH, angle_HOH = compute_OH_bond_lengths_angles(
            coords=coords_curr,
            H_to_OO_pairs=H_to_OO_pairs,
            state_hydrogens=state_curr,
            box_lengths=box_lengths_curr,
        )
        print("\n=== From updated structure ===")
        print("O–H bond lengths (Å):\n", np.round(distance_OH, 4))
        print("H–O–H angles (deg):\n", np.round(angle_HOH, 3))
        
        # ase.io.write("test_update_loop.vasp", atoms, format="vasp")

        # Plot energy trajectory
        import matplotlib.pyplot as plt

        energy_traj = np.array(energy_traj)
        
        plt.figure(figsize=(8, 4), dpi=300)
        plt.plot(np.arange(len(energy_traj)), energy_traj / num_O + 16, ".-", lw=1)

        plt.xlabel("Monte Carlo step", fontsize=12)
        plt.ylabel("Energy per H₂O (eV)", fontsize=12)
        plt.title(f"Energy trajectory at {temperature_K:.0f} K", fontsize=13)
        plt.grid(True, ls="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

    # ==============================================================================================
    # Test mode: metropolis_only_loop
    # ==============================================================================================
    elif test_mode == "metropolis_only_loop":

        # # Compute bond lengths and angles for the loaded structure
        # distance_OH, angle_HOH = compute_OH_bond_lengths_angles(
        #     coords=coords,
        #     H_to_OO_pairs=H_to_OO_pairs,
        #     state_hydrogens=state_hydrogens,
        #     box_lengths=box_lengths,
        # )

        # print("\n=== From vasp structure ===")
        # print("O–H bond lengths (Å):\n", np.round(distance_OH, 4))
        # print("H–O–H angles (deg):\n", np.round(angle_HOH, 3))
            
        # Prepare MACE model
        mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel251125/mace_iceIh_128x0e128x1o_r4.5_float32_seed144_cueq.model"
        mace_dtype = "float32"
        mace_device = "cuda"

        mace_inference = initialize_mace_model(
            mace_model_path,
            mace_dtype,
            mace_device,
        )

        # Build the Metropolis short-loop update function
        _, perform_loop_flip, metropolis_only_loop_update = make_metropolis_loop_update_functions(
            O_neighbors=O_neighbors,
            H_to_OO_pairs=H_to_OO_pairs,
            mace_inference=mace_inference,
        )
        
        # ----------------------------------------------------------------------------------------------
        # Monte Carlo loop test (Metropolis with short-loop proposals)
        # ----------------------------------------------------------------------------------------------
        state_curr = state_hydrogens.copy()

        num_steps = 1000
        temperature_K = 100.0
        temperature_in_eV = temperature_K * units.K_B_EV_PER_K

        accepts = 0
        attempts = 0

        # Force perform loop flip
        for step in range(1, 20):
            state_curr, atoms = perform_loop_flip(state_curr, atoms)
            print(
                f"[FORCE] Step {step:5d}  "
                f"state = {bitstring_to_hexstr(state_to_bitstring(state_curr))}",
                flush=True,
            )

        H_selected = H2_candidates[np.arange(len(state_curr)), state_curr]
        coords_curr = np.concatenate([atomcoords_O, H_selected], axis=0)
        # Compute initial potential energy
        atoms.set_positions(coords_curr)
        energy_curr, _, _ = mace_inference(atoms, compute_force=False)
        print(f"[INIT] Energy = {energy_curr:.12f} eV")
        energy_traj = [energy_curr]    

        t0 = time.time()

        for step in range(1, num_steps + 1):
            # Randomly select a starting oxygen
            start_O = int(O_neighbors[np.random.randint(len(O_neighbors)), 0])

            # Propose and (internally) accept/reject; always returns a valid current state
            state_curr, energy_curr, atoms, accepted = metropolis_only_loop_update(
                state_hydrogens=state_curr,
                start_O=start_O,
                potential_energy=energy_curr,
                temperature_in_eV=temperature_in_eV,
                atoms=atoms,
                atomcoords_O=atomcoords_O,
                H2_candidates=H2_candidates,
            )

            attempts += 1
            accepts += int(accepted)
            energy_traj.append(energy_curr)

            # Progress print
            if (step % 1 == 0) or (step == num_steps):
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                acc_rate = accepts / attempts if attempts > 0 else float("nan")
                print(
                    f"[{ts}] Step {step:5d}/{num_steps:5d}  "
                    f"E = {energy_curr: .12f} eV  "
                    f"acc_rate = {acc_rate:.3f} ({accepts}/{attempts})  "
                    f"state = {bitstring_to_hexstr(state_to_bitstring(state_curr))}",
                    flush=True,
                )

        t1 = time.time()
        print("\n[MC Summary]")
        print(f"  Steps attempted : {attempts}")
        print(f"  Steps accepted  : {accepts}")
        print(f"  Acceptance rate : {accepts/attempts:.3f}")
        print(f"  Final energy    : {energy_curr:.12f} eV")
        print(f"  Temperature     : {temperature_K:.1f} K  (kT = {temperature_in_eV:.6f} eV)")
        print(f"  Total time      : {t1 - t0:.2f} s")

        print("MC loop finished, structure updated in `atoms`.")

        # Compute bond lengths and angles for the final structure
        coords_curr = atoms.get_positions()
        box_lengths_curr = atoms.cell.lengths()
        distance_OH, angle_HOH = compute_OH_bond_lengths_angles(
            coords=coords_curr,
            H_to_OO_pairs=H_to_OO_pairs,
            state_hydrogens=state_curr,
            box_lengths=box_lengths_curr,
        )
        print("\n=== From updated structure ===")
        print("O–H bond lengths (Å):\n", np.round(distance_OH, 4))
        print("H–O–H angles (deg):\n", np.round(angle_HOH, 3))
        
        # ase.io.write("test_update_loop.vasp", atoms, format="vasp")

        # Plot energy trajectory
        import matplotlib.pyplot as plt

        energy_traj = np.array(energy_traj)
        
        plt.figure(figsize=(8, 4), dpi=300)
        plt.plot(np.arange(len(energy_traj)), energy_traj / num_O + 16, ".-", lw=1)

        plt.xlabel("Monte Carlo step", fontsize=12)
        plt.ylabel("Energy per H₂O (eV)", fontsize=12)
        plt.title(f"Energy trajectory at {temperature_K:.0f} K", fontsize=13)
        plt.grid(True, ls="--", alpha=0.5)

        plt.tight_layout()
        plt.show()


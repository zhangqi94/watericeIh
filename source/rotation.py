import numpy as np
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
def get_loop_O_H_pairs_from_maps(
    H_to_OO_pairs, loop_O, state_before, state_after
):
    ## h2o_idx_move[i] = [O_i, moved_before_H, moved_after_H, stationary_H]
    Omap_before = classify_h_by_oxygen(H_to_OO_pairs, state_before)
    Omap_after  = classify_h_by_oxygen(H_to_OO_pairs, state_after)

    result = []
    for O in loop_O:
        near_b = set(Omap_before[O][0])  # before: covalent H
        near_a = set(Omap_after[O][0])   # after:  covalent H

        moved_before = list(near_b - near_a)  # lost by O
        moved_after  = list(near_a - near_b)  # gained by O
        stationary   = list(near_b & near_a)  # stayed with O

        # convert 1-element lists → scalars (ice rule ensures size=1)
        moved_before = moved_before[0] if moved_before else -1
        moved_after  = moved_after[0]  if moved_after  else -1
        stationary   = stationary[0]   if stationary   else -1

        result.append([int(O), moved_before, moved_after, stationary])

    return np.array(result, dtype=int)

####################################################################################################
def plane_angle_signed_deg(coords: np.ndarray,
                           Oim1: int, Oi: int, Oip1: int, Hstat: int,
                           box_lengths: np.ndarray,
                           use_obtuse: bool = False,
                           eps: float = 1e-15) -> float:
    """
    Signed dihedral angle (deg) between planes (Oim1, Oi, Hstat) → (Oi, Oip1, Hstat),
    about the axis Oi→Hstat (right-hand rule). Positive = CCW when looking from Oi to Hstat.

    If use_obtuse=True, returns the complementary signed angle (e.g., 120° instead of 60°),
    preserving the sign/direction.
    """
    # Vectors rooted at Oi
    v1 = mic_vec(coords[Oim1]  - coords[Oi],  box_lengths)  # Oi→Oim1
    w  = mic_vec(coords[Hstat] - coords[Oi],  box_lengths)  # Oi→Hstat (axis)
    v2 = mic_vec(coords[Oip1]  - coords[Oi],  box_lengths)  # Oi→Oip1

    # Plane normals (both ⟂ w)
    n1 = np.cross(v1, w)
    n2 = np.cross(w,  v2)

    n1_hat = normalize_vec(n1, eps)
    n2_hat = normalize_vec(n2, eps)
    w_hat  = normalize_vec(w,  eps)

    # Principal (unsigned) angle
    ang = np.degrees(np.arctan2(
        np.linalg.norm(np.cross(n1_hat, n2_hat)),
        np.clip(np.dot(n1_hat, n2_hat), -1.0, 1.0)
    ))

    # Sign via right-hand rule about axis w_hat: from n1 to n2
    sign = np.sign(np.dot(np.cross(n1_hat, n2_hat), w_hat))
    ang_signed = sign * ang

    if use_obtuse:
        # Keep direction; use complementary magnitude
        ang_signed = np.copysign(180.0 - abs(ang_signed), ang_signed)

    return float(ang_signed)

####################################################################################################
def rodrigues_rotate(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate vec around (normalized) axis by angle_deg (degrees)."""
    k = normalize_vec(axis)
    ang = np.deg2rad(angle_deg)
    return (vec * np.cos(ang)
            + np.cross(k, vec) * np.sin(ang)
            + k * np.dot(k, vec) * (1.0 - np.cos(ang)))

####################################################################################################
def rotate_H_coordinate(coords: np.ndarray,
                        Oi: int, Hstat: int, Hmove: int,
                        angle_deg: float,
                        box_lengths: np.ndarray) -> np.ndarray:
    """
    Rotate Hmove around axis (Oi→Hstat) by angle_deg (deg).
    Returns the single rotated coordinate for Hmove (no wrap).
    """
    axis = mic_vec(coords[Hstat] - coords[Oi], box_lengths)
    rH   = mic_vec(coords[Hmove] - coords[Oi], box_lengths)
    rH_rot = rodrigues_rotate(rH, axis, angle_deg)
    return coords[Oi] + rH_rot

####################################################################################################
def rotate_Hmove_from_plane1_to_plane2(coords: np.ndarray,
                                       Oim1: int, Oi: int, Oip1: int,
                                       Hstat: int, Hmove: int,
                                       box_lengths: np.ndarray,
                                       magnitude: str = "obtuse") -> tuple[np.ndarray, float]:
    """
    Convenience wrapper:
    - Computes the signed angle about axis Oi→Hstat that rotates Plane1=(Oim1, Oi, Hstat)
      into Plane2=(Oi, Oip1, Hstat), with the same direction (from plane1 to plane2).
    - Rotates Hmove by that signed angle.
    - magnitude: "acute" (e.g., 60°) or "obtuse" (e.g., 120°). Default "obtuse".

    Returns:
        (new_coord_for_Hmove, signed_angle_deg)
    """
    use_obtuse = (magnitude.lower() == "obtuse")
    theta_signed = plane_angle_signed_deg(coords, Oim1, Oi, Oip1, Hstat,
                                          box_lengths, use_obtuse=use_obtuse)
    Hmove_new = rotate_H_coordinate(coords, Oi, Hstat, Hmove, theta_signed, box_lengths)
    return Hmove_new, theta_signed

####################################################################################################
# --------------------------------- main function -----------
def update_loop_H_coordinates(coords: np.ndarray,
                              loop_O: np.ndarray,
                              h2o_idx_move: np.ndarray,
                              box_lengths: np.ndarray,
                              magnitude: str = "obtuse") -> np.ndarray:
    """
    Rotate proton loops: for each O in loop_O, rotate H_before into H_after.
    """
    coords_new = coords.copy()
    k = len(loop_O)

    for idx in range(k):
        Oi       = int(h2o_idx_move[idx, 0])
        H_before = int(h2o_idx_move[idx, 1])   # old covalent hydrogen (leaves Oi)
        H_after  = int(h2o_idx_move[idx, 2])   # new covalent hydrogen (enters Oi)
        H_stat   = int(h2o_idx_move[idx, 3])   # stationary hydrogen

        Oim1 = int(h2o_idx_move[(idx-1) % k, 0])
        Oip1 = int(h2o_idx_move[(idx+1) % k, 0])

        # signed angle from (Oim1,Oi,H_stat) → (Oi,Oip1,H_stat)
        use_obtuse = (magnitude.lower() == "obtuse")
        theta = plane_angle_signed_deg(coords, Oim1, Oi, Oip1, H_stat,
                                       box_lengths, use_obtuse=use_obtuse)

        # rotate vector from Oi to H_before, then write to H_after
        axis = mic_vec(coords[H_stat] - coords[Oi], box_lengths)
        r = mic_vec(coords[H_before] - coords[Oi], box_lengths)
        r_rot = rodrigues_rotate(r, axis, theta)

        coords_new[H_after] = coords[Oi] + r_rot   # ✅ write into H_after index

    coords_new = wrap_pos(coords_new, box_lengths)  # wrap back to box
    return coords_new

####################################################################################################
####################################################################################################
if __name__ == "__main__":
    """
    Example usage of the short-loop hydrogen-bond network update.

    Steps:
      1. Load initial ice structure and metadata
      2. Print initial H-bond network (state_before)
      3. Perform one short-loop update on hydrogen orientations
      4. Print updated network (state_after)
      5. Derive per-O hydrogen transitions within the loop
      6. Update hydrogen coordinates to maintain water geometry
      7. Validate topology using nearest-distance O–H method
    """

    import time
    from ckpt import load_structure_from_json
    from createcrystal import classify_h_by_oxygen
    from crystaltools import two_nearest_H_per_O
    from updateloop import short_loop_update, state_to_bitstring, build_bond_map

    # ----------------------------------------------------------------------------------------------
    # 0) Load structure and initialize
    # ----------------------------------------------------------------------------------------------
    initial_structure_file = (
        "structure/initstru/supercell_211_n_16.json"
        # "structure/initstru/supercell_212_n_32.json"
        # "structure/initstru/supercell_222_n_64.json"
    )

    # Load ASE Atoms object and metadata dictionary
    atoms, data = load_structure_from_json(initial_structure_file)

    # Extract atomic coordinates and topology arrays
    coords = atoms.get_positions()                         # (n_atoms, 3)
    O_neighbors = data["O_neighbors"]                      # (n_O, 4) or (n_O, 5) if first col is O
    H_to_OO_pairs = data["H_to_OO_pairs"]                  # (n_H, 3): [H, O1, O2]
    state_hydrogens = data["state_hydrogens"]              # (n_H,): 0/1 for O1/O2
    num_O = atoms.get_chemical_symbols().count("O")        # Number of O atoms
    box_lengths = atoms.cell.lengths()                     # Orthorhombic box lengths (Å)

    print(f"\n[LOAD] {num_O} O atoms, {len(state_hydrogens)} H atoms  |  box = {box_lengths}\n")

    state_before = state_hydrogens.copy()
    bond_map = build_bond_map(H_to_OO_pairs)

    # ----------------------------------------------------------------------------------------------
    # 1) Print H-bond network state before update
    # ----------------------------------------------------------------------------------------------
    print("[STATE before]", state_to_bitstring(state_before))
    O_map_before = classify_h_by_oxygen(H_to_OO_pairs, state_before)

    print("Covalent H per O (before):")
    for O in sorted(O_map_before):
        near, far = O_map_before[O]
        print(f"  O {O:3d} : near={sorted(near)}, far={sorted(far)}")

    # ----------------------------------------------------------------------------------------------
    # 2) Perform one short-loop update
    # ----------------------------------------------------------------------------------------------
    print("\n[Performing short-loop update...]")
    state_after, loop_O, flipped_H_indices = short_loop_update(
        O_neighbors, H_to_OO_pairs, state_before, start_O=None, bond_map=bond_map,
    )

    print(f"Loop O sequence : {loop_O.tolist()}")
    print(f"Flipped H index : {flipped_H_indices.tolist()}\n")

    # ----------------------------------------------------------------------------------------------
    # 3) Print updated H-bond network
    # ----------------------------------------------------------------------------------------------
    print("[STATE after ]", state_to_bitstring(state_after))
    O_map_after = classify_h_by_oxygen(H_to_OO_pairs, state_after)

    print("Covalent H per O (after):")
    for O in sorted(O_map_after):
        near, far = O_map_after[O]
        print(f"  O {O:3d} : near={sorted(near)}, far={sorted(far)}")

    # ----------------------------------------------------------------------------------------------
    # 4) Derive per-O hydrogen transitions along the loop
    # ----------------------------------------------------------------------------------------------
    h2o_idx_move = get_loop_O_H_pairs_from_maps(H_to_OO_pairs, loop_O, state_before, state_after)
    print("\n[Loop atom mapping  (O_i, H_before, H_after, H_stat)]")
    for row in h2o_idx_move:
        print(f"  {row}")

    # ----------------------------------------------------------------------------------------------
    # 5) Rotate H-atom coordinates to preserve molecular geometry
    # ----------------------------------------------------------------------------------------------
    coords_updated = update_loop_H_coordinates(
        coords,
        loop_O,
        h2o_idx_move,
        box_lengths,
        magnitude="obtuse",  # typically ~120°
    )
    atoms.set_positions(coords_updated)

    # ----------------------------------------------------------------------------------------------
    # 6) Validate topology: check nearest O–H assignments
    # ----------------------------------------------------------------------------------------------
    print("\n[Check O–H geometry vs. state_after covalent assignment]")

    # (a) Geometric nearest-neighbor method
    h_to_oo_list = two_nearest_H_per_O(atoms, cutoff=1.2, strict=True)

    # (b) Covalent-bond classification from state_after
    O_map_after = classify_h_by_oxygen(H_to_OO_pairs, state_after)

    print("O  |  nearest-two-H   |  covalent-near-from-state  | match?")
    print("-------------------------------------------------------------")

    mismatch = 0
    for O, h1, h2 in h_to_oo_list:
        near_geom = {int(h1), int(h2)}
        near_state = set(O_map_after[int(O)][0])
        ok = (near_geom == near_state)
        if not ok:
            mismatch += 1
        print(f"{O:2d} | {sorted(near_geom)}   | {sorted(near_state)}   | {'OK' if ok else 'DIFF'}")

    if mismatch == 0:
        print("\n✅  All O atoms have consistent hydrogens: geometry matches state_after.")
    else:
        print(f"\n❌  {mismatch} O atoms differ: geometry and state_after are inconsistent.")
        
        
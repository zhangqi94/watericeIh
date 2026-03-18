import numpy as np
from typing import Callable, Any, Optional, Tuple, Dict, List

####################################################################################################
# --------------------------------- small utils ---------------------------------
try:
    from tools import mic_vec, wrap_pos, normalize_vec
except Exception:
    from source.tools import mic_vec, wrap_pos, normalize_vec

####################################################################################################
# ---------- H-only relaxation with MACE ----------
def relax_H_with_MACE(
    atoms: Any,
    mace_inference: Callable[..., Tuple[float, np.ndarray, Any]],
    init_coords: np.ndarray,
    lr: float = 0.05,
    max_iter: int = 200,
    f_tol: float = 0.05,
    verbose: bool = True,
    return_traj: bool = False,
    create_neighborlist_device: str = "gpu",
) -> Tuple:
    """Relax hydrogens only by simple gradient descent using MACE forces.

    Update rule (F in eV/Å): X_H <- X_H + lr * F_H. Oxygens stay fixed (assumed O first).

    Args:
        atoms: ASE-like Atoms (get_chemical_symbols, get_cell, set_positions).
        mace_inference: Callable returning (E, F, extra).
        init_coords: (n_atoms, 3) initial coordinates.
        lr: Step size for H updates (Å per eV).
        max_iter: Max iterations.
        f_tol: Stop when max ||F_H|| < f_tol (eV/Å).
        verbose: If True, print progress.
        return_traj: If True, also return E_traj and Fmax_traj.

    Returns:
        If return_traj == False:
            (coords_relaxed, energy_final, forces_final)
        If return_traj == True:
            (coords_relaxed, energy_final, forces_final, E_traj, Fmax_traj)
    """
    X = np.asarray(init_coords, float).copy()
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("init_coords must be (n_atoms, 3).")

    symbols = atoms.get_chemical_symbols()
    num_O = symbols.count("O")
    if num_O <= 0 or num_O >= X.shape[0]:
        raise ValueError("Expect O first then H; found invalid O/H split.")
    L = atoms.cell.lengths()

    atoms.set_positions(X)
    E, F, _ = mace_inference(atoms, compute_force=True, create_neighborlist_device=create_neighborlist_device)

    # trajectories
    if return_traj:
        E_traj = [float(E)]
        Fmax_traj = []

    for it in range(1, max_iter + 1):
        F = np.asarray(F, float)
        if F.shape != X.shape:
            raise ValueError("mace_inference must return F of shape (n_atoms, 3).")

        FH = F[num_O:]
        maxF = 0.0 if FH.size == 0 else float(np.max(np.linalg.norm(FH, axis=1)))

        if verbose:
            print(f"[GD-H] {it:3d}  E={float(E):.8f}  max|F_H|={maxF:.6f} eV/Å")

        if return_traj:
            Fmax_traj.append(maxF)

        if maxF < f_tol:
            break

        # update coordinates
        X[num_O:] = wrap_pos(X[num_O:] + lr * FH, L)
        atoms.set_positions(X)

        E, F, _ = mace_inference(atoms, compute_force=True, create_neighborlist_device=create_neighborlist_device)

        if return_traj:
            E_traj.append(float(E))

    # final return
    if return_traj:
        return (
            X, float(E), np.asarray(F, float),
            np.asarray(E_traj, float),
            np.asarray(Fmax_traj, float),
        )
    else:
        return X, float(E), np.asarray(F, float)
    
####################################################################################################
####################################################################################################
# ---------- All-atom relaxation with MACE ----------
def relax_all_with_MACE(
    atoms: Any,
    mace_inference: Callable[..., Tuple[float, np.ndarray, Any]],
    init_coords: np.ndarray,
    lr: float = 0.05,
    max_iter: int = 200,
    f_tol: float = 0.05,
    verbose: bool = True,
    return_traj: bool = False,
    create_neighborlist_device: str = "gpu",
) -> Tuple:
    """Relax all atoms by simple gradient descent using MACE forces.

    Update rule (F in eV/Å): X <- X + lr * F.

    Args:
        atoms: ASE-like Atoms (get_chemical_symbols, get_cell, set_positions).
        mace_inference: Callable returning (E, F, extra).
        init_coords: (n_atoms, 3) initial coordinates.
        lr: Step size for coordinate updates (Å per eV).
        max_iter: Max iterations.
        f_tol: Stop when max ||F|| < f_tol (eV/Å).
        verbose: If True, print progress.
        return_traj: If True, return (E_traj, Fmax_traj).

    Returns:
        If return_traj == False:
            (coords_relaxed, energy_final, forces_final)
        If return_traj == True:
            (coords_relaxed, energy_final, forces_final, E_traj, Fmax_traj)
    """
    X = np.asarray(init_coords, float).copy()
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("init_coords must be (n_atoms, 3).")

    # Periodic cell lengths
    L = atoms.cell.lengths()

    # Initial energy & forces
    atoms.set_positions(X)
    E, F, _ = mace_inference(atoms, compute_force=True, create_neighborlist_device=create_neighborlist_device)

    # trajectory buffers
    if return_traj:
        E_traj = [float(E)]
        Fmax_traj = []

    for it in range(1, max_iter + 1):
        F = np.asarray(F, float)
        if F.shape != X.shape:
            raise ValueError("mace_inference must return F of shape (n_atoms, 3).")

        # Maximum atomic force
        maxF = float(np.max(np.linalg.norm(F, axis=1)))

        if verbose:
            print(f"[GD-all] {it:3d}  E={float(E):.8f}  max|F|={maxF:.6f} eV/Å")

        if return_traj:
            Fmax_traj.append(maxF)

        # Convergence check
        if maxF < f_tol:
            break

        # Update ALL atomic coordinates
        X = wrap_pos(X + lr * F, L)
        atoms.set_positions(X)

        # Recompute energy & forces
        E, F, _ = mace_inference(atoms, compute_force=True, create_neighborlist_device=create_neighborlist_device)

        if return_traj:
            E_traj.append(float(E))

    # Outputs
    if return_traj:
        return (
            X,
            float(E),
            np.asarray(F, float),
            np.asarray(E_traj, float),
            np.asarray(Fmax_traj, float),
        )
    else:
        return X, float(E), np.asarray(F, float)

####################################################################################################
####################################################################################################
if __name__ == "__main__":
    # ---- 0) Imports only needed for the demo/run ----
    from pathlib import Path
    import ase.io
    from ckpt import load_structure_from_json
    from potentialmace_cueq import initialize_mace_model
    from crystaltools import compute_OH_bond_lengths_angles
    import matplotlib.pyplot as plt
    
    # ---- 1) Load structure & topology ----
    init_json = Path("/home/zq/zqcodeml/watericeIh-mc-master/source/structure/initstru/sc_322_n_96_rho_933.json")
    # relax_mode = "H"
    relax_mode = "all"

    atoms, data = load_structure_from_json(str(init_json))

    coords = atoms.get_positions()                               # (n_atoms, 3), O first then H
    H_to_OO_pairs = np.asarray(data["H_to_OO_pairs"], dtype=int) # (n_H, 3): [H, O1, O2]
    state_hydrogens = np.asarray(data["state_hydrogens"], int)   # (n_H,)
    num_O = atoms.get_chemical_symbols().count("O")
    box_lengths = atoms.cell.lengths()                           # (3,), orthorhombic lengths

    print(f"[LOAD] structure={init_json.name} | O={num_O}, H={len(state_hydrogens)}")
    print(f"[LOAD] box lengths (Å): {np.round(box_lengths, 6)}")

    # ---- 2) Initialize MACE inference handle ----
    mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel251113/mace_iceIh_128x0e128x1o_r5.0_float32_seed153.model"
    
    mace_inference = initialize_mace_model(
        mace_model_path=mace_model_path,
        mace_dtype="float32",
        mace_device="cuda",
    )
    print("[MACE] model initialized.")

    # ---- 3) Geometry before relaxation ----
    distance_OH, angle_HOH = compute_OH_bond_lengths_angles(
        coords=coords,
        H_to_OO_pairs=H_to_OO_pairs,
        state_hydrogens=state_hydrogens,
        box_lengths=box_lengths,
    )
    print("\n[INIT] geometry")
    print("  O–H (Å):\n", np.round(distance_OH, 4))
    print("  H–O–H (deg):\n", np.round(angle_HOH, 3))

    if relax_mode == "H":
        # ---- 4) Relax H only (O fixed) using MACE forces ----
        relaxed_coords, E_relaxed, F_relaxed, E_traj, Fmax_traj = relax_H_with_MACE(
            atoms=atoms,
            mace_inference=mace_inference,
            init_coords=coords,
            lr=0.05,
            max_iter=100,
            f_tol=1e-4,
            verbose=True,
            return_traj=True,
        )
        print(f"\n[RELAX] done (H-only). Final energy = {E_relaxed:.6f} eV")

    elif relax_mode == "all":
        # ---- 4) Relax all atoms using MACE forces ----
        relaxed_coords, E_relaxed, F_relaxed, E_traj, Fmax_traj = relax_all_with_MACE(
            atoms=atoms,
            mace_inference=mace_inference,
            init_coords=coords,
            lr=0.01,
            max_iter=200,
            f_tol=1e-4,
            verbose=True,
            return_traj=True,
        )
        print(f"\n[RELAX] done (all atoms). Final energy = {E_relaxed:.6f} eV")
        

    # ---- 5) Geometry after relaxation ----
    distance_OH, angle_HOH = compute_OH_bond_lengths_angles(
        coords=relaxed_coords,
        H_to_OO_pairs=H_to_OO_pairs,
        state_hydrogens=state_hydrogens,
        box_lengths=box_lengths,
    )
    print("\n[FINAL] geometry")
    print("  O–H (Å):\n", np.round(distance_OH, 4))
    print("  H–O–H (deg):\n", np.round(angle_HOH, 3))

    # ---- 6) Save relaxed structure ----
    atoms.set_positions(relaxed_coords)
    # ase.io.write(str(out_vasp), atoms, format="vasp")
    # print(f"[SAVE] wrote {out_vasp.resolve()}")

    # ---- 7) (Optional) Plot relaxation trajectory ----

    # Iteration indices: energy has an initial value at step 0,
    # forces are defined from the first update step onward.
    steps_E = np.arange(len(E_traj))              # 0, 1, ..., n
    steps_F = np.arange(1, len(Fmax_traj) + 1)    # 1, 2, ..., n

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), dpi = 300, sharex=True)

    # Energy trajectory
    ax1.plot(steps_E, E_traj, marker=".")
    ax1.set_ylabel("Energy (eV)")
    ax1.set_title("H-only relaxation with MACE")

    # Max force trajectory
    ax2.plot(steps_F, Fmax_traj, marker=".")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("max |F_H| (eV/Å)")

    plt.tight_layout()
    plt.show()
    
    
import numpy as np
from typing import Callable, Any, Optional, Tuple, Dict, List

####################################################################################################
# --------------------------------- small utils ---------------------------------
try:
    from tools import mic_vec, wrap_pos, normalize_vec
    from crystalrelaxcoord import relax_H_with_MACE, relax_all_with_MACE
except Exception:
    from source.tools import mic_vec, wrap_pos, normalize_vec
    from source.crystalrelaxcoord import relax_H_with_MACE, relax_all_with_MACE
    
####################################################################################################

def relax_cell_abc_with_MACE(
    atoms: Any,
    mace_inference: Callable[..., Tuple[float, np.ndarray, Any]],
    init_coords: np.ndarray,
    lr: float = 1.0e-3,
    max_iter: int = 50,
    stress_tol: float = 1.0e-3,
    target_stress: Optional[np.ndarray] = None,
    return_traj: bool = False,
    verbose: bool = True,
    create_neighborlist_device: str = "gpu",
) -> Tuple:
    """Relax orthorhombic cell lengths (a,b,c) using diagonal stress (xx,yy,zz).

    Update rule: L <- L * (1 - lr * (stress_diag - target)).
    Positions are scaled with the cell at each step.
    Stress and target_stress are in eV/Å^3.

    Returns:
        If return_traj == False:
            (coords_relaxed, box_lengths, stress_vec)
        If return_traj == True:
            (coords_relaxed, box_lengths, stress_vec, E_traj, stress_traj, abc_traj, density_traj)
    """
    X = np.asarray(init_coords, float).copy()
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("init_coords must be (n_atoms, 3).")

    atoms.set_positions(X)

    target = np.zeros(3, dtype=float) if target_stress is None else np.asarray(target_stress, float)
    if target.shape != (3,):
        raise ValueError("target_stress must be shape (3,) for xx, yy, zz.")

    if return_traj:
        E_traj: List[float] = []
        stress_traj: List[np.ndarray] = []
        abc_traj: List[np.ndarray] = []
        density_traj: List[float] = []

    last_stress = np.zeros(6, dtype=float)
    for it in range(1, max_iter + 1):
        E, _, stress_vec = mace_inference(
            atoms, compute_force=True, create_neighborlist_device=create_neighborlist_device
        )
        last_stress = np.asarray(stress_vec, float)
        s = last_stress[:3] - target
        max_s = float(np.max(np.abs(s)))

        if return_traj:
            L = atoms.cell.lengths()
            volume_m3 = float(atoms.get_volume()) * 1.0e-30
            mass_total_kg = float(np.sum(atoms.get_masses())) * 1.660539066e-27
            density = mass_total_kg / volume_m3
            E_traj.append(float(E))
            stress_traj.append(last_stress.copy())
            abc_traj.append(L.copy())
            density_traj.append(float(density))

        if verbose:
            print(f"[CELL] {it:3d}  stress_diag={s}  max|stress|={max_s:.6f} eV/Å^3")

        if max_s < stress_tol:
            break

        L = atoms.cell.lengths()
        scale = 1.0 - lr * s
        if np.any(scale <= 0.0):
            raise ValueError("Cell update produced non-positive length. Reduce lr.")

        L = L * scale
        atoms.set_cell(np.diag(L), scale_atoms=True)
        X = atoms.get_positions()

    if return_traj:
        return (
            X,
            atoms.cell.lengths(),
            last_stress,
            np.asarray(E_traj, float),
            np.asarray(stress_traj, float),
            np.asarray(abc_traj, float),
            np.asarray(density_traj, float),
        )
    else:
        return X, atoms.cell.lengths(), last_stress


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
    # init_json = Path("/home/zq/zqcodeml/watericeIh-mc-master/source/structure/initstru/sc_322_n_96_rho_933.json")

    init_json = Path("/home/zq/zqcodeml/watericeIh_data/fig_energy_hist_voerlap/sc_422_n_128_rho_933_state_10000_relax.json")


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
    mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_cueq.model"
    
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

    # ---- 4.5) Cell relaxation using stress (diagonal a,b,c only) ----
    # target stress: 1e-4 GPa (approx. 1 atm) -> eV/Å^3
    target_stress_gpa = 1.0e-4
    gpa_to_ev_a3 = 1.0 / 160.21766208
    target_stress = np.array([target_stress_gpa, target_stress_gpa, target_stress_gpa]) * gpa_to_ev_a3

    (
        relaxed_coords,
        box_lengths,
        stress_vec,
        E_cell_traj,
        stress_cell_traj,
        abc_traj,
        density_traj,
    ) = relax_cell_abc_with_MACE(
        atoms=atoms,
        mace_inference=mace_inference,
        init_coords=relaxed_coords,
        lr=1.0e-1,
        max_iter=300,
        stress_tol=1.0e-8,
        target_stress=target_stress,
        return_traj=True,
        verbose=True,
    )

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
    ax1.set_title("relaxation with MACE")

    # Max force trajectory
    ax2.plot(steps_F, Fmax_traj, marker=".")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("max |F_H| (eV/Å)")

    plt.tight_layout()
    plt.show()

    # ---- 8) Plot cell relaxation trajectories ----
    it_cell = np.arange(len(E_cell_traj))

    fig, axes = plt.subplots(4, 1, figsize=(7, 10), dpi=300, sharex=True)

    axes[0].plot(it_cell, E_cell_traj, marker=".")
    axes[0].set_ylabel("Energy (eV)")
    axes[0].set_title("Cell relaxation trajectories")

    axes[1].plot(it_cell, stress_cell_traj[:, 0], label="Sxx")
    axes[1].plot(it_cell, stress_cell_traj[:, 1], label="Syy")
    axes[1].plot(it_cell, stress_cell_traj[:, 2], label="Szz")
    axes[1].set_ylabel("Stress (eV/Å^3)")
    axes[1].legend()

    axes[2].plot(it_cell, abc_traj[:, 0], label="a")
    axes[2].plot(it_cell, abc_traj[:, 1], label="b")
    axes[2].plot(it_cell, abc_traj[:, 2], label="c")
    axes[2].set_ylabel("Cell length (Å)")
    axes[2].legend()

    axes[3].plot(it_cell, density_traj, marker=".")
    axes[3].set_ylabel("Density (kg/m^3)")
    axes[3].set_xlabel("Iteration")

    plt.tight_layout()
    plt.show()
    
    

    print(f"\n[CELL] final abc (Å): {np.round(box_lengths, 6)}")
    print(f"[CELL] final density (kg/m^3): {density_traj[-1]:.6f}")
    print(f"[CELL] final stress (eV/Å^3): {np.array2string(stress_vec, precision=6)}")

    ev_a3_to_gpa = 160.21766208

    print(f"\n[CELL] final abc (Å): {np.round(box_lengths, 6)}")
    print(f"[CELL] final density (kg/m^3): {density_traj[-1]:.6f}")

    # stress_vec is 6 components (xx, yy, zz, yz, xz, xy) in eV/Å^3
    stress_gpa = np.asarray(stress_vec, float) * ev_a3_to_gpa

    print(f"[CELL] final stress (eV/Å^3): {np.array2string(stress_vec, precision=6)}")
    print(f"[CELL] final stress (GPa):    {np.array2string(stress_gpa, precision=6)}")
    print(f"[CELL] final stress diag (GPa): {np.array2string(stress_gpa[:3], precision=6)}")
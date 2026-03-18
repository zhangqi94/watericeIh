import numpy as np
import ase
import ase.io
from typing import Callable, Any, Tuple, Dict, List

try:
    import units
    from tools import mic_vec, wrap_pos, normalize_vec
    from crystaltools import two_nearest_H_per_O, compute_OH_bond_lengths_angles
except Exception:
    from source import units
    from source.tools import mic_vec, wrap_pos, normalize_vec
    from source.crystaltools import two_nearest_H_per_O, compute_OH_bond_lengths_angles

####################################################################################################
## useful functions
def suggest_mc_width(T: float, T_ref: float = 100.0, width_ref: float = 0.010) -> float:
    """Suggest an MC proposal width based on temperature.

    Args:
        T: Target temperature in Kelvin.
        T_ref: Reference temperature in Kelvin.
        width_ref: Reference width in Angstrom at T_ref.

    Returns:
        Suggested proposal width in Angstrom.
    """
    return float(width_ref) * np.sqrt(float(T) / float(T_ref))


####################################################################################################
def make_metropolis_mala_update_functions(
    num_O: int,
    mace_inference: Callable,
    create_neighborlist_device: str = "gpu"
) -> Callable:
    """Build a single-step MALA updater that moves only hydrogens (O fixed).

    Proposal on H rows only:
        x' = x + 0.5 * sigma^2 * (beta * F_old) + sigma * N(0, I)
    Acceptance uses the standard MALA asymmetric correction with MIC displacements.

    Assumptions:
        1) Global ordering is O first, H after (N = num_O + num_H).
        2) Orthorhombic PBC; use project-provided `wrap_pos` and `mic_vec`.

    Args:
        num_O: Number of oxygen atoms (0 < num_O < N_atoms).
        mace_inference: Callable (atoms, coords, compute_force=True)
            -> (E_eV, F_eV_per_Angstrom, stress_vec).

    Returns:
        Callable:
            mala_step(energy_curr, force_curr, stress_curr, temperature_in_eV, atoms,
                      mc_width=0.05)
            -> (energy_new, force_new, stress_new, atoms, accepted).
    """
    
    def _log_q_diag(diff, F_old_red, sigma_tensor):
        """
        Compute log q(x'|x) for a diagonal-covariance Langevin proposal:
        x' = x + 0.5 * sigma^2 * F_old_red + sigma * N(0, I)
        where all arrays are (..., 3) shaped and operations are elementwise.

        diff        : MIC(x' - x)         [Å]
        F_old_red   : beta * F_old         [1/Å]
        sigma_tensor: per-dimension sigma  [Å]

        Only sums over components where sigma>0.
        """
        # valid mask for moved dims
        move = (sigma_tensor > 0.0)
        if not np.any(move):
            return 0.0  # no moved dofs -> degenerate, but fine

        # Only slice moved components before doing division
        diff_m  = diff[move]
        Fm      = F_old_red[move]
        sigma_m = sigma_tensor[move]

        var_m   = sigma_m**2
        mean_m  = 0.5 * var_m * Fm
        z_m     = (diff_m - mean_m) / sigma_m

        z2_sum  = np.sum(z_m * z_m)
        logdet  = np.sum(np.log(2.0 * np.pi * var_m))

        return -0.5 * (z2_sum + logdet)

    #===============================================================================================
    # --- Build full MALA updater that only moves H atoms ---
    def mala_step_only_hydrogens(
        energy_curr: float,
        force_curr: np.ndarray,     # (N_atoms, 3), eV/Angstrom
        stress_curr: np.ndarray,    # (6,), eV/Angstrom^3
        temperature_in_eV: float,   # k_B T in eV (> 0)
        atoms: ase.Atoms,
        mc_width: float = 0.05,     # sigma in Angstrom
    ) -> Tuple[float, np.ndarray, np.ndarray, ase.Atoms, bool]:
        """
        Perform one MALA update step for the hydrogen atoms.

        Only hydrogen positions are moved (oxygen atoms remain fixed).
        The proposal follows a Langevin-like drift + noise scheme and is
        accepted or rejected using the standard MALA asymmetric correction.

        Args:
            energy_curr: Current total potential energy (eV).
            force_curr: Current atomic forces (eV/Angstrom), shape (N_atoms, 3).
            stress_curr: Current stress vector (6,) in eV/Angstrom^3.
            temperature_in_eV: Thermal energy k_B*T in eV (must be > 0).
            atoms: ASE Atoms object (positions updated in-place).
            mc_width: Proposal width (sigma) for MALA step (Angstrom).

        Returns:
            Tuple containing:
                E_new (float): Updated potential energy (eV).
                F_new (np.ndarray): Updated atomic forces (eV/Angstrom).
                stress_new (np.ndarray): Updated stress vector (6,) in eV/Angstrom^3.
                atoms (ase.Atoms): Atoms object with updated coordinates.
                accepted (bool): True if proposal accepted, else False.
        """
        # 1) Validate & prepare
        coords = np.asarray(atoms.get_positions(), dtype=float)
        L = np.asarray(atoms.cell.lengths(), dtype=float)
        F_old  = np.asarray(force_curr, dtype=float)
        S_old  = np.asarray(stress_curr, dtype=float)
        E_old  = float(energy_curr)

        if F_old.shape != coords.shape:
            raise ValueError("F_old must have the same shape as coords.")

        beta  = 1.0 / float(temperature_in_eV)
        sigma = float(mc_width)

        # Build H-only sigma tensor at runtime (O fixed, H move)
        # N = coords.shape[0]
        sigma_tensor = np.zeros_like(coords, dtype=float)  # (N,3)
        sigma_tensor[num_O:, :] = sigma                    # H rows: σ; O rows: 0

        # 2) Propose
        F_old_red = beta * F_old
        eta = np.random.normal(0.0, 1.0, size=coords.shape)
        drift = 0.5 * (sigma_tensor**2) * F_old_red
        step  = drift + sigma_tensor * eta
        coords_prop = wrap_pos(coords + step, L)

        # 3) Evaluate
        # Evaluate at the proposed coordinates (use a copy to avoid side effects)
        atoms_prop = atoms.copy()
        atoms_prop.set_positions(coords_prop)
        E_prop, F_prop, S_prop = mace_inference(atoms_prop, compute_force=True, create_neighborlist_device=create_neighborlist_device)
        F_prop = np.asarray(F_prop, dtype=float)
        S_prop = np.asarray(S_prop, dtype=float)
        if F_prop.shape != coords.shape:
            raise ValueError("mace_inference returned forces with unexpected shape.")

        F_prop_red = beta * F_prop

        # 4) Robust asymmetric correction via explicit log q
        # Forward: q(x'|x)
        diff_fwd = mic_vec(coords_prop - coords, L)
        log_q_fwd = _log_q_diag(diff_fwd, F_old_red, sigma_tensor)

        # Backward: q(x|x')
        diff_bwd = mic_vec(coords - coords_prop, L)
        log_q_bwd = _log_q_diag(diff_bwd, F_prop_red, sigma_tensor)

        # MH test
        dE = float(E_prop - E_old)
        log_alpha = (log_q_bwd - log_q_fwd) - beta * dE
        accepted = bool(np.log(np.random.rand()) < log_alpha)

        if accepted:
            atoms.set_positions(coords_prop)
            return float(E_prop), F_prop, S_prop, atoms, True
        else:
            atoms.set_positions(coords)
            return float(E_old), F_old, S_old, atoms, False

    #===============================================================================================
    # --- Build full MALA updater that moves all atoms (O and H) ---
    #===============================================================================================
    # --- Move BOTH O and H with per-species scaling: s_O=0.25, s_H=1.0 (no mask) ---
    def mala_step_all_atoms(
        energy_curr: float,
        force_curr: np.ndarray,     # (N_atoms, 3), eV/Angstrom
        stress_curr: np.ndarray,    # (6,), eV/Angstrom^3
        temperature_in_eV: float,   # k_B T in eV (> 0)
        atoms: ase.Atoms,
        mc_width: float = 0.05,     # interpret as sigma_H in Angstrom
    ) -> Tuple[float, np.ndarray, np.ndarray, ase.Atoms, bool]:

        # 1) Validate & prepare
        coords = np.asarray(atoms.get_positions(), dtype=float)
        L = np.asarray(atoms.cell.lengths(), dtype=float)
        F_old  = np.asarray(force_curr, dtype=float)
        S_old  = np.asarray(stress_curr, dtype=float)
        E_old  = float(energy_curr)

        if F_old.shape != coords.shape:
            raise ValueError("F_old must have the same shape as coords.")

        beta  = 1.0 / float(temperature_in_eV)
        sigma_H = float(mc_width)
        sigma_O = 0.25 * sigma_H  # per your default

        # Build per-species sigma tensor at runtime
        # N = coords.shape[0]
        sigma_tensor = np.empty_like(coords, dtype=float)  # (N,3)
        sigma_tensor[:num_O, :]  = sigma_O
        sigma_tensor[num_O:, :]  = sigma_H

        # 2) Propose
        F_old_red = beta * F_old
        eta = np.random.normal(0.0, 1.0, size=coords.shape)
        drift = 0.5 * (sigma_tensor**2) * F_old_red
        step  = drift + sigma_tensor * eta
        coords_prop = wrap_pos(coords + step, L)

        # 3) Evaluate
        # Evaluate at the proposed coordinates (use a copy to avoid side effects)
        atoms_prop = atoms.copy()
        atoms_prop.set_positions(coords_prop)
        E_prop, F_prop, S_prop = mace_inference(atoms_prop, compute_force=True, create_neighborlist_device=create_neighborlist_device)
        F_prop = np.asarray(F_prop, dtype=float)
        S_prop = np.asarray(S_prop, dtype=float)
        if F_prop.shape != coords.shape:
            raise ValueError("mace_inference returned forces with unexpected shape.")

        F_prop_red = beta * F_prop

        # 4) Robust asymmetric correction
        diff_fwd = mic_vec(coords_prop - coords, L)
        log_q_fwd = _log_q_diag(diff_fwd, F_old_red, sigma_tensor)

        diff_bwd = mic_vec(coords - coords_prop, L)
        log_q_bwd = _log_q_diag(diff_bwd, F_prop_red, sigma_tensor)

        dE = float(E_prop - E_old)
        log_alpha = (log_q_bwd - log_q_fwd) - beta * dE
        accepted = bool(np.log(np.random.rand()) < log_alpha)

        if accepted:
            atoms.set_positions(coords_prop)
            return float(E_prop), F_prop, S_prop, atoms, True
        else:
            atoms.set_positions(coords)
            return float(E_old), F_old, S_old, atoms, False

    return mala_step_only_hydrogens, mala_step_all_atoms

####################################################################################################
####################################################################################################
if __name__ == "__main__":
    
    # ----------------------------------------------------------------------------------------------
    # Load initial structure and metadata from JSON file
    import time
    from datetime import datetime
    from ckpt import load_structure_from_json
    from createcrystal import classify_h_by_oxygen
    from potentialmace_cueq import initialize_mace_model

    # Path to initial structure JSON file
    initial_structure_file = (
        "/home/zq/zqcodeml/watericeIh-mc-master/source/structure/initstru/sc_322_n_96.json"
    )

    # Load ASE Atoms object and metadata dict
    atoms, data = load_structure_from_json(initial_structure_file)

    # Extract coordinates and structure information
    coords = atoms.get_positions()                         # (n_atoms, 3)
    O_neighbors = data["O_neighbors"]                      # (n_O, 4) or (n_O, 5) with first column O
    H_to_OO_pairs = data["H_to_OO_pairs"]                  # (n_H, 3)
    state_hydrogens = data["state_hydrogens"]              # (n_H,)
    num_O = atoms.get_chemical_symbols().count("O")        # number of oxygen atoms

    # Orthorhombic box lengths (Å)
    box_lengths = atoms.cell.lengths()  
    
    print(f"[LOAD] Loaded structure: {num_O} O atoms, {len(state_hydrogens)} H atoms.")
    print(f"[LOAD] Box lengths (Å): {box_lengths}")

    # ----------------------------------------------------------------------------------------------
    # Prepare MACE model and initial energy
    # ----------------------------------------------------------------------------------------------
    mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel251125/mace_iceIh_128x0e128x1o_r4.5_float32_seed144_cueq.model"
    mace_dtype = "float32"
    mace_device = "cuda"

    mace_inference = initialize_mace_model(
        mace_model_path,
        mace_dtype,
        mace_device,
    )

    # Initial energy/force @ current coords
    coords = atoms.get_positions()
    coords = coords + 0.01 * np.random.randn(*coords.shape)
    atoms.set_positions(coords)
    energy_curr, force_curr, stress_curr = mace_inference(atoms, compute_force=True)
    print(f"[INIT] Energy = {energy_curr:.12f} eV")
    print(f"[INIT] Stress (eV/Angstrom^3) = {np.array2string(np.asarray(stress_curr, float), precision=6)}")

    # ----------------------------------------------------------------------------------------------
    # Build Monte Carlo mala test (MALA on H with O fixed)
    # ----------------------------------------------------------------------------------------------
    mala_step_only_hydrogens, mala_step_all_atoms = make_metropolis_mala_update_functions(
        num_O=num_O,
        mace_inference=mace_inference,
    )
    # mala_step = mala_step_only_hydrogens
    mala_step = mala_step_all_atoms

    # ----------------------------------------------------------------------------------------------
    # Monte Carlo mala test (MALA on H with O fixed)
    # ----------------------------------------------------------------------------------------------
    energy_curr = float(energy_curr)
    force_curr = np.asarray(force_curr, dtype=float)
    stress_curr = np.asarray(stress_curr, dtype=float)

    # Calculate pressure using the tools function
    from tools import calculate_pressure_from_stress
    num_atoms = len(coords)
    volume = np.prod(atoms.cell.lengths())
    temperature_K = 100.0       # simulation temperature (K)
    # temperature_K = 20.0       # simulation temperature (K)
    k_B_eV_per_K = 8.617333262145e-5
    temperature_in_eV = temperature_K * k_B_eV_per_K

    P_kinetic, P_virial, pressure_eV_A3 = calculate_pressure_from_stress(
        stress=stress_curr,
        volume=volume,
        num_atoms=num_atoms,
        temperature_in_eV=temperature_in_eV,
    )
    pressure_curr = pressure_eV_A3 / units.GPA_TO_EV_PER_ANGSTROM3  # Convert to GPa
    stress_diag_curr = stress_curr[:3] / units.GPA_TO_EV_PER_ANGSTROM3  # Diagonal components in GPa

    num_steps = 500             # total MC steps
    
    mc_width = suggest_mc_width(temperature_K)
    print(f"[MC] Using MC width = {mc_width:.6f} Å")
    
    accepts = 0
    attempts = 0

    energy_traj = [energy_curr]
    stress_traj = [stress_curr.copy()]
    pressure_traj = [pressure_curr]
    stress_diag_traj = [np.asarray(stress_diag_curr, dtype=float).copy()]
    volume_traj = [volume]
    print_interval = 20

    t0 = time.time()
    for step in range(1, num_steps + 1):
        energy_curr, force_curr, stress_curr, atoms, accepted = mala_step(
            energy_curr=energy_curr,
            force_curr=force_curr,
            stress_curr=stress_curr,
            temperature_in_eV=float(temperature_in_eV),
            atoms=atoms,
            mc_width=mc_width,
        )
        accepts += int(accepted)
        attempts += 1
        energy_traj.append(energy_curr)
        stress_traj.append(np.asarray(stress_curr, dtype=float).copy())

        # Calculate pressure for current step
        volume = np.prod(atoms.cell.lengths())
        P_kinetic, P_virial, pressure_eV_A3 = calculate_pressure_from_stress(
            stress=stress_curr,
            volume=volume,
            num_atoms=num_atoms,
            temperature_in_eV=temperature_in_eV,
        )
        pressure_curr = pressure_eV_A3 / units.GPA_TO_EV_PER_ANGSTROM3  # Convert to GPa
        stress_diag_curr = stress_curr[:3] / units.GPA_TO_EV_PER_ANGSTROM3  # Diagonal components in GPa

        pressure_traj.append(pressure_curr)
        stress_diag_traj.append(np.asarray(stress_diag_curr, dtype=float).copy())
        volume_traj.append(volume)

        if print_interval > 0 and ((step % print_interval == 0) or (step == 1) or (step == num_steps)):
            acc_rate = accepts / attempts
            print(f"[MALA] step={step:6d}/{num_steps:6d}  E={energy_curr:.12f} eV  acc_rate={acc_rate:5.3f}")

    t1 = time.time()
    print(f"[DONE] Steps={attempts}, Accepts={accepts}, Final acc_rate={accepts/attempts:5.3f}, Elapsed={t1 - t0:.2f} s")

    # Calculate density trajectory
    mass_H2O_g = units.calculate_mass_h2o_g(1)
    volume_traj_arr = np.array(volume_traj)
    density_traj = (num_O * mass_H2O_g) / (volume_traj_arr * units.ANGSTROM3_TO_CM3)
    final_density = density_traj[-1]

    print("\n[MC Summary]")
    print(f"  Steps attempted : {attempts}")
    print(f"  Steps accepted  : {accepts}")
    print(f"  Acceptance rate : {accepts/attempts:.3f}")
    print(f"  Final energy    : {energy_curr:.12f} eV")
    print(f"  Final pressure  : {pressure_curr:.6f} GPa")
    print(f"  Final stress diag (GPa): {np.array2string(stress_diag_curr, precision=6)}")
    print(f"  Final density   : {final_density:.4f} g/cm³")
    print(f"  Temperature     : {temperature_K:.1f} K  (kT = {temperature_in_eV:.6f} eV)")
    print(f"  Total time      : {t1 - t0:.2f} s")

    # ----------------------------------------------------------------------------------------------
    # Compute bond lengths and angles for the final structure
    # ----------------------------------------------------------------------------------------------
    # coords_final = atoms.get_positions()
    # box_lengths_final = atoms.cell.lengths()
    # distance_OH, angle_HOH = compute_OH_bond_lengths_angles(
    #     coords=coords_final,
    #     H_to_OO_pairs=H_to_OO_pairs,
    #     state_hydrogens=state_hydrogens,
    #     box_lengths=box_lengths_final,
    # )
    # print("\n=== From updated structure ===")
    # print("O–H bond lengths (Å):\n", np.round(distance_OH, 4))
    # print("H–O–H angles (deg):\n", np.round(angle_HOH, 3))

    # ----------------------------------------------------------------------------------------------
    # Plot energy trajectory
    # ----------------------------------------------------------------------------------------------
    import matplotlib.pyplot as plt

    energy_traj = np.array(energy_traj)
    pressure_traj = np.array(pressure_traj, dtype=float)
    stress_diag_traj = np.asarray(stress_diag_traj, dtype=float)

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(np.arange(len(energy_traj)), energy_traj / num_O, ".-", lw=1)
    plt.xlabel("Monte Carlo step", fontsize=12)
    plt.ylabel("Energy per H₂O (eV)", fontsize=12)
    # plt.title(f"{temperature_K:.0f} K", fontsize=13)
    plt.title(f"H₂O={num_O} | T={temperature_K:.0f}K | acc={acc_rate:.3f}", fontsize=13)
    plt.grid(True, ls="--", alpha=0.5)
    # plt.ylim([-0.330, -0.290])
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------------------------------------
    # Plot pressure trajectory (GPa)
    # ----------------------------------------------------------------------------------------------
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(np.arange(len(pressure_traj)), pressure_traj, ".-", lw=1)
    plt.xlabel("Monte Carlo step", fontsize=12)
    plt.ylabel("Pressure (GPa)", fontsize=12)
    plt.title(f"Pressure | T={temperature_K:.0f}K | acc={acc_rate:.3f}", fontsize=13)
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------------------------------------
    # Plot diagonal stress trajectory (GPa)
    # ----------------------------------------------------------------------------------------------
    plt.figure(figsize=(8, 4), dpi=300)
    steps = np.arange(len(stress_diag_traj))
    plt.plot(steps, stress_diag_traj[:, 0], ".-", lw=1, label="Sxx")
    plt.plot(steps, stress_diag_traj[:, 1], ".-", lw=1, label="Syy")
    plt.plot(steps, stress_diag_traj[:, 2], ".-", lw=1, label="Szz")
    plt.xlabel("Monte Carlo step", fontsize=12)
    plt.ylabel("Stress diag (GPa)", fontsize=12)
    plt.title(f"Stress diag | T={temperature_K:.0f}K | acc={acc_rate:.3f}", fontsize=13)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------------------------------------
    # Plot density trajectory (g/cm³)
    # ----------------------------------------------------------------------------------------------
    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(np.arange(len(density_traj)), density_traj, ".-", lw=1)
    plt.xlabel("Monte Carlo step", fontsize=12)
    plt.ylabel("Density (g/cm³)", fontsize=12)
    plt.title(f"Density | T={temperature_K:.0f}K | acc={acc_rate:.3f}", fontsize=13)
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()



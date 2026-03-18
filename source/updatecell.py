import numpy as np
import ase
import ase.io
from typing import Callable, Any, Tuple, Dict, List

try:
    from tools import mic_vec, wrap_pos, normalize_vec
    import units
except Exception:
    from source.tools import mic_vec, wrap_pos, normalize_vec
    from source import units

####################################################################################################
## useful functions
def suggest_cell_mc_width(T, volume, T_ref=100.0, V_ref=11258.135, width_ref=0.002):
    """Suggest MC width for cell parameter updates scaled by temperature and cell volume.

    Args:
        T: Temperature in K.
        volume: Current cell volume in Angstrom^3.
        T_ref: Reference temperature in K.
        V_ref: Reference volume in Angstrom^3 for scaling (default from your chosen run).
        width_ref: Reference width (fractional change in cell parameter).

    Returns:
        Suggested MC width (fractional).
    """
    T_scale = np.sqrt(float(T) / float(T_ref))
    V_scale = (float(V_ref) / float(volume)) ** (1.0 / 3.0)
    return float(width_ref) * T_scale * V_scale

####################################################################################################
def make_metropolis_cell_update_functions(
    mace_inference: Callable,
    create_neighborlist_device: str = "gpu"
) -> Callable:
    """Build a single-step cell updater for orthorhombic lattice (updates a, b, c).

    Proposal:
        For isotropic: scale all three dimensions by same factor
        For anisotropic: scale each dimension independently

    When cell changes, atomic positions are scaled accordingly to maintain
    fractional coordinates.

    Assumptions:
        1) Orthorhombic cell (only a, b, c vary; angles fixed at 90°)
        2) Use project-provided `wrap_pos` for PBC

    Args:
        mace_inference: Callable (atoms, compute_force=True)
            -> (E_eV, F_eV_per_Angstrom, stress_vec).

    Returns:
        Callable:
            cell_step(energy_curr, force_curr, stress_curr, temperature_in_eV, atoms,
                      pressure_eV_A3=0.0, mc_width=0.001, output_force_stress=True)
            -> (energy_new, force_new, stress_new, atoms, accepted).
    """

    #===============================================================================================
    # --- Isotropic cell update: scale all dimensions by same factor ---
    #===============================================================================================
    def cell_step_isotropic(
        energy_curr: float,
        force_curr: np.ndarray,     # (N_atoms, 3), eV/Angstrom
        stress_curr: np.ndarray,    # (6,), eV/Angstrom^3
        temperature_in_eV: float,   # k_B T in eV (> 0)
        atoms: ase.Atoms,
        pressure_eV_A3: float = 0.0,  # External pressure in eV/Angstrom^3
        mc_width: float = 0.001,    # width in log-space for cell scaling
        output_force_stress: bool = True,
    ):
        """
        Perform one isotropic cell update step.

        All three cell parameters (a, b, c) are scaled by the same factor.
        Atomic positions are scaled to maintain fractional coordinates.

        Args:
            energy_curr: Current total potential energy (eV).
            force_curr: Current atomic forces (eV/Angstrom), shape (N_atoms, 3).
            stress_curr: Current stress vector (6,) in eV/Angstrom^3.
            temperature_in_eV: Thermal energy k_B*T in eV (must be > 0).
            atoms: ASE Atoms object (cell and positions updated in-place).
            pressure_eV_A3: External pressure in eV/Angstrom^3 (default 0.0 for NVT-like).
            mc_width: Proposal width (fractional) for cell scaling.
            output_force_stress: If True, compute and return force and stress for accepted moves.

        Returns:
            Tuple containing:
                energy_new (float): Updated potential energy (eV).
                force_new (np.ndarray): Updated atomic forces (eV/Angstrom).
                stress_new (np.ndarray): Updated stress vector (6,) in eV/Angstrom^3.
                atoms (ase.Atoms): Atoms object with updated cell and coordinates.
                accepted (bool): True if proposal accepted, else False.

        Notes:
            If output_force_stress is False, force_new and stress_new are returned
            as the input values without recomputation.
        """
        P_eV_per_A3 = float(pressure_eV_A3)

        # 1) Validate & prepare - get coords from atoms
        coords = atoms.get_positions()
        coords = np.asarray(coords, dtype=float)
        E_old = float(energy_curr)
        F_old = np.asarray(force_curr, dtype=float)
        S_old = np.asarray(stress_curr, dtype=float)
        N_atoms = coords.shape[0]

        beta = 1.0 / float(temperature_in_eV)
        sigma = float(mc_width)

        # Get current cell parameters
        L_old = np.asarray(atoms.cell.lengths(), dtype=float)  # (a, b, c)
        V_old = np.prod(L_old)  # volume

        # 2) Propose isotropic scaling in log-space: scale = exp(delta_log)
        # Symmetric in ln(scale) => Hastings term becomes +ln(V_new/V_old)
        delta_log = np.random.normal(0.0, sigma)
        scale = np.exp(delta_log)

        # Reject if scale would make cell invalid
        if scale <= 0.0:
            return float(E_old), F_old, S_old, atoms, False

        L_new = L_old * scale
        V_new = np.prod(L_new)

        # Scale atomic positions to maintain fractional coordinates
        # frac = coords / L_old, then coords_new = frac * L_new
        frac_coords = coords / L_old
        coords_prop = frac_coords * L_new
        coords_prop = wrap_pos(coords_prop, L_new)

        # Update atoms object with new cell
        atoms_prop = atoms.copy()
        atoms_prop.set_cell([L_new[0], L_new[1], L_new[2]], scale_atoms=False)
        atoms_prop.set_positions(coords_prop)

        # 3) Evaluate energy (and optionally force/stress) at new configuration
        E_prop, F_prop, S_prop = mace_inference(
            atoms_prop,
            compute_force=output_force_stress,
            create_neighborlist_device=create_neighborlist_device,
        )
        E_prop = float(E_prop)
        if output_force_stress:
            F_prop = np.asarray(F_prop, dtype=float)
            S_prop = np.asarray(S_prop, dtype=float)
        else:
            F_prop = F_old
            S_prop = S_old

        # 4) Metropolis acceptance with NPT ensemble
        # ΔH = ΔE + P*ΔV - N*kT*ln(V_new/V_old)
        # For isotropic scaling in 3D: ln(V_new/V_old) = 3*ln(scale)
        dE = E_prop - E_old
        dV = V_new - V_old
        log_V_ratio = 3.0 * np.log(scale)

        # Acceptance criterion: exp(-β * ΔH)
        # log-space proposal => extra +ln(V_new/V_old) Hastings term
        # ΔH = ΔE + P*ΔV - (N+1) * kT * ln(V_new/V_old)
        delta_H = (
            dE
            + P_eV_per_A3 * dV
            - (N_atoms + 1) * temperature_in_eV * log_V_ratio
        )
        log_alpha = -beta * delta_H

        accepted = bool(np.log(np.random.rand()) < log_alpha)

        if accepted:
            atoms.set_cell([L_new[0], L_new[1], L_new[2]], scale_atoms=False)
            atoms.set_positions(coords_prop)
            return float(E_prop), F_prop, S_prop, atoms, True
        else:
            atoms.set_positions(coords)
            return float(E_old), F_old, S_old, atoms, False

    #===============================================================================================
    # --- Anisotropic cell update: scale each dimension independently ---
    #===============================================================================================
    def cell_step_anisotropic(
        energy_curr: float,
        force_curr: np.ndarray,     # (N_atoms, 3), eV/Angstrom
        stress_curr: np.ndarray,    # (6,), eV/Angstrom^3
        temperature_in_eV: float,   # k_B T in eV (> 0)
        atoms: ase.Atoms,
        pressure_eV_A3: float = 0.0,  # External pressure in eV/Angstrom^3
        mc_width: float = 0.001,    # width in log-space for cell scaling
        output_force_stress: bool = True,
    ):
        """
        Perform one anisotropic cell update step.

        Each cell parameter (a, b, c) is scaled independently.
        Atomic positions are scaled to maintain fractional coordinates.

        Args:
            energy_curr: Current total potential energy (eV).
            force_curr: Current atomic forces (eV/Angstrom), shape (N_atoms, 3).
            stress_curr: Current stress vector (6,) in eV/Angstrom^3.
            temperature_in_eV: Thermal energy k_B*T in eV (must be > 0).
            atoms: ASE Atoms object (cell and positions updated in-place).
            pressure_eV_A3: External pressure in eV/Angstrom^3 (default 0.0 for NVT-like).
            mc_width: Proposal width (fractional) for cell scaling.
            output_force_stress: If True, compute and return force and stress for accepted moves.

        Returns:
            Tuple containing:
                energy_new (float): Updated potential energy (eV).
                force_new (np.ndarray): Updated atomic forces (eV/Angstrom).
                stress_new (np.ndarray): Updated stress vector (6,) in eV/Angstrom^3.
                atoms (ase.Atoms): Atoms object with updated cell and coordinates.
                accepted (bool): True if proposal accepted, else False.

        Notes:
            If output_force_stress is False, force_new and stress_new are returned
            as the input values without recomputation.
        """
        P_eV_per_A3 = float(pressure_eV_A3)

        # 1) Validate & prepare - get coords from atoms
        coords = atoms.get_positions()
        coords = np.asarray(coords, dtype=float)
        E_old = float(energy_curr)
        F_old = np.asarray(force_curr, dtype=float)
        S_old = np.asarray(stress_curr, dtype=float)
        N_atoms = coords.shape[0]

        beta = 1.0 / float(temperature_in_eV)
        sigma = float(mc_width)

        # Get current cell parameters
        L_old = np.asarray(atoms.cell.lengths(), dtype=float)  # (a, b, c)
        V_old = np.prod(L_old)  # volume

        # 2) Propose anisotropic scaling in log-space: scale = exp(delta_log)
        # Symmetric in ln(scale_i) => Hastings term becomes +ln(V_new/V_old)
        delta_log = np.random.normal(0.0, sigma, size=3)
        scale = np.exp(delta_log)  # (scale_a, scale_b, scale_c)

        # Reject if any scale would make cell invalid
        if np.any(scale <= 0.0):
            return float(E_old), F_old, S_old, atoms, False

        L_new = L_old * scale
        V_new = np.prod(L_new)

        # Scale atomic positions to maintain fractional coordinates
        frac_coords = coords / L_old
        coords_prop = frac_coords * L_new
        coords_prop = wrap_pos(coords_prop, L_new)

        # Update atoms object with new cell
        atoms_prop = atoms.copy()
        atoms_prop.set_cell([L_new[0], L_new[1], L_new[2]], scale_atoms=False)
        atoms_prop.set_positions(coords_prop)

        # 3) Evaluate energy (and optionally force/stress) at new configuration
        E_prop, F_prop, S_prop = mace_inference(
            atoms_prop,
            compute_force=output_force_stress,
            create_neighborlist_device=create_neighborlist_device,
        )
        E_prop = float(E_prop)
        if output_force_stress:
            F_prop = np.asarray(F_prop, dtype=float)
            S_prop = np.asarray(S_prop, dtype=float)
        else:
            F_prop = F_old
            S_prop = S_old

        # 4) Metropolis acceptance with NPT ensemble
        # For anisotropic scaling: ln(V_new/V_old) = sum(ln(scale_i))
        dE = E_prop - E_old
        dV = V_new - V_old
        log_V_ratio = np.sum(np.log(scale))

        # Acceptance criterion
        # log-space proposal => extra +ln(V_new/V_old) Hastings term
        # ΔH = ΔE + P*ΔV - (N+1) * kT * ln(V_new/V_old)
        delta_H = (
            dE
            + P_eV_per_A3 * dV
            - (N_atoms + 1) * temperature_in_eV * log_V_ratio
        )
        log_alpha = -beta * delta_H

        accepted = bool(np.log(np.random.rand()) < log_alpha)

        if accepted:
            atoms.set_cell([L_new[0], L_new[1], L_new[2]], scale_atoms=False)
            atoms.set_positions(coords_prop)
            return float(E_prop), F_prop, S_prop, atoms, True
        else:
            atoms.set_positions(coords)
            return float(E_old), F_old, S_old, atoms, False

    return cell_step_isotropic, cell_step_anisotropic

####################################################################################################
####################################################################################################
if __name__ == "__main__":

    # ----------------------------------------------------------------------------------------------
    # Load initial structure and metadata from JSON file
    import time
    from datetime import datetime
    from ckpt import load_structure_from_json
    from potentialmace_cueq import initialize_mace_model

    # Path to initial structure JSON file
    initial_structure_file = (
        # "/home/zq/zqcodeml/watericeIh-mc-master/source/structure/initstru/sc_322_n_96.json"
        "/home/zq/zqcodeml/watericeIh-mc-master/source/structure/initstru/density_992/sc_322_n_96.json"
        # "/home/zq/zqcodeml/watericeIh_data/energy_traj_overlap/energy_traj_v2/sc_422_n_128_rho_933_state_10000_relax.json"
    )

    # Load ASE Atoms object and metadata dict
    atoms, data = load_structure_from_json(initial_structure_file)

    # Extract coordinates and structure information
    coords = atoms.get_positions()                         # (n_atoms, 3)
    num_O = atoms.get_chemical_symbols().count("O")        # number of oxygen atoms

    # Orthorhombic box lengths (Å)
    box_lengths = atoms.cell.lengths()

    print(f"[LOAD] Loaded structure: {num_O} O atoms, {len(coords)} total atoms.")
    print(f"[LOAD] Initial box lengths (Å): {box_lengths}")

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

    # Initial energy/force/stress @ current coords
    atoms.set_positions(coords)
    e_curr, f_curr, s_curr = mace_inference(atoms, compute_force=True)
    print(f"[INIT] Energy = {e_curr:.12f} eV")

    # ----------------------------------------------------------------------------------------------
    # Build Monte Carlo cell updater
    # ----------------------------------------------------------------------------------------------
    cell_step_isotropic, cell_step_anisotropic = make_metropolis_cell_update_functions(
        mace_inference=mace_inference,
    )

    # Choose update mode: isotropic or anisotropic
    # cell_step = cell_step_isotropic  # or cell_step_anisotropic
    cell_step = cell_step_anisotropic

    # ----------------------------------------------------------------------------------------------
    # Monte Carlo cell update test
    # ----------------------------------------------------------------------------------------------
    energy_curr = float(e_curr)
    force_curr = np.asarray(f_curr, dtype=float)
    stress_curr = np.asarray(s_curr, dtype=float)

    num_steps = 5000             # total MC steps
    temperature_K = 200.0       # simulation temperature (K)
    temperature_in_eV = temperature_K * units.K_B_EV_PER_K
    pressure_GPa = 0.000101325  # External pressure in GPa (1 atm = 101325 Pa)
    # pressure_GPa = 0.0  # External pressure in GPa (1 atm = 101325 Pa)

    mc_width = suggest_cell_mc_width(temperature_K, atoms.get_volume(), width_ref=0.002)
    print(f"[MC] Using cell MC width = {mc_width:.6f} (fractional)")
    print(f"[MC] Pressure = {pressure_GPa:.3f} GPa")

    # Convert pressure from GPa to eV/Angstrom^3
    pressure_eV_A3 = pressure_GPa * units.GPA_TO_EV_PER_ANGSTROM3

    accepts = 0
    attempts = 0

    energy_traj = [energy_curr]
    volume_traj = [np.prod(atoms.cell.lengths())]
    cell_a_traj = [atoms.cell.lengths()[0]]
    cell_b_traj = [atoms.cell.lengths()[1]]
    cell_c_traj = [atoms.cell.lengths()[2]]
    stress_traj = [stress_curr.copy()]  # Store stress (6-component vector)

    print_interval = 20

    t0 = time.time()
    for step in range(1, num_steps + 1):
        energy_curr, force_curr, stress_curr, atoms, accepted = cell_step(
            energy_curr=energy_curr,
            force_curr=force_curr,
            stress_curr=stress_curr,
            temperature_in_eV=float(temperature_in_eV),
            atoms=atoms,
            pressure_eV_A3=pressure_eV_A3,
            mc_width=mc_width,
        )

        # Get box lengths from atoms
        box_lengths = atoms.cell.lengths()

        accepts += int(accepted)
        attempts += 1
        energy_traj.append(energy_curr)
        volume_traj.append(np.prod(box_lengths))
        cell_a_traj.append(box_lengths[0])
        cell_b_traj.append(box_lengths[1])
        cell_c_traj.append(box_lengths[2])
        stress_traj.append(stress_curr.copy())

        if print_interval > 0 and ((step % print_interval == 0) or (step == 1) or (step == num_steps)):
            acc_rate = accepts / attempts
            V = np.prod(box_lengths)
            print(f"[CELL] step={step:6d}/{num_steps:6d}  E={energy_curr:.12f} eV  "
                  f"V={V:.3f} Å³  acc_rate={acc_rate:5.3f}")

    t1 = time.time()
    # Calculate final density
    final_volume = np.prod(box_lengths)
    mass_H2O_g = units.calculate_mass_h2o_g(1)  # grams per H2O molecule
    final_density = (num_O * mass_H2O_g) / (final_volume * units.ANGSTROM3_TO_CM3)  # g/cm^3

    print(f"[DONE] Steps={attempts}, Accepts={accepts}, Final acc_rate={accepts/attempts:5.3f}, "
          f"Elapsed={t1 - t0:.2f} s")
    print("\n[MC Summary]")
    print(f"  Steps attempted : {attempts}")
    print(f"  Steps accepted  : {accepts}")
    print(f"  Acceptance rate : {accepts/attempts:.3f}")
    print(f"  Final energy    : {energy_curr:.12f} eV")
    print(f"  Energy per H₂O  : {(energy_curr/num_O + 16.0)*1000:.3f} meV")
    print(f"  Final box (Å)   : a={box_lengths[0]:.4f}, b={box_lengths[1]:.4f}, c={box_lengths[2]:.4f}")
    print(f"  Final volume    : {final_volume:.3f} Å³")
    print(f"  Final density   : {final_density:.4f} g/cm³")
    print(f"  Temperature     : {temperature_K:.1f} K  (kT = {temperature_in_eV:.6f} eV)")
    print(f"  Pressure        : {pressure_GPa:.3f} GPa")
    print(f"  Total time      : {t1 - t0:.2f} s")

    # ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------
    # Plot trajectories
    # ----------------------------------------------------------------------------------------------
    import matplotlib.pyplot as plt

    energy_traj = np.array(energy_traj)
    volume_traj = np.array(volume_traj)
    cell_a_traj = np.array(cell_a_traj)
    cell_b_traj = np.array(cell_b_traj)
    cell_c_traj = np.array(cell_c_traj)
    stress_traj = np.array(stress_traj)  # Shape: (num_steps+1, 6)

    # Calculate density trajectory (g/cm^3)
    mass_H2O_g = units.calculate_mass_h2o_g(1)  # grams per H2O molecule
    density_traj = (num_O * mass_H2O_g) / (volume_traj * units.ANGSTROM3_TO_CM3)  # g/cm^3

    # Calculate pressure trajectory from stress using the tools function
    from tools import calculate_pressure_from_stress
    num_atoms = len(coords)
    P_kinetic, P_virial, pressure_traj = calculate_pressure_from_stress(
        stress=stress_traj,
        volume=volume_traj,
        num_atoms=num_atoms,
        temperature_in_eV=temperature_in_eV,
    )
    pressure_traj_GPa = pressure_traj / units.GPA_TO_EV_PER_ANGSTROM3

    # Convert stress components to GPa
    stress_xx_GPa = stress_traj[:, 0] / units.GPA_TO_EV_PER_ANGSTROM3
    stress_yy_GPa = stress_traj[:, 1] / units.GPA_TO_EV_PER_ANGSTROM3
    stress_zz_GPa = stress_traj[:, 2] / units.GPA_TO_EV_PER_ANGSTROM3

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), dpi=150)

    # Energy trajectory (in meV per H2O)
    energy_per_h2o_meV = (energy_traj / num_O + 16.0) * 1000  # convert to meV
    axes[0, 0].plot(np.arange(len(energy_per_h2o_meV)), energy_per_h2o_meV, ".-", lw=1)
    axes[0, 0].set_xlabel("Monte Carlo step", fontsize=11)
    axes[0, 0].set_ylabel("Energy per H₂O (meV)", fontsize=11)
    axes[0, 0].set_title(f"Energy | T={temperature_K:.0f}K | acc={accepts/attempts:.3f}", fontsize=12)
    axes[0, 0].grid(True, ls="--", alpha=0.5)

    # Volume trajectory
    axes[0, 1].plot(np.arange(len(volume_traj)), volume_traj, ".-", lw=1, color="orange")
    axes[0, 1].set_xlabel("Monte Carlo step", fontsize=11)
    axes[0, 1].set_ylabel("Volume (Å³)", fontsize=11)
    axes[0, 1].set_title(f"Volume | P_ext={pressure_GPa:.6f} GPa", fontsize=12)
    axes[0, 1].grid(True, ls="--", alpha=0.5)

    # Cell parameters a, b, c
    axes[1, 0].plot(np.arange(len(cell_a_traj)), cell_a_traj, ".-", lw=1, label="a", alpha=0.7)
    axes[1, 0].plot(np.arange(len(cell_b_traj)), cell_b_traj, ".-", lw=1, label="b", alpha=0.7)
    axes[1, 0].plot(np.arange(len(cell_c_traj)), cell_c_traj, ".-", lw=1, label="c", alpha=0.7)
    axes[1, 0].set_xlabel("Monte Carlo step", fontsize=11)
    axes[1, 0].set_ylabel("Cell parameter (Å)", fontsize=11)
    axes[1, 0].set_title("Cell parameters (a, b, c)", fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, ls="--", alpha=0.5)

    # Density trajectory
    axes[1, 1].plot(np.arange(len(density_traj)), density_traj, ".-", lw=1, color="green")
    axes[1, 1].set_xlabel("Monte Carlo step", fontsize=11)
    axes[1, 1].set_ylabel("Density (g/cm³)", fontsize=11)
    axes[1, 1].set_title(f"Density | ρ_final={density_traj[-1]:.4f} g/cm³", fontsize=12)
    axes[1, 1].grid(True, ls="--", alpha=0.5)

    # Pressure trajectory
    axes[2, 0].plot(np.arange(len(pressure_traj_GPa)), pressure_traj_GPa, ".-", lw=1, color="purple")
    axes[2, 0].set_xlabel("Monte Carlo step", fontsize=11)
    axes[2, 0].set_ylabel("Pressure (GPa)", fontsize=11)
    axes[2, 0].set_title(f"Pressure | P_final={pressure_traj_GPa[-1]:.6f} GPa", fontsize=12)
    axes[2, 0].grid(True, ls="--", alpha=0.5)

    # Directional stress components
    axes[2, 1].plot(np.arange(len(stress_xx_GPa)), stress_xx_GPa, ".-", lw=1, label="σ_xx", alpha=0.7)
    axes[2, 1].plot(np.arange(len(stress_yy_GPa)), stress_yy_GPa, ".-", lw=1, label="σ_yy", alpha=0.7)
    axes[2, 1].plot(np.arange(len(stress_zz_GPa)), stress_zz_GPa, ".-", lw=1, label="σ_zz", alpha=0.7)
    axes[2, 1].set_xlabel("Monte Carlo step", fontsize=11)
    axes[2, 1].set_ylabel("Stress (GPa)", fontsize=11)
    axes[2, 1].set_title("Directional stress components", fontsize=12)
    axes[2, 1].legend(fontsize=10)
    axes[2, 1].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

    # ========== Moving Average Plots ==========
    def moving_average(data, window_size):
        """Compute moving average using convolution."""
        if len(data) < window_size:
            return data
        kernel = np.ones(window_size) / window_size
        return np.convolve(data, kernel, mode='valid')

    window = 200

    # Apply moving average to all trajectories
    energy_smooth = moving_average(energy_per_h2o_meV, window)
    volume_smooth = moving_average(volume_traj, window)
    cell_a_smooth = moving_average(cell_a_traj, window)
    cell_b_smooth = moving_average(cell_b_traj, window)
    cell_c_smooth = moving_average(cell_c_traj, window)
    density_smooth = moving_average(density_traj, window)
    pressure_smooth = moving_average(pressure_traj_GPa, window)
    stress_xx_smooth = moving_average(stress_xx_GPa, window)
    stress_yy_smooth = moving_average(stress_yy_GPa, window)
    stress_zz_smooth = moving_average(stress_zz_GPa, window)

    # X-axis for smoothed data (offset by window/2 to center the moving average)
    x_smooth = np.arange(window - 1, len(energy_per_h2o_meV))

    # Create new figure for smoothed data
    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 12), dpi=150)

    # Energy trajectory (smoothed)
    axes2[0, 0].plot(x_smooth, energy_smooth, "-", lw=2)
    axes2[0, 0].set_xlabel("Monte Carlo step", fontsize=11)
    axes2[0, 0].set_ylabel("Energy per H₂O (meV)", fontsize=11)
    axes2[0, 0].set_title(f"Energy (MA-{window}) | T={temperature_K:.0f}K | acc={accepts/attempts:.3f}", fontsize=12)
    axes2[0, 0].grid(True, ls="--", alpha=0.5)

    # Volume trajectory (smoothed)
    axes2[0, 1].plot(x_smooth, volume_smooth, "-", lw=2, color="orange")
    axes2[0, 1].set_xlabel("Monte Carlo step", fontsize=11)
    axes2[0, 1].set_ylabel("Volume (Å³)", fontsize=11)
    axes2[0, 1].set_title(f"Volume (MA-{window}) | P_ext={pressure_GPa:.6f} GPa", fontsize=12)
    axes2[0, 1].grid(True, ls="--", alpha=0.5)

    # Cell parameters a, b, c (smoothed)
    axes2[1, 0].plot(x_smooth, cell_a_smooth, "-", lw=2, label="a", alpha=0.8)
    axes2[1, 0].plot(x_smooth, cell_b_smooth, "-", lw=2, label="b", alpha=0.8)
    axes2[1, 0].plot(x_smooth, cell_c_smooth, "-", lw=2, label="c", alpha=0.8)
    axes2[1, 0].set_xlabel("Monte Carlo step", fontsize=11)
    axes2[1, 0].set_ylabel("Cell parameter (Å)", fontsize=11)
    axes2[1, 0].set_title(f"Cell parameters (MA-{window})", fontsize=12)
    axes2[1, 0].legend(fontsize=10)
    axes2[1, 0].grid(True, ls="--", alpha=0.5)

    # Density trajectory (smoothed)
    axes2[1, 1].plot(x_smooth, density_smooth, "-", lw=2, color="green")
    axes2[1, 1].set_xlabel("Monte Carlo step", fontsize=11)
    axes2[1, 1].set_ylabel("Density (g/cm³)", fontsize=11)
    axes2[1, 1].set_title(f"Density (MA-{window}) | ρ_final={density_traj[-1]:.4f} g/cm³", fontsize=12)
    axes2[1, 1].grid(True, ls="--", alpha=0.5)

    # Pressure trajectory (smoothed)
    axes2[2, 0].plot(x_smooth, pressure_smooth, "-", lw=2, color="purple")
    axes2[2, 0].set_xlabel("Monte Carlo step", fontsize=11)
    axes2[2, 0].set_ylabel("Pressure (GPa)", fontsize=11)
    axes2[2, 0].set_title(f"Pressure (MA-{window}) | P_final={pressure_traj_GPa[-1]:.6f} GPa", fontsize=12)
    axes2[2, 0].grid(True, ls="--", alpha=0.5)

    # Directional stress components (smoothed)
    axes2[2, 1].plot(x_smooth, stress_xx_smooth, "-", lw=2, label="σ_xx", alpha=0.8)
    axes2[2, 1].plot(x_smooth, stress_yy_smooth, "-", lw=2, label="σ_yy", alpha=0.8)
    axes2[2, 1].plot(x_smooth, stress_zz_smooth, "-", lw=2, label="σ_zz", alpha=0.8)
    axes2[2, 1].set_xlabel("Monte Carlo step", fontsize=11)
    axes2[2, 1].set_ylabel("Stress (GPa)", fontsize=11)
    axes2[2, 1].set_title(f"Directional stress (MA-{window})", fontsize=12)
    axes2[2, 1].legend(fontsize=10)
    axes2[2, 1].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

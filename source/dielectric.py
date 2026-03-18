import numpy as np
import ase
from typing import Tuple

####################################################################################################
# --------------------------------- Small utility imports ---------------------------------

try:
    from tools import mic_vec, wrap_pos, normalize_vec
    from crystaltools import two_nearest_H_per_O
    import units
except Exception:
    from source.tools import mic_vec, wrap_pos, normalize_vec
    from source.crystaltools import two_nearest_H_per_O
    from source import units

####################################################################################################

def compute_h2o_dipoles(
    atoms: ase.Atoms,
    model: str = "TIP4P-ICE",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-molecule and total dipole moments of a water system.

    This function estimates the dipole moments of each H₂O molecule
    using a rigid-point-charge model (default: TIP4P-ICE).
    The dipoles are returned in units of e·Å, with the total dipole 
    obtained as the vector sum over all molecules.

    Args:
        atoms: ASE Atoms object containing O and H positions.
        model: Water model name. Currently only "TIP4P-ICE" is supported.

    Returns:
        orient_unit (np.ndarray): (N, 3) array of unit orientation vectors
            for each water molecule (bisector of the two O–H bonds).
        dip_eA (np.ndarray): (N, 3) array of individual molecular dipoles (e·Å).
        M_total_eA (np.ndarray): (3,) array of total system dipole (e·Å).
    """

    # --- TIP4P-ICE parameters ---
    if model.upper() == "TIP4P-ICE":
        q_H = +0.5897   # charge on H (e)
        q_M = -1.1794   # charge on M site (−2 × q_H)
        d_OM = 0.1577   # O–M distance in Å
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Identify the two nearest hydrogens for each oxygen
    o_hh = two_nearest_H_per_O(atoms, cutoff=1.2, strict=True)
    N = o_hh.shape[0]

    pos = atoms.get_positions()  # atomic positions (Å)
    L = atoms.get_cell().lengths()  # box lengths (Å)

    orient_unit = np.zeros((N, 3))
    dip_eA = np.zeros((N, 3))  # dipoles in e·Å

    for i, (o, h1, h2) in enumerate(o_hh):
        # Minimum-image O→H displacement vectors (Å)
        v1 = mic_vec(pos[h1] - pos[o], L)
        v2 = mic_vec(pos[h2] - pos[o], L)

        # Unit vectors along O–H bonds
        u1 = normalize_vec(v1)
        u2 = normalize_vec(v2)

        # Bisector unit vector defining molecular orientation
        n_bis = normalize_vec(u1 + u2).ravel()
        orient_unit[i] = n_bis

        # Position of the negative M-site relative to O (Å)
        rM = d_OM * n_bis

        # Molecular dipole moment (sum of point charges × displacements)
        dip = q_H * v1 + q_H * v2 + q_M * rM
        dip_eA[i] = dip

    # Total system dipole (vector sum)
    M_total_eA = np.sum(dip_eA, axis=0)

    return orient_unit, dip_eA, M_total_eA


####################################################################################################

def compute_correlation_parameter(
    atoms: ase.Atoms,
    model: str = "TIP4P-ICE",
) -> Tuple[np.ndarray, float, float, float]:
    """
    Compute per-snapshot quantities for the correlation parameter G.

    Parameters
    ----------
    atoms
        ASE Atoms of the water system.
    model
        Water model passed through to `compute_h2o_dipoles` (default: "TIP4P-ICE").

    Returns
    -------
    M_vec_eA : (3,) np.ndarray
        Total dipole vector of the simulation cell in e·Å.
    M2_eA2 : float
        Squared magnitude |M|^2 in (e·Å)^2.
    mu2_mean_eA2 : float
        Frame-averaged molecular dipole squared ⟨|μ|²⟩ in (e·Å)^2.
    correlation_G : float
        Single-frame correlation parameter G = |M|^2 / (N * ⟨|μ|²⟩).

    Notes
    -----
    - These numbers are sufficient to assemble the ensemble estimator
      G = ⟨|M|²⟩ / (N ⟨|μ|²⟩) elsewhere in your analysis.
    - Polarization P can be computed from M_vec_eA / volume when needed.
    """
    # Reuse your dipole routine
    _, dip_eA, M_vec_eA = compute_h2o_dipoles(atoms, model=model)

    # Ensure clean float arrays
    M_vec_eA = np.asarray(M_vec_eA, dtype=float)
    dip_eA   = np.asarray(dip_eA,   dtype=float)

    # |M|^2
    M2_eA2 = float(np.dot(M_vec_eA, M_vec_eA))
    # ⟨|μ|^2⟩ over molecules in this frame
    mu2_mean_eA2 = float(np.mean(np.einsum("ij,ij->i", dip_eA, dip_eA)))

    # Single-frame correlation parameter: G_frame = |M|^2 / (N * ⟨|μ|²⟩)
    num_molecule = len(atoms) // 3
    correlation_G = M2_eA2 / (num_molecule * mu2_mean_eA2)

    return M_vec_eA, M2_eA2, mu2_mean_eA2, correlation_G

####################################################################################################
####################################################################################################
if __name__ == "__main__":

    from ckpt import load_structure_from_json

    init_stru_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/structure/initstru/sc_212_n_32.json"


    # Load ASE Atoms object and metadata dict
    atoms, data = load_structure_from_json(init_stru_path)
    num_molecule = len(atoms) // 3  # assume 3 atoms per water molecule
    print(f"Number of water molecules: {num_molecule}")


    # -----------------------------
    # Unit conversions
    # -----------------------------
    DEBYE_PER_eA = units.E_ANGSTROM_TO_DEBYE
    C_PER_M2_PER_eA2 = units.E_CHARGE / (units.ANGSTROM_TO_M ** 2)  # (e/Angstrom^2) -> (C/m^2)
    
    # ----------------------------------------------------------------------------------------------
    # Compute correlation parameter (your existing routine)
    # ----------------------------------------------------------------------------------------------
    moment_vec_eA, moment2_eA2, mu2_mean_eA2, correlation_G = compute_correlation_parameter(atoms)

    # Magnitudes / conversions for M
    M_mag_eA = float(np.linalg.norm(moment_vec_eA))
    M_vec_D = moment_vec_eA * DEBYE_PER_eA
    M_mag_D = M_mag_eA * DEBYE_PER_eA

    # Per-molecule "net projection" (|M|/N) in D (useful order-parameter-like number)
    mu_net_per_mol_D = (M_mag_eA / num_molecule) * DEBYE_PER_eA

    # RMS single-molecule dipole from <|mu|^2>_frame
    mu_rms_eA = float(np.sqrt(mu2_mean_eA2))
    mu_rms_D = mu_rms_eA * DEBYE_PER_eA

    print("\n--- Correlation snapshot values ---")
    print(f"Total dipole M (e·Å):        {moment_vec_eA}")
    print(f"|M| (e·Å):                  {M_mag_eA:.6f}")
    print(f"Total dipole M (D):          {M_vec_D}")
    print(f"|M| (D):                    {M_mag_D:.6f}")
    print(f"|M|/N (D per H2O):           {mu_net_per_mol_D:.6f}")

    print(f"|M|^2 (e·Å)^2:               {moment2_eA2:.6e}")
    print(f"<|μ|^2>_frame (e·Å)^2:       {mu2_mean_eA2:.6e}")
    print(f"μ_rms = sqrt(<|μ|^2>) (D):   {mu_rms_D:.6f}")
    print(f"Correlation parameter G:     {correlation_G:.6f}")

    # ----------------------------------------------------------------------------------------------
    # Polarization: P = M / V
    # ----------------------------------------------------------------------------------------------
    V_A3 = float(atoms.get_volume())          # Å^3 (robust also for non-orthogonal cells)
    P_vec_eA2 = moment_vec_eA / V_A3          # e/Å^2  (since e·Å / Å^3)
    P_mag_eA2 = float(np.linalg.norm(P_vec_eA2))

    # Also show in C/m^2 (commonly reported)
    P_vec_Cm2 = P_vec_eA2 * C_PER_M2_PER_eA2
    P_mag_Cm2 = float(np.linalg.norm(P_vec_Cm2))

    # Optional: "equivalent dipole per molecule" from P*V_per_mol (in D)
    V_per_mol_A3 = V_A3 / num_molecule
    mu_equiv_from_P_eA = P_mag_eA2 * V_per_mol_A3     # e·Å
    mu_equiv_from_P_D = mu_equiv_from_P_eA * DEBYE_PER_eA

    print("\n--- Polarization ---")
    print(f"Volume V (Å^3):              {V_A3:.6f}")
    print(f"V per H2O (Å^3):              {V_per_mol_A3:.6f}")

    print(f"P (e/Å^2):                    {P_vec_eA2}")
    print(f"|P| (e/Å^2):                  {P_mag_eA2:.6f}")

    print(f"P (C/m^2):                    {P_vec_Cm2}")
    print(f"|P| (C/m^2):                  {P_mag_Cm2:.6f}")

    print(f"Equivalent dipole from |P| (D per H2O): {mu_equiv_from_P_D:.6f}")
    
    # # ----------------------------------------------------------------------------------------------
    # # Compute correlation parameter
    # # ----------------------------------------------------------------------------------------------
    # moment_vec, moment2, mu2_mean, correlation_G = compute_correlation_parameter(atoms)
    # print("\n--- Correlation snapshot values ---")
    # print(f"Total dipole M (e·Å):        {moment_vec}")
    # print(f"|M|^2 (e·Å)^2:               {moment2:.6e}")
    # print(f"<|μ|^2>_frame (e·Å)^2:       {mu2_mean:.6e}")
    # print(f"Correlation parameter G:     {correlation_G:.6f}")
    
    
    # # # ----------------------------------------------------------------------------------------------
    # # # Compute dipoles using the TIP4P-ICE model
    # # # ----------------------------------------------------------------------------------------------
    # box_lengths = atoms.get_cell().lengths()
    # V = atoms.get_volume()            # 推荐：对非正交也正确，单位 Å^3

    # P_e_per_A2 = moment_vec / V       # e/Å^2（向量）
    # P_mag = np.linalg.norm(moment_vec) / V

    # print("Volume V (Å^3):", V)
    # print("Polarization P (e / Å²):", P_e_per_A2)
    # print("|P| (e / Å²):", P_mag)
        
    # # ----------------------------------------------------------------------------------------------
    # # Convert polarization into standard physical units
    # # ----------------------------------------------------------------------------------------------
    # P_C_per_m2 = P_e_per_A2 * Units.e_charge / (Units.Ang_2_m ** 2)
    # P_uC_per_cm2 = P_C_per_m2 * 1e6 / 1e4  # 1 C/m² = 100 μC/cm²

    # print("Polarization P (C/m²):", P_C_per_m2)
    # print("Polarization P (μC/cm²):", P_uC_per_cm2)
    
    
    
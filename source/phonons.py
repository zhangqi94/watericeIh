from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Literal

import numpy as np
import ase


MaceInference = Callable[..., Tuple[float, np.ndarray, object]]


@dataclass(frozen=True)
class PhononResult:
    hessian: np.ndarray          # (3N,3N) eV/Å^2
    dyn: np.ndarray              # (3N,3N) eV/Å^2/amu
    eigenvalues: np.ndarray      # (3N,)
    eigenvectors: np.ndarray     # (3N,3N)
    frequencies_cm1: np.ndarray  # (3N,)


# -------------------------
# Mass helpers (H2O-friendly)
# -------------------------
MASS_AMU_DEFAULT = {
    "H": 1.007825031898,   # 1H
    "D": 2.014101777844,   # 2H (if you ever use D symbol)
    "O": 15.994914619257,  # 16O
}

MASS_AMU_H2O = {
    "H": MASS_AMU_DEFAULT["H"],
    "O": MASS_AMU_DEFAULT["O"],
}

MASS_AMU_D2O = {
    "H": MASS_AMU_DEFAULT["D"],  # keep symbol H, but use D mass
    "O": MASS_AMU_DEFAULT["O"],
}


def set_isotope_masses(atoms: ase.Atoms, mass_map: Optional[dict[str, float]] = None) -> None:
    """Set atomic masses explicitly.

    This is useful for isotope control (e.g., H2O vs D2O) and for ensuring consistent
    masses across environments.

    Args:
        atoms: ASE Atoms object to modify in-place.
        mass_map: Mapping from chemical symbol (e.g., "H", "O") to mass in amu.
            If None, MASS_AMU_DEFAULT is used.
    """
    mm = MASS_AMU_DEFAULT if mass_map is None else mass_map
    for i, sym in enumerate(atoms.get_chemical_symbols()):
        if sym in mm:
            atoms[i].mass = float(mm[sym])


# -------------------------
# Force evaluation (stable + optional averaging)
# -------------------------
def _forces(
    atoms: ase.Atoms,
    positions: np.ndarray,
    mace_inference: MaceInference,
    repeats: int = 1,
    reduce: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Evaluate forces; optionally repeat and reduce to fight float32 noise.

    Args:
        atoms: ASE Atoms object (positions will be temporarily overwritten).
        positions: Target positions with shape (N, 3).
        mace_inference: Callable that returns (energy, forces, extra).
        repeats: Number of repeated evaluations.
        reduce: Reduction across repeats: "mean" or "median".

    Returns:
        Forces array with shape (N, 3) in eV/Angstrom.

    Raises:
        ValueError: If force shape mismatches positions.
    """
    x = np.asarray(positions, dtype=np.float64)

    x_old = atoms.get_positions().copy()
    try:
        Fs = []
        for _ in range(int(repeats)):
            atoms.set_positions(x)
            try:
                _, f, _ = mace_inference(atoms, x, compute_force=True)
            except TypeError:
                _, f, _ = mace_inference(atoms, x)

            f = np.asarray(f, dtype=np.float64)
            if f.shape != x.shape:
                raise ValueError(f"Force shape {f.shape} != position shape {x.shape}")
            Fs.append(f)

        Fstack = np.stack(Fs, axis=0)  # (repeats, N, 3)
        if reduce == "median":
            return np.median(Fstack, axis=0)
        return np.mean(Fstack, axis=0)
    finally:
        atoms.set_positions(x_old)


# -------------------------
# Hessian by FD of forces
# -------------------------
def compute_hessian_fd_forces(
    atoms: ase.Atoms,
    mace_inference: MaceInference,
    step: float = 2.0e-2,
    symmetrize: bool = True,
    repeats: int = 1,
    reduce: Literal["mean", "median"] = "mean",
) -> np.ndarray:
    """Compute Hessian via central finite differences of forces.

    H_ij = d2E/dx_i dx_j = - dF_i/dx_j
    H[:, j] ≈ -(F(x+h e_j) - F(x-h e_j)) / (2h)

    Args:
        atoms: ASE Atoms object.
        mace_inference: Force provider.
        step: Finite difference step in Angstrom.
        symmetrize: If True, symmetrize Hessian as 0.5*(H+H^T).
        repeats: Repeated force evaluations per displacement.
        reduce: Reduction across repeats.

    Returns:
        Hessian matrix with shape (3N, 3N) in eV/Angstrom^2.

    Raises:
        ValueError: If atoms positions are not shape (N, 3).
    """
    x0 = np.asarray(atoms.get_positions(), dtype=np.float64)
    if x0.ndim != 2 or x0.shape[1] != 3:
        raise ValueError("atoms positions must have shape (N, 3).")

    n = x0.shape[0]
    dof = 3 * n
    h = float(step)

    H = np.zeros((dof, dof), dtype=np.float64)

    for j in range(dof):
        a = j // 3
        c = j % 3

        xp = x0.copy()
        xm = x0.copy()
        xp[a, c] += h
        xm[a, c] -= h

        fp = _forces(atoms, xp, mace_inference, repeats=repeats, reduce=reduce)
        fm = _forces(atoms, xm, mace_inference, repeats=repeats, reduce=reduce)

        dF = (fp.reshape(-1) - fm.reshape(-1)) / (2.0 * h)  # dF_i/dx_j
        H[:, j] = -dF

    if symmetrize:
        H = 0.5 * (H + H.T)

    return H


# -------------------------
# Dynamical matrix + ASR
# -------------------------
def build_dynamical_matrix(atoms: ase.Atoms, hessian: np.ndarray) -> np.ndarray:
    """Build mass-weighted dynamical matrix from Hessian.

    Args:
        atoms: ASE Atoms with masses already set.
        hessian: Hessian matrix (3N, 3N) in eV/Angstrom^2.

    Returns:
        Dynamical matrix (3N, 3N) in eV/Angstrom^2/amu.

    Raises:
        ValueError: If shape mismatch.
    """
    n = len(atoms)
    dof = 3 * n
    if hessian.shape != (dof, dof):
        raise ValueError(f"Hessian shape {hessian.shape} != {(dof, dof)}")

    masses_amu = np.array([atoms[i].mass for i in range(n)], dtype=np.float64)
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses_amu), 3)  # (3N,)
    D = (hessian * inv_sqrt_m[None, :]) * inv_sqrt_m[:, None]
    return 0.5 * (D + D.T)


def apply_asr(dyn: np.ndarray, atoms: ase.Atoms) -> np.ndarray:
    """Apply Acoustic Sum Rule by projecting out 3 translational modes.

    Args:
        dyn: Dynamical matrix (3N, 3N) in mass-weighted coordinates.
        atoms: ASE Atoms with masses set.

    Returns:
        Projected dynamical matrix (3N, 3N), symmetrized.
    """
    n = len(atoms)
    dof = 3 * n
    masses = np.array([atoms[i].mass for i in range(n)], dtype=np.float64)

    T = np.zeros((dof, 3), dtype=np.float64)
    for i, mi in enumerate(masses):
        s = np.sqrt(mi)
        T[3 * i + 0, 0] = s
        T[3 * i + 1, 1] = s
        T[3 * i + 2, 2] = s

    Q, _ = np.linalg.qr(T)  # (3N, 3)
    P = np.eye(dof) - Q @ Q.T
    Dp = P @ dyn @ P
    return 0.5 * (Dp + Dp.T)


# -------------------------
# Units: eigenvalues -> cm^-1
# -------------------------
def eigen_to_frequencies_cm1(eigenvalues: np.ndarray) -> np.ndarray:
    """Convert dynamical-matrix eigenvalues to frequencies in cm^-1.

    Args:
        eigenvalues: Eigenvalues in (eV/Angstrom^2/amu).

    Returns:
        Frequencies in cm^-1. Imaginary modes are returned as negative values.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)

    conv_to_s2 = 1.602176634e-19 / (1.0e-20 * 1.66053906660e-27)
    omega2 = lam * conv_to_s2

    omega = np.sign(omega2) * np.sqrt(np.abs(omega2))
    c_cm_s = 2.99792458e10
    return omega / (2.0 * np.pi * c_cm_s)


def phonons_fd(
    atoms: ase.Atoms,
    mace_inference: MaceInference,
    step: float = 2.0e-2,
    symmetrize_hessian: bool = True,
    apply_asr_translation: bool = True,
    repeats: int = 1,
    reduce: Literal["mean", "median"] = "mean",
    sort_by_frequency: bool = True,
    set_masses_for_h2o: bool = True,
    isotope: Optional[Literal["H2O", "D2O"]] = None,
    mass_map: Optional[dict[str, float]] = None,
) -> PhononResult:
    """Finite-difference phonons using force FD Hessian.

    Mass control priority (highest to lowest):
      1) mass_map (if provided)
      2) isotope ("H2O" or "D2O", if provided)
      3) set_masses_for_h2o (if True -> default H2O masses)
      4) otherwise keep atoms' existing masses

    Args:
        atoms: ASE Atoms object (copied/modified outside if needed).
        mace_inference: MACE inference callable.
        step: FD displacement in Angstrom.
        symmetrize_hessian: Symmetrize Hessian.
        apply_asr_translation: Apply ASR to remove translational modes.
        repeats: Repeat force evaluations per displacement.
        reduce: Reduction across repeats.
        sort_by_frequency: Sort modes by frequency.
        set_masses_for_h2o: Backward-compatible flag to set default H2O isotope masses.
        isotope: If "H2O" or "D2O", set masses accordingly (without changing symbols).
        mass_map: Explicit masses in amu, e.g. {"H": 2.014..., "O": 15.994...}.

    Returns:
        PhononResult with frequencies in cm^-1.
    """
    if mass_map is not None:
        set_isotope_masses(atoms, mass_map=mass_map)
    elif isotope is not None:
        if isotope == "H2O":
            set_isotope_masses(atoms, mass_map=MASS_AMU_H2O)
        elif isotope == "D2O":
            set_isotope_masses(atoms, mass_map=MASS_AMU_D2O)
        else:
            raise ValueError(f"Unsupported isotope={isotope}")
    elif set_masses_for_h2o:
        set_isotope_masses(atoms, mass_map=MASS_AMU_H2O)

    H = compute_hessian_fd_forces(
        atoms=atoms,
        mace_inference=mace_inference,
        step=step,
        symmetrize=symmetrize_hessian,
        repeats=repeats,
        reduce=reduce,
    )
    D = build_dynamical_matrix(atoms=atoms, hessian=H)

    if apply_asr_translation:
        D = apply_asr(D, atoms)

    evals, evecs = np.linalg.eigh(D)
    freqs = eigen_to_frequencies_cm1(evals)

    if sort_by_frequency:
        order = np.argsort(freqs)
        evals = evals[order]
        freqs = freqs[order]
        evecs = evecs[:, order]

    return PhononResult(
        hessian=H,
        dyn=D,
        eigenvalues=evals,
        eigenvectors=evecs,
        frequencies_cm1=freqs,
    )

####################################################################################################
####################################################################################################

if __name__ == "__main__":
    import os
    import ase.io
    import numpy as np

    # -----------------------
    # 1) Read first frame
    # -----------------------
    stru_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_relax.xyz"
    frames = ase.io.read(stru_file, index=":")   # list of Atoms
    if isinstance(frames, ase.Atoms):
        frames = [frames]
    atoms = frames[0].copy()  # pick the first atoms object

    print(f"[INFO] read file: {stru_file}")
    print(f"[INFO] natoms = {len(atoms)}  pbc={atoms.get_pbc()}")

    # If your extxyz does not include cell/pbc, set them here (adjust to your system).
    # atoms.set_pbc([True, True, True])
    # atoms.set_cell([...])

    # -----------------------
    # 2) Prepare MACE model
    # -----------------------
    from potentialmace_cueq import initialize_mace_model

    mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel251125/mace_iceIh_128x0e128x1o_r4.5_float32_seed144_cueq.model"
    mace_dtype = "float32"
    mace_device = "cuda"

    mace_inference = initialize_mace_model(
        mace_model_path,
        mace_dtype,
        mace_device,
    )

    # -----------------------
    # 3) Phonons (FD Hessian)
    # -----------------------
    # float32 force field: step should not be too small, or noise/δ amplification causes spurious imaginary modes
    step = 2e-2        # 0.02 Å (recommended starting point)
    repeats = 2        # 1~4; increase if many imaginary modes, decrease if too slow
    reduce = "mean"    # or "median"

    res = phonons_fd(
        atoms=atoms,
        mace_inference=mace_inference,
        step=step,
        repeats=repeats,
        reduce=reduce,
        apply_asr_translation=True,
        symmetrize_hessian=True,
        sort_by_frequency=True,
        set_masses_for_h2o=True,
    )

    freqs = res.frequencies_cm1

    print(f"[INFO] phonons done. step={step} Å repeats={repeats} reduce={reduce}")
    print("[INFO] lowest 12 frequencies (cm^-1):")
    print(np.array2string(freqs[:12], precision=4, suppress_small=False))

    print("[INFO] highest 12 frequencies (cm^-1):")
    print(np.array2string(freqs[-12:], precision=2, suppress_small=False))

    # Simple diagnostic: check for obvious small imaginary modes (beyond a few cm^-1 numerical noise)
    n_imag = int(np.sum(freqs < -1.0))  # you can adjust the threshold, e.g. -5
    print(f"[INFO] imaginary modes (< -1 cm^-1): {n_imag} / {len(freqs)}")

    # Also print the lowest 6 modes (periodic systems should have ~3 near zero; clusters have more near zero)
    print("[INFO] lowest 6 modes:")
    print(np.array2string(freqs[:6], precision=6, suppress_small=False))

    
    import matplotlib.pyplot as plt
    from tools import phonon_dos_lorentz
    # -----------------------
    # Plot DOS
    # -----------------------
    gamma = 20.0      # you can try 5 / 10 / 20 cm^-1
    drop0 = 1.0       # filter near-zero acoustic modes with |freq|<5 cm^-1 (adjust as needed)
    w, dos, f_pos, f_neg = phonon_dos_lorentz(freqs, gamma_cm1=gamma, drop_below=drop0, positive_only=True)

    print(f"[DOS] using {len(f_pos)} positive modes; {len(f_neg)} imaginary modes (filtered by |f|<{drop0} cm^-1 removed).")

    plt.figure(figsize=(8, 4), dpi=300)
    plt.plot(w, dos, lw=1.5)
    plt.xlabel(r"Frequency (cm$^{-1}$)")
    plt.ylabel(r"Phonon DOS (arb. units)")
    plt.title(f"Lorentzian-broadened DOS  (gamma={gamma} cm$^{{-1}}$, drop|f|<{drop0})")
    plt.tight_layout()
    plt.show()

from __future__ import annotations
import numpy as np
import json
import pandas as pd
from pathlib import Path
import sys
from typing import Tuple, Literal, Dict, Any, Sequence
import ase

_UNITS_DIR = Path(__file__).resolve().parent
if str(_UNITS_DIR) not in sys.path:
    sys.path.insert(0, str(_UNITS_DIR))
import units as units
    
    
    
####################################################################################################
####################################################################################################
## Functions for reading and trimming log files
def read_log_trim(
    log_path: str | Path, 
    drop: float | int = 0.1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read a Monte Carlo log file and drop the initial part (by ratio or count).

    Args:
        log_path (str | Path): Path to CSV-like log file.
        drop (float | int): If 0–1, drop that fraction of rows; 
            if ≥1, drop that many rows (default 0.1).
        verbose (bool): Whether to print read summary (default True).

    Returns:
        pd.DataFrame: Remaining data after dropping the first part.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If `drop` is not a valid float or int.
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    # Use a regex separator to handle optional spaces around commas.
    df = pd.read_csv(
        path,
        comment="#",
        on_bad_lines="skip",
        sep=r"\s*,\s*",
        engine="python",
    )
    n_total = len(df)

    # Determine number of rows to drop
    if isinstance(drop, float) and 0.0 <= drop < 1.0:
        n_drop = int(n_total * drop)
    elif isinstance(drop, int) and drop >= 1:
        n_drop = min(drop, n_total)
    else:
        raise ValueError("`drop` must be a float in [0,1) or int ≥1")

    df = df.iloc[n_drop:].reset_index(drop=True)

    if verbose:
        print(f"[INFO] Read file: {path}")
        print(f"[INFO] Total {n_total} rows, dropped {n_drop}, remaining {len(df)} rows.")

    return df

####################################################################################################
def ensure_temperature_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a unified temperature column named 't_k' exists.

    The new MC logs store temperature in 'temperature_K'. Older logs may use 't_k'.
    This helper adds 't_k' if needed.
    """
    if "t_k" in df.columns:
        return df
    if "temperature_K" in df.columns:
        df = df.copy()
        df["t_k"] = df["temperature_K"]
        return df
    raise KeyError("Missing temperature column: expected 't_k' or 'temperature_K'.")

####################################################################################################
def ensure_m2_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure M^2 column exists as 'm2' using mx,my,mz if needed."""
    if "m2" in df.columns:
        return df
    for c in ("mx", "my", "mz"):
        if c not in df.columns:
            raise KeyError(
                "Missing dipole components; need 'mx','my','mz' to build 'm2'."
            )
    df = df.copy()
    mx = df["mx"].to_numpy(dtype=float)
    my = df["my"].to_numpy(dtype=float)
    mz = df["mz"].to_numpy(dtype=float)
    df["m2"] = mx * mx + my * my + mz * mz
    return df

####################################################################################################
def safe_div(num, den):
    """Safe divide that returns NaN where denominator <= 0."""
    return np.where(den > 0, num / den, np.nan)

def _count_leading_comment_lines(path: str | Path, comment: str = "#") -> int:
    """Count leading comment lines in a text file."""
    count = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith(comment):
                count += 1
            else:
                break
    return count

####################################################################################################
# ---- moving average helpers ----
def moving_average(y, window: int):
    y = np.asarray(y, dtype=float)
    w = int(window)
    if w <= 1 or w > y.size:
        return y
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y, kernel, mode="valid")


def plot_with_ma(ax, x, y, *, ma_x, ma_window: int, color: str, ylabel: str, title: str):
    ax.plot(x, y, color="0.7", lw=0.8, alpha=0.6)
    y_ma = moving_average(y, ma_window)
    ax.plot(ma_x, y_ma, color=color, lw=1.5)
    ax.set_xlabel("Block")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.5)


def plot_multi_with_ma(ax, x, ys, labels, colors, *, ma_x, ma_window: int, ylabel: str, title: str):
    for y, label, color in zip(ys, labels, colors):
        ax.plot(x, y, color=color, lw=0.8, alpha=0.35)
        y_ma = moving_average(y, ma_window)
        ax.plot(ma_x, y_ma, color=color, lw=1.5, label=label)
    ax.set_xlabel("Block")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, ls="--", alpha=0.5)

####################################################################################################

def load_structure_from_json(json_path: str | Path) -> Tuple[ase.Atoms, Dict[str, Any]]:
    """
    Load a structure and related arrays from a JSON file into ASE and NumPy objects.

    Args:
        json_path (str | Path): Path to the JSON file containing structure data.

    Returns:
        Tuple[ase.Atoms, Dict[str, Any]]:
            - atoms: ASE Atoms object with atomic symbols, positions, cell, and PBC.
            - info: Dictionary containing:
                {
                    "supercell_size": np.ndarray,
                    "density": float,
                    "O_neighbors": np.ndarray,
                    "H_to_OO_pairs": np.ndarray,
                    "state_hydrogens": np.ndarray,
                }

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        KeyError: If required keys are missing in the JSON data.
        ValueError: If array shapes or data types are inconsistent.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    required_keys = [
        "atoms", 
        "supercell_size", 
        "density", 
        "O_neighbors", 
        "H_to_OO_pairs",
    ]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in JSON file {json_path}")

    atoms_info = data["atoms"]
    try:
        atoms = ase.Atoms(
            symbols=atoms_info["symbols"],
            positions=np.array(atoms_info["positions"], dtype=float),
            cell=np.array(atoms_info["cell"], dtype=float),
            pbc=np.array(atoms_info["pbc"], dtype=bool),
        )
    except Exception as e:
        raise ValueError(f"Failed to construct ASE Atoms: {e}")

    data = {
        "supercell_size": np.array(data["supercell_size"], dtype=int),
        "density": float(data["density"]),
        "O_neighbors": np.array(data["O_neighbors"], dtype=int),
        "H_to_OO_pairs": np.array(data["H_to_OO_pairs"], dtype=int),
        "state_hydrogens": np.array(data["state_hydrogens"], dtype=int),
    }

    return atoms, data


####################################################################################################
def detect_energy_outliers(
    energy: np.ndarray,
    *,
    zmax: float = 6.0,
    use_mad: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Detect outliers in an energy time series using a robust z-score.

    Args:
        energy : 1D array of energies (eV).
        zmax   : Threshold on |z| above which a point is considered an outlier.
        use_mad: If True, use median/MAD; otherwise use mean/std.

    Returns:
        mask_good : bool array, True for non-outlier points.
        z_score   : array of z-scores (same shape as energy, NaN for non-finite entries).
        stats     : dict with basic statistics used in detection.
    """
    e = np.asarray(energy, dtype=float)
    n = e.size
    mask_finite = np.isfinite(e)

    z = np.full_like(e, np.nan, dtype=float)
    mask_good = mask_finite.copy()  # start by rejecting non-finite

    # Too few points -> skip detection
    if np.count_nonzero(mask_finite) < 5:
        stats = dict(
            center=np.nan,
            scale=np.nan,
            method="none",
            zmax=zmax,
        )
        return mask_good, z, stats

    e_finite = e[mask_finite]

    if use_mad:
        # Robust center and scale
        center = float(np.median(e_finite))
        abs_dev = np.abs(e_finite - center)
        mad = float(np.median(abs_dev))

        if mad <= 0.0:
            # Fallback to std if MAD is degenerate
            scale = float(np.std(e_finite, ddof=1))
            method = "std-fallback"
            if scale <= 0.0:
                stats = dict(center=center, scale=scale, method=method, zmax=zmax)
                return mask_good, z, stats
            z[mask_finite] = (e_finite - center) / scale
        else:
            # 0.6745... is the factor that makes MAD equal to sigma for Gaussian
            scale = mad / 0.6744897501960817
            method = "mad"
            z[mask_finite] = (e_finite - center) / scale
    else:
        center = float(np.mean(e_finite))
        scale = float(np.std(e_finite, ddof=1))
        method = "std"
        if scale <= 0.0:
            stats = dict(center=center, scale=scale, method=method, zmax=zmax)
            return mask_good, z, stats
        z[mask_finite] = (e_finite - center) / scale

    # Mark outliers
    mask_outlier = np.abs(z) > float(zmax)
    mask_good[mask_outlier] = False

    stats = dict(
        center=center,
        scale=scale,
        method=method,
        zmax=zmax,
        n_total=int(n),
        n_finite=int(np.count_nonzero(mask_finite)),
        n_outliers=int(np.count_nonzero(mask_outlier)),
    )
    return mask_good, z, stats

####################################################################################################
def compute_acceptance_rate(accepts: np.ndarray, attempts: np.ndarray) -> float:
    """Compute total acceptance rate = sum(accepts) / sum(attempts).

    Args:
        accepts (np.ndarray): Number of accepted moves per block.
        attempts (np.ndarray): Number of attempted moves per block.

    Returns:
        float: Overall acceptance rate (0–1). Returns NaN if sum(attempts)=0.
    """
    a = np.asarray(accepts, dtype=float)
    b = np.asarray(attempts, dtype=float)
    total_accepts = np.sum(a)
    total_attempts = np.sum(b)
    return total_accepts / total_attempts if total_attempts > 0 else float("nan")

####################################################################################################
## Functions for computing energy statistics
def compute_energy_mean_error(
    energy: np.ndarray,
    num_molecule: float | None = None,
    energy_shift: float = 0.0,
) -> tuple[float, float]:
    """Compute mean energy and its statistical error, with optional normalization and shift.

    Args:
        energy (np.ndarray): Array of sampled energies in eV.
        num_molecule (float | None): If provided, divide energy by this value (per molecule).
        energy_shift (float): Constant energy shift added to all samples before statistics (default 0.0).

    Returns:
        tuple[float, float]: (mean_energy_eV, error_eV)
            - mean_energy_eV: Average energy (possibly per molecule and/or shifted).
            - error_eV: Standard error of the mean (std / sqrt(N)).
    """
    e = np.asarray(energy, dtype=float)
    if num_molecule is not None:
        e = e / float(num_molecule) + float(energy_shift)

    n = e.size
    if n == 0:
        raise ValueError("Energy array is empty.")

    e_mean = float(np.mean(e))
    e_std = float(np.std(e, ddof=1)) if n > 1 else 0.0
    e_err = float(e_std / np.sqrt(n)) if n > 1 else 0.0

    return e_mean, e_err


####################################################################################################
def compute_heat_capacity(
    energy: np.ndarray,
    temperature_K: float,
    num_molecules: int,
    *,
    energy_basis: Literal["total", "per-molecule"] = "total",
    output_unit:   Literal["eV_per_K", "J_per_mol_K"] = "J_per_mol_K",
    cv_sem_method: Literal["normal-iid"] = "normal-iid",
) -> Tuple[float, float]:
    """Compute heat capacity via fluctuation formula with a correct basis and units.

    The variance MUST be computed on TOTAL energy:
        C_total = Var(E_total) / (k_B * T^2)

    If `energy_basis == "per-molecule"`, this function rebuilds E_total = e_per_mol * N
    before taking the variance. Final reporting can be per-molecule (J/mol/K) or eV/K.

    Args:
        energy: 1D array of energies. If energy_basis="total", these are total energies (eV).
                If energy_basis="per-molecule", these are per-molecule energies (eV/molecule).
        temperature_K: Temperature in Kelvin (> 0).
        num_molecules: Number of molecules N in the simulation cell (int > 0).
        energy_basis: "total" or "per-molecule" describing the input `energy`.
        output_unit:  "eV_per_K"  -> returns per-molecule in eV/K;
                       "J_per_mol_K" -> returns per-molecule in J/mol/K (recommended).
        cv_sem_method: Currently supports "normal-iid": Var(s^2)=2*s^4/(n-1).

    Returns:
        (Cv, Cv_sem): Heat capacity and its standard error bar in the requested unit.
                      Both are **per molecule** when output_unit is "eV_per_K" or "J_per_mol_K".
    """
    e = np.asarray(energy, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("`energy` must be 1D with at least 2 samples.")
    if not (temperature_K > 0.0):
        raise ValueError("`temperature_K` must be positive.")
    if not (isinstance(num_molecules, int) and num_molecules > 0):
        raise ValueError("`num_molecules` must be a positive integer.")

    n = e.size
    # 1) Build TOTAL-energy series for variance
    if energy_basis == "per-molecule":
        E_total = e * float(num_molecules)  # rebuild total energy
    elif energy_basis == "total":
        E_total = e
    else:
        raise ValueError("energy_basis must be 'total' or 'per-molecule'.")

    # 2) Unbiased sample variance of TOTAL energy
    s2 = float(np.var(E_total, ddof=1))

    # 3) Cv on TOTAL basis (eV/K)
    kB = units.K_B_EV_PER_K
    T = float(temperature_K)
    Cv_total_eV_per_K = s2 / (kB * (T ** 2))

    # 4) SEM for Cv under normal-iid assumption
    if cv_sem_method != "normal-iid":
        raise ValueError(f"Unsupported cv_sem_method: {cv_sem_method}")
    var_s2 = 2.0 * (s2 ** 2) / (n - 1)          # Var of sample variance for normal data
    Cv_sem_total_eV_per_K = (var_s2 ** 0.5) / (kB * (T ** 2))

    # 5) Convert to per-molecule basis (divide by N)
    invN = 1.0 / float(num_molecules)
    Cv_per_mol_eV_per_K = Cv_total_eV_per_K * invN
    Cv_sem_per_mol_eV_per_K = Cv_sem_total_eV_per_K * invN

    # 6) Unit conversion
    if output_unit == "eV_per_K":
        Cv_out = Cv_per_mol_eV_per_K
        Cv_sem_out = Cv_sem_per_mol_eV_per_K
    elif output_unit == "J_per_mol_K":
        conv = units.EV_TO_J * units.N_A  # eV/K per molecule -> J/mol/K
        Cv_out = Cv_per_mol_eV_per_K * conv
        Cv_sem_out = Cv_sem_per_mol_eV_per_K * conv
    else:
        raise ValueError("output_unit must be 'eV_per_K' or 'J_per_mol_K'.")

    return float(Cv_out), float(Cv_sem_out)

####################################################################################################
def compute_kinetic_energy_total(
    temperature_K: float,
    num_molecules: int,
    *,
    atoms_per_molecule: int = 3,
) -> float:
    """Classical kinetic energy (total, eV) using equipartition.

    Default assumes H2O with 3 atoms per molecule:
        E_kin_total = (3/2) * N_atoms * k_B * T
    """
    if not (temperature_K > 0.0):
        raise ValueError("temperature_K must be positive.")
    if not (isinstance(num_molecules, int) and num_molecules > 0):
        raise ValueError("num_molecules must be a positive integer.")
    if not (isinstance(atoms_per_molecule, int) and atoms_per_molecule > 0):
        raise ValueError("atoms_per_molecule must be a positive integer.")

    n_atoms = atoms_per_molecule * num_molecules
    return 1.5 * n_atoms * units.K_B_EV_PER_K * float(temperature_K)

####################################################################################################
def compute_polarization_magnitude_cpm2(
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    cell_a: np.ndarray,
    cell_b: np.ndarray,
    cell_c: np.ndarray,
) -> np.ndarray:
    """Compute |P| in C/m^2 from dipole components and cell lengths.

    Assumes:
        M components (mx,my,mz) are in e·Å,
        cell lengths (a,b,c) are in Å.
    """
    mx = np.asarray(mx, dtype=float)
    my = np.asarray(my, dtype=float)
    mz = np.asarray(mz, dtype=float)
    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)

    V_A3 = cell_a * cell_b * cell_c
    if np.any(V_A3 <= 0.0):
        raise ValueError("Cell volume must be positive.")

    M_mag_eA = np.sqrt(mx * mx + my * my + mz * mz)
    # P = M / V, convert: (e·Å)/(Å^3) -> C/m^2
    eA_to_Cm = units.E_CHARGE * units.ANGSTROM_TO_M
    V_m3 = V_A3 * units.ANGSTROM3_TO_M3
    P_mag = (M_mag_eA * eA_to_Cm) / V_m3
    return P_mag

####################################################################################################

def compute_Gcorr(m2, mu2, num_molecule):
    """
    Compute correlation factor Gcorr = <M^2> / (N * <mu^2>)
    and estimate its uncertainty via standard error propagation.

    Args:
        m2 (array-like): M^2 values from each block or sample.
        mu2 (array-like): μ^2 values from each block or sample.
        num_molecule (int or float): Number of molecules N.

    Returns:
        tuple[float, float]: (Gcorr_mean, Gcorr_err)
    """
    m2 = np.asarray(m2, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    N = float(num_molecule)

    # means
    m2_mean = np.nanmean(m2)
    mu2_mean = np.nanmean(mu2)

    # std / sqrt(n)
    m2_err = np.nanstd(m2, ddof=1) / np.sqrt(np.sum(~np.isnan(m2)))
    mu2_err = np.nanstd(mu2, ddof=1) / np.sqrt(np.sum(~np.isnan(mu2)))

    # Gcorr
    Gcorr = m2_mean / (N * mu2_mean)

    # error propagation
    rel_err = np.sqrt((m2_err / m2_mean) ** 2 + (mu2_err / mu2_mean) ** 2)
    Gcorr_err = Gcorr * rel_err

    return Gcorr, Gcorr_err

####################################################################################################

def compute_ordering(
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    mu2: np.ndarray,
    num_molecule: int,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute all component products <Mα Mβ> with α,β∈{x,y,z} after normalizing
    the total dipole by N * mu_ref, where mu_ref = sqrt(<mu^2>_all).

    Inputs
    ------
    mx, my, mz : (n_snap,) arrays
        Total dipole components per snapshot in e·Å.
    mu2 : array-like
        Samples of per-molecule dipole-squared (μ^2). Can be:
          - (n_snap, N_mol) per snapshot & per molecule,
          - (n_snap,) snapshot-averaged μ^2,
          - scalar (already averaged).
        The function uses mean over all finite entries.
    num_molecule : int
        Number of molecules N.

    Returns
    -------
    out : dict[str, (mean, sem)]
        Keys: "<Mxx>", "<Mxy>", "<Mxz>", "<Myy>", "<Myz>", "<Mzz>".
        Values: (mean, standard error of the mean), computed over snapshots.
    """
    # Assemble M and drop rows with any non-finite component
    M = np.column_stack([mx, my, mz]).astype(float)
    mask = np.all(np.isfinite(M), axis=1)
    M = M[mask]
    n = len(M)
    if n < 2:
        raise ValueError("Need at least two finite dipole samples after masking.")

    # Aggregate <mu^2> over all provided entries (any shape)
    mu2 = np.asarray(mu2, dtype=float)
    mu2_mean = float(np.nanmean(mu2))
    if not np.isfinite(mu2_mean) or mu2_mean <= 0.0:
        raise ValueError("Mean(mu^2) must be positive and finite.")

    # Normalize total dipole by N * mu_ref, with mu_ref = sqrt(<mu^2>)
    N = float(num_molecule)
    mu_ref = np.sqrt(mu2_mean)
    M_norm = M / (N * mu_ref)  # shape (n_snap, 3)

    # Allocate tensors
    order_mean = np.zeros((3,3), dtype=float)
    order_err  = np.zeros((3,3), dtype=float)

    for i in range(3):
        for j in range(i, 3):
            prod = M_norm[:, i] * M_norm[:, j]
            mean = float(np.mean(prod))
            sem  = float(np.std(prod, ddof=1) / np.sqrt(n))
            order_mean[i,j] = order_mean[j,i] = mean
            order_err[i,j]  = order_err[j,i]  = sem

    return order_mean, order_err


####################################################################################################

def compute_mean_sem(x: np.ndarray) -> Tuple[float, float]:
    """
    Return (mean, standard error of mean) for finite samples of x.
    """
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(arr[0]), 0.0
    mean = float(np.mean(arr))
    sem  = float(np.std(arr, ddof=1) / np.sqrt(n))
    return mean, sem

####################################################################################################
def _choose_rebin_block_size(n_samples: int, min_blocks: int = 8) -> int:
    """Choose a power-of-two block size so that n_blocks >= min_blocks."""
    n = int(n_samples)
    if n <= 0:
        return 1
    mb = int(min_blocks)
    if mb < 1:
        mb = 1
    block = 1
    while (n // (block * 2)) >= mb:
        block *= 2
    return block


def _iter_block_slices(n_samples: int, block_size: int):
    n = int(n_samples)
    b = int(block_size)
    if b <= 0:
        raise ValueError("block_size must be positive.")
    n_blocks = n // b
    for i in range(n_blocks):
        start = i * b
        end = start + b
        yield slice(start, end)


def _mean_sem_from_blocks(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean, sem

####################################################################################################
def compute_density_g_cm3(
    cell_a: np.ndarray,
    cell_b: np.ndarray,
    cell_c: np.ndarray,
    num_molecules: int,
    *,
    mass_H_amu: float | None = None,
    mass_O_amu: float | None = None,
) -> np.ndarray:
    """Compute density in g/cm^3 from cell lengths and molecule count.

    Assumes H2O with masses from analysis/units.py unless overridden.
    """
    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)
    cell_c = np.asarray(cell_c, dtype=float)

    if not (isinstance(num_molecules, int) and num_molecules > 0):
        raise ValueError("num_molecules must be a positive integer.")

    # Mass per H2O in amu
    mH = units.MASS_H if mass_H_amu is None else float(mass_H_amu)
    mO = units.MASS_O if mass_O_amu is None else float(mass_O_amu)
    mass_h2o_amu = 2.0 * mH + mO
    total_mass_g = float(num_molecules) * mass_h2o_amu * units.AMU_TO_G

    # Volume in Å^3
    V_A3 = cell_a * cell_b * cell_c
    if np.any(V_A3 <= 0.0):
        raise ValueError("Cell volume must be positive.")
    return units.calculate_density_g_per_cm3(total_mass_g, V_A3)

####################################################################################################


def compute_eps_mean(
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    V_A3: float,
    T_K: float,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute the static dielectric tensor and its isotropic average from dipole fluctuations.

    Fluctuation formula (conducting boundary conditions / tin-foil):
        ε = I + Cov(M) / (ε0 * V * k_B * T)

    Parameters
    ----------
    mx, my, mz : (n_snap,) arrays
        Total dipole components per snapshot in e·Å.
    V_A3 : float
        Simulation cell volume in Å³.
    T_K : float
        Temperature in Kelvin.
    Returns
    -------
    eps_iso : float
        Isotropic dielectric constant (trace(ε)/3).
    eps_iso_err : float
        Simple uncertainty estimate based on the variance of |ΔM|².
    eps_diag : (3,) ndarray
        Eigenvalues of ε (principal dielectric constants).
    eps_tensor : (3,3) ndarray
        Full dielectric tensor.

    Notes
    -----
    - The covariance here uses a 1/n normalization (population second moment),
      which matches the standard derivation of the fluctuation formula. If you
      prefer an unbiased estimator for small samples, switch to 1/(n-1).
    """
    # --- Stack and filter finite snapshots ---
    M = np.column_stack([mx, my, mz]).astype(float)
    mask = np.all(np.isfinite(M), axis=1)
    M = M[mask]
    n = len(M)
    if n < 2:
        raise ValueError("Need at least two finite dipole samples.")
    if V_A3 <= 0.0 or T_K <= 0.0:
        raise ValueError("Volume and temperature must be positive.")

    # --- Unit conversions ---
    # e·Å -> C·m
    eA_to_Cm = units.E_CHARGE * units.ANGSTROM_TO_M
    M_SI = M * eA_to_Cm
    V_m3 = V_A3 * units.ANGSTROM3_TO_M3
    T = float(T_K)

    # --- Fluctuations ---
    M_mean = np.mean(M_SI, axis=0)
    dM = M_SI - M_mean
    cov = np.einsum("ni,nj->ij", dM, dM) / n  # (C·m)^2

    # --- Dielectric tensor ---
    denom = units.EPS0_F_PER_M * V_m3 * units.K_B_J_PER_K * T
    eps_tensor = np.eye(3) + cov / denom
    eps_diag = np.linalg.eigvalsh(eps_tensor)
    eps_iso = float(np.trace(eps_tensor) / 3.0)

    # --- Crude uncertainty estimate for ε_iso ---
    s = np.einsum("ij,ij->i", dM, dM)  # |ΔM|^2 per snapshot
    s_err = np.std(s, ddof=1) / np.sqrt(n)
    eps_iso_err = float(s_err / (3.0 * denom))

    return eps_iso, eps_iso_err, eps_diag, eps_tensor

####################################################################################################
def compute_Q(m2: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute Binder-like ratio Q = <M^2>^2 / <M^4> from M^2 samples.

    Also returns a normalized Binder cumulant for a 3-component vector order 
    parameter, rescaled so that:
        U = 0  in the high-T Gaussian limit (Q = 3/5)
        U = 1  in the perfectly ordered limit (Q = 1)

    The final definition used is:
        U = (5/2) - (3/(2 Q))

    Args:
        m2: Array-like of M^2 samples (one value per snapshot or block).

    Returns:
        Tuple[float, float, float, float]:
            - Q_mean: Binder ratio Q (dimensionless).
            - Q_err:  Standard error of Q.
            - U_mean: Binder cumulant U (normalized).
            - U_err:  Standard error of U.

    Raises:
        ValueError: If insufficient samples or <M^2>, <M^4> not positive.
    """
    m2_arr = np.asarray(m2, dtype=float)
    mask = np.isfinite(m2_arr)
    m2_arr = m2_arr[mask]

    n = m2_arr.size
    if n < 2:
        raise ValueError("compute_Q: need at least two finite M^2 samples.")

    # Per-snapshot M^4 from M^2
    m4_arr = m2_arr ** 2

    # Sample means
    m2_mean = float(np.mean(m2_arr))
    m4_mean = float(np.mean(m4_arr))

    if m2_mean <= 0.0 or m4_mean <= 0.0:
        raise ValueError("compute_Q: <M^2> and <M^4> must both be positive.")

    # Standard errors of the means
    m2_sem = float(np.std(m2_arr, ddof=1) / np.sqrt(n))
    m4_sem = float(np.std(m4_arr, ddof=1) / np.sqrt(n))

    # Binder-like ratio Q = <M^2>^2 / <M^4>
    Q_mean = (m2_mean ** 2) / m4_mean

    # Error propagation (assuming Cov ≈ 0)
    dQ_dm2 = 2.0 * m2_mean / m4_mean
    dQ_dm4 = -(m2_mean ** 2) / (m4_mean ** 2)

    Q_var = (dQ_dm2 ** 2) * (m2_sem ** 2) + (dQ_dm4 ** 2) * (m4_sem ** 2)
    Q_err = float(np.sqrt(Q_var))

    # --- Normalized Binder cumulant for N = 3 components ---
    # U_norm = (5/2) - (3/(2Q))
    U_mean = (5.0 / 2.0) - (3.0 / (2.0 * Q_mean))

    # Propagate uncertainty:
    # dU/dQ = 3/(2 Q^2)
    dU_dQ = 3.0 / (2.0 * (Q_mean ** 2))
    U_err = float(abs(dU_dQ) * Q_err)

    return float(Q_mean), float(Q_err), float(U_mean), float(U_err)

####################################################################################################
def compute_R(
    energy: np.ndarray,
    center: bool = True
) -> Tuple[float, float, float, float]:
    """
    Compute the energy moment ratio R and derived V.

    Two definitions are supported:
        center = True  (default):
            R = <(E - <E>)^2>^2 / <(E - <E>)^4>
            V = 1 - 1/R
            This is shift-invariant (recommended for ML potentials).

        center = False:
            R = <E^2>^2 / <E^4>
            V = 1 - 1/R
            This is the raw-moment definition used in some literature.

    Args:
        energy: Array of energy samples.
        center: Whether to subtract the mean energy.

    Returns:
        (R_mean, R_err, V_mean, V_err)
    """
    arr = np.asarray(energy, dtype=float)
    mask = np.isfinite(arr)
    arr = arr[mask]
    n = arr.size
    if n < 2:
        raise ValueError("compute_R: need at least two samples")

    # Choose centered or raw moments
    if center:
        x = arr - np.mean(arr)
        desc = "centered"
    else:
        x = arr
        desc = "raw"

    # Compute second and fourth moments
    e2 = x ** 2
    e4 = x ** 4

    x_mean = float(np.mean(e2))
    y_mean = float(np.mean(e4))

    if x_mean <= 0.0 or y_mean <= 0.0:
        raise ValueError(f"compute_R ({desc}): moments must be positive")

    # Standard error estimates
    x_sem = float(np.std(e2, ddof=1) / np.sqrt(n))
    y_sem = float(np.std(e4, ddof=1) / np.sqrt(n))

    # Moment ratio
    R_mean = (x_mean ** 2) / y_mean

    # Error propagation
    dR_dx = 2.0 * x_mean / y_mean
    dR_dy = -(x_mean ** 2) / (y_mean ** 2)
    R_var = (dR_dx ** 2) * (x_sem ** 2) + (dR_dy ** 2) * (y_sem ** 2)
    R_err = float(np.sqrt(R_var))

    # V = 1 - 1/R
    V_mean = 1.0 - 1.0 / R_mean
    dV_dR = 1.0 / (R_mean ** 2)
    V_err = abs(dV_dR) * R_err

    return R_mean, R_err, V_mean, V_err

####################################################################################################
####################################################################################################

# from scipy.interpolate import interp1d, PchipInterpolator, Akima1DInterpolator, UnivariateSpline
# from scipy.integrate import cumulative_trapezoid

# def compute_entropy(
#     T, Cv,
#     method="pchip",
#     n_grid=2000,
#     spline_s=None,
#     ensure_nonnegative=True,
#     T0=None,      # <<< NEW
# ):
#     """Compute entropy by integrating Cv(T)/T with optional reference temperature T0.

#     If T0 is None:
#         S(T) = ∫_0^T Cv/T dT   with S(0)=0.
#     If T0 is given:
#         S(T0)=0 and
#         S(T) = ∫_{T0}^T Cv/T dT  (negative for T < T0, positive for T > T0).

#     Returns
#     -------
#     T_dense : ndarray
#     Cv_dense : ndarray
#     S_dense : ndarray
#         A single continuous entropy curve satisfying S(T0)=0.
#     """

#     # -------------------------
#     # Sanitize input
#     # -------------------------
#     T = np.asarray(T, dtype=float)
#     Cv = np.asarray(Cv, dtype=float)
#     if T.shape != Cv.shape:
#         raise ValueError("T and Cv must have same shape.")

#     mask = np.isfinite(T) & np.isfinite(Cv)
#     T, Cv = T[mask], Cv[mask]
#     if T.size < 2:
#         raise ValueError("Need at least two valid points.")

#     # Sort + merge duplicates
#     idx = np.argsort(T)
#     T, Cv = T[idx], Cv[idx]
#     Tu, inv = np.unique(T, return_inverse=True)
#     if Tu.size != T.size:
#         Cv = np.bincount(inv, weights=Cv) / np.bincount(inv)
#         T = Tu

#     # Zero anchor for interpolation (only if T0=None)
#     if T0 is None:
#         if T[0] > 1e-8:
#             T = np.insert(T, 0, 0.0)
#             Cv = np.insert(Cv, 0, 0.0)
#         else:
#             Cv[0] = 0.0

#     # -------------------------
#     # Interpolant
#     # -------------------------
#     method = method.lower()

#     if method == "pchip":
#         f = PchipInterpolator(T, Cv, extrapolate=False)
#         eval_f = lambda x: f(x)

#     elif method == "akima":
#         f = Akima1DInterpolator(T, Cv)
#         def eval_f(x):
#             y = f(x)
#             y[(x < T.min()) | (x > T.max())] = np.nan
#             return y

#     elif method == "cubic":
#         f = interp1d(T, Cv, kind="cubic", bounds_error=True)
#         eval_f = lambda x: f(x)

#     elif method == "linear":
#         f = interp1d(T, Cv, kind="linear", bounds_error=True)
#         eval_f = lambda x: f(x)

#     elif method == "spline":
#         s = 0.0 if spline_s is None else float(spline_s)
#         f = UnivariateSpline(T, Cv, s=s)
#         eval_f = lambda x: f(x)

#     elif method == "logcubic":
#         if np.any(Cv <= 0):
#             raise ValueError("logcubic requires Cv>0")
#         flog = interp1d(T, np.log(Cv), kind="cubic", bounds_error=True)
#         eval_f = lambda x: np.exp(flog(x))

#     elif method == "logspline":
#         if np.any(Cv <= 0):
#             raise ValueError("logspline requires Cv>0")
#         s = 0.0 if spline_s is None else float(spline_s)
#         f = UnivariateSpline(T, np.log(Cv), s=s)
#         eval_f = lambda x: np.exp(f(x))

#     else:
#         raise ValueError(f"Unknown method: {method}")

#     # -------------------------
#     # Dense grid
#     # -------------------------
#     T_dense = np.linspace(T.min(), T.max(), int(n_grid))
#     Cv_dense = eval_f(T_dense)

#     valid = np.isfinite(Cv_dense)
#     T_dense = T_dense[valid]
#     Cv_dense = Cv_dense[valid]

#     if ensure_nonnegative:
#         Cv_dense = np.maximum(Cv_dense, 0.0)

#     # integrand = Cv/T
#     T_eps = 1e-12
#     integrand = Cv_dense / np.maximum(T_dense, T_eps)

#     # -------------------------
#     # Integration cases
#     # -------------------------
#     if T0 is None:
#         # Original: integrate from 0
#         S_dense = cumulative_trapezoid(integrand, T_dense, initial=0.0)
#         return T_dense, Cv_dense, S_dense

#     # ---- Custom reference: S(T0) = 0 ----
#     T0 = float(T0)

#     S_raw = cumulative_trapezoid(integrand, T_dense, initial=0.0)

#     # Find nearest T to T0
#     i0 = np.argmin(np.abs(T_dense - T0))
#     S_T0 = S_raw[i0]

#     # Shift so that S(T0)=0
#     S_dense = S_raw - S_T0

#     return T_dense, Cv_dense, S_dense

####################################################################################################

def paulings_entropy(x: float = 1.5069):
    """
    Compute Pauling's configurational entropy for proton-disordered ice.

    Uses natural logarithm: S = R * ln(x).
    """
    R = 8.31446261815324  # J/mol/K
    S_molar = R * np.log(x)      # ln(x)
    S_per_molecule = np.log(x)   # ln(x), in units of k_B
    return S_molar, S_per_molecule

####################################################################################################
def count_unique_states(states) -> int:
    """
    Return the number of distinct (non-null) states.
    """
    s = pd.Series(states, dtype="object")
    s = s[s.notna()]       # remove NaN
    return s.nunique()

####################################################################################################
####################################################################################################
def analyze_log_summary_multirun(
    log_paths: Sequence[str | Path],
    drop: float | int = 0.1,
    verbose: int = 1,
    use_energy_outlier_detection: bool = False,
) -> dict[str, Any]:
    """Summarize multiple Monte Carlo log files (same system and temperature).

    This function:
        - Reads and trims each log file (burn-in removal).
        - Concatenates all runs.
        - Optionally detects and removes energy outliers.
        - Checks consistency of temperature and system size.
        - Computes ensemble-averaged observables.
        - Counts distinct proton states (if column exists).

    Args:
        log_paths: Paths to MC log files.
        drop: Burn-in trimming per log.
        verbose: Print summary.
        use_energy_outlier_detection: If False, disable energy outlier removal.

    Returns:
        dict[str, Any] with all observables and stats.
    """
    log_paths = [Path(p) for p in log_paths]
    if len(log_paths) == 0:
        raise ValueError("analyze_log_summary_multirun: log_paths is empty.")

    # ------------------------------------------------------------
    # Step 1: Read and trim per run
    # ------------------------------------------------------------
    df_list: list[pd.DataFrame] = []
    per_run_counts: list[tuple[str, int]] = []

    for p in log_paths:
        df_i = read_log_trim(p, drop, verbose=False)
        if df_i.empty:
            continue
        per_run_counts.append((p.name, len(df_i)))
        df_list.append(df_i)

    if len(df_list) == 0:
        raise RuntimeError("All logs are empty after trimming.")

    # ------------------------------------------------------------
    # Step 2: Concatenate all logs
    # ------------------------------------------------------------
    df = pd.concat(df_list, ignore_index=True)
    df = ensure_temperature_column(df)
    df = ensure_m2_column(df)

    # ------------------------------------------------------------
    # Step 3: (Optional) Detect and remove energy outliers
    # ------------------------------------------------------------
    energy_outlier_stats: dict[str, Any]

    if use_energy_outlier_detection:
        energy_raw = df["energy"].to_numpy()
        mask_good, z_energy, stats = detect_energy_outliers(
            energy_raw,
            zmax=6.0,
            use_mad=True,
        )
        num_outliers = int(stats.get("n_outliers", 0))
        energy_outlier_stats = stats

        if num_outliers > 0:
            print("[WARN] Detected energy outliers:")
            print(
                f"       method={stats['method']}, "
                f"center={stats['center']:.6f} eV, "
                f"scale={stats['scale']:.6f} eV, "
                f"zmax={stats['zmax']}"
            )
            print(
                f"       total={stats['n_total']}, "
                f"finite={stats['n_finite']}, "
                f"outliers={stats['n_outliers']}"
            )

        df = df.iloc[mask_good].reset_index(drop=True)

    else:
        # ----- outlier detection disabled -----
        energy_outlier_stats = {"enabled": False}
        # no filtering at all

    total_samples = len(df)

    # ------------------------------------------------------------
    # Step 4: Count distinct proton states
    # ------------------------------------------------------------
    if "state" in df.columns:
        num_states = pd.Series(df["state"]).nunique()
    else:
        num_states = 0

    # ------------------------------------------------------------
    # NEW Step: Compute time statistics
    # ------------------------------------------------------------
    # if "time" in df.columns:
    #     total_time_sec = float(df["time"].sum())
    #     mean_time_per_sample = (total_time_sec / total_samples)
    # else:
    #     total_time_sec = float("nan")
    #     mean_time_per_sample = float("nan")

    # ------------------------------------------------------------
    # Step 5: Consistency checks
    # ------------------------------------------------------------
    num_molecule = int(df["num_molecule"].iloc[0])
    temperature = float(df["t_k"].iloc[0])

    if not np.allclose(df["t_k"], temperature):
        raise ValueError("Inconsistent temperature among runs.")
    if not np.all(df["num_molecule"] == num_molecule):
        raise ValueError("Inconsistent num_molecule among runs.")

    # ------------------------------------------------------------
    # Step 6: Compute observables
    # ------------------------------------------------------------
    loop_acc_rate = compute_acceptance_rate(
        df["loop_accepts_block"], df["loop_attempts_block"]
    ) if ("loop_accepts_block" in df.columns and "loop_attempts_block" in df.columns) else float("nan")
    mala_acc_rate = compute_acceptance_rate(
        df["mala_accepts_block"], df["mala_attempts_block"]
    ) if ("mala_accepts_block" in df.columns and "mala_attempts_block" in df.columns) else float("nan")
    cell_acc_rate = compute_acceptance_rate(
        df["cell_accepts_block"], df["cell_attempts_block"]
    ) if ("cell_accepts_block" in df.columns and "cell_attempts_block" in df.columns) else float("nan")

    # --- Energies (potential/kinetic/total) ---
    energy_pot = df["energy"].to_numpy(dtype=float)
    E_pot_mean, E_pot_err = compute_energy_mean_error(energy_pot, num_molecule)

    E_kin_total = compute_kinetic_energy_total(temperature, num_molecule)
    energy_tot = energy_pot + E_kin_total
    E_tot_mean, E_tot_err = compute_energy_mean_error(energy_tot, num_molecule)
    E_kin_mean = E_kin_total / float(num_molecule)

    # Heat capacities (per molecule, J/mol/K)
    C_pot_mean, C_pot_err = compute_heat_capacity(
        energy_pot, temperature, num_molecule, energy_basis="total", output_unit="J_per_mol_K"
    )
    C_tot_mean, C_tot_err = compute_heat_capacity(
        energy_tot, temperature, num_molecule, energy_basis="total", output_unit="J_per_mol_K"
    )
    
    enthalpy = energy_tot
    H_mean, H_err = compute_energy_mean_error(enthalpy, num_molecule)
    C_H_mean, C_H_err = compute_heat_capacity(
        enthalpy, temperature, num_molecule, energy_basis="total", output_unit="J_per_mol_K"
    )
    # PV_mean, PV_err = compute_energy_mean_error(pv_term, num_molecule)

    # --- Dipole-related ---
    G_mean, G_err = compute_Gcorr(df["m2"], df["mu2_mean"], num_molecule)
    Q_mean, Q_err, U_mean, U_err = compute_Q(df["m2"].to_numpy())

    # ===== NEW: <m2> and <mu2> summary (for polarization etc.) =====
    m2_mean, m2_err = compute_mean_sem(df["m2"].to_numpy())

    order_mean, order_err = compute_ordering(
        df["mx"].to_numpy(),
        df["my"].to_numpy(),
        df["mz"].to_numpy(),
        df["mu2_mean"].to_numpy(),
        num_molecule,
    )

    R_mean, R_err, V_mean, V_err = compute_R(energy_pot, center=True)

    # Polarization magnitude |P| (C/m^2)
    P_mag = compute_polarization_magnitude_cpm2(
        df["mx"].to_numpy(),
        df["my"].to_numpy(),
        df["mz"].to_numpy(),
        df["cell_a"].to_numpy(),
        df["cell_b"].to_numpy(),
        df["cell_c"].to_numpy(),
    )
    P_mag_mean, P_mag_err = compute_mean_sem(P_mag)

    # --- Cell / pressure / stress / density ---
    if all(c in df.columns for c in ("cell_a", "cell_b", "cell_c")):
        cell_a_mean, cell_a_err = compute_mean_sem(df["cell_a"].to_numpy())
        cell_b_mean, cell_b_err = compute_mean_sem(df["cell_b"].to_numpy())
        cell_c_mean, cell_c_err = compute_mean_sem(df["cell_c"].to_numpy())

        density_arr = compute_density_g_cm3(
            df["cell_a"].to_numpy(),
            df["cell_b"].to_numpy(),
            df["cell_c"].to_numpy(),
            num_molecule,
        )
        density_mean, density_err = compute_mean_sem(density_arr)
    else:
        cell_a_mean = cell_a_err = float("nan")
        cell_b_mean = cell_b_err = float("nan")
        cell_c_mean = cell_c_err = float("nan")
        density_mean = density_err = float("nan")

    if "pressure_GPa" in df.columns:
        pressure_mean, pressure_err = compute_mean_sem(df["pressure_GPa"].to_numpy())
    else:
        pressure_mean = pressure_err = float("nan")

    stress_xx_mean = stress_xx_err = float("nan")
    stress_yy_mean = stress_yy_err = float("nan")
    stress_zz_mean = stress_zz_err = float("nan")
    if "stress_xx" in df.columns:
        stress_xx_mean, stress_xx_err = compute_mean_sem(df["stress_xx"].to_numpy())
    if "stress_yy" in df.columns:
        stress_yy_mean, stress_yy_err = compute_mean_sem(df["stress_yy"].to_numpy())
    if "stress_zz" in df.columns:
        stress_zz_mean, stress_zz_err = compute_mean_sem(df["stress_zz"].to_numpy())

    # ------------------------------------------------------------
    # Step 7: Package results
    # ------------------------------------------------------------
    result: dict[str, Any] = dict(
        loop_acc_rate=float(loop_acc_rate),
        mala_acc_rate=float(mala_acc_rate),
        cell_acc_rate=float(cell_acc_rate),
        # --- Energy (potential) ---
        E_mean=float(E_pot_mean),
        E_err=float(E_pot_err),
        # --- Energy split ---
        E_pot_mean=float(E_pot_mean),
        E_pot_err=float(E_pot_err),
        E_kin_mean=float(E_kin_mean),
        E_tot_mean=float(E_tot_mean),
        E_tot_err=float(E_tot_err),
        H_mean=float(H_mean),
        H_err=float(H_err),
        # PV_mean=float(PV_mean),
        # PV_err=float(PV_err),
        # --- Heat capacity ---
        C_mean=float(C_pot_mean),
        C_err=float(C_pot_err),
        C_pot_mean=float(C_pot_mean),
        C_pot_err=float(C_pot_err),
        C_tot_mean=float(C_tot_mean),
        C_tot_err=float(C_tot_err),
        C_H_mean=float(C_H_mean),
        C_H_err=float(C_H_err),
        # --- Dipole/binder ---
        G_mean=float(G_mean),
        G_err=float(G_err),
        Q_mean=float(Q_mean),
        Q_err=float(Q_err),
        U_mean=float(U_mean),
        U_err=float(U_err),
        R_mean=float(R_mean),
        R_err=float(R_err),
        V_mean=float(V_mean),
        V_err=float(V_err),
        m2_mean=float(m2_mean),
        m2_err=float(m2_err),
        # --- Polarization magnitude ---
        P_mag_mean=float(P_mag_mean),
        P_mag_err=float(P_mag_err),
        # --- Cell / pressure / stress / density ---
        cell_a_mean=float(cell_a_mean),
        cell_a_err=float(cell_a_err),
        cell_b_mean=float(cell_b_mean),
        cell_b_err=float(cell_b_err),
        cell_c_mean=float(cell_c_mean),
        cell_c_err=float(cell_c_err),
        pressure_mean=float(pressure_mean),
        pressure_err=float(pressure_err),
        stress_xx_mean=float(stress_xx_mean),
        stress_xx_err=float(stress_xx_err),
        stress_yy_mean=float(stress_yy_mean),
        stress_yy_err=float(stress_yy_err),
        stress_zz_mean=float(stress_zz_mean),
        stress_zz_err=float(stress_zz_err),
        density_mean=float(density_mean),
        density_err=float(density_err),
        order_mean=order_mean,
        order_err=order_err,
        num_samples=total_samples,
        per_run_sample_counts=per_run_counts,
        energy_outlier_stats=energy_outlier_stats,
        num_states=num_states,
        # ---- new: time statistics ----
        # total_time_sec=total_time_sec,
        # mean_time_per_sample=mean_time_per_sample,
    )

    # ------------------------------------------------------------
    # Step 8: Verbose printout
    # ------------------------------------------------------------
    if verbose:
        print("=" * 90)
        print("[INFO] Files included:")
        for p in log_paths:
            print(f"   {p}")

        print("\n[INFO] Samples per run (after trimming):")
        for name, count in per_run_counts:
            print(f"   {name:40s}  {count:6d}")

        print(f"\n[INFO] Total samples after trim + outlier removal: {total_samples}")
        print(f"[INFO] Temperature:               {temperature:.6f} K")
        print(f"[INFO] Num molecules:             {num_molecule}")
        print(f"[INFO] Distinct states:           {num_states}")
        print(f"Loop acceptance rate:             {loop_acc_rate:.6f}")
        print(f"MALA acceptance rate:             {mala_acc_rate:.6f}")
        print(f"CELL acceptance rate:             {cell_acc_rate:.6f}")
        print(f"Energy (pot):                     {E_pot_mean:.6f} ± {E_pot_err:.6f} eV/molecule")
        print(f"Energy (kin):                     {E_kin_mean:.6f} eV/molecule")
        print(f"Energy (total):                   {E_tot_mean:.6f} ± {E_tot_err:.6f} eV/molecule")
        print(f"Energy (PV):                      {PV_mean:.6f} ± {PV_err:.6f} eV/molecule")
        print(f"Enthalpy (H):                     {H_mean:.6f} ± {H_err:.6f} eV/molecule")
        print(f"Heat capacity (pot):              {C_pot_mean:.6f} ± {C_pot_err:.6f} J/mol/K")
        print(f"Heat capacity (total):            {C_tot_mean:.6f} ± {C_tot_err:.6f} J/mol/K")
        print(f"Heat capacity (H):                {C_H_mean:.6f} ± {C_H_err:.6f} J/mol/K")
        print(f"Gcorr:                            {G_mean:.6f} ± {G_err:.6f}")
        print(f"Binder ratio Q:                   {Q_mean:.6f} ± {Q_err:.6f}")
        print(f"Binder cumulant U:                {U_mean:.6f} ± {U_err:.6f}")
        print(f"Energy moment ratio R:            {R_mean:.6f} ± {R_err:.6f}")
        print(f"Energy cumulant-like V = 1-1/R:   {V_mean:.6f} ± {V_err:.6f}")
        print(f"|P| magnitude:                    {P_mag_mean:.6f} ± {P_mag_err:.6f} C/m^2")
        print(f"cell_a:                           {cell_a_mean:.6f} ± {cell_a_err:.6f} Å")
        print(f"cell_b:                           {cell_b_mean:.6f} ± {cell_b_err:.6f} Å")
        print(f"cell_c:                           {cell_c_mean:.6f} ± {cell_c_err:.6f} Å")
        print(f"pressure:                         {pressure_mean:.6f} ± {pressure_err:.6f} GPa")
        print(f"stress_xx:                        {stress_xx_mean:.6e} ± {stress_xx_err:.6e} eV/Å^3")
        print(f"stress_yy:                        {stress_yy_mean:.6e} ± {stress_yy_err:.6e} eV/Å^3")
        print(f"stress_zz:                        {stress_zz_mean:.6e} ± {stress_zz_err:.6e} eV/Å^3")
        print(f"density:                          {density_mean:.6f} ± {density_err:.6f} g/cm^3")
        print("=" * 90)

    return result

####################################################################################################
def analyze_log_summary_multirun_rebin(
    log_paths: Sequence[str | Path],
    drop: float | int = 0.1,
    verbose: int = 1,
    use_energy_outlier_detection: bool = False,
    *,
    block_size: int | None = None,
    min_blocks: int = 8,
) -> dict[str, Any]:
    """Summarize multiple MC logs with full-sample means and block-jackknife errors.

    Means are computed from the full concatenated samples (after per-run trim
    and optional outlier removal). Error bars are estimated via block
    jackknife over per-chain blocks (no block crosses chain boundaries).

    Args:
        log_paths: Paths to MC log files.
        drop: Burn-in trimming per log.
        verbose: Print summary.
        use_energy_outlier_detection: If False, disable energy outlier removal.
        block_size: Block length for rebinning. If None, choose a power-of-two
            block size so that n_blocks >= min_blocks when possible.
        min_blocks: Minimum number of blocks for auto block size selection.
    """
    log_paths = [Path(p) for p in log_paths]
    if len(log_paths) == 0:
        raise ValueError("analyze_log_summary_multirun_rebin: log_paths is empty.")

    # ------------------------------------------------------------
    # Step 1: Read and trim per run (keep per-chain order)
    # ------------------------------------------------------------
    df_list: list[pd.DataFrame] = []
    path_list: list[Path] = []
    per_run_counts: list[tuple[str, int]] = []
    per_chain_meta: list[dict[str, Any]] = []
    per_chain_outliers: list[dict[str, Any]] = []

    for p in log_paths:
        df_i = read_log_trim(p, drop, verbose=False)
        if df_i.empty:
            continue

        df_i = ensure_temperature_column(df_i)
        df_i = ensure_m2_column(df_i)

        if use_energy_outlier_detection:
            energy_raw = df_i["energy"].to_numpy()
            mask_good, z_energy, stats = detect_energy_outliers(
                energy_raw,
                zmax=6.0,
                use_mad=True,
            )
            stats = dict(stats)
            stats["name"] = p.name
            per_chain_outliers.append(stats)
            df_i = df_i.iloc[mask_good].reset_index(drop=True)

        if df_i.empty:
            continue

        per_run_counts.append((p.name, len(df_i)))
        df_list.append(df_i)
        path_list.append(p)

    if len(df_list) == 0:
        raise RuntimeError("All logs are empty after trimming.")

    # ------------------------------------------------------------
    # Step 2: Consistency checks across chains
    # ------------------------------------------------------------
    num_molecule = int(df_list[0]["num_molecule"].iloc[0])
    temperature = float(df_list[0]["t_k"].iloc[0])
    for df_i in df_list[1:]:
        if not np.allclose(df_i["t_k"], temperature):
            raise ValueError("Inconsistent temperature among runs.")
        if not np.all(df_i["num_molecule"] == num_molecule):
            raise ValueError("Inconsistent num_molecule among runs.")

    # ------------------------------------------------------------
    # Step 3: Aggregate samples for acceptance rates and counts
    # ------------------------------------------------------------
    df_all = pd.concat(df_list, ignore_index=True)
    total_samples = len(df_all)

    if "state" in df_all.columns:
        num_states = pd.Series(df_all["state"]).nunique()
    else:
        num_states = 0

    # ------------------------------------------------------------
    # Step 4: Full-sample means (iid errors as fallback)
    # ------------------------------------------------------------
    energy_pot_all = df_all["energy"].to_numpy(dtype=float)
    E_pot_mean, E_pot_err = compute_energy_mean_error(energy_pot_all, num_molecule)

    E_kin_total = compute_kinetic_energy_total(temperature, num_molecule)
    energy_tot_all = energy_pot_all + E_kin_total
    E_tot_mean, E_tot_err = compute_energy_mean_error(energy_tot_all, num_molecule)
    H_mean, H_err = compute_energy_mean_error(energy_tot_all, num_molecule)

    C_pot_mean, C_pot_err = compute_heat_capacity(
        energy_pot_all, temperature, num_molecule,
        energy_basis="total", output_unit="J_per_mol_K"
    )
    C_tot_mean, C_tot_err = compute_heat_capacity(
        energy_tot_all, temperature, num_molecule,
        energy_basis="total", output_unit="J_per_mol_K"
    )
    C_H_mean, C_H_err = compute_heat_capacity(
        energy_tot_all, temperature, num_molecule,
        energy_basis="total", output_unit="J_per_mol_K"
    )

    m2_all = df_all["m2"].to_numpy(dtype=float)
    m2_mean, m2_err = compute_mean_sem(m2_all)
    Q_mean, Q_err, U_mean, U_err = compute_Q(m2_all)

    if "mu2_mean" in df_all.columns:
        mu2_all = df_all["mu2_mean"].to_numpy(dtype=float)
        G_mean, G_err = compute_Gcorr(m2_all, mu2_all, num_molecule)
    else:
        mu2_all = None
        G_mean = G_err = float("nan")

    R_mean, R_err, V_mean, V_err = compute_R(energy_pot_all, center=True)

    has_mx = all(c in df_all.columns for c in ("mx", "my", "mz"))
    has_cell = all(c in df_all.columns for c in ("cell_a", "cell_b", "cell_c"))
    has_pressure = "pressure_GPa" in df_all.columns
    has_stress_xx = "stress_xx" in df_all.columns
    has_stress_yy = "stress_yy" in df_all.columns
    has_stress_zz = "stress_zz" in df_all.columns

    if has_mx and (mu2_all is not None):
        order_mean, order_err = compute_ordering(
            df_all["mx"].to_numpy(dtype=float),
            df_all["my"].to_numpy(dtype=float),
            df_all["mz"].to_numpy(dtype=float),
            mu2_all,
            num_molecule,
        )
    else:
        order_mean = np.full((3, 3), np.nan)
        order_err = np.full((3, 3), np.nan)

    if has_cell:
        cell_a_mean, cell_a_err = compute_mean_sem(df_all["cell_a"].to_numpy(dtype=float))
        cell_b_mean, cell_b_err = compute_mean_sem(df_all["cell_b"].to_numpy(dtype=float))
        cell_c_mean, cell_c_err = compute_mean_sem(df_all["cell_c"].to_numpy(dtype=float))
        density_arr_all = compute_density_g_cm3(
            df_all["cell_a"].to_numpy(dtype=float),
            df_all["cell_b"].to_numpy(dtype=float),
            df_all["cell_c"].to_numpy(dtype=float),
            num_molecule,
        )
        density_mean, density_err = compute_mean_sem(density_arr_all)
    else:
        cell_a_mean = cell_a_err = float("nan")
        cell_b_mean = cell_b_err = float("nan")
        cell_c_mean = cell_c_err = float("nan")
        density_arr_all = None
        density_mean = density_err = float("nan")

    if has_pressure:
        pressure_mean, pressure_err = compute_mean_sem(df_all["pressure_GPa"].to_numpy(dtype=float))
    else:
        pressure_mean = pressure_err = float("nan")

    if has_stress_xx:
        stress_xx_mean, stress_xx_err = compute_mean_sem(df_all["stress_xx"].to_numpy(dtype=float))
    else:
        stress_xx_mean = stress_xx_err = float("nan")
    if has_stress_yy:
        stress_yy_mean, stress_yy_err = compute_mean_sem(df_all["stress_yy"].to_numpy(dtype=float))
    else:
        stress_yy_mean = stress_yy_err = float("nan")
    if has_stress_zz:
        stress_zz_mean, stress_zz_err = compute_mean_sem(df_all["stress_zz"].to_numpy(dtype=float))
    else:
        stress_zz_mean = stress_zz_err = float("nan")

    if has_mx and has_cell:
        P_mag_all = compute_polarization_magnitude_cpm2(
            df_all["mx"].to_numpy(dtype=float),
            df_all["my"].to_numpy(dtype=float),
            df_all["mz"].to_numpy(dtype=float),
            df_all["cell_a"].to_numpy(dtype=float),
            df_all["cell_b"].to_numpy(dtype=float),
            df_all["cell_c"].to_numpy(dtype=float),
        )
        P_mag_mean, P_mag_err = compute_mean_sem(P_mag_all)
    else:
        P_mag_all = None
        P_mag_mean = P_mag_err = float("nan")

    # ------------------------------------------------------------
    # Step 5: Block jackknife (per chain, no cross-chain blocks)
    # ------------------------------------------------------------
    block_n = []
    block_sum_e = []
    block_sum_e2 = []
    block_sum_e3 = []
    block_sum_e4 = []
    block_sum_m2 = []
    block_sum_m2_2 = []

    block_sum_mu2 = [] if mu2_all is not None else None
    block_sum_mx2 = [] if has_mx else None
    block_sum_my2 = [] if has_mx else None
    block_sum_mz2 = [] if has_mx else None
    block_sum_mxmy = [] if has_mx else None
    block_sum_mxmz = [] if has_mx else None
    block_sum_mymz = [] if has_mx else None

    block_sum_cell_a = [] if has_cell else None
    block_sum_cell_b = [] if has_cell else None
    block_sum_cell_c = [] if has_cell else None
    block_sum_density = [] if has_cell else None
    block_sum_pressure = [] if has_pressure else None
    block_sum_stress_xx = [] if has_stress_xx else None
    block_sum_stress_yy = [] if has_stress_yy else None
    block_sum_stress_zz = [] if has_stress_zz else None
    block_sum_Pmag = [] if (has_mx and has_cell) else None

    for (p, df_i) in zip(path_list, df_list):
        n_total = len(df_i)
        if n_total <= 0:
            continue

        if block_size is None or int(block_size) <= 0:
            block_size_i = _choose_rebin_block_size(n_total, min_blocks=min_blocks)
        else:
            block_size_i = int(block_size)

        if block_size_i < 1:
            block_size_i = 1

        n_blocks_i = n_total // block_size_i
        if n_blocks_i < 1:
            n_blocks_i = 1
            block_size_i = n_total

        n_used_i = n_blocks_i * block_size_i
        n_drop_i = n_total - n_used_i

        per_chain_meta.append(
            dict(
                name=p.name,
                num_samples_total=int(n_total),
                block_size=int(block_size_i),
                num_blocks=int(n_blocks_i),
                num_samples_used=int(n_used_i),
                num_samples_dropped=int(n_drop_i),
            )
        )

        if n_used_i <= 0:
            continue

        energy_pot = df_i["energy"].to_numpy(dtype=float)[:n_used_i]
        m2 = df_i["m2"].to_numpy(dtype=float)[:n_used_i]
        mu2 = df_i["mu2_mean"].to_numpy(dtype=float)[:n_used_i] if mu2_all is not None else None

        mx = df_i["mx"].to_numpy(dtype=float)[:n_used_i] if has_mx else None
        my = df_i["my"].to_numpy(dtype=float)[:n_used_i] if has_mx else None
        mz = df_i["mz"].to_numpy(dtype=float)[:n_used_i] if has_mx else None

        cell_a = df_i["cell_a"].to_numpy(dtype=float)[:n_used_i] if has_cell else None
        cell_b = df_i["cell_b"].to_numpy(dtype=float)[:n_used_i] if has_cell else None
        cell_c = df_i["cell_c"].to_numpy(dtype=float)[:n_used_i] if has_cell else None

        pressure = df_i["pressure_GPa"].to_numpy(dtype=float)[:n_used_i] if has_pressure else None
        stress_xx = df_i["stress_xx"].to_numpy(dtype=float)[:n_used_i] if has_stress_xx else None
        stress_yy = df_i["stress_yy"].to_numpy(dtype=float)[:n_used_i] if has_stress_yy else None
        stress_zz = df_i["stress_zz"].to_numpy(dtype=float)[:n_used_i] if has_stress_zz else None

        P_mag = None
        if (mx is not None) and (cell_a is not None):
            P_mag = compute_polarization_magnitude_cpm2(mx, my, mz, cell_a, cell_b, cell_c)

        density_arr = None
        if cell_a is not None:
            density_arr = compute_density_g_cm3(cell_a, cell_b, cell_c, num_molecule)

        for sl in _iter_block_slices(n_used_i, block_size_i):
            e_b = energy_pot[sl]
            if e_b.size == 0:
                continue

            block_n.append(int(e_b.size))
            block_sum_e.append(float(np.sum(e_b)))
            block_sum_e2.append(float(np.sum(e_b ** 2)))
            block_sum_e3.append(float(np.sum(e_b ** 3)))
            block_sum_e4.append(float(np.sum(e_b ** 4)))

            m2_b = m2[sl]
            block_sum_m2.append(float(np.sum(m2_b)))
            block_sum_m2_2.append(float(np.sum(m2_b ** 2)))

            if mu2 is not None:
                block_sum_mu2.append(float(np.sum(mu2[sl])))
            if mx is not None:
                mx_b = mx[sl]
                my_b = my[sl]
                mz_b = mz[sl]
                block_sum_mx2.append(float(np.sum(mx_b * mx_b)))
                block_sum_my2.append(float(np.sum(my_b * my_b)))
                block_sum_mz2.append(float(np.sum(mz_b * mz_b)))
                block_sum_mxmy.append(float(np.sum(mx_b * my_b)))
                block_sum_mxmz.append(float(np.sum(mx_b * mz_b)))
                block_sum_mymz.append(float(np.sum(my_b * mz_b)))

            if cell_a is not None:
                block_sum_cell_a.append(float(np.sum(cell_a[sl])))
                block_sum_cell_b.append(float(np.sum(cell_b[sl])))
                block_sum_cell_c.append(float(np.sum(cell_c[sl])))
                block_sum_density.append(float(np.sum(density_arr[sl])))

            if pressure is not None:
                block_sum_pressure.append(float(np.sum(pressure[sl])))
            if stress_xx is not None:
                block_sum_stress_xx.append(float(np.sum(stress_xx[sl])))
            if stress_yy is not None:
                block_sum_stress_yy.append(float(np.sum(stress_yy[sl])))
            if stress_zz is not None:
                block_sum_stress_zz.append(float(np.sum(stress_zz[sl])))
            if P_mag is not None:
                block_sum_Pmag.append(float(np.sum(P_mag[sl])))

    def _jk_err(theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        theta = theta[np.isfinite(theta)]
        if theta.size < 2:
            return float("nan")
        tbar = float(np.mean(theta))
        var = (theta.size - 1) / theta.size * float(np.sum((theta - tbar) ** 2))
        return float(np.sqrt(var))

    # --- Jackknife errors ---
    if len(block_n) >= 2:
        bn = np.asarray(block_n, dtype=float)
        n_tot = float(np.sum(bn))
        n_loo = n_tot - bn
        valid = n_loo > 0

        sum_e = np.asarray(block_sum_e, dtype=float)
        sum_e2 = np.asarray(block_sum_e2, dtype=float)
        sum_e3 = np.asarray(block_sum_e3, dtype=float)
        sum_e4 = np.asarray(block_sum_e4, dtype=float)
        sum_m2 = np.asarray(block_sum_m2, dtype=float)
        sum_m2_2 = np.asarray(block_sum_m2_2, dtype=float)

        sum_e_tot = float(np.sum(sum_e))
        sum_e2_tot = float(np.sum(sum_e2))
        sum_e3_tot = float(np.sum(sum_e3))
        sum_e4_tot = float(np.sum(sum_e4))
        sum_m2_tot = float(np.sum(sum_m2))
        sum_m2_2_tot = float(np.sum(sum_m2_2))

        mean_e_loo = (sum_e_tot - sum_e) / n_loo
        mean_e2_loo = (sum_e2_tot - sum_e2) / n_loo
        mean_e3_loo = (sum_e3_tot - sum_e3) / n_loo
        mean_e4_loo = (sum_e4_tot - sum_e4) / n_loo

        e_pot_mean_loo = mean_e_loo / float(num_molecule)
        e_pot_err_jk = _jk_err(e_pot_mean_loo[valid])
        if np.isfinite(e_pot_err_jk):
            E_pot_err = e_pot_err_jk
            E_tot_err = e_pot_err_jk
            H_err = e_pot_err_jk

        # Heat capacity via variance (total energy basis)
        var_e_loo = mean_e2_loo - mean_e_loo ** 2
        Cv_total_eV_per_K = var_e_loo / (units.K_B_EV_PER_K * (temperature ** 2))
        Cv_per_mol_eV_per_K = Cv_total_eV_per_K / float(num_molecule)
        Cv_per_mol_J_per_K = Cv_per_mol_eV_per_K * units.EV_TO_J * units.N_A
        cv_err_jk = _jk_err(Cv_per_mol_J_per_K[valid])
        if np.isfinite(cv_err_jk):
            C_pot_err = cv_err_jk
            C_tot_err = cv_err_jk
            C_H_err = cv_err_jk

        # m2 mean
        mean_m2_loo = (sum_m2_tot - sum_m2) / n_loo
        m2_err_jk = _jk_err(mean_m2_loo[valid])
        if np.isfinite(m2_err_jk):
            m2_err = m2_err_jk

        # Binder Q/U
        mean_m4_loo = (sum_m2_2_tot - sum_m2_2) / n_loo
        mask_q = valid & (mean_m2_loo > 0.0) & (mean_m4_loo > 0.0)
        if np.any(mask_q):
            Q_loo = (mean_m2_loo[mask_q] ** 2) / mean_m4_loo[mask_q]
            U_loo = (5.0 / 2.0) - (3.0 / (2.0 * Q_loo))
            Q_err_jk = _jk_err(Q_loo)
            U_err_jk = _jk_err(U_loo)
            if np.isfinite(Q_err_jk):
                Q_err = Q_err_jk
            if np.isfinite(U_err_jk):
                U_err = U_err_jk

        # Energy moment ratio R/V (centered)
        mu2_loo = mean_e2_loo - mean_e_loo ** 2
        mu4_loo = (
            mean_e4_loo
            - 4.0 * mean_e_loo * mean_e3_loo
            + 6.0 * (mean_e_loo ** 2) * mean_e2_loo
            - 3.0 * (mean_e_loo ** 4)
        )
        mask_r = valid & (mu2_loo > 0.0) & (mu4_loo > 0.0)
        if np.any(mask_r):
            R_loo = (mu2_loo[mask_r] ** 2) / mu4_loo[mask_r]
            V_loo = 1.0 - 1.0 / R_loo
            R_err_jk = _jk_err(R_loo)
            V_err_jk = _jk_err(V_loo)
            if np.isfinite(R_err_jk):
                R_err = R_err_jk
            if np.isfinite(V_err_jk):
                V_err = V_err_jk

        # Gcorr
        if block_sum_mu2 is not None:
            sum_mu2 = np.asarray(block_sum_mu2, dtype=float)
            sum_mu2_tot = float(np.sum(sum_mu2))
            mean_mu2_loo = (sum_mu2_tot - sum_mu2) / n_loo
            mask_g = valid & (mean_mu2_loo > 0.0)
            if np.any(mask_g):
                G_loo = mean_m2_loo[mask_g] / (float(num_molecule) * mean_mu2_loo[mask_g])
                G_err_jk = _jk_err(G_loo)
                if np.isfinite(G_err_jk):
                    G_err = G_err_jk

        # Order tensor
        if block_sum_mu2 is not None and block_sum_mx2 is not None:
            sum_mu2 = np.asarray(block_sum_mu2, dtype=float)
            sum_mu2_tot = float(np.sum(sum_mu2))
            mean_mu2_loo = (sum_mu2_tot - sum_mu2) / n_loo
            denom = (float(num_molecule) ** 2) * mean_mu2_loo
            mask_o = valid & (denom > 0.0)
            if np.any(mask_o):
                smx2 = np.asarray(block_sum_mx2, dtype=float)
                smy2 = np.asarray(block_sum_my2, dtype=float)
                smz2 = np.asarray(block_sum_mz2, dtype=float)
                smxmy = np.asarray(block_sum_mxmy, dtype=float)
                smxmz = np.asarray(block_sum_mxmz, dtype=float)
                smymz = np.asarray(block_sum_mymz, dtype=float)
                smx2_tot = float(np.sum(smx2))
                smy2_tot = float(np.sum(smy2))
                smz2_tot = float(np.sum(smz2))
                smxmy_tot = float(np.sum(smxmy))
                smxmz_tot = float(np.sum(smxmz))
                smymz_tot = float(np.sum(smymz))

                mean_mx2_loo = (smx2_tot - smx2) / n_loo
                mean_my2_loo = (smy2_tot - smy2) / n_loo
                mean_mz2_loo = (smz2_tot - smz2) / n_loo
                mean_mxmy_loo = (smxmy_tot - smxmy) / n_loo
                mean_mxmz_loo = (smxmz_tot - smxmz) / n_loo
                mean_mymz_loo = (smymz_tot - smymz) / n_loo

                o_xx = mean_mx2_loo[mask_o] / denom[mask_o]
                o_yy = mean_my2_loo[mask_o] / denom[mask_o]
                o_zz = mean_mz2_loo[mask_o] / denom[mask_o]
                o_xy = mean_mxmy_loo[mask_o] / denom[mask_o]
                o_xz = mean_mxmz_loo[mask_o] / denom[mask_o]
                o_yz = mean_mymz_loo[mask_o] / denom[mask_o]

                order_err = np.full((3, 3), np.nan)
                order_err[0, 0] = _jk_err(o_xx)
                order_err[1, 1] = _jk_err(o_yy)
                order_err[2, 2] = _jk_err(o_zz)
                order_err[0, 1] = order_err[1, 0] = _jk_err(o_xy)
                order_err[0, 2] = order_err[2, 0] = _jk_err(o_xz)
                order_err[1, 2] = order_err[2, 1] = _jk_err(o_yz)

        # Linear means
        if block_sum_cell_a is not None:
            s_ca = np.asarray(block_sum_cell_a, dtype=float)
            s_cb = np.asarray(block_sum_cell_b, dtype=float)
            s_cc = np.asarray(block_sum_cell_c, dtype=float)
            s_ca_tot = float(np.sum(s_ca))
            s_cb_tot = float(np.sum(s_cb))
            s_cc_tot = float(np.sum(s_cc))
            cell_a_err_jk = _jk_err(((s_ca_tot - s_ca) / n_loo)[valid])
            cell_b_err_jk = _jk_err(((s_cb_tot - s_cb) / n_loo)[valid])
            cell_c_err_jk = _jk_err(((s_cc_tot - s_cc) / n_loo)[valid])
            if np.isfinite(cell_a_err_jk):
                cell_a_err = cell_a_err_jk
            if np.isfinite(cell_b_err_jk):
                cell_b_err = cell_b_err_jk
            if np.isfinite(cell_c_err_jk):
                cell_c_err = cell_c_err_jk

            s_den = np.asarray(block_sum_density, dtype=float)
            s_den_tot = float(np.sum(s_den))
            density_err_jk = _jk_err(((s_den_tot - s_den) / n_loo)[valid])
            if np.isfinite(density_err_jk):
                density_err = density_err_jk

        if block_sum_Pmag is not None:
            s_pm = np.asarray(block_sum_Pmag, dtype=float)
            s_pm_tot = float(np.sum(s_pm))
            P_mag_err_jk = _jk_err(((s_pm_tot - s_pm) / n_loo)[valid])
            if np.isfinite(P_mag_err_jk):
                P_mag_err = P_mag_err_jk

        if block_sum_pressure is not None:
            s_p = np.asarray(block_sum_pressure, dtype=float)
            s_p_tot = float(np.sum(s_p))
            pressure_err_jk = _jk_err(((s_p_tot - s_p) / n_loo)[valid])
            if np.isfinite(pressure_err_jk):
                pressure_err = pressure_err_jk

        if block_sum_stress_xx is not None:
            s_xx = np.asarray(block_sum_stress_xx, dtype=float)
            s_xx_tot = float(np.sum(s_xx))
            stress_xx_err_jk = _jk_err(((s_xx_tot - s_xx) / n_loo)[valid])
            if np.isfinite(stress_xx_err_jk):
                stress_xx_err = stress_xx_err_jk

        if block_sum_stress_yy is not None:
            s_yy = np.asarray(block_sum_stress_yy, dtype=float)
            s_yy_tot = float(np.sum(s_yy))
            stress_yy_err_jk = _jk_err(((s_yy_tot - s_yy) / n_loo)[valid])
            if np.isfinite(stress_yy_err_jk):
                stress_yy_err = stress_yy_err_jk

        if block_sum_stress_zz is not None:
            s_zz = np.asarray(block_sum_stress_zz, dtype=float)
            s_zz_tot = float(np.sum(s_zz))
            stress_zz_err_jk = _jk_err(((s_zz_tot - s_zz) / n_loo)[valid])
            if np.isfinite(stress_zz_err_jk):
                stress_zz_err = stress_zz_err_jk

    # ------------------------------------------------------------
    # Step 6: Acceptance rates (overall)
    # ------------------------------------------------------------
    loop_acc_rate = compute_acceptance_rate(
        df_all["loop_accepts_block"], df_all["loop_attempts_block"]
    ) if ("loop_accepts_block" in df_all.columns and "loop_attempts_block" in df_all.columns) else float("nan")
    mala_acc_rate = compute_acceptance_rate(
        df_all["mala_accepts_block"], df_all["mala_attempts_block"]
    ) if ("mala_accepts_block" in df_all.columns and "mala_attempts_block" in df_all.columns) else float("nan")
    cell_acc_rate = compute_acceptance_rate(
        df_all["cell_accepts_block"], df_all["cell_attempts_block"]
    ) if ("cell_accepts_block" in df_all.columns and "cell_attempts_block" in df_all.columns) else float("nan")

    # ------------------------------------------------------------
    # Step 7: Package results
    # ------------------------------------------------------------
    PV_mean = float("nan")
    PV_err = float("nan")

    result: dict[str, Any] = dict(
        loop_acc_rate=float(loop_acc_rate),
        mala_acc_rate=float(mala_acc_rate),
        cell_acc_rate=float(cell_acc_rate),
        # --- Energy (potential) ---
        E_mean=float(E_pot_mean),
        E_err=float(E_pot_err),
        # --- Energy split ---
        E_pot_mean=float(E_pot_mean),
        E_pot_err=float(E_pot_err),
        E_kin_mean=float(E_kin_total / float(num_molecule)),
        E_tot_mean=float(E_tot_mean),
        E_tot_err=float(E_tot_err),
        H_mean=float(H_mean),
        H_err=float(H_err),
        # PV_mean=float(PV_mean),
        # PV_err=float(PV_err),
        # --- Heat capacity ---
        C_mean=float(C_pot_mean),
        C_err=float(C_pot_err),
        C_pot_mean=float(C_pot_mean),
        C_pot_err=float(C_pot_err),
        C_tot_mean=float(C_tot_mean),
        C_tot_err=float(C_tot_err),
        C_H_mean=float(C_H_mean),
        C_H_err=float(C_H_err),
        # --- Dipole/binder ---
        G_mean=float(G_mean),
        G_err=float(G_err),
        Q_mean=float(Q_mean),
        Q_err=float(Q_err),
        U_mean=float(U_mean),
        U_err=float(U_err),
        R_mean=float(R_mean),
        R_err=float(R_err),
        V_mean=float(V_mean),
        V_err=float(V_err),
        m2_mean=float(m2_mean),
        m2_err=float(m2_err),
        # --- Polarization magnitude ---
        P_mag_mean=float(P_mag_mean),
        P_mag_err=float(P_mag_err),
        # --- Cell / pressure / stress / density ---
        cell_a_mean=float(cell_a_mean),
        cell_a_err=float(cell_a_err),
        cell_b_mean=float(cell_b_mean),
        cell_b_err=float(cell_b_err),
        cell_c_mean=float(cell_c_mean),
        cell_c_err=float(cell_c_err),
        pressure_mean=float(pressure_mean),
        pressure_err=float(pressure_err),
        stress_xx_mean=float(stress_xx_mean),
        stress_xx_err=float(stress_xx_err),
        stress_yy_mean=float(stress_yy_mean),
        stress_yy_err=float(stress_yy_err),
        stress_zz_mean=float(stress_zz_mean),
        stress_zz_err=float(stress_zz_err),
        density_mean=float(density_mean),
        density_err=float(density_err),
        order_mean=order_mean,
        order_err=order_err,
        num_samples=total_samples,
        per_run_sample_counts=per_run_counts,
        energy_outlier_stats=(
            {"enabled": True, "per_chain": per_chain_outliers}
            if use_energy_outlier_detection
            else {"enabled": False}
        ),
        num_states=num_states,
        # --- Rebin metadata ---
        rebin_block_size=int(block_size) if block_size is not None else None,
        rebin_num_blocks=int(sum(m["num_blocks"] for m in per_chain_meta)),
        rebin_num_samples_used=int(sum(m["num_samples_used"] for m in per_chain_meta)),
        rebin_num_samples_dropped=int(sum(m["num_samples_dropped"] for m in per_chain_meta)),
        rebin_per_chain=per_chain_meta,
    )

    # ------------------------------------------------------------
    # Step 8: Verbose printout
    # ------------------------------------------------------------
    if verbose:
        print("=" * 90)
        print("[INFO] Files included:")
        for p in path_list:
            print(f"   {p}")

        print("\n[INFO] Samples per run (after trimming):")
        for name, count in per_run_counts:
            print(f"   {name:40s}  {count:6d}")

        print(f"\n[INFO] Total samples after trim + outlier removal: {total_samples}")
        if block_size is None:
            print(f"[INFO] Rebin block size:          auto (chains={len(per_chain_meta)})")
        else:
            print(f"[INFO] Rebin block size:          {block_size} (chains={len(per_chain_meta)})")
        print(f"[INFO] Temperature:               {temperature:.6f} K")
        print(f"[INFO] Num molecules:             {num_molecule}")
        print(f"[INFO] Distinct states:           {num_states}")
        print(f"Loop acceptance rate:             {loop_acc_rate:.6f}")
        print(f"MALA acceptance rate:             {mala_acc_rate:.6f}")
        print(f"CELL acceptance rate:             {cell_acc_rate:.6f}")
        print(f"Energy (pot):                     {E_pot_mean:.6f} ± {E_pot_err:.6f} eV/molecule")
        print(f"Energy (kin):                     {E_kin_total / float(num_molecule):.6f} eV/molecule")
        print(f"Energy (total):                   {E_tot_mean:.6f} ± {E_tot_err:.6f} eV/molecule")
        print(f"Energy (PV):                      {PV_mean:.6f} ± {PV_err:.6f} eV/molecule")
        print(f"Enthalpy (H):                     {H_mean:.6f} ± {H_err:.6f} eV/molecule")
        print(f"Heat capacity (pot):              {C_pot_mean:.6f} ± {C_pot_err:.6f} J/mol/K")
        print(f"Heat capacity (total):            {C_tot_mean:.6f} ± {C_tot_err:.6f} J/mol/K")
        print(f"Heat capacity (H):                {C_H_mean:.6f} ± {C_H_err:.6f} J/mol/K")
        print(f"Gcorr:                            {G_mean:.6f} ± {G_err:.6f}")
        print(f"Binder ratio Q:                   {Q_mean:.6f} ± {Q_err:.6f}")
        print(f"Binder cumulant U:                {U_mean:.6f} ± {U_err:.6f}")
        print(f"Energy moment ratio R:            {R_mean:.6f} ± {R_err:.6f}")
        print(f"Energy cumulant-like V = 1-1/R:   {V_mean:.6f} ± {V_err:.6f}")
        print(f"|P| magnitude:                    {P_mag_mean:.6f} ± {P_mag_err:.6f} C/m^2")
        print(f"cell_a:                           {cell_a_mean:.6f} ± {cell_a_err:.6f} Å")
        print(f"cell_b:                           {cell_b_mean:.6f} ± {cell_b_err:.6f} Å")
        print(f"cell_c:                           {cell_c_mean:.6f} ± {cell_c_err:.6f} Å")
        print(f"pressure:                         {pressure_mean:.6f} ± {pressure_err:.6f} GPa")
        print(f"stress_xx:                        {stress_xx_mean:.6e} ± {stress_xx_err:.6e} eV/Å^3")
        print(f"stress_yy:                        {stress_yy_mean:.6e} ± {stress_yy_err:.6e} eV/Å^3")
        print(f"stress_zz:                        {stress_zz_mean:.6e} ± {stress_zz_err:.6e} eV/Å^3")
        print(f"density:                          {density_mean:.6f} ± {density_err:.6f} g/cm^3")
        print("=" * 90)

    return result

####################################################################################################
####################################################################################################

def analyze_energy_distribution_multirun(
    log_paths: Sequence[str | Path],
    drop: float | int = 0.1,
    bins: int = 120,
    energy_range: tuple[float, float] | None = None,
    density: bool = True,
    zmax: float = 6.0,
    use_mad: bool = True,
    use_energy_outlier_detection: bool = False,
    verbose: int = 0,
) -> dict[str, Any]:
    """Build energy and m^2 probability distributions from multiple MC logs.

    Steps:
        - Read and trim each log.
        - Concatenate all runs.
        - Optionally detect and remove energy outliers.
        - Check consistency of temperature and system size.
        - Build energy histogram P(E).
        - Build m^2 histogram P(m^2) using the same cleaned sample set.

    Args:
        log_paths: List of MC log files.
        drop: Fraction or number of initial samples to drop per run
            (thermalization).
        bins: Number of histogram bins (used for both energy and m^2).
        energy_range: Optional (emin, emax) in eV. If None, the range is
            inferred from the energy data after outlier removal (or from all
            data if outlier detection is disabled).
        density: If True, return normalized probability density P(x)
            (np.histogram(density=True)) for both E and m^2.
        zmax: Z-score threshold for energy outlier detection. Ignored if
            use_energy_outlier_detection is False.
        use_mad: If True, use MAD-based robust z-score in
            detect_energy_outliers. Ignored if use_energy_outlier_detection
            is False.
        use_energy_outlier_detection: If False, disable energy outlier
            removal (all finite samples are kept).
        verbose: If non-zero, print basic information and outlier stats.

    Returns:
        dict[str, Any]: Dictionary with keys

            Energy-related:
                - "bin_edges": np.ndarray
                    Histogram bin edges for energy (size = bins + 1).
                - "bin_centers": np.ndarray
                    Histogram bin centers for energy (size = bins).
                - "hist": np.ndarray
                    Energy histogram (counts or density, size = bins).
                - "energy_raw": np.ndarray
                    Raw energies before outlier removal.
                - "energy_good": np.ndarray
                    Energies after outlier removal (or equal to energy_raw if
                    outlier detection is disabled).
                - "mask_good": np.ndarray[bool]
                    Mask of good samples (same length as energy_raw).
                - "energy_outlier_stats": dict
                    Diagnostics returned by detect_energy_outliers, or
                    {"enabled": False} if outlier detection is disabled.
                - "num_samples_raw": int
                    Number of raw samples.
                - "num_samples_good": int
                    Number of samples after outlier removal (or equal to
                    num_samples_raw if detection is disabled).

            m^2-related:
                - "m2_bin_edges": np.ndarray
                    Histogram bin edges for m^2 (size = bins + 1).
                - "m2_bin_centers": np.ndarray
                    Histogram bin centers for m^2 (size = bins).
                - "m2_hist": np.ndarray
                    m^2 histogram (counts or density, size = bins).
                - "m2_raw": np.ndarray
                    Raw m^2 values before energy outlier removal.
                - "m2_good": np.ndarray
                    m^2 values after applying the same energy-based mask
                    (or equal to m2_raw if detection is disabled).

            Shared metadata:
                - "temperature": float
                    Common temperature in logs (K).
                - "num_molecule": int
                    Common number of molecules.

    Raises:
        ValueError:
            - If log_paths is empty.
            - If the logs are missing the "m2" column.
            - If temperature or num_molecule are inconsistent among runs.
        RuntimeError:
            If all logs are empty after trimming.
    """

    log_paths = [Path(p) for p in log_paths]
    if len(log_paths) == 0:
        raise ValueError("analyze_energy_distribution_multirun: log_paths is empty.")

    # ------------------------------------------------------------
    # Step 1: Read + trim per run
    # ------------------------------------------------------------
    df_list: list[pd.DataFrame] = []

    for p in log_paths:
        df_i = read_log_trim(p, drop, verbose=False)
        if df_i.empty:
            continue
        df_list.append(df_i)

    if len(df_list) == 0:
        raise RuntimeError("All logs are empty after trimming.")

    # ------------------------------------------------------------
    # Step 2: Concatenate all logs
    # ------------------------------------------------------------
    df = pd.concat(df_list, ignore_index=True)
    df = ensure_temperature_column(df)
    df = ensure_m2_column(df)

    if "m2" not in df.columns:
        raise ValueError(
            "analyze_energy_distribution_multirun: required column 'm2' "
            "is missing from the logs."
        )

    # Raw arrays before outlier removal
    energy_raw = df["energy"].to_numpy()
    m2_raw = df["m2"].to_numpy()

    # ------------------------------------------------------------
    # Step 3: (Optional) Detect and remove energy outliers
    # ------------------------------------------------------------
    if use_energy_outlier_detection:
        mask_good, z_energy, stats = detect_energy_outliers(
            energy_raw,
            zmax=zmax,
            use_mad=use_mad,
        )

        num_samples_raw = int(energy_raw.size)
        num_samples_good = int(mask_good.sum())
        num_outliers = int(stats.get("n_outliers", 0))

        if num_outliers > 0:
            print("[WARN] Detected energy outliers:")
            print(
                f"       method={stats['method']}, "
                f"center={stats['center']:.6f} eV, "
                f"scale={stats['scale']:.6f} eV, "
                f"zmax={stats['zmax']}"
            )
            print(
                f"       total={stats['n_total']}, "
                f"finite={stats['n_finite']}, "
                f"outliers={stats['n_outliers']}"
            )

            out_idx = np.where(~mask_good)[0]
            print("       First few outliers (index, energy[eV], z):")
            for i in out_idx[:10]:
                print(f"         {i:7d}  {energy_raw[i]: .8f}  z={z_energy[i]: .3f}")
    else:
        # Outlier detection disabled: keep all finite samples
        mask_good = np.isfinite(energy_raw)
        stats = {"enabled": False}
        num_samples_raw = int(energy_raw.size)
        num_samples_good = int(mask_good.sum())

    # Keep only good samples
    df = df.iloc[mask_good].reset_index(drop=True)
    energy_good = df["energy"].to_numpy()
    m2_good = df["m2"].to_numpy()

    # ------------------------------------------------------------
    # Step 4: Consistency checks (temperature, num_molecule)
    # ------------------------------------------------------------
    num_molecule = int(df["num_molecule"].iloc[0])
    temperature = float(df["t_k"].iloc[0])

    if not np.allclose(df["t_k"], temperature):
        raise ValueError("Inconsistent temperature among runs.")
    if not np.all(df["num_molecule"] == num_molecule):
        raise ValueError("Inconsistent num_molecule among runs.")

    # ------------------------------------------------------------
    # Step 5: Build energy histogram (probability distribution)
    # ------------------------------------------------------------
    if energy_range is None:
        e_min = float(energy_good.min())
        e_max = float(energy_good.max())
        energy_range = (e_min, e_max)

    hist, bin_edges = np.histogram(
        energy_good,
        bins=bins,
        range=energy_range,
        density=density,
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # ------------------------------------------------------------
    # Step 6: Build m^2 histogram (probability distribution)
    # ------------------------------------------------------------
    m2_min = float(m2_good.min())
    m2_max = float(m2_good.max())
    m2_range = (m2_min, m2_max)

    m2_hist, m2_bin_edges = np.histogram(
        m2_good,
        bins=bins,
        range=m2_range,
        density=density,
    )
    m2_bin_centers = 0.5 * (m2_bin_edges[:-1] + m2_bin_edges[1:])

    # ------------------------------------------------------------
    # Verbose summary
    # ------------------------------------------------------------
    if verbose >= 1:
        print("=" * 90)
        print("[INFO] Energy and m^2 distribution analysis")
        print("[INFO] Files included:")
        for p in log_paths:
            print(f"   {p}")
        print(f"\n[INFO] Temperature:            {temperature:.6f} K")
        print(f"[INFO] Num molecules:          {num_molecule}")
        print(f"[INFO] Raw samples:            {num_samples_raw}")
        print(f"[INFO] Samples after outliers: {num_samples_good}")
        print(f"[INFO] Histogram bins:         {bins}")
        print(
            f"[INFO] Energy range:           "
            f"[{energy_range[0]:.6f}, {energy_range[1]:.6f}] eV"
        )
        print(
            f"[INFO] m^2 range:              "
            f"[{m2_range[0]:.6e}, {m2_range[1]:.6e}]"
        )
        print("=" * 90)

    result: dict[str, Any] = dict(
        # Energy
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        hist=hist,
        energy_raw=energy_raw,
        energy_good=energy_good,
        mask_good=mask_good,
        energy_outlier_stats=stats,
        num_samples_raw=num_samples_raw,
        num_samples_good=num_samples_good,
        # m^2
        m2_bin_edges=m2_bin_edges,
        m2_bin_centers=m2_bin_centers,
        m2_hist=m2_hist,
        m2_raw=m2_raw,
        m2_good=m2_good,
        # Metadata
        temperature=temperature,
        num_molecule=num_molecule,
    )

    return result

####################################################################################################
####################################################################################################

def get_gr(x, y, L, bins=100, alpha=1.0, rm_dist=1e-6):
    """
        Compute radial distribution function (RDF) between two particle sets x and y.
    Input:
        x: array, shape (batchsize, nx, dim), positions of particles in the first set
        y: array, shape (batchsize, ny, dim), positions of particles in the second set
        L: float, box size
        bins: int, number of bins for histogram
        alpha: float, exponent for non-linear binning
        rm_dist: float, minimum distance to consider (to avoid same particle zero distance)
    Output:
        rmesh: array, shape (bins,), midpoints of the bins
        gr: array, shape (bins,), radial distribution function values
    """

    # Validate input shapes
    batchsize, nx, dim = x.shape
    batchsize, ny, dim = y.shape

    # Calculate all pairwise differences with periodic boundary conditions
    rij = x.reshape(-1, nx, 1, dim) - y.reshape(-1, 1, ny, dim)  # Shape: (batchsize, nx, ny, dim)
    rij = rij - L * np.rint(rij / L)  # Apply periodic boundary correction

    # Compute pairwise distances
    dij = np.linalg.norm(rij, axis=-1)  # Shape: (batchsize, nx, ny)
    dij_flat = dij.reshape(-1)  # Flatten to 1D array
    dij_flat = dij_flat[dij_flat > rm_dist] # Remove zero distances

    # non-linear bin edges
    i = np.linspace(0, bins, bins + 1)
    bin_edges = (i / bins) ** alpha * (L / 2)

    # Generate distance histogram
    hist, bin_edges = np.histogram(dij_flat, bins=bin_edges)

    # Calculate expected particle count in ideal gas
    volume_bins = (4/3) * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)  # Spherical shell volumes
    density_pairs = (nx * ny * batchsize) / (L**3)  # Pair density
    h_id = volume_bins * density_pairs  # Expected count in ideal gas

    # Compute radial distribution function
    rmesh = (bin_edges[1:] + bin_edges[:-1]) / 2
    gr = hist / h_id  # Normalize actual counts by ideal gas expectation
    
    return rmesh, gr




####################################################################################################
####################################################################################################

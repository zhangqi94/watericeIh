import numpy as np

####################################################################################################
def safe_div(numerator: int, denominator: int) -> float:
    """Safely divide two integers; return NaN if denominator == 0."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator

def fmt_rate(value: float) -> str:
    """Format acceptance rate; return 'n/a' if value is NaN or infinite."""
    if not np.isfinite(value):  # handles NaN and ±inf
        value = 0.0
    return f"{value:.8f}"

def str2bool(v: str) -> bool:
    """Convert common string representations of truth to bool."""
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1", "y"):
        return True
    elif v in ("no", "false", "f", "0", "n"):
        return False
    else:
        raise ValueError(f"Invalid boolean value: {v}")

####################################################################################################
def mic_vec(d: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """
    Map displacement vectors into the **minimum-image convention** under orthorhombic PBC.
    """
    d = np.asarray(d, dtype=float)
    L = np.asarray(box_lengths, dtype=float)
    return d - np.round(d / L) * L

def wrap_pos(pos: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """
    Wrap Cartesian positions into the simulation box [0, L) under orthorhombic PBC.
    """
    pos = np.asarray(pos, dtype=float)
    L = np.asarray(box_lengths, dtype=float)
    return pos - np.floor(pos / L) * L

def normalize_vec(v: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Normalize vectors along the last axis with numerical safety.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

####################################################################################################
def calculate_pressure_from_stress(
    stress: np.ndarray,
    volume: np.ndarray,
    num_atoms: int,
    temperature_in_eV: float,
) -> tuple:
    """
    Calculate pressure from stress tensor with kinetic and virial contributions.

    In Monte Carlo simulations, the total pressure has two components:
    1. Kinetic contribution: P_kinetic = NkT/V (from thermal motion)
    2. Virial contribution: P_virial = -(σ_xx + σ_yy + σ_zz)/3 (from forces)

    The stress tensor from ML potentials (like MACE) only contains the virial
    contribution, so we must add the kinetic term separately.

    Args:
        stress: Stress tensor, shape (6,) or (n_steps, 6) in eV/Å³.
                Components are [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz].
        volume: Volume in Å³, scalar or array of shape (n_steps,).
        num_atoms: Total number of atoms in the system.
        temperature_in_eV: Temperature in eV (k_B * T).

    Returns:
        tuple: (P_kinetic, P_virial, P_total) all in eV/Å³
            - P_kinetic: Kinetic contribution (NkT/V)
            - P_virial: Virial contribution from stress tensor
            - P_total: Total pressure (P_kinetic + P_virial)
    """
    stress = np.asarray(stress, dtype=float)
    volume = np.asarray(volume, dtype=float)

    # Kinetic contribution: P = NkT/V
    P_kinetic = num_atoms * temperature_in_eV / volume

    # Virial contribution: P = -(σ_xx + σ_yy + σ_zz)/3
    # Handle both single stress tensor (6,) and trajectory (n_steps, 6)
    if stress.ndim == 1:
        P_virial = -(stress[0] + stress[1] + stress[2]) / 3.0
    else:
        P_virial = -(stress[:, 0] + stress[:, 1] + stress[:, 2]) / 3.0

    # Total pressure
    P_total = P_kinetic + P_virial

    return P_kinetic, P_virial, P_total

####################################################################################################
def lorentzian(x, x0, gamma):
    """
    Normalized Lorentzian line shape.

    The integral over x is equal to 1.

    Args:
        x: Evaluation grid.
        x0: Center frequency.
        gamma: Full width at half maximum (FWHM).

    Returns:
        Lorentzian evaluated at x.
    """
    return (1.0 / np.pi) * (0.5 * gamma) / ((x - x0) ** 2 + (0.5 * gamma) ** 2)


def phonon_dos_lorentz(
    freqs_cm1,
    gamma_cm1=10.0,
    wmin=0.0,
    wmax=None,
    nw=4000,
    drop_below=1.0,
    positive_only=True,
):
    """
    Construct a Lorentzian-broadened phonon density of states (DOS).

    Args:
        freqs_cm1: Phonon frequencies (3N,) in cm^-1.
        gamma_cm1: Lorentzian FWHM (cm^-1).
        wmin: Minimum frequency of the DOS grid (cm^-1).
        wmax: Maximum frequency of the DOS grid (cm^-1).
              If None, it is set automatically from the data.
        nw: Number of frequency grid points.
        drop_below: Discard modes with |frequency| < drop_below (cm^-1),
                    typically used to remove near-zero acoustic modes.
        positive_only: If True, only positive frequencies are included
                       in the DOS. Negative (imaginary) modes are returned
                       separately.

    Returns:
        w: Frequency grid (cm^-1).
        dos: Lorentzian-broadened phonon DOS on grid w.
        f_pos: Positive frequencies included in the DOS.
        f_neg: Negative (imaginary) frequencies excluded from the DOS.
    """
    f = np.asarray(freqs_cm1, dtype=float)

    # Remove near-zero modes (e.g. acoustic modes around Gamma)
    f = f[np.abs(f) >= drop_below]

    if positive_only:
        f_pos = f[f > 0.0]
        f_neg = f[f < 0.0]
    else:
        f_pos = f
        f_neg = np.array([])

    # Automatically determine the upper bound of the frequency grid
    if wmax is None:
        wmax = (
            max(10.0, float(np.max(np.abs(f_pos))) + 5.0 * gamma_cm1)
            if f_pos.size
            else 4000.0
        )

    w = np.linspace(wmin, wmax, int(nw))
    dos = np.zeros_like(w)

    # Vectorized evaluation:
    # DOS(w) = sum_i L(w; f_i, gamma)
    if f_pos.size:
        dos = np.sum(
            lorentzian(w[:, None], f_pos[None, :], gamma_cm1),
            axis=1,
        )
        # Each mode integrates to 1, so the total area is approximately
        # equal to the number of modes.
        # To normalize the total area to 1 instead, use:
        # dos /= np.trapz(dos, w)

    return w, dos, f_pos, f_neg
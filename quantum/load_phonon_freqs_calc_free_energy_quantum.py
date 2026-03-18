#%%
####################################################################################################
import os
import sys

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (one level up)
work_dir = os.path.dirname(current_dir)

# Change current working directory
os.chdir(work_dir)
print(f"[INFO] Working directory set to: {os.getcwd()}")

# Add the parent directory to Python's module search path
# This allows importing project modules from anywhere
if work_dir not in sys.path:
    sys.path.insert(0, work_dir)

####################################################################################################
import json
import ase
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from source.tools import phonon_dos_lorentz


#%%
####################################################################################################

# freq_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_freq_H2O.json"
# xyz_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_relax.xyz"

freq_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_freq_H2O.json"
xyz_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_relax.xyz"

# freq_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_freq_D2O.json"
# xyz_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_relax.xyz"

# freq_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_freq_D2O.json"
# xyz_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_relax.xyz"


# Load phonon json
with open(freq_file, "r") as f:
    data = json.load(f)

print("[INFO] JSON keys:", data.keys())
print("[INFO] JSON nframes:", len(data["frames"]))

# Load extxyz energies E0
xyz_frames = ase.io.read(xyz_file, index=":")
if isinstance(xyz_frames, ase.Atoms):
    xyz_frames = [xyz_frames]

print("[INFO] XYZ nframes:", len(xyz_frames))

n_json = len(data["frames"])
n_xyz  = len(xyz_frames)
assert n_json == n_xyz, f"Frame mismatch: json={n_json}, xyz={n_xyz}"

E0 = np.zeros(n_xyz, dtype=float)
for i, at in enumerate(xyz_frames):
    if "energy" in at.info:
        E0[i] = float(at.info["energy"])
    else:
        # fallback if not stored in info
        E0[i] = float(at.get_potential_energy())

print("[INFO] E0 first 5 (eV):", E0[:5])


#%%
####################################################################################################
#%%
####################################################################################################
# Vibrational free energy from phonon frequencies (cm^-1)
kB_eV_per_K = 8.617333262145e-5          # eV/K
hc_eV_cm    = 1.2398419843320026e-4      # eV*cm  (E = h c * (cm^-1))

def vib_free_energy_from_cm1(freqs_cm1, T, drop_below=1.0, imag_policy="skip"):
    """
    Compute F_vib(T) = sum_k [ (eps_k/2) + kB T ln(1 - exp(-eps_k/(kB T))) ]
    where eps_k = h c * freq_k, freq in cm^-1.

    freqs_cm1: array-like, can include negative/near-zero.
    T: temperature in K (T=0 allowed -> ZPE only)
    drop_below: filter |freq| < drop_below (cm^-1) to remove near-zero modes
    imag_policy:
        - "skip": skip imaginary modes (freq < 0)
        - "abs":  use |freq| (diagnostic; not physical if truly unstable)
        - "error": raise if any imaginary mode exists after filtering
    """
    freqs = np.asarray(freqs_cm1, dtype=float)

    # filter small |f|
    mask_small = np.abs(freqs) < float(drop_below)
    freqs2 = freqs[~mask_small]
    n_skipped_small = int(mask_small.sum())

    # handle imaginary
    imag = freqs2 < 0.0
    n_imag = int(imag.sum())
    if n_imag > 0:
        if imag_policy == "skip":
            freqs3 = freqs2[~imag]
        elif imag_policy == "abs":
            freqs3 = np.abs(freqs2)
        elif imag_policy == "error":
            raise ValueError(f"Found {n_imag} imaginary modes (<0 cm^-1) after filtering.")
        else:
            raise ValueError(f"Unknown imag_policy={imag_policy}")
    else:
        freqs3 = freqs2

    eps = hc_eV_cm * freqs3  # eV

    if T <= 0:
        F = 0.5 * np.sum(eps)
        return float(F), int(len(eps)), n_imag, n_skipped_small

    beta = 1.0 / (kB_eV_per_K * T)
    x = beta * eps

    # stable computation: log(1-exp(-x)) = log1p(-exp(-x))
    thermal = (kB_eV_per_K * T) * np.log1p(-np.exp(-x))
    F = np.sum(0.5 * eps + thermal)

    return float(F), int(len(eps)), n_imag, n_skipped_small


#%%
####################################################################################################
# ---- temperature grid ----
Tmin, Tmax, dT = 10.0, 250.0, 0.1


Tgrid = np.arange(Tmin, Tmax + 1e-12, dT)
nT = len(Tgrid)

drop0 = 1.0
imag_policy = "skip"

nframes = len(data["frames"])
assert nframes == len(E0), f"Frame mismatch: json={nframes}, E0={len(E0)}"
assert nframes >= 2, "Need at least 2 frames to average frames>=1."

# ---- residual entropy (applied to frames i>=1 only) ----
kB_eV_per_K = 8.617333262145e-5
# ln32 = np.log(3.0 / 2.0)
ln32 = np.log(1.507)
# ln32 = np.log(1.5)

natoms0 = len(xyz_frames[0])
N_h2o = natoms0 / 3.0   # 如果 xyz 是 O-only，就改成 N_h2o = natoms0
S_H_eV_per_K = N_h2o * kB_eV_per_K * ln32

print(f"[INFO] frame0 natoms={natoms0}, inferred N_h2o={N_h2o} (=natoms/3). <-- sanity-check")
print(f"[INFO] residual entropy applied to frames i>=1: S_H={S_H_eV_per_K:.6e} eV/K")

# ---- helper to get freqs ----
def get_freqs(i):
    return np.array(data["frames"][i]["frequencies_cm1"], dtype=float)

# ---- compute curve for frame 0 ----
F0 = np.zeros(nT, dtype=float)
freqs0 = get_freqs(0)
for j, T in enumerate(Tgrid):
    Fvib0, *_ = vib_free_energy_from_cm1(freqs0, float(T), drop_below=drop0, imag_policy=imag_policy)
    F0[j] = E0[0] + Fvib0

# ---- compute mean curve over frames 1..end ----
Fmean = np.zeros(nT, dtype=float)
for j, T in enumerate(Tgrid):
    Fi = []
    for i in range(1, nframes):
        freqs = get_freqs(i)
        Fvib, *_ = vib_free_energy_from_cm1(freqs, float(T), drop_below=drop0, imag_policy=imag_policy)
        Fi.append(E0[i] + Fvib - T * S_H_eV_per_K)
    Fmean[j] = float(np.mean(Fi))


#%%

# ---- plot both ----
# ---- plot both ----
plt.figure(figsize=(7, 4), dpi=300)
plt.plot(Tgrid, F0, lw=2.0, label="Ice XI : $E_0+F_{vib}$ (no $S_H$)")
plt.plot(Tgrid, Fmean, lw=2.0, label="Ice Ih : $E_0+F_{vib}-T S_H$")

plt.xlabel("Temperature (K)")
plt.ylabel(r"$F$ (eV)")
plt.title("Free energy curves")

plt.grid(True, which="both", ls="--", lw=0.6, alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# ---- plot relative curves (shifted) ----
plt.figure(figsize=(7, 4), dpi=300)
plt.plot(Tgrid, F0 - F0, lw=2.0, label="Ice XI (reference)")
plt.plot(Tgrid, Fmean - F0, lw=2.0, label="Ice Ih minus Ice XI")

plt.xlabel("Temperature (K)")
plt.ylabel(r"$\Delta F(T) = F(T) - F_{\mathrm{Ice\ XI}}(T)$ (eV)")
plt.title("Free-energy difference: Ice Ih vs Ice XI")

plt.grid(True, which="both", ls="--", lw=0.6, alpha=0.4)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%

# ΔF(T)
dF = Fmean - F0

# 找绝对值最小的位置
idx0 = np.argmin(np.abs(dF))
Tc_est = Tgrid[idx0]

print(f"[rough] Crossing near T ≈ {Tc_est:.3f} K, ΔF = {dF[idx0]:.3e} eV")

# %%



#%%
####################################################################################################
# ---- print ZPE info (per frame + stats) ----

def zpe_from_frame(i: int) -> tuple[float, int, int, int]:
    """Compute ZPE for a given frame index.

    Args:
        i: Frame index in the JSON frames list.

    Returns:
        A tuple (zpe_eV, n_used_modes, n_imag, n_skipped_small):
          - zpe_eV: Zero-point energy in eV (sum_k 0.5 * h c * nu_k).
          - n_used_modes: Number of modes included after filters and imag handling.
          - n_imag: Number of imaginary modes (<0 cm^-1) encountered after small-mode filtering.
          - n_skipped_small: Number of modes skipped due to |freq| < drop0.
    """
    freqs = get_freqs(i)
    zpe_eV, n_used, n_imag, n_skipped_small = vib_free_energy_from_cm1(
        freqs_cm1=freqs,
        T=0.0,
        drop_below=drop0,
        imag_policy=imag_policy,
    )
    return zpe_eV, n_used, n_imag, n_skipped_small


print(
    "\n[ZPE] Settings: "
    f"drop_below={drop0} cm^-1, imag_policy={imag_policy}, T=0 -> ZPE only"
)

# frame 0
zpe0_eV, n0_used, n0_imag, n0_small = zpe_from_frame(0)
print(
    f"[ZPE] frame 0: ZPE={zpe0_eV: .8f} eV | "
    f"used_modes={n0_used} | imag={n0_imag} | dropped_small={n0_small}"
)

# frames 1..end
zpe_list: list[float] = []
for i in range(1, nframes):
    zpe_i, n_used, n_imag, n_small = zpe_from_frame(i)
    zpe_list.append(zpe_i)
    print(
        f"[ZPE] frame {i:4d}: ZPE={zpe_i: .8f} eV | "
        f"used_modes={n_used} | imag={n_imag} | dropped_small={n_small}"
    )

zpe_arr = np.array(zpe_list, dtype=float) if len(zpe_list) > 0 else np.array([], dtype=float)

# optional stats for frames 1..end
if zpe_arr.size > 0:
    zpe_mean = float(zpe_arr.mean())
    zpe_std = float(zpe_arr.std(ddof=1)) if zpe_arr.size > 1 else 0.0
    zpe_min = float(zpe_arr.min())
    zpe_max = float(zpe_arr.max())
    print(
        f"[ZPE] frames 1..end stats: mean={zpe_mean: .8f} eV, "
        f"std={zpe_std: .8f} eV, min={zpe_min: .8f} eV, max={zpe_max: .8f} eV"
    )

# Optional: ZPE-corrected static energy (E0 + ZPE)
print(f"[ZPE] frame 0: E0={E0[0]: .8f} eV, E0+ZPE={E0[0] + zpe0_eV: .8f} eV")
if zpe_arr.size > 0:
    e0p_mean = float((E0[1:] + zpe_arr).mean())
    print(f"[ZPE] frames 1..end: mean(E0+ZPE)={e0p_mean: .8f} eV")


#%%
####################################################################################################
# ---- final ZPE comparison: frame 0 vs averaged others ----

#%%
####################################################################################################
# ---- final ZPE comparison: frame 0 vs averaged others ----

print("\n[ZPE] ===== Final comparison =====")

# ---- frame 0 ----
print(f"[ZPE] frame 0 (reference, cell): ZPE = {zpe0_eV: .8f} eV")

zpe0_per_h2o = zpe0_eV / N_h2o
print(f"[ZPE] frame 0 (reference, per H2O): ZPE = {zpe0_per_h2o: .8f} eV/H2O")

# ---- frames 1..end ----
if zpe_arr.size > 0:
    zpe_mean_rest = zpe_arr.mean()
    zpe_std_rest = zpe_arr.std(ddof=1) if zpe_arr.size > 1 else 0.0

    zpe_mean_rest_per_h2o = zpe_mean_rest / N_h2o
    zpe_std_rest_per_h2o = zpe_std_rest / N_h2o

    print(
        f"[ZPE] frames 1..end (cell): "
        f"ZPE = {zpe_mean_rest: .8f} ± {zpe_std_rest: .8f} eV"
    )
    print(
        f"[ZPE] frames 1..end (per H2O): "
        f"ZPE = {zpe_mean_rest_per_h2o: .8f} ± {zpe_std_rest_per_h2o: .8f} eV/H2O"
    )

    # ---- difference Ih - ref ----
    delta_zpe_cell = zpe_mean_rest - zpe0_eV
    delta_zpe_per_h2o = delta_zpe_cell / N_h2o

    print(f"[ZPE] ΔZPE (Ih − ref, cell): {delta_zpe_cell: .8f} eV")
    print(f"[ZPE] ΔZPE (Ih − ref, per H2O): {delta_zpe_per_h2o: .8f} eV/H2O")

else:
    print("[ZPE] no frames >=1 available for averaging.")# %%

# %%

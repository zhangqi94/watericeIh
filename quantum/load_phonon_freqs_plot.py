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
import numpy as np
import matplotlib.pyplot as plt
from source.tools import phonon_dos_lorentz


#%%
####################################################################################################

file_name = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_freq_D2O.json"
# file_name = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_freq_H2O.json"

with open(file_name, "r") as f:
    data = json.load(f)

print("[INFO] keys:", data.keys())
print("[INFO] nframes:", len(data["frames"]))

#%%
####################################################################################################

iframe = 10

freqs = np.array(data["frames"][iframe]["frequencies_cm1"])
n_imag = data["frames"][iframe]["n_imag_lt_-1"]
min_freq = data["frames"][iframe]["min_freq_cm1"]

print(f"[FRAME {iframe}]")
print(f"  min freq = {min_freq:.4f} cm^-1")
print(f"  n_imag   = {n_imag}")
print(f"  nmodes   = {len(freqs)}")
print("  lowest 12 freqs:")
print(freqs[:12])


#%%
####################################################################################################

gamma = 20.0   # cm^-1
drop0 = 1.0    # 过滤 |freq| < drop0 的近零模

# 你想画哪些 frame（自己改）
frames_to_plot = [0, 5, 10, 20, 30]   # 例子

plt.figure(figsize=(8, 4), dpi=300)

for iframe in frames_to_plot:
    freqs = np.array(data["frames"][iframe]["frequencies_cm1"], dtype=float)

    w, dos, f_pos, f_neg = phonon_dos_lorentz(
        freqs,
        gamma_cm1=gamma,
        drop_below=drop0,
        positive_only=True,
    )

    plt.plot(
        w, dos,
        lw=1.5,
        label=f"stru={iframe:4d}  (N={len(f_pos)})"
    )

    print(
        f"[DOS] stru={iframe:4d} : "
        f"{len(f_pos)} positive modes, "
        f"{len(f_neg)} imaginary modes "
        f"(filtered |f|<{drop0})"
    )

plt.xlim([0, 3800])
plt.xlabel(r"Frequency (cm$^{-1}$)")
plt.ylabel(r"Phonon DOS (arb. units)")
plt.title(rf"Lorentzian-broadened phonon DOS")
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.show()

#%%
####################################################################################################

# ---- ZPE (zero-point energy) per frame ----
hc_eV_cm = 1.2398419843320026e-4  # eV*cm  (E = h c * (cm^-1))

def zpe_from_cm1(freqs_cm1, drop_below=1.0, imag_policy="skip"):
    """Return (zpe_eV, n_used_modes, n_imag, n_skipped_small)."""
    freqs = np.asarray(freqs_cm1, dtype=float)

    # filter small |f|
    mask_small = np.abs(freqs) < float(drop_below)
    freqs2 = freqs[~mask_small]
    n_skipped_small = int(mask_small.sum())

    # handle imaginary modes
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
    zpe_eV = 0.5 * np.sum(eps)
    return float(zpe_eV), int(len(eps)), n_imag, n_skipped_small


print("\n[ZPE] per-frame zero-point energy (T=0)")
imag_policy = "skip"
for iframe in range(len(data["frames"])):
    freqs = np.array(data["frames"][iframe]["frequencies_cm1"], dtype=float)
    zpe_eV, n_used, n_imag, n_small = zpe_from_cm1(
        freqs, drop_below=drop0, imag_policy=imag_policy
    )
    print(
        f"[ZPE] frame {iframe:4d}: ZPE={zpe_eV: .8f} eV | "
        f"used_modes={n_used} | imag={n_imag} | dropped_small={n_small}"
    )

#%%
zpe_all = []
for iframe in range(len(data["frames"])):
    freqs = np.array(data["frames"][iframe]["frequencies_cm1"], dtype=float)
    zpe_eV, *_ = zpe_from_cm1(freqs, drop_below=drop0, imag_policy=imag_policy)
    zpe_all.append(zpe_eV)

zpe_all = np.array(zpe_all, dtype=float)
if zpe_all.size > 0:
    zpe_mean = float(zpe_all.mean())
    zpe_std = float(zpe_all.std(ddof=1)) if zpe_all.size > 1 else 0.0
    zpe_min = float(zpe_all.min())
    zpe_max = float(zpe_all.max())
    print(
        f"[ZPE] all frames stats: mean={zpe_mean: .8f} eV, "
        f"std={zpe_std: .8f} eV, min={zpe_min: .8f} eV, max={zpe_max: .8f} eV"
    )

# ---- requested summary: frame 0 отдельно, 平均 frames 1..100 ----
if zpe_all.size > 0:
    zpe0 = zpe_all[0]
    print(f"[ZPE] frame 0 (single): ZPE={zpe0: .8f} eV")

    start_i = 1
    end_i = min(100, zpe_all.size - 1)
    if end_i >= start_i:
        zpe_1_100 = zpe_all[start_i:end_i + 1]
        zpe_1_100_mean = float(zpe_1_100.mean())
        zpe_1_100_std = float(zpe_1_100.std(ddof=1)) if zpe_1_100.size > 1 else 0.0
        print(
            f"[ZPE] frames {start_i}..{end_i} mean: "
            f"{zpe_1_100_mean: .8f} eV (std={zpe_1_100_std: .8f} eV)"
        )
    else:
        print("[ZPE] frames 1..100 not available (need at least 2 frames).")

#%%

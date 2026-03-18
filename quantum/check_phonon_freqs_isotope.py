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
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from source.phonons import phonons_fd
from source.potentialmace_cueq import initialize_mace_model
from source.tools import phonon_dos_lorentz


#%%
####################################################################################################
# -----------------------
# 1) Read first frame
# -----------------------
stru_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_relax.xyz"
# stru_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_relax.xyz"

frames = ase.io.read(stru_file, index=":")   # list of Atoms
if isinstance(frames, ase.Atoms):
    frames = [frames]
atoms = frames[0].copy()  # 选第一个 atoms 对象 Cmc21
# atoms = frames[10].copy()  # 选第一个 atoms 对象

print(f"[INFO] read file: {stru_file}")
print(f"[INFO] natoms = {len(atoms)}  pbc={atoms.get_pbc()}")

# -----------------------
# 2) Prepare MACE model
# -----------------------
mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel251125/mace_iceIh_128x0e128x1o_r4.5_float32_seed144_cueq.model"
mace_dtype = "float32"
mace_batch_size = 1
mace_device = "cuda"

mace_inference = initialize_mace_model(
    mace_model_path,
    mace_batch_size,
    mace_dtype,
    mace_device,
)

# -----------------------
# 3) Phonons (FD Hessian)
# -----------------------
#%%
####################################################################################################
step = 0.01
repeats = 2
reduce = "mean"

gamma = 20.0   # cm^-1
drop0 = 1.0
positive_only = True

results = {}

# -----------------------
# Loop over isotopes
# -----------------------
for isotope in ["H2O", "D2O"]:
    print("\n" + "="*80)
    print(f"[RUN] isotope = {isotope}   step = {step:.3f} Å")

    atoms_i = atoms.copy()

    # ---- phonons ----
    res = phonons_fd(
        atoms=atoms_i,
        mace_inference=mace_inference,
        step=step,
        repeats=repeats,
        reduce=reduce,
        apply_asr_translation=True,
        symmetrize_hessian=True,
        sort_by_frequency=True,
        set_masses_for_h2o=False,
        isotope=isotope, 
    )

    freqs = np.asarray(res.frequencies_cm1, dtype=float)

    n_imag = int(np.sum(freqs < -1.0))
    print(f"[INFO] imag(<-1) = {n_imag:4d} / {len(freqs)}")
    print(f"[INFO] min freq  = {freqs.min():.4f} cm^-1")
    print("[INFO] lowest 12 freqs:")
    print(np.array2string(freqs[:12], precision=4))

    # ---- DOS ----
    w, dos, f_pos, f_neg = phonon_dos_lorentz(
        freqs,
        gamma_cm1=gamma,
        drop_below=drop0,
        positive_only=positive_only,
    )

    print(
        f"[DOS] {isotope}: "
        f"{len(f_pos)} positive modes, "
        f"{len(f_neg)} imaginary modes (filtered |f|<{drop0})"
    )

    results[isotope] = {
        "freqs": freqs,
        "w": w,
        "dos": dos,
    }

#%%
####################################################################################################
# Plot DOS: H2O vs D2O

plt.figure(figsize=(8, 4), dpi=300)

for isotope in ["H2O", "D2O"]:
    plt.plot(
        results[isotope]["w"],
        results[isotope]["dos"],
        lw=1.8,
        label=f"{isotope}  step={step:.2f} Å",
    )

plt.xlabel(r"Frequency (cm$^{-1}$)")
plt.ylabel("Phonon DOS (arb. units)")
plt.title(
    rf"Phonon DOS (Lorentz, $\gamma$={gamma:.1f} cm$^{{-1}}$, drop |f|<{drop0})"
)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()



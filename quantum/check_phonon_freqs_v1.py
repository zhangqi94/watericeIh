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
atoms = frames[10].copy()  # 选第一个 atoms 对象

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
# float32 力场：step 不要太小，否则噪声/δ 放大导致假虚频
steps = [0.005, 0.01, 0.02, 0.04]
repeats = 2
reduce = "mean"

results = {}   # 保存每个 step 的 freqs

for step in steps:
    print("\n" + "="*80)
    print(f"[RUN] step = {step:.3f} Å")

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
    results[step] = freqs

    print(f"[INFO] phonons done. step={step} Å repeats={repeats} reduce={reduce}")
    print("[INFO] lowest 12 frequencies (cm^-1):")
    print(np.array2string(freqs[:12], precision=4, suppress_small=False))
    # print("[INFO] lowest 6 modes:")
    # print(np.array2string(freqs[:6], precision=6, suppress_small=False))

    n_imag = int(np.sum(freqs < -1.0))
    print(f"[INFO] imaginary modes (< -1 cm^-1): {n_imag} / {len(freqs)}")
    

#%%
####################################################################################################

gamma = 20.0   # cm^-1
drop0 = 1.0    # 过滤 |freq| < drop0 的近零模

plt.figure(figsize=(8, 4), dpi=300)

for step in steps:
    freqs = results[step]
    w, dos, f_pos, f_neg = phonon_dos_lorentz(
        freqs,
        gamma_cm1=gamma,
        drop_below=drop0,
        positive_only=True,
    )

    plt.plot(
        w, dos,
        lw=1.5,
        label=f"step = {step:.02f} Å  (N={len(f_pos)})"
    )

    print(
        f"[DOS] step={step:.02f} Å : "
        f"{len(f_pos)} positive modes, "
        f"{len(f_neg)} imaginary modes "
        f"(filtered |f|<{drop0})"
    )

plt.xlabel(r"Frequency (cm$^{-1}$)")
plt.ylabel(r"Phonon DOS (arb. units)")
plt.title(
    rf"Lorentzian-broadened phonon DOS "
    #rf"(γ={gamma} cm$^{{-1}}$, repeats={repeats})"
)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

#%%

n_show = len(results[steps[0]])   # 打印前多少个低频 mode

print("\n🌲 Low-frequency comparison (cm^-1)")
print("├─ mode │ " + " │ ".join([f"step={s:.2f}" for s in steps]))
print("├" + "─"*6 + "┼" + "─"*12*len(steps))

for i in range(n_show):
    line = f"│ {i:4d} │ "
    for s in steps:
        line += f"{results[s][i]:9.4f} │ "
    print(line)

print("└" + "─"*6 + "┴" + "─"*12*len(steps))
# %%

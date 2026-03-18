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
# stru_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_relax.xyz"
# save_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_freq.json"
# isotope = "H2O"

# stru_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_relax.xyz"
# save_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_freq.json"
# isotope = "H2O"


stru_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_relax.xyz"
save_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_freq_D2O.json"
isotope = "D2O"

# stru_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_relax.xyz"
# save_file = "/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_freq_D2O.json"
# isotope = "D2O"

frames = ase.io.read(stru_file, index=":")   # list of Atoms
if isinstance(frames, ase.Atoms):
    frames = [frames]

print(f"[INFO] read file: {stru_file}")
print(f"[INFO] nframes = {len(frames)}")


####################################################################################################
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

#%%
####################################################################################################
# -----------------------
# 3) Phonons (FD Hessian)
# -----------------------
# float32 力场：step 不要太小，否则噪声/δ 放大导致假虚频
step = 0.01
repeats = 2
reduce = "mean"

results_by_frame = []   # list of dicts per frame

for iframe, atoms0 in enumerate(frames):
    atoms = atoms0.copy()

    print("\n" + "="*90)
    print(f"[RUN] frame={iframe:4d}  natoms={len(atoms)}  step={step:.3f} Å")

    res = phonons_fd(
        atoms=atoms,
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

    freqs = np.array(res.frequencies_cm1, dtype=float)
    n_imag = int(np.sum(freqs < -1.0))

    results_by_frame.append({
        "frame": iframe,
        "freqs_cm1": freqs,
        "n_imag": n_imag,
        "min_freq": float(freqs.min()),
    })

    print(f"[INFO] done frame={iframe:4d}   imag(<-1)={n_imag:4d}/{len(freqs)}   min={freqs.min():.4f} cm^-1")
    print("[INFO] lowest 12 freqs (cm^-1):")
    print(np.array2string(freqs[:12], precision=4, suppress_small=False))

####################################################################################################
# 4) Summary table
print("\n" + "#"*90)
print("Summary (per frame):")
print("frame | imag(<-1) | min_freq(cm^-1)")
print("------+-----------+----------------")
for r in results_by_frame:
    print(f"{r['frame']:5d} | {r['n_imag']:9d} | {r['min_freq']:14.4f}")

# # 可选：把所有帧的频率堆成一个 2D 数组方便后处理
# all_freqs = np.stack([r["freqs_cm1"] for r in results_by_frame], axis=0)  # (nframe, nmodes)

# # 可选：保存


#%%
####################################################################################################
# 5) Save phonon results to JSON

import json

json_file = save_file
os.makedirs(os.path.dirname(json_file), exist_ok=True)

data = {
    "meta": {
        "source_structure": stru_file,
        "save_file": json_file,
        "mace_model": mace_model_path,
        "step_A": step,
        "repeats": repeats,
        "reduce": reduce,
        "nframes": len(results_by_frame),
        "units": {
            "frequency": "cm^-1",
            "length": "angstrom",
        },
    },
    "frames": [],
}

for r in results_by_frame:
    iframe = r["frame"]
    freqs = r["freqs_cm1"]

    data["frames"].append({
        "frame": iframe,
        "natoms": len(frames[iframe]),
        "n_imag_lt_-1": int(r["n_imag"]),
        "min_freq_cm1": float(r["min_freq"]),
        "frequencies_cm1": freqs.tolist(),   # numpy -> list
    })

with open(json_file, "w") as f:
    json.dump(data, f, indent=2)

print(f"[SAVE] phonon frequencies written to JSON:")
print(f"  {json_file}")


#%%    
"""
conda activate jax0503-torch280-mace314
cd /home/zq/zqcodeml/watericeIh-mc-master/quantum
python3 run_script_calc_phonon_freqs.py

"""


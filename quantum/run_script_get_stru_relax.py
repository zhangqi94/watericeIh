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
    

#%%
####################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import ase
import ase.io
from pathlib import Path
from source.potentialmace_cueq import initialize_mace_model
from source.crystalrelax import relax_all_with_MACE


#%%

# -------------------------
# Input / output
# -------------------------
traj_path = Path("/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop.xyz")
save_path = Path("/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop_relax.xyz")

traj_path = Path("/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop.xyz")
save_path = Path("/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop_relax.xyz")


# -------------------------
# MACE
# -------------------------
mace_model_path = "source/potential/macemodel251125/mace_iceIh_128x0e128x1o_r4.5_float32_seed144_cueq.model"
mace_dtype = "float32"
mace_batch_size = 1
mace_device = "cuda"

mace_inference = initialize_mace_model(
    mace_model_path,
    mace_batch_size,
    mace_dtype,
    mace_device,
)


# -------------------------
# Relax settings
# -------------------------
# lr = 0.0005
# max_iter = 5000
# f_tol = 1e-4

# -------------------------
# Read all frames
# -------------------------
frames = ase.io.read(str(traj_path), index=":")
if isinstance(frames, ase.Atoms):
    frames = [frames]

print(f"[INFO] total frames = {len(frames)}")

# -------------------------
# Relax all frames and save to ONE extxyz trajectory
# -------------------------
relaxed_traj_path = str(save_path)
if os.path.exists(relaxed_traj_path):
    os.remove(relaxed_traj_path)

for i, atoms in enumerate(frames):
    
    E_traj_all = []
    Fmax_traj_all = []   
     
    # ===== stage 1 =====
    atoms = atoms.copy()
    coords0 = atoms.get_positions().copy()
    coords_fin, energy_fin, forces_fin, E_traj, Fmax_traj = relax_all_with_MACE(
        atoms=atoms,
        mace_inference=mace_inference,
        init_coords=coords0,
        lr=0.01,
        max_iter=200,
        f_tol=1e-4,
        verbose=False,
        return_traj=True,
    )

    # E_traj_all.append(E_traj)
    # Fmax_traj_all.append(Fmax_traj)

    atoms.set_positions(coords_fin)
    atoms.info["energy"] = float(energy_fin)

    # ===== stage 2 =====
    atoms = atoms.copy()
    coords0 = atoms.get_positions().copy()

    coords_fin, energy_fin, forces_fin, E_traj, Fmax_traj = relax_all_with_MACE(
        atoms=atoms,
        mace_inference=mace_inference,
        init_coords=coords0,
        lr=0.001,
        max_iter=1000,
        f_tol=1e-4,
        verbose=False,
        return_traj=True,
    )

    # # ⚠️ 去掉第 0 步，避免能量重复
    E_traj_all.append(E_traj[1:])
    Fmax_traj_all.append(Fmax_traj)

    atoms.set_positions(coords_fin)
    atoms.info["energy"] = float(energy_fin)

    # ===== stage 3 =====
    atoms = atoms.copy()
    coords0 = atoms.get_positions().copy()

    coords_fin, energy_fin, forces_fin, E_traj, Fmax_traj = relax_all_with_MACE(
        atoms=atoms,
        mace_inference=mace_inference,
        init_coords=coords0,
        lr=0.0002,
        max_iter=3000,
        f_tol=1e-4,
        verbose=False,
        return_traj=True,
    )


    # # ⚠️ 去掉第 0 步，避免能量重复
    E_traj_all.append(E_traj[1:])
    Fmax_traj_all.append(Fmax_traj)

    atoms.set_positions(coords_fin)
    atoms.info["energy"] = float(energy_fin)


    # ---- STORE THIS FRAME ----
    ase.io.write(
        relaxed_traj_path,
        atoms,
        format="extxyz",
        append=True,
    )

    maxF = float(np.max(np.linalg.norm(forces_fin, axis=1)))
    print(
        f"[RELAX] frame={i:4d}  "
        f"E={energy_fin: .8f} eV  "
        f"max|F|={maxF:.3e} eV/Å",
        flush=True,
    )
    
    E_traj_all = np.concatenate([np.asarray(x) for x in E_traj_all])
    Fmax_traj_all = np.concatenate([np.asarray(x) for x in Fmax_traj_all])
    # ---- plotting (unchanged) ----
    fig, ax1 = plt.subplots(figsize=(8, 4), dpi=300)
    # it = np.arange(len(E_traj_all))
    ax1.plot(E_traj_all, "-o", ms=2)
    ax1.set_xlabel("Relax iteration (all stages)")
    ax1.set_ylabel("Energy (eV)")
    ax1.grid(True, ls="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(Fmax_traj_all, "-r", lw=1)
    ax2.set_ylabel("max |F| (eV/Å)")

    plt.title(f"Relax traj (all stages, frame {i})")
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    # ---------------------------


#%%    
"""
conda activate jax0503-torch280-mace314
cd /home/zq/zqcodeml/watericeIh-mc-master/quantum
python3 run_script_get_stru_relax.py

"""

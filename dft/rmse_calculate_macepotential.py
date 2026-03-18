#%%
####################################################################################################
import os
import sys

work_dir = "/home/zq/zqcodeml/watericeIh-mc-master"

# Change current working directory
os.chdir(work_dir)
print(f"[INFO] Working directory set to: {os.getcwd()}")
if work_dir not in sys.path:
    sys.path.insert(0, work_dir)
    
####################################################################################################
import os, sys, time
import numpy as np
import pandas as pd
import ase
import ase.io
from typing import Callable, Any, Optional, Tuple, Dict, List

from datetime import datetime
from source.potentialmace_cueq import initialize_mace_model

#%%
####################################################################################################
# Inputs
xyz_file = "/home/zq/zqcodeml/watericeIh_data/dft_data/data_xyz_260201/watericeIh_test.xyz"

mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-github/source/potential/macemodel260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_cueq.model"
out_npz = "/home/zq/zqcodeml/watericeIh_data/dft_data/data_xyz_260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_testset_gpu_v2.npz"
create_neighborlist_device = "cpu"

# out_npz = "/home/zq/zqcodeml/watericeIh_data/dft_data/data_xyz_251112/mace_pred_watericeIh_testset_gpu_v2.npz"
# Create inference callable (DO NOT redefine mace_inference manually)
mace_inference = initialize_mace_model(mace_model_path, "float32", "cuda")


#%%
####################################################################################################
def parse_energy_forces_mace(out):
    """Parse MACE output -> (E_tot, F[N,3] or None)"""
    E = None
    F = None

    if isinstance(out, dict):
        for k in ["energy", "E", "total_energy", "E_total"]:
            if k in out and out[k] is not None:
                E = float(np.asarray(out[k]).reshape(-1)[0])
                break
        for k in ["forces", "F", "force"]:
            if k in out and out[k] is not None:
                F = np.asarray(out[k], dtype=float)
                break
    elif isinstance(out, (tuple, list)):
        if len(out) >= 1 and out[0] is not None:
            E = float(np.asarray(out[0]).reshape(-1)[0])
        if len(out) >= 2 and out[1] is not None:
            F = np.asarray(out[1], dtype=float)

    if E is None:
        raise RuntimeError(f"Cannot parse energy from MACE output type={type(out)}")
    return E, F



def get_ref_energy(atoms):
    """Read reference total energy from extxyz metadata (e.g. TotEnergy=...)."""
    # 1) exact keys first
    keys_try = [
        "TotEnergy", "totenergy", "TotalEnergy", "total_energy", "E_total",
        "energy", "E", "dft_energy", "DFT_energy", "E_dft", "ref_energy",
        "PotentialEnergy",
    ]
    for k in keys_try:
        if k in atoms.info:
            try:
                return float(np.asarray(atoms.info[k]).reshape(-1)[0])
            except Exception:
                pass

    # 2) case-insensitive scan over atoms.info
    try:
        lower_map = {str(k).lower(): k for k in atoms.info.keys()}
        for lk in ["totenergy", "energy", "potentialenergy", "dft_energy", "e_dft", "e"]:
            if lk in lower_map:
                k = lower_map[lk]
                try:
                    return float(np.asarray(atoms.info[k]).reshape(-1)[0])
                except Exception:
                    pass
    except Exception:
        pass

    # 3) last resort
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        return float("nan")


def get_ref_forces(atoms):
    """Read reference forces from extxyz. Your file uses Properties=...:force:R:3:..."""
    # 1) extxyz Properties usually go into atoms.arrays
    keys_try = [
        "force",      # <-- your file
        "forces",
        "F",
        "dft_forces", "DFT_forces",
        "ref_forces", "RefForces",
    ]
    for k in keys_try:
        if k in atoms.arrays:
            try:
                F = np.asarray(atoms.arrays[k], dtype=float)
                if F.ndim == 2 and F.shape[1] == 3 and F.shape[0] == len(atoms):
                    return F
            except Exception:
                pass

    # 2) case-insensitive scan over atoms.arrays
    try:
        lower_map = {str(k).lower(): k for k in atoms.arrays.keys()}
        for lk in ["force", "forces", "dft_forces", "f"]:
            if lk in lower_map:
                k = lower_map[lk]
                try:
                    F = np.asarray(atoms.arrays[k], dtype=float)
                    if F.ndim == 2 and F.shape[1] == 3 and F.shape[0] == len(atoms):
                        return F
                except Exception:
                    pass
    except Exception:
        pass

    # 3) rare: forces stored in info as flat list
    for k in ["force", "forces", "F"]:
        if k in atoms.info:
            try:
                F = np.asarray(atoms.info[k], dtype=float).reshape((-1, 3))
                if F.shape[0] == len(atoms) and F.shape[1] == 3:
                    return F
            except Exception:
                pass

    return None

#%%
####################################################################################################
E_ref_list = []
E_mace_list = []
F_ref_list = []
F_mace_list = []
n_atoms_list = []

t0 = time.time()
frames = ase.io.iread(xyz_file, index=":")

for iframe, atoms in enumerate(frames):
    pos = atoms.get_positions()
    n_atoms = len(atoms)

    # ---- ref from xyz ----
    E_ref = get_ref_energy(atoms)
    F_ref = get_ref_forces(atoms)
    if F_ref is None:
        F_ref = np.full((n_atoms, 3), np.nan, dtype=float)

    # ---- MACE ----
    out = mace_inference(
        atoms=atoms,
        compute_force=True,
        create_neighborlist_device=create_neighborlist_device,
    )

    E_mace, F_mace = parse_energy_forces_mace(out)
    if F_mace is None:
        F_mace = np.full((n_atoms, 3), np.nan, dtype=float)

    # collect
    E_ref_list.append(float(E_ref))
    E_mace_list.append(float(E_mace))
    F_ref_list.append(F_ref)
    F_mace_list.append(F_mace)
    n_atoms_list.append(int(n_atoms))

    if (iframe + 1) % 50 == 0:
        dt = time.time() - t0
        print(f"[INFO] {iframe+1} frames | {dt:.1f}s | {dt/(iframe+1):.3f}s/frame")

#%%
####################################################################################################
E_ref = np.asarray(E_ref_list, dtype=float)
E_mace = np.asarray(E_mace_list, dtype=float)
n_atoms = np.asarray(n_atoms_list, dtype=int)

# stack if same N, else object arrays
sameN = (len(set(n_atoms.tolist())) == 1)
if sameN:
    F_ref = np.stack(F_ref_list, axis=0)   # (n_frames, N, 3)
    F_mace = np.stack(F_mace_list, axis=0) # (n_frames, N, 3)
else:
    F_ref = np.array(F_ref_list, dtype=object)
    F_mace = np.array(F_mace_list, dtype=object)

np.savez_compressed(
    out_npz,
    E_ref=E_ref,
    E_mace=E_mace,
    F_ref=F_ref,
    F_mace=F_mace,
    n_atoms=n_atoms,
)

dt = time.time() - t0
print(f"[DONE] frames = {len(E_ref)} | wall time = {dt:.1f}s")
print(f"[SAVE] {out_npz}")



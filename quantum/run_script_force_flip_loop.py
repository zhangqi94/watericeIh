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
import numpy as np
from typing import Tuple
from pathlib import Path

from source.ckpt import load_structure_from_json
from source.ckpt import save_snapshot_json, append_xyz_snapshot


def make_snap_path_from_traj(traj_path: Path, index: int = 0) -> Path:
    traj_stem = traj_path.stem
    snap_name = f"snap_{traj_stem}_{index}.json"
    return traj_path.parent / snap_name


def ensure_parent_dir_exists(path: str) -> None:
    parent = os.path.dirname(path)
    if parent != "":
        os.makedirs(parent, exist_ok=True)


#%%

initial_structure_file = "source/structure/initstru/sc_422_n_128_rho_933.json"
traj_path = Path("/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_422_n_128/traj_loop.xyz")

# initial_structure_file = "source/structure/initstru/sc_533_n_360_rho_933.json"
# traj_path = Path("/home/zq/zqcodeml/watericeIh_data/quantum_effect/stru/sc_533_n_360/traj_loop.xyz")

snap_path = make_snap_path_from_traj(traj_path, index=0)
save_xyz = True

thermal_loop = 500
num_snapshots = 101          # number of xyz frames to write AFTER thermal
xyz_interval = 500           # write one xyz frame every 100 loop flips

print(traj_path)
print(snap_path)

#%%

# Load ASE Atoms object and metadata dict
atoms, data = load_structure_from_json(initial_structure_file)

# Extract coordinates and structure information
coords = atoms.get_positions()                         # (n_atoms, 3)
O_neighbors = data["O_neighbors"]                      # (n_O, 4) or (n_O, 5) with first column O
H_to_OO_pairs = data["H_to_OO_pairs"]                  # (n_H, 3)
state_hydrogens = data["state_hydrogens"]              # (n_H,)
num_O = atoms.get_chemical_symbols().count("O")        # number of oxygen atoms

# Orthorhombic box lengths (Å)
box_lengths = atoms.cell.lengths()  

print(f"[LOAD] Loaded structure: {num_O} O atoms, {len(state_hydrogens)} H atoms.")
print(f"[LOAD] Box lengths (Å): {box_lengths}")


#%%
####################################################################################################
# ----------------------------------------------------------------------------------------------
# Prepare MACE model and initial energy
# ----------------------------------------------------------------------------------------------
from source.potentialmace_cueq import initialize_mace_model

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

#%%
####################################################################################################
# ----------------------------------------------------------------------------------------------
# Build the Metropolis short-loop update function
# ----------------------------------------------------------------------------------------------

from source.updateloop import make_metropolis_loop_update_functions, state_to_bitstring, bitstring_to_hexstr
                               
metropolis_loop_update, perform_loop_flip, _ = make_metropolis_loop_update_functions(
    O_neighbors=O_neighbors,
    H_to_OO_pairs=H_to_OO_pairs,
    box_lengths=box_lengths,
    mace_inference=mace_inference,
)


#%%
####################################################################################################
coords_curr = coords.copy()
state_curr = state_hydrogens.copy()

# Prepare output paths
ensure_parent_dir_exists(traj_path)
ensure_parent_dir_exists(snap_path)

# Reset trajectory file (we will start writing AFTER thermal)
if save_xyz and os.path.exists(traj_path):
    os.remove(traj_path)

# Track energies at saved frames (not every MC step)
E_frames = []

#%%
# ----------------------------------------------------------------------------------------------
# 1) Save ONE snap (after thermal), then start xyz recording
# ----------------------------------------------------------------------------------------------
save_snapshot_json(
    json_path=snap_path,
    atoms=atoms,
    supercell_size=data["supercell_size"],
    density=data["density"],
    O_neighbors=O_neighbors,
    H_to_OO_pairs=H_to_OO_pairs,
    state_hydrogens=state_curr,
    atomcoords_O=data["atomcoords_O"],
    H2_candidates=data["H2_candidates"],
)
print(f"[SAVE] snap -> {snap_path}", flush=True)

# Write frame 0 (post-thermal)
energy_curr, _, _ = mace_inference(atoms, coords_curr, compute_force=False)
E_frames.append(float(energy_curr))

if save_xyz:
    append_xyz_snapshot(
        atoms=atoms,
        xyz_path=traj_path,
    )
print(f"[XYZ] frame=000  E={energy_curr: .8f} eV  -> {traj_path}", flush=True)


#%%
# ----------------------------------------------------------------------------------------------
# 0) Thermalization: DO NOT write xyz
# ----------------------------------------------------------------------------------------------
for step in range(int(thermal_loop)):
    state_curr, coords_curr, atoms = perform_loop_flip(state_curr, coords_curr, atoms)

    if (step + 1) % 100 == 0 or step == 0:
        energy_curr, _, _ = mace_inference(atoms, coords_curr, compute_force=False)
        hex_state = bitstring_to_hexstr(state_to_bitstring(state_curr))
        print(
            f"[THERM] step={step + 1:6d}/{thermal_loop:6d}  "
            f"E={energy_curr: .8f} eV  state={hex_state}",
            flush=True,
        )

#%%
# ----------------------------------------------------------------------------------------------
# 2) Production: every xyz_interval steps append ONE xyz frame, total num_snapshots frames
# ----------------------------------------------------------------------------------------------
for frame_idx in range(1, int(num_snapshots)):
    for _ in range(int(xyz_interval)):
        state_curr, coords_curr, atoms = perform_loop_flip(state_curr, coords_curr, atoms)

    energy_curr, _, _ = mace_inference(atoms, coords_curr, compute_force=False)
    E_frames.append(float(energy_curr))

    if save_xyz:
        append_xyz_snapshot(
            atoms=atoms,
            xyz_path=traj_path,
        )

    hex_state = bitstring_to_hexstr(state_to_bitstring(state_curr))
    print(
        f"[XYZ] frame={frame_idx:03d}/{num_snapshots - 1:03d}  "
        f"E={energy_curr: .8f} eV  state={hex_state}",
        flush=True,
    )

#%%
# ----------------------------------------------------------------------------------------------
# Plot energy trajectory (per saved frame)
# ----------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt

energy_traj_fig = np.array(E_frames)
x = np.arange(len(energy_traj_fig))

plt.figure(figsize=(6, 6), dpi=300)
plt.plot(x, energy_traj_fig, ".-", lw=1)
plt.xlabel("Saved frame index", fontsize=12)
plt.ylabel("Energy (eV)", fontsize=12)
plt.title("Energy trajectory (saved xyz frames)", fontsize=13)
plt.grid(True, ls="--", alpha=0.5)
plt.tight_layout()
plt.show()

print(f"final - first = {energy_traj_fig[-1] - energy_traj_fig[0]:.8f}")


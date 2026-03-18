

####################################################################################################
import os
import sys
from pathlib import Path
import numpy as np
import time
from datetime import datetime

import torch
import ase
from importlib.metadata import version
print("python version:", sys.version, flush=True)
print("torch.version:", version("torch"), flush=True)
print("numpy.version:", version("numpy"), flush=True)
print("ase.version:", version("ase"), flush=True)

####################################################################################################

import argparse
from source.tools import str2bool
parser = argparse.ArgumentParser(description= "finite-temperature Monte Carlo simulation for water ice Ih (loop + mala update)")

# ----------------------------------------------------------------------------------------------
# Structure / system
# ----------------------------------------------------------------------------------------------
parser.add_argument(
    "--init_stru_path",
    type=Path,
    default=Path("/home/zq/zqcodeml/watericeIh-mc-master/source/structure/initstru/supercell_211_n_16.json"),
    help="Path to initial structure JSON file.",
)
parser.add_argument(
    "--save_file_path",
    type=Path,
    default=Path("/home/zq/zqcodeml/data_ice_mc/test/supercell_211_n_16_T_100"),
    help="Path to save the final structure json & txt file.",
)
parser.add_argument(
    "--save_xyz",
    type=str2bool,
    nargs="?",
    const=True,
    default=False,
    help="Whether to save multi-frame XYZ trajectory (True/False). Default: False.",
)
parser.add_argument(
    "--target_temperature_K",
    type=float,
    default=100.0,
    help="Target temperature in Kelvin.",
)

# ----------------------------------------------------------------------------------------------
# MACE model
# ----------------------------------------------------------------------------------------------
parser.add_argument(
    "--mace_model_path",
    type=Path,  
    default=Path("/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel/mace_iceIh_128x0e128x1o_r5.0_float32_k43_mace314.model"),
    help="Path to pretrained MACE model file.",
)
parser.add_argument(
    "--mace_device",
    type=str,
    choices=["cuda", "cpu", "dcu"],
    default="cuda",
    help="Computation device for MACE model.",
)
parser.add_argument(
    "--mace_dtype",
    type=str,
    choices=["float32", "float64"],
    default="float32",
    help="Floating-point dtype for MACE model.",
)

# ----------------------------------------------------------------------------------------------
# Monte Carlo control parameters
# ----------------------------------------------------------------------------------------------
parser.add_argument(
    "--num_blocks",
    type=int,
    default=1000,
    help="Number of Monte Carlo blocks to run.",
)

parser.add_argument(
    "--num_loop_steps",
    type=int,
    default=5,
    help="Number of short-loop steps per loop block.",
)

parser.add_argument(
    "--thermal_force_loop",
    type=int,
    default=100,
    help="Number of thermalization force loop steps.",
)

parser.add_argument(
    "--create_neighborlist_device",
    type=str,
    default="gpu",
    choices=["gpu", "cpu"],
    help="Device to use for creating neighborlist (gpu or cpu).",
)

# ----------------------------------------------------------------------------------------------
# Print control parameters
# ----------------------------------------------------------------------------------------------
parser.add_argument(
    "--print_interval_loop",
    type=int,
    default=1,
    help="Print frequency (in sweeps) during each LOOP block. Set 0 to disable intermediate prints.",
)

####################################################################################################
from source import units
from source.ckpt import format_log_preamble
args = parser.parse_args()
log_preamble_text = format_log_preamble(args)

# --- File paths ---
init_stru_path: Path = args.init_stru_path
save_file_path: Path = args.save_file_path
save_xyz: bool = args.save_xyz

target_temperature_K: float = args.target_temperature_K

# --- Model paths ---
mace_model_path: Path = args.mace_model_path
mace_device: str = args.mace_device
mace_dtype: str = args.mace_dtype

# --- MC control parameters ---
num_blocks: int = args.num_blocks
num_loop_steps: int = args.num_loop_steps

thermal_force_loop: int = args.thermal_force_loop
create_neighborlist_device: str = args.create_neighborlist_device

# --- Print control ---
print_interval_loop: int = args.print_interval_loop

# ======== Print parsed configuration ========
print("\n========== Parsed Configuration ==========")
print(f"{'Initial structure:':22s} {init_stru_path}")
print(f"{'Save file path:':22s} {save_file_path}")
print(f"")

print(f"{'Temperature:':22s} {target_temperature_K:.3f} K ({target_temperature_K * units.K_B_EV_PER_K:.8f} eV)")
print(f"{'MACE model path:':22s} {mace_model_path}")
print(f"{'MACE device:':22s} {mace_device}")
print(f"{'MACE dtype:':22s} {mace_dtype}")
print(f"{'MC total blocks:':22s} {num_blocks}")

print(f"{'LOOP steps/block:':22s} {num_loop_steps}")
print(f"{'Thermalization (loop):':22s} {thermal_force_loop}")
print(f"{'Create NL device:':22s} {create_neighborlist_device}")
print(f"{'Print interval (loop):':22s} {print_interval_loop}")

print("==========================================\n")


####################################################################################################
# Initialize structure
####################################################################################################
print("\n========== Initialize structure ==========", flush=True)
from source.ckpt import load_structure_from_json

# Load ASE Atoms object and metadata dict
atoms, data = load_structure_from_json(init_stru_path)

# Extract coordinates and structure information
coords = atoms.get_positions()                         # (n_atoms, 3)
supercell_size = data["supercell_size"]                # (3,)
density = data["density"]                              # 
O_neighbors = data["O_neighbors"]                      # (n_O, 4) or (n_O, 5) with first column O
H_to_OO_pairs = data["H_to_OO_pairs"]                  # (n_H, 3)
state_hydrogens = data["state_hydrogens"]              # (n_H,)

atomcoords_O = data["atomcoords_O"]                    # (n_O, 3)
H2_candidates = data["H2_candidates"]                  # (n_O, 2, 3)

num_O = atoms.get_chemical_symbols().count("O")        # number of oxygen atoms
num_H = atoms.get_chemical_symbols().count("H")        # number of hydrogen atoms
num_molecule = len(O_neighbors)                         # number of O-H pairs

# --- fix labels/comments around counts ---
num_molecule = len(O_neighbors)  # number of H2O molecules (== number of O atoms)
if num_O != len(O_neighbors):
    raise ValueError("Number of O atoms and rows in O_neighbors do not match.")
if num_H != len(H_to_OO_pairs):
    raise ValueError("Number of H atoms and rows in H_to_OO_pairs do not match.")
if num_H != len(state_hydrogens):
    raise ValueError("Number of H atoms and state_hydrogens do not match.")
if (2 * num_O) != num_H:
    raise ValueError("Number of H atoms is not twice the number of O atoms.")

# Orthorhombic box lengths (Å)
box_lengths = atoms.cell.lengths()  

print(f"[LOAD] Loaded structure: {num_O} O atoms, {num_H} H atoms.")
print(f"[LOAD] Box lengths (Å): {box_lengths}")


####################################################################################################
# Initialize MACE model
####################################################################################################
print("\n========== Initialize MACE model ==========", flush=True)
if mace_device == "cuda":
    from source.potentialmace_cueq import initialize_mace_model
    print(f"[INIT] Using CUDA for MACE model")
# elif mace_device == "dcu":
#     from source.potentialmace_oeq import initialize_mace_model
#     print(f"[INIT] Using DCU for MACE model")

mace_inference = initialize_mace_model(
    mace_model_path,
    mace_dtype,
    mace_device="cuda",
)

# Initial energy/force @ current coords
atoms.set_positions(coords)
energy_curr, force_curr, _ = mace_inference(atoms, compute_force=True, create_neighborlist_device=create_neighborlist_device)

print(f"[INIT]  Energy            : {energy_curr: .12f} eV")
print(f"[INIT]  Force array shape : {force_curr.shape}   (should be (N_atoms, 3))")
print(f"[INIT]  |F|_max           : {np.max(np.linalg.norm(force_curr, axis=1)): .6f} eV/Å")


####################################################################################################
# ----------------------------------------------------------------------------------------------
# Build the Metropolis mala & short-loop update function
# ----------------------------------------------------------------------------------------------
print("\n========== Initialize only loop update ==========", flush=True)

from source.updateloop import (
    make_metropolis_loop_update_functions, state_to_bitstring,
    bitstring_to_hexstr, hexstr_to_bitstring)
_, perform_loop_flip, metropolis_only_loop_update = make_metropolis_loop_update_functions(
    O_neighbors=O_neighbors,
    H_to_OO_pairs=H_to_OO_pairs,
    mace_inference=mace_inference,
    create_neighborlist_device=create_neighborlist_device,
)

####################################################################################################
# Prepare saving paths (log + snapshot)
####################################################################################################
print("\n========== Prepare saving paths ==========", flush=True)
from source.ckpt import resolve_save_paths, auto_rename_log_file
from source.ckpt import save_snapshot_json, append_block_summary_line, append_xyz_snapshot

log_txt_path, snap_json_path, traj_xyz_path = resolve_save_paths(save_file_path)

# --- automatically rename if same log already exists ---
log_txt_path = auto_rename_log_file(log_txt_path)
snap_json_path = auto_rename_log_file(snap_json_path)
traj_xyz_path = auto_rename_log_file(traj_xyz_path)

print(f"[INIT] Log file:   {log_txt_path}", flush=True)
print(f"[INIT] Snap file:  {snap_json_path}", flush=True)
if save_xyz:
    print(f"[INIT] Trajectory: {traj_xyz_path}", flush=True)

####################################################################################################
# Monte Carlo loop with proposals
####################################################################################################
print("\n========== Initialize Metropolis Monte Carlo ==========", flush=True)
from source.updateblock import run_mc_only_loop_block
from source.tools import safe_div, fmt_rate, calculate_pressure_from_stress
from source.crystaltools import check_hydrogen_consistency

# User-tunable parameters
temperature_in_eV = units.K_B_EV_PER_K * target_temperature_K  # Convert T (K) to energy (eV)

# ---- Neat summary ----
print(
    f"[INIT] Blocks                 : {num_blocks}\n"
    f"[INIT] Loop steps / block     : {num_loop_steps}\n"
    f"[INIT] Temperature            : {target_temperature_K:.1f} K  (kT = {temperature_in_eV:.6f} eV)\n",
    flush=True,
)


####################################################################################################
# Thermalization
####################################################################################################
print("\n========== Thermalization ==========", flush=True)

# Copy current hydrogen state and coordinates for thermalization
state_curr = state_hydrogens.copy()
coords_curr = coords.copy()

# ----------------------------------------------------------------------------------------------
# 1) Discrete loop flips (hydrogen-bond network relaxation)
# ----------------------------------------------------------------------------------------------
for step in range(int(thermal_force_loop)):
    state_curr, atoms = perform_loop_flip(
        state_curr,
        atoms,
    )
    # Print compact hydrogen-orientation state (bitstring → hex)
    hex_state = bitstring_to_hexstr(state_to_bitstring(state_curr))
    print(f"[LOOP FLIP]  step={step:6d}  state={hex_state}", flush=True)

# ----------------------------------------------------------------------------------------------
# 2) geometry/state consistency check
# ----------------------------------------------------------------------------------------------
mismatch = check_hydrogen_consistency(
    atoms=atoms,
    H_to_OO_pairs=H_to_OO_pairs,
    state_curr=state_curr,
    cutoff=1.2,
)

# ----------------------------------------------------------------------------------------------
# 4) compute initial polarization P (e / Å²)
# ----------------------------------------------------------------------------------------------
from source.dielectric import compute_correlation_parameter
moment_vec, moment2, mu2_mean, correlation_G = compute_correlation_parameter(atoms)
print("\n--- Correlation snapshot values ---")
print(f"Total dipole M (e*Angstrom):      {moment_vec}")
print(f"|M|^2 (e*Angstrom)^2:             {moment2:.6e}")
print(f"<|mu|^2>_frame (e*Angstrom)^2:    {mu2_mean:.6e}")
print(f"P (e/Angstrom^2):                 {moment_vec / atoms.get_volume()}")

####################################################################################################
# Monte Carlo production loop
####################################################################################################
# Initialize global acceptance counters for reporting
loop_accepts_total = 0
loop_attempts_total = 0
mala_accepts_total = 0
mala_attempts_total = 0
cell_accepts_total = 0
cell_attempts_total = 0

# ==============================================================================================
# Main production phase: alternating discrete (LOOP) and continuous (MALA) updates
# Each block represents one MC macro-step: {loop-sweeps → MALA relaxation}.
# ==============================================================================================
for block in range(1, num_blocks + 1):
    block_t0 = time.perf_counter()
    
    # if update_mode == "sequential":
    state_before = state_curr.copy()
    # ----------------------------- LOOP stage -----------------------------
    # Performs short-loop hydrogen reorientations with Metropolis testing.
    print("\n" + "=" * 80, flush=True)
    print(f"[BLOCK {block:8d}/{num_blocks:8d}] LOOP part: steps × {num_loop_steps}", flush=True)
    energy_curr, state_curr, atoms, loop_accepts_block, loop_attempts_block = run_mc_only_loop_block(
        num_loop_steps=num_loop_steps,
        num_O=num_O,
        energy_curr=energy_curr,
        state_curr=state_curr,
        atoms=atoms,
        temperature_in_eV=temperature_in_eV,
        metropolis_only_loop_update=metropolis_only_loop_update,
        atomcoords_O=atomcoords_O,
        H2_candidates=H2_candidates,
        print_interval_loop=print_interval_loop,
    )

    # Refresh force field for subsequent MALA stage
    energy_curr, force_curr, stress_curr = mace_inference(atoms, compute_force=True, create_neighborlist_device=create_neighborlist_device)
    loop_accepts_total += loop_accepts_block
    loop_attempts_total += loop_attempts_block

    # ----------------------------- MALA stage -----------------------------
    mala_accepts_block = 0
    mala_attempts_block = 0
    mala_accepts_total += mala_accepts_block
    mala_attempts_total += mala_attempts_block

    # ----------------------------- Cell stage -----------------------------
    cell_accepts_block = 0
    cell_attempts_block = 0
    cell_accepts_total += cell_accepts_block
    cell_attempts_total += cell_attempts_block
    
    # updater_label = "(ONLY LOOP)"

    # ------------------------------------------------------------------------------------------
    # Diagnostics and reporting
    # Compute acceptance ratios, elapsed time, and structural change flags
    # ------------------------------------------------------------------------------------------
    loop_acc_block = safe_div(loop_accepts_block, loop_attempts_block)
    loop_acc_total = safe_div(loop_accepts_total, loop_attempts_total)

    block_dt = time.perf_counter() - block_t0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    state_changed = not np.array_equal(state_curr, state_before)
    changed_flag = "YES" if state_changed else "NO"
    hex_state = bitstring_to_hexstr(state_to_bitstring(state_curr))

    # Check for hydrogen consistency
    mismatch = check_hydrogen_consistency(
        atoms=atoms,
        H_to_OO_pairs=H_to_OO_pairs,
        state_curr=state_curr,
        cutoff=1.2,
    )

    # Compute observables
    moment_vec, moment2, mu2_mean, correlation_G = compute_correlation_parameter(atoms)

    # ------------------------------------------------------------------------------------------
    # Block summary
    # ------------------------------------------------------------------------------------------
    cell_a, cell_b, cell_c = atoms.cell.lengths()
    current_volume = atoms.get_volume()
    current_density = units.calculate_density_g_per_cm3(
        units.calculate_mass_h2o_g(num_molecule), current_volume
    )

    # Calculate pressure from stress
    num_atoms = len(atoms)
    P_kinetic, P_virial, pressure_eV_A3_curr = calculate_pressure_from_stress(
        stress=stress_curr,
        volume=current_volume,
        num_atoms=num_atoms,
        temperature_in_eV=temperature_in_eV,
    )
    current_pressure_GPa = pressure_eV_A3_curr / units.GPA_TO_EV_PER_ANGSTROM3
    current_pressure_kinetic_GPa = P_kinetic / units.GPA_TO_EV_PER_ANGSTROM3
    current_pressure_virial_GPa = P_virial / units.GPA_TO_EV_PER_ANGSTROM3

    print(
        f"[BLOCK {block:8d}/{num_blocks:8d}] SUMMARY ({timestamp})\n"
        f"  Time used:       {block_dt:.3f} s\n"
        f"  Energy / H2O:    {energy_curr:.12f} eV    {(energy_curr/num_molecule+16)*1000:.6f} meV/H2O\n"
        f"  Cell (a,b,c):    {cell_a:.6f}, {cell_b:.6f}, {cell_c:.6f} A\n"
        f"  Density:         {current_density:.6f} g/cm^3\n"
        f"  Pressure:        {current_pressure_GPa:.6f} GPa\n"
        f"  LOOP acc:        block={fmt_rate(loop_acc_block)}, total={fmt_rate(loop_acc_total)}"
        f"    (block: {loop_accepts_block}/{loop_attempts_block}, total: {loop_accepts_total}/{loop_attempts_total})\n"
        f"  state_changed:   {changed_flag}  ({hex_state})\n"
        f"  Dielectric:  G={correlation_G:.6f}   |M|^2={moment2:.6e}   <|mu|^2>={mu2_mean:.6e}\n"
        f"               M=[{moment_vec[0]:.3e}, {moment_vec[1]:.3e}, {moment_vec[2]:.3e}]",
        flush=True,
    )

    # Persist one line of block metrics (note: state_bitstring now stores the int)
    append_block_summary_line(
        log_path=log_txt_path,
        block_idx=block,
        num_blocks=num_blocks,
        time_s=block_dt,
        energy_eV=energy_curr,
        loop_accepts_block=loop_accepts_block,
        loop_attempts_block=loop_attempts_block,
        loop_accepts_total=loop_accepts_total,
        loop_attempts_total=loop_attempts_total,
        mala_accepts_block=mala_accepts_block,
        mala_attempts_block=mala_attempts_block,
        mala_accepts_total=mala_accepts_total,
        mala_attempts_total=mala_attempts_total,
        cell_accepts_block=cell_accepts_block,
        cell_attempts_block=cell_attempts_block,
        cell_accepts_total=cell_accepts_total,
        cell_attempts_total=cell_attempts_total,
        moment_vec=moment_vec,
        mu2_mean=mu2_mean,
        num_molecule=num_molecule,
        temperature_K=target_temperature_K,
        cell_lengths=atoms.cell.lengths(),
        pressure_GPa=current_pressure_GPa,
        stress_xx=float(stress_curr[0]),
        stress_yy=float(stress_curr[1]),
        stress_zz=float(stress_curr[2]),
        state_bitstring=hex_state,
        preamble_text=log_preamble_text,
    )

    # Snapshot structure
    save_snapshot_json(
        json_path=snap_json_path,
        atoms=atoms,
        supercell_size=supercell_size,
        density=current_density,
        O_neighbors=O_neighbors,
        H_to_OO_pairs=H_to_OO_pairs,
        state_hydrogens=state_curr,
        atomcoords_O=data["atomcoords_O"],
        H2_candidates=data["H2_candidates"],
    )

    if save_xyz:
        append_xyz_snapshot(
        atoms=atoms,
        xyz_path=traj_xyz_path,
        )


####################################################################################################

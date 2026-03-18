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
parser = argparse.ArgumentParser(description= "finite-temperature Monte Carlo simulation for water ice Ih (loop + mala + cell update)")

# ----------------------------------------------------------------------------------------------
# Structure / system
# ----------------------------------------------------------------------------------------------
parser.add_argument(
    "--init_stru_path",
    type=Path,
    default=Path("/home/zq/zqcodeml/watericeIh-mc-master/source/structure/initstru/sc_211_n_16.json"),
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
    default=Path("/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_cueq.model"),
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
    "--num_cont_steps",
    type=int,
    default=100,
    help="Number of continuous (MALA/CELL) steps per block.",
)
parser.add_argument(
    "--mala_width_ref",
    type=float,
    default=0.010,
    help="Reference MALA proposal width (in Å).",
)

parser.add_argument(
    "--create_neighborlist_device",
    type=str,
    default="gpu",
    choices=["gpu", "cpu"],
    help="Device to use for creating neighborlist (gpu or cpu).",
)

parser.add_argument(
    "--thermal_loop_force_flip",
    type=int,
    default=0,
    help="Number of thermalization steps (force filp hydrogens).",
)
parser.add_argument(
    "--thermal_loop",
    type=int,
    default=100,
    help="Number of thermalization steps.",
)
parser.add_argument(
    "--thermal_cont",
    type=int,
    default=100,
    help="Number of thermalization steps for continuous updates (MALA/CELL).",
)

parser.add_argument(
    "--update_mala_mode",
    type=str,
    choices=["all", "hydrogens"],
    default="all",
    help="Which atoms to update in MALA step.",
)


parser.add_argument(
    "--p_mala",
    type=float,
    default=0.5,
    help="Probability to choose MALA update after LOOP (1 - p_mala gives CELL update probability).",
)

# ----------------------------------------------------------------------------------------------
# Cell update parameters
# ----------------------------------------------------------------------------------------------
parser.add_argument(
    "--cell_mode",
    type=str,
    choices=["isotropic", "anisotropic"],
    default="anisotropic",
    help="Cell update mode: 'isotropic' scales all dimensions equally, 'anisotropic' scales independently.",
)
parser.add_argument(
    "--cell_width_ref",
    type=float,
    default=0.001,
    help="Reference MC width for cell parameter updates (fractional).",
)
parser.add_argument(
    "--pressure_GPa",
    type=float,
    default=0.0,
    help="External pressure in GPa for NPT ensemble.",
)
# ----------------------------------------------------------------------------------------------
# Print control parameters
# ----------------------------------------------------------------------------------------------
parser.add_argument(
    "--print_interval_loop",
    type=int,
    default=2,
    help="Print frequency (in sweeps) during each LOOP block. Set 0 to disable intermediate prints.",
)

parser.add_argument(
    "--print_interval_mala",
    type=int,
    default=50,
    help="Print frequency (in steps) during each MALA block. Set 0 to disable intermediate prints.",
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

# --- Temperature ---
target_temperature_K: float = args.target_temperature_K

# --- Model paths ---
mace_model_path: Path = args.mace_model_path
mace_device: str = args.mace_device
mace_dtype: str = args.mace_dtype

# --- MC control parameters ---
num_blocks: int = args.num_blocks
num_loop_steps: int = args.num_loop_steps
num_cont_steps: int = args.num_cont_steps
mala_width_ref: float = args.mala_width_ref

thermal_loop_force_flip: int = args.thermal_loop_force_flip
thermal_loop: int = args.thermal_loop
thermal_cont: int = args.thermal_cont

update_mala_mode: str = args.update_mala_mode
p_mala: float = args.p_mala

# --- Cell update parameters ---
cell_mode: str = args.cell_mode
cell_width_ref: float = args.cell_width_ref
pressure_GPa: float = args.pressure_GPa
create_neighborlist_device: str = args.create_neighborlist_device

# --- Print control ---
print_interval_loop: int = args.print_interval_loop
print_interval_mala: int = args.print_interval_mala

# ======== Print parsed configuration ========
print("\n========== Parsed Configuration ==========")
print(f"{'Initial structure:':22s} {init_stru_path}")
print(f"{'Save file path:':22s} {save_file_path}")
print(f"")

print(f"{'Temperature:':22s} {target_temperature_K:.3f} K ({target_temperature_K * units.K_B_EV_PER_K:.8f} eV)")
print(f"{'MACE model path:':22s} {mace_model_path}")
print(f"{'MACE device:':22s} {mace_device}")
print(f"{'MACE dtype:':22s} {mace_dtype}")
print(f"{'Neighborlist device:':22s} {create_neighborlist_device}")
print(f"{'MC total blocks:':22s} {num_blocks}")

print(f"{'LOOP steps/block:':22s} {num_loop_steps}")
print(f"{'CONT steps/block:':22s} {num_cont_steps}")
print(f"{'MALA width ref:':22s} {mala_width_ref:.3f}")

print(f"{'Thermalization (force flip):':22s} {thermal_loop_force_flip}")
print(f"{'Thermalization (loop):':22s} {thermal_loop}")
print(f"{'Thermalization (cont):':22s} {thermal_cont}")

print(f"{'Print interval (loop):':22s} {print_interval_loop}")
print(f"{'Print interval (mala):':22s} {print_interval_mala}")

print(f"{'update mode (mala):':22s} {update_mala_mode}")
print(f"{'Update structure:':22s} LOOP → CONT(MALA/CELL)")
print(f"{'MALA probability:':22s} {p_mala:.2f}")
print(f"{'CELL probability:':22s} {1 - p_mala:.2f}")
print(f"{'CELL mode:':22s} {cell_mode}")
print(f"{'CELL width ref:':22s} {cell_width_ref:.6f}")
print(f"{'Pressure:':22s} {pressure_GPa:.6f} GPa")
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
elif mace_device == "dcu":
    from source.potentialmace_oeq import initialize_mace_model
    print(f"[INIT] Using DCU for MACE model")

mace_inference = initialize_mace_model(
    mace_model_path,
    mace_dtype,
    mace_device="cuda",
)

# Initial energy/force @ current coords
energy_curr, force_curr, _ = mace_inference(
    atoms,
    compute_force=True,
    create_neighborlist_device=create_neighborlist_device,
)

print(f"[INIT]  Energy            : {energy_curr: .12f} eV")
print(f"[INIT]  Force array shape : {force_curr.shape}   (should be (N_atoms, 3))")
print(f"[INIT]  |F|_max           : {np.max(np.linalg.norm(force_curr, axis=1)): .6f} eV/Å")


####################################################################################################
# ----------------------------------------------------------------------------------------------
# Build the Metropolis mala & short-loop update function
# ----------------------------------------------------------------------------------------------
print("\n========== Initialize loop & mala update ==========", flush=True)

# ---------------------------- mala update function
from source.updatemala import make_metropolis_mala_update_functions, suggest_mc_width
mala_step_only_hydrogens, mala_step_all_atoms = make_metropolis_mala_update_functions(
    num_O=num_O,
    mace_inference=mace_inference,
    create_neighborlist_device=create_neighborlist_device,
)

if   update_mala_mode == "all":
    mala_step = mala_step_all_atoms
    print(f"[INIT]  MALA update mode  : update all atoms")
elif update_mala_mode == "hydrogens":
    mala_step = mala_step_only_hydrogens
    print(f"[INIT]  MALA update mode  : update only hydrogens")
else:
    raise ValueError(f"Invalid update_mala_mode: {update_mala_mode}")

# ---------------------------- loop update function
from source.updateloop import (
    make_metropolis_loop_update_functions,
    state_to_bitstring,
    bitstring_to_hexstr,
    )
from source.buildh2 import create_h2_candidates_by_midpoint_flip_vectorized
metropolis_loop_update, perform_loop_flip, metropolis_only_loop_update = make_metropolis_loop_update_functions(
    O_neighbors=O_neighbors,
    H_to_OO_pairs=H_to_OO_pairs,
    mace_inference=mace_inference,
    create_neighborlist_device=create_neighborlist_device,
)

# ---------------------------- cell update function
from source.updatecell import make_metropolis_cell_update_functions, suggest_cell_mc_width
cell_step_isotropic, cell_step_anisotropic = make_metropolis_cell_update_functions(
    mace_inference=mace_inference,
    create_neighborlist_device=create_neighborlist_device,
)

if cell_mode == "isotropic":
    cell_step = cell_step_isotropic
    print(f"[INIT]  CELL update mode  : isotropic (scale all dimensions equally)")
elif cell_mode == "anisotropic":
    cell_step = cell_step_anisotropic
    print(f"[INIT]  CELL update mode  : anisotropic (scale dimensions independently)")
else:
    raise ValueError(f"Invalid cell_mode: {cell_mode}")

# Calculate cell MC width based on temperature and current volume
cell_mc_width = suggest_cell_mc_width(
    target_temperature_K,
    atoms.get_volume(),
    width_ref=cell_width_ref,
)
print(f"[INIT]  CELL MC width     : {cell_mc_width:.6f} (fractional)")

# Convert pressure from GPa to eV/Angstrom^3
pressure_eV_A3 = pressure_GPa * units.GPA_TO_EV_PER_ANGSTROM3
print(f"[INIT]  Pressure          : {pressure_GPa:.6f} GPa ({pressure_eV_A3:.6e} eV/Å³)")


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
from source.updateblock import run_mc_loop_block, run_mc_continuous_block, run_mc_only_loop_block
from source.tools import safe_div, fmt_rate, calculate_pressure_from_stress
from source.crystaltools import check_hydrogen_consistency

# User-tunable parameters
temperature_in_eV = units.K_B_EV_PER_K * target_temperature_K  # Convert T (K) to energy (eV)
mc_width = suggest_mc_width(target_temperature_K, width_ref=mala_width_ref)     # Initial MALA step size scaled ∝ sqrt(T)

# ---- Neat summary ----
print(
    f"[INIT] Blocks                 : {num_blocks}\n"
    f"[INIT] Loop steps / block     : {num_loop_steps}\n"
    f"[INIT] Temperature            : {target_temperature_K:.1f} K  (kT = {temperature_in_eV:.6f} eV)\n"
    f"[INIT] Initial mc_width       : {mc_width:.6f} Å\n",
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
print("\n" + "-" * 80, flush=True)
for step in range(int(thermal_loop_force_flip)):
    state_curr, atoms = perform_loop_flip(
        state_curr,
        atoms,
    )
    # Print compact hydrogen-orientation state (bitstring → hex)
    hex_state = bitstring_to_hexstr(state_to_bitstring(state_curr))
    print(f"[LOOP FLIP]  step={step:6d}  state={hex_state}", flush=True)

# ----------------------------------------------------------------------------------------------
print("\n" + "-" * 80, flush=True)
atomcoords_O, H2_candidates = create_h2_candidates_by_midpoint_flip_vectorized(
    atom_coords=atoms.get_positions(),
    H_to_OO_pairs=H_to_OO_pairs,
    box_lengths=atoms.cell.lengths(),
)
energy_curr, _, _ = mace_inference(atoms, compute_force=False, create_neighborlist_device=create_neighborlist_device)
accepts = 0
attempts = 0

for step in range(int(thermal_loop)):
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
        print_interval_loop=10000000,
    )
    coords_curr = atoms.get_positions()
    attempts += loop_attempts_block
    accepts += loop_accepts_block
    hex_state = bitstring_to_hexstr(state_to_bitstring(state_curr))
    print(f"[LOOP THERMAL] step={step:6d}  E={energy_curr:.12f} eV  acc={accepts:8d}/{attempts:8d}  state={hex_state}", flush=True)

# ----------------------------------------------------------------------------------------------
# 2) Continuous warm-up (MALA/CELL thermalization with probability-based selection)
# ----------------------------------------------------------------------------------------------
print("\n" + "-" * 80, flush=True)
if int(thermal_cont) > 0:
    coords_curr = atoms.get_positions()
    coords_curr = coords_curr + np.random.normal(scale=0.01, size=coords_curr.shape)
    atoms.set_positions(coords_curr)

energy_curr, force_curr, stress_curr = mace_inference(atoms, compute_force=True, create_neighborlist_device=create_neighborlist_device)

cont_mala_accepts = 0
cont_mala_attempts = 0
cont_cell_accepts = 0
cont_cell_attempts = 0

for step in range(int(thermal_cont)):
    energy_curr, state_curr, force_curr, stress_curr, atoms, mala_accepts_thermal, mala_attempts_thermal, cell_accepts_thermal, cell_attempts_thermal = run_mc_continuous_block(
        num_cont_steps=num_cont_steps,
        energy_curr=energy_curr,
        force_curr=force_curr,
        stress_curr=stress_curr,
        state_curr=state_curr,
        atoms=atoms,
        temperature_in_eV=temperature_in_eV,
        mala_step=mala_step,
        cell_step=cell_step,
        mc_width=mc_width,
        p_mala=p_mala,
        print_interval_cont=10000000,
        pressure_eV_A3=pressure_eV_A3,
        mc_width_cell=cell_mc_width,
    )
    coords_curr = atoms.get_positions()
    cont_mala_accepts += mala_accepts_thermal
    cont_mala_attempts += mala_attempts_thermal
    cont_cell_accepts += cell_accepts_thermal
    cont_cell_attempts += cell_attempts_thermal
    cont_attempts = cont_mala_attempts + cont_cell_attempts
    cont_accepts = cont_mala_accepts + cont_cell_accepts
    print(
        f"[CONT THERMAL] step={step:6d}  E={energy_curr:.12f} eV  "
        f"acc={cont_accepts:8d}/{cont_attempts:8d} "
        f"(M:{cont_mala_accepts}/{cont_mala_attempts} C:{cont_cell_accepts}/{cont_cell_attempts})",
        flush=True,
    )
    
# ----------------------------------------------------------------------------------------------
# 3) geometry/state consistency check
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

    state_before = state_curr.copy()
    # ----------------------------- LOOP stage -----------------------------
    # Performs short-loop hydrogen reorientations with Metropolis testing.
    print("\n" + "=" * 80, flush=True)
    energy_curr, state_curr, coords_curr, atoms, loop_accepts_block, loop_attempts_block = run_mc_loop_block(
        num_loop_steps=num_loop_steps,
        energy_curr=energy_curr,
        state_curr=state_curr,
        atoms=atoms,
        temperature_in_eV=temperature_in_eV,
        metropolis_loop_update=metropolis_loop_update,
        print_interval_loop=print_interval_loop,
    )

    # Refresh force and stress for subsequent continuous (MALA/CELL) stage
    energy_curr, force_curr, stress_curr = mace_inference(atoms, compute_force=True, create_neighborlist_device=create_neighborlist_device)
    loop_accepts_total += loop_accepts_block
    loop_attempts_total += loop_attempts_block

    # ----------------------------- Continuous (MALA/CELL) stage -----------------------------
    # Performs mixed MALA and CELL updates with probability-based selection
    energy_curr, state_curr, force_curr, stress_curr, atoms, mala_accepts_block, mala_attempts_block, cell_accepts_block, cell_attempts_block = run_mc_continuous_block(
        num_cont_steps=num_cont_steps,
        energy_curr=energy_curr,
        force_curr=force_curr,
        stress_curr=stress_curr,
        state_curr=state_curr,
        atoms=atoms,
        temperature_in_eV=temperature_in_eV,
        mala_step=mala_step,
        cell_step=cell_step,
        mc_width=mc_width,
        p_mala=p_mala,
        print_interval_cont=print_interval_mala,
        pressure_eV_A3=pressure_eV_A3,
        mc_width_cell=cell_mc_width,
    )
    mala_accepts_total += mala_accepts_block
    mala_attempts_total += mala_attempts_block
    cell_accepts_total += cell_accepts_block
    cell_attempts_total += cell_attempts_block
    updater_label = "LOOP→CONT"

    # Update coords_curr from atoms
    coords_curr = atoms.get_positions()

    # ------------------------------------------------------------------------------------------
    # Diagnostics and reporting
    # Compute acceptance ratios, elapsed time, and structural change flags
    # ------------------------------------------------------------------------------------------
    loop_acc_block = safe_div(loop_accepts_block, loop_attempts_block)
    loop_acc_total = safe_div(loop_accepts_total, loop_attempts_total)
    mala_acc_block = safe_div(mala_accepts_block, mala_attempts_block)
    mala_acc_total = safe_div(mala_accepts_total, mala_attempts_total)
    cell_acc_block = safe_div(cell_accepts_block, cell_attempts_block)
    cell_acc_total = safe_div(cell_accepts_total, cell_attempts_total)

    block_dt = time.perf_counter() - block_t0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    state_changed = not np.array_equal(state_curr, state_before)
    changed_flag = "YES" if state_changed else "NO"
    hex_state = bitstring_to_hexstr(state_to_bitstring(state_curr))

    # Check for hydrogen consistency
    # t1 = time.perf_counter()
    mismatch = check_hydrogen_consistency(
        atoms=atoms,
        H_to_OO_pairs=H_to_OO_pairs,
        state_curr=state_curr,
        cutoff=1.2,
    )
    # t2 = time.perf_counter()
    # print(f"[CHECK] Hydrogen consistency check took {t2 - t1:.6f} s")
    
    # t1 = time.perf_counter()
    # Compute observables
    moment_vec, moment2, mu2_mean, correlation_G = compute_correlation_parameter(atoms)
    # t2 = time.perf_counter()
    # print(f"[CHECK] Correlation parameter computation took {t2 - t1:.6f} s")

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

    atomcoords_O, H2_candidates = create_h2_candidates_by_midpoint_flip_vectorized(
        atom_coords=atoms.get_positions(),
        H_to_OO_pairs=H_to_OO_pairs,
        box_lengths=atoms.cell.lengths(),
    )

    print(
        f"[BLOCK] [{block:8d}/{num_blocks:8d}] SUMMARY ({timestamp}) [{updater_label}]\n"
        f"  Time used:       {block_dt:.3f} s\n"
        f"  Energy / H2O:    {energy_curr:.12f} eV    {(energy_curr/num_molecule+16)*1000:.6f} meV/H2O\n"
        f"  Cell (a,b,c):    {cell_a:.6f}, {cell_b:.6f}, {cell_c:.6f} A\n"
        f"  Density:         {current_density:.6f} g/cm^3\n"
        f"  Pressure:        {current_pressure_GPa:.6f} GPa\n"
        f"  LOOP acc:        block={fmt_rate(loop_acc_block)}, total={fmt_rate(loop_acc_total)}"
        f"    (block: {loop_accepts_block}/{loop_attempts_block}, total: {loop_accepts_total}/{loop_attempts_total})\n"
        f"  MALA acc:        block={fmt_rate(mala_acc_block)}, total={fmt_rate(mala_acc_total)}"
        f"    (block: {mala_accepts_block}/{mala_attempts_block}, total: {mala_accepts_total}/{mala_attempts_total})\n"
        f"  CELL acc:        block={fmt_rate(cell_acc_block)}, total={fmt_rate(cell_acc_total)}"
        f"    (block: {cell_accepts_block}/{cell_attempts_block}, total: {cell_accepts_total}/{cell_attempts_total})\n"
        f"  state_changed:   {changed_flag}  ({hex_state})",
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
        atomcoords_O=atomcoords_O,
        H2_candidates=H2_candidates,
    )

    if save_xyz:
        append_xyz_snapshot(
        atoms=atoms,
        xyz_path=traj_xyz_path,
        )


####################################################################################################

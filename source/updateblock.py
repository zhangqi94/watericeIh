import numpy as np
from typing import Callable, Any, Optional, Tuple, Dict, List

try:
    from tools import safe_div
    from updateloop import bitstring_to_hexstr, state_to_bitstring
except Exception:
    from source.tools import safe_div
    from source.updateloop import bitstring_to_hexstr, state_to_bitstring
    
####################################################################################################

def run_mc_loop_block(
    *,
    num_loop_steps: int,
    energy_curr: float,
    state_curr: np.ndarray,
    atoms: Any,
    temperature_in_eV: float,
    metropolis_loop_update: Callable,
    print_interval_loop: int,
) -> Tuple[float, np.ndarray, np.ndarray, Any, int, int]:
    """
    Run one Monte Carlo loop block consisting of multiple short-loop steps.

    Args:
        num_loop_steps: Number of loop steps in this MC block.
        num_O: Number of oxygen atoms (defines the range of loop starting points).
        energy_curr: Current potential energy at block start.
        state_curr: Current hydrogen orientation state array.
        atoms: ASE Atoms-like object.
        temperature_in_eV: Temperature expressed as k_B * T in eV.
        metropolis_loop_update: Callable performing one Metropolis loop update.
        print_interval_loop: Print progress every this many steps.

    Returns:
        Tuple containing:
            energy_now (float): Final energy after the block.
            state_now (np.ndarray): Final hydrogen state array.
            coords_now (np.ndarray): Final atomic coordinates.
            atoms_now (Any): Updated atoms object.
            accepts_block (int): Total accepted loop updates in this block.
            attempts_block (int): Total loop update attempts in this block.
    """
    # ------------------------------ Initialize ------------------------------
    accepts_block = 0      # number of accepted moves
    attempts_block = 0     # total number of trial moves

    # Make working copies to avoid modifying inputs directly
    energy_now = float(energy_curr)
    state_now = state_curr.copy()
    atoms_now = atoms.copy()

    # ------------------------------ Loop steps ------------------------------
    for sweep in range(1, num_loop_steps + 1):
        state_now, energy_now, atoms_now, accepted = metropolis_loop_update(
            state_hydrogens=state_now,
            start_O=None,
            potential_energy=energy_now,
            temperature_in_eV=float(temperature_in_eV),
            atoms=atoms_now,
        )

        # Count acceptance statistics
        attempts_block += 1
        accepts_block += int(bool(accepted))

        # Periodic printing of block status
        if print_interval_loop > 0 and ((sweep % print_interval_loop == 0) or (sweep == num_loop_steps)):
            print(
                f"[LOOP]  step={sweep:6d}  "
                f"E={energy_now:.12f} eV  acc={accepts_block:3d}/{attempts_block:3d}",
                flush=True,
            )

    # Get final coordinates from atoms
    coords_now = atoms_now.get_positions()

    # Return final state after all steps
    return energy_now, state_now, coords_now, atoms_now, accepts_block, attempts_block


####################################################################################################

def run_mc_continuous_block(
    *,
    num_cont_steps: int,
    energy_curr: float,
    force_curr: np.ndarray,
    stress_curr: np.ndarray,
    state_curr: np.ndarray,
    atoms: Any,
    temperature_in_eV: float,
    mala_step: Callable[..., Tuple[float, np.ndarray, np.ndarray, Any, bool]],
    cell_step: Callable[..., Tuple[float, np.ndarray, np.ndarray, Any, bool]],
    mc_width: float,
    p_mala: float,
    print_interval_cont: int,
    pressure_eV_A3: float = 0.0,
    mc_width_cell: float = 0.001,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, Any, int, int, int, int]:
    """Run one continuous (MALA/cell) block with random selection between MALA and cell updates.

    Args:
        num_cont_steps: Number of continuous steps to perform in this block (>= 0).
        energy_curr: Current total potential energy (eV) at the start of the block.
        force_curr: Current atomic forces array, shape (N_atoms, 3).
        stress_curr: Current stress vector (6,) in eV/Angstrom^3.
        state_curr: Discrete hydrogen-bond state array (unchanged by MALA/cell).
        atoms: ASE Atoms-like object supporting `.copy()` and `.set_positions()`.
        temperature_in_eV: Temperature expressed as k_B*T in eV.
        mala_step: Callable implementing a single MALA step.
        cell_step: Callable implementing a single cell update step.
        mc_width: Step-size (sigma) for the MALA move.
        p_mala: Probability of performing MALA update (p_cell = 1 - p_mala).
        print_interval_cont: Print progress every this many steps.
        pressure_eV_A3: External pressure in eV/Angstrom^3 for cell updates.
        mc_width_cell: Step-size (fractional) for cell updates.

    Returns:
        Tuple:
            energy_now (float): Final energy after all steps.
            state_now (np.ndarray): Same hydrogen state array (unchanged).
            force_now (np.ndarray): Final atomic forces.
            stress_now (np.ndarray): Final stress vector.
            atoms_now (Any): Updated Atoms object.
            mala_accepts (int): Number of accepted MALA moves.
            mala_attempts (int): Total number of attempted MALA moves.
            cell_accepts (int): Number of accepted cell moves.
            cell_attempts (int): Total number of attempted cell moves.
    """
    # ------------------------------ Initialize ------------------------------
    mala_accepts = 0
    mala_attempts = 0
    cell_accepts = 0
    cell_attempts = 0

    energy_now = float(energy_curr)
    force_now = np.asarray(force_curr, dtype=float)
    stress_now = np.asarray(stress_curr, dtype=float)
    atoms_now = atoms.copy()
    state_now = state_curr.copy()

    # ------------------------------ Mixed MALA/cell iterations -------------------------
    for step in range(1, num_cont_steps + 1):
        # Randomly choose between MALA and cell update
        if np.random.rand() < p_mala:
            # Perform one MALA step (internal accept/reject)
            energy_now, force_now, stress_now, atoms_now, accepted = mala_step(
                energy_curr=energy_now,
                force_curr=force_now,
                stress_curr=stress_now,
                temperature_in_eV=float(temperature_in_eV),
                atoms=atoms_now,
                mc_width=mc_width,
            )
            mala_attempts += 1
            mala_accepts += int(bool(accepted))
            update_type = "MALA"
        else:
            # Perform one cell update step (internal accept/reject)
            energy_now, force_now, stress_now, atoms_now, accepted = cell_step(
                energy_curr=energy_now,
                force_curr=force_now,
                stress_curr=stress_now,
                temperature_in_eV=float(temperature_in_eV),
                atoms=atoms_now,
                pressure_eV_A3=pressure_eV_A3,
                mc_width=mc_width_cell,
            )
            cell_attempts += 1
            cell_accepts += int(bool(accepted))
            update_type = "CELL"

        # Periodic status print
        if print_interval_cont > 0 and ((step % print_interval_cont == 0) or (step == num_cont_steps)):
            total_attempts = mala_attempts + cell_attempts
            total_accepts = mala_accepts + cell_accepts
            print(
                f"[CONT]  step={step:6d}  "
                f"E={energy_now:.12f} eV  "
                f"acc={total_accepts:3d}/{total_attempts:3d} "
                f"(M:{mala_accepts}/{mala_attempts} C:{cell_accepts}/{cell_attempts})",
                flush=True,
            )

    # Return final state after all steps
    return energy_now, state_now, force_now, stress_now, atoms_now, mala_accepts, mala_attempts, cell_accepts, cell_attempts



####################################################################################################
####################################################################################################

def run_mc_only_loop_block(
    *,
    num_loop_steps: int,
    num_O: int,
    energy_curr: float,
    state_curr: np.ndarray,
    atoms: Any,
    temperature_in_eV: float,
    metropolis_only_loop_update: Callable,
    atomcoords_O: np.ndarray,
    H2_candidates: np.ndarray,
    print_interval_loop: int,
) -> Tuple[float, np.ndarray, Any, int, int]:
    """
    Run one Monte Carlo loop block consisting of multiple short-loop steps.
    """
    # ------------------------------ Initialize ------------------------------
    accepts_block = 0      # number of accepted moves
    attempts_block = 0     # total number of trial moves

    # Make working copies to avoid modifying inputs directly
    energy_now = float(energy_curr)
    state_now = state_curr.copy()
    atoms_now = atoms.copy()

    # ------------------------------ Loop steps ------------------------------
    for sweep in range(1, num_loop_steps + 1):

        state_now, energy_now, atoms_now, accepted = metropolis_only_loop_update(
            state_hydrogens=state_now,
            start_O=None,
            potential_energy=energy_now,
            temperature_in_eV=float(temperature_in_eV),
            atoms=atoms_now,
            atomcoords_O=atomcoords_O,
            H2_candidates=H2_candidates,
        )

        H_selected = H2_candidates[np.arange(len(state_now)), state_now]
        coords_now = np.concatenate([atomcoords_O, H_selected], axis=0)
        atoms_now.set_positions(coords_now)

        # Count acceptance statistics
        attempts_block += 1
        accepts_block += int(bool(accepted))

        # Periodic printing of block status
        if print_interval_loop > 0 and ((sweep % print_interval_loop == 0) or (sweep == num_loop_steps)):
            print(
                f"[LOOP]  step={sweep:6d}  "
                f"E={energy_now:.12f} eV  acc={accepts_block:3d}/{attempts_block:3d}",
                flush=True,
            )

    # Return final state after all steps
    return energy_now, state_now, atoms_now, accepts_block, attempts_block

####################################################################################################
####################################################################################################
"""
Examples:

# Loop block example
energy_curr, state_curr, coords_curr, atoms, loop_accepts_block, loop_attempts_block = run_mc_loop_block(
    num_loop_steps=num_loop_steps,
    num_O=num_O,
    energy_curr=energy_curr,
    state_curr=state_curr,
    atoms=atoms,
    temperature_in_eV=temperature_in_eV,
    metropolis_loop_update=metropolis_loop_update,
    print_interval_loop=print_interval_loop,
)

# Mixed MALA/cell block example
energy_curr, force_curr, stress_curr = mace_inference(atoms, compute_force=True)
energy_curr, state_curr, force_curr, stress_curr, atoms, mala_accepts, mala_attempts, cell_accepts, cell_attempts = run_mc_continuous_block(
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
    p_mala=0.8,
    print_interval_cont=print_interval_cont,
    pressure_eV_A3=pressure_eV_A3,
    mc_width_cell=mc_width_cell,
)

"""



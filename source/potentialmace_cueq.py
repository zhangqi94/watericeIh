import torch
import mace
from mace.calculators import MACECalculator
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from copy import deepcopy

import ase
import argparse
import numpy as np
from typing import Callable, Any, Optional, Tuple, Dict, List

try:
    from potential_neighborlist import build_atomic_data_gpu
except Exception:
    from source.potential_neighborlist import build_atomic_data_gpu

####################################################################################################

def make_mace_calculator(mace_model_path: str,
                         mace_dtype = "float32",
                         mace_device: str = "cuda",
                         enable_cueq: bool = True,
                         ):
    """Create a MACE ASE calculator.

    Args:
        mace_model_path: Path to a pretrained MACE model file.
        mace_dtype: Default dtype used by the calculator, e.g. "float32".
        mace_device: Device string, e.g. "cuda" or "cpu".
        enable_cueq: Whether to enable Cueq acceleration when available.

    Returns:
        A configured `MACECalculator`.
    """
        
    calc = MACECalculator(model_paths = mace_model_path,
                          device = mace_device,
                          default_dtype = mace_dtype,
                          enable_cueq = enable_cueq,
                          )
    
    return calc

####################################################################################################
def initialize_mace_model(
    mace_model_path: str,
    mace_dtype: str = "float32",
    mace_device: str = "cuda",
):
    """Load a pretrained MACE model and return a batched inference function."""
    
    # Parse model arguments
    mace_args = argparse.Namespace(default_type = mace_dtype,
                                    model = mace_model_path,
                                    device = mace_device,
                                    )

    # Set the default data type for PyTorch and initialize the computation device
    torch_tools.set_default_dtype(mace_args.default_type)
    device = torch_tools.init_device(mace_args.device)

    # Load the pretrained model and move it to the specified device
    model = torch.load(f=mace_args.model, map_location=mace_args.device)
    model_path = str(mace_args.model)
    if model_path.endswith("_cueq.model"):
        print("[INFO] Model filename ends with '_cueq.model'; skipping conversion.")
    elif mace_args.device == "cuda":
        print("[INFO] Converting model from E3NN to CUEQ format...")
        model = run_e3nn_to_cueq(deepcopy(model), device=device)
        print("[INFO] Conversion complete.")    

    for param in model.parameters():
        param.requires_grad = False

    atomic_numbers = model.atomic_numbers
    r_max = model.r_max
    try:
        heads = model.heads
    except AttributeError:
        heads = None
    
    model = model.to(mace_args.device)
    model.eval()

    ####################################################################################################
    def mace_inference(
        atoms: ase.Atoms,
        compute_force: bool = True,
        create_neighborlist_device: str = "gpu",
    ):
        """Run MACE inference for a single atoms object.
        Args:
            atoms: An ASE `Atoms` object with positions already set.
            compute_force: If True, compute forces and stresses; else energies only.

        Returns:
            (energy: float, forces: (N, 3), stress_vec: (6,))
        """

        compute_stress = compute_force  # Keep the same switch for forces/stress

        # ---------------- Build a list of ASE Atoms for single frame ----------------
        atoms_list = [atoms]
            
        # ---------------- Convert ASE Atoms to MACE input format ----------------
        configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
        z_table = utils.AtomicNumberTable([int(z) for z in atomic_numbers])

        if create_neighborlist_device == "cpu":
            data_set = [data.AtomicData.from_config(
                        config, z_table=z_table, cutoff=float(r_max), heads=heads
                        )
                        for config in configs
                        ]
        elif create_neighborlist_device == "gpu":
            data_set = build_atomic_data_gpu(
                configs=configs, 
                cutoff=float(r_max), 
                atomic_numbers=atomic_numbers,
                device="cuda",
                )
            
        # Create a data loader for batched processing of input data
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset = data_set,
            batch_size = 1,
            shuffle = False,
            drop_last = False,
        )

        # ---------------- Prepare containers for batched outputs ----------------
        energies_list: List[np.ndarray] = []
        stresses_list: List[np.ndarray] = []
        forces_collection: List[List[np.ndarray]] = []

        # ---------------- Forward pass through MACE model ----------------
        for batch in data_loader:

            batch = batch.to(device)

            output = model(batch,
                        compute_force=compute_force,
                        compute_stress=compute_stress,
                        )

            energies_list.append(torch_tools.to_numpy(output["energy"]))

            if compute_stress:
                stresses_list.append(torch_tools.to_numpy(output["stress"]))

                forces = torch_tools.to_numpy(output["forces"])
                forces_collection.append(forces)

        # ---------------- Extract single-frame results ----------------
        energy = float(energies_list[0][0])

        if compute_stress:
            stress = stresses_list[0][0].astype(np.float64)
            forces = forces_collection[0].astype(np.float64)

            # Convert full stress tensor to 6-component vector (xx, yy, zz, xy, xz, yz)
            stress_vec = np.array([
                stress[0, 0], stress[1, 1], stress[2, 2],
                stress[0, 1], stress[0, 2], stress[1, 2]
            ])
        else:
            forces = np.zeros((len(atoms), 3), dtype=np.float64)
            stress_vec = np.zeros(6, dtype=np.float64)

        return energy, forces, stress_vec

    return mace_inference


####################################################################################################
# Test section
####################################################################################################

if __name__ == "__main__":
    import time

    print("mace version:", mace.__version__)

    ################################################################################################
    # --- Model setup ---
    mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_cueq.model"

    mace_dtype = "float32"
    mace_device = "cuda"

    # --- Load structure ---
    # stru_file = "structure/initstru/sc_322_n_96.vasp"
    stru_file = "structure/initstru/sc_533_n_360.vasp"
    
    atoms = ase.io.read(stru_file)
    positions_init = atoms.get_positions()
    num_atoms, dim = positions_init.shape
    rng = np.random.default_rng(seed=0)

    # --- Initialize model ---
    mace_inference = initialize_mace_model(
        mace_model_path=mace_model_path,
        mace_dtype=mace_dtype,
        mace_device=mace_device,
    )

    # --- Warmup ---
    print("\n[Warmup]")
    atoms.set_positions(positions_init)
    E_warmup, F_warmup, S_warmup = mace_inference(
        atoms=atoms,
        compute_force=True,
        create_neighborlist_device="cpu",
    )
    print(f"Energy: {E_warmup:.8f} eV")
    print(f"Forces shape: {F_warmup.shape}")
    print(f"Stress shape: {S_warmup.shape}")

    # --- Test parameters ---
    num_iter = 100

    ################################################################################################
    # Test 1: GPU neighbor list with force
    ################################################################################################
    print("\n" + "="*70)
    print("Test 1: GPU neighbor list WITH force computation")
    print("="*70)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed=42)

    for i in range(num_iter):
        atomcoords = positions_init + 0.10 * rng.uniform(size=(num_atoms, dim))
        atoms.set_positions(atomcoords)
        E, F, S = mace_inference(
            atoms,
            compute_force=True,
            create_neighborlist_device="gpu",
        )

    t1 = time.perf_counter()
    gpu_force_time = t1 - t0
    gpu_force_avg = gpu_force_time / num_iter

    print(f"Total time for {num_iter} evaluations: {gpu_force_time:.6f} s")
    print(f"Average time per evaluation: {gpu_force_avg:.6f} s")
    print(f"Last energy: {E:.8f} eV")

    ################################################################################################
    # Test 2: GPU neighbor list without force
    ################################################################################################
    print("\n" + "="*70)
    print("Test 2: GPU neighbor list WITHOUT force computation")
    print("="*70)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed=42)

    for i in range(num_iter):
        atomcoords = positions_init + 0.10 * rng.uniform(size=(num_atoms, dim))
        atoms.set_positions(atomcoords)
        E, F, S = mace_inference(
            atoms,
            compute_force=False,
            create_neighborlist_device="gpu",
        )

    t1 = time.perf_counter()
    gpu_noforce_time = t1 - t0
    gpu_noforce_avg = gpu_noforce_time / num_iter

    print(f"Total time for {num_iter} evaluations: {gpu_noforce_time:.6f} s")
    print(f"Average time per evaluation: {gpu_noforce_avg:.6f} s")
    print(f"Last energy: {E:.8f} eV")

    ################################################################################################
    # Test 3: CPU neighbor list with force
    ################################################################################################
    print("\n" + "="*70)
    print("Test 3: CPU neighbor list WITH force computation")
    print("="*70)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed=42)

    for i in range(num_iter):
        atomcoords = positions_init + 0.10 * rng.uniform(size=(num_atoms, dim))
        atoms.set_positions(atomcoords)
        E, F, S = mace_inference(
            atoms,
            compute_force=True,
            create_neighborlist_device="cpu",
        )

    t1 = time.perf_counter()
    cpu_force_time = t1 - t0
    cpu_force_avg = cpu_force_time / num_iter

    print(f"Total time for {num_iter} evaluations: {cpu_force_time:.6f} s")
    print(f"Average time per evaluation: {cpu_force_avg:.6f} s")
    print(f"Last energy: {E:.8f} eV")

    ################################################################################################
    # Test 4: CPU neighbor list without force
    ################################################################################################
    print("\n" + "="*70)
    print("Test 4: CPU neighbor list WITHOUT force computation")
    print("="*70)

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed=42)

    for i in range(num_iter):
        atomcoords = positions_init + 0.10 * rng.uniform(size=(num_atoms, dim))
        atoms.set_positions(atomcoords)
        E, F, S = mace_inference(
            atoms,
            compute_force=False,
            create_neighborlist_device="cpu",
        )

    t1 = time.perf_counter()
    cpu_noforce_time = t1 - t0
    cpu_noforce_avg = cpu_noforce_time / num_iter

    print(f"Total time for {num_iter} evaluations: {cpu_noforce_time:.6f} s")
    print(f"Average time per evaluation: {cpu_noforce_avg:.6f} s")
    print(f"Last energy: {E:.8f} eV")

    ################################################################################################
    # Summary
    ################################################################################################
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"GPU with force:    {gpu_force_avg*1000:.3f} ms/eval")
    print(f"GPU without force: {gpu_noforce_avg*1000:.3f} ms/eval")
    print(f"CPU with force:    {cpu_force_avg*1000:.3f} ms/eval")
    print(f"CPU without force: {cpu_noforce_avg*1000:.3f} ms/eval")
    print(f"\nSpeedup (GPU vs CPU, with force):    {cpu_force_avg/gpu_force_avg:.2f}x")
    print(f"Speedup (GPU vs CPU, without force): {cpu_noforce_avg/gpu_noforce_avg:.2f}x")


######################################################################################################


"""
#### test on 4090
singularity exec --nv --no-home \
    --bind /home/zq/zqcodeml:/home/zq/zqcodeml \
    /home/zq/zqdata/images/cuda12.8-jax0503-torch280-mace314.sif bash -c \
"
source /jaxtorchmace/bin/activate
cd /home/zq/zqcodeml/watericeIh-mc-master/source
python3 potentialmace_cueq.py
"

#### test on t02
srun -p home --cpus-per-task=16 --mem=32GB --gres=gpu:NV5090:1 -t 0-01:00 --pty /bin/bash
singularity exec --no-home --nv --bind /home/user_zhangqi/private/homefile/t02codeml:/t02codeml \
    /home/user_zhangqi/private/homefile/images/cuda12.8-jax0503-torch280-mace314.sif bash -c \
"
source /jaxtorchmace/bin/activate
cd /t02codeml/watericeIh-mc-master/source
python3 potentialmace_cueq.py
"

##########
sudo singularity shell --nv --no-home \
  --bind /home/zq/zqcodeml:/home/zq/zqcodeml \
  /home/zq/zqdata/images/cuda12.8-jax0503-torch280-mace314.sif
source /jaxtorchmace/bin/activate
cd /home/zq/zqcodeml/watericeIh-mc-master/potential
python3 potentialmace_cueq.py
"""

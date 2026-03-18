import torch
import torch.nn as nn
import os, warnings

import mace
from mace.calculators import MACECalculator
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
### from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq
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

warnings.filterwarnings(
    "ignore",
    message=".*weights_only=False.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*copy construct from a tensor.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.jit\._check",
)

####################################################################################################

def make_mace_calculator(mace_model_path: str,
                         mace_dtype = "float32",
                         mace_device: str = "cuda",
                         enable_oeq=True,
                         ):
    """Create a MACE ASE calculator.
    """
        
    calc = MACECalculator(model_paths = mace_model_path,
                          device = mace_device,
                          default_dtype = mace_dtype,
                          enable_oeq = enable_oeq,
                          )

    def _force_tp_compat(calc):

        class _TPCompat(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, X, edge_attrs, W, edge_index):
                src, dst = edge_index[0].long(), edge_index[1].long()
                Xe = X[src]
                msg = self.base(Xe, edge_attrs, W)  # e3nn/OEq TensorProduct(x, y, w)

                N = X.size(0)
                out = torch.zeros((N, msg.size(-1)), dtype=msg.dtype, device=msg.device)
                out.index_add_(0, dst, msg)
                return out

        ## Collect all possible model modules (single-head or multi-head)
        models = []
        if hasattr(calc, "model") and isinstance(calc.model, nn.Module):
            models.append(calc.model)
        if hasattr(calc, "models"):
            if isinstance(calc.models, dict):
                models.extend([m for m in calc.models.values() if isinstance(m, nn.Module)])
            elif isinstance(calc.models, (list, tuple)):
                models.extend([m for m in calc.models if isinstance(m, nn.Module)])

        ## Inject compatibility wrapper into all convolution TP modules
        for mdl in models:
            for _, module in mdl.named_modules():
                if hasattr(module, "conv_tp"):
                    module.conv_tp = _TPCompat(module.conv_tp)

    # Apply the TP compatibility patch
    _force_tp_compat(calc)

    return calc

####################################################################################################
def initialize_mace_model(
    mace_model_path: str,
    mace_dtype: str = "float32",
    mace_device: str = "cuda",
):
    """Load a pretrained MACE model and return a single-structure inference function."""
    
    # Parse model arguments
    mace_args = argparse.Namespace(default_type = mace_dtype,
                                    model = mace_model_path,
                                    device = mace_device,
                                    )

    # Set the default data type for PyTorch and initialize the computation device
    torch_tools.set_default_dtype(mace_args.default_type)
    device = torch_tools.init_device(mace_args.device)

    # # # Load the pretrained model and move it to the specified device
    # # ----------------------------------------------------------------------------------------------
    # # Retrieve the underlying nn.Module model depending on input type
    # # ----------------------------------------------------------------------------------------------
    print("[INFO] Converting model from E3NN to OEQ format...")
    calc = make_mace_calculator(mace_model_path = mace_model_path,
                                mace_dtype = mace_dtype,
                                mace_device = mace_device,
                                enable_oeq = True,
                                )
    model = calc.models[0]
    print("[INFO] Conversion complete.")   

    ##----------------------------------------------------------------------------------------------

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
            create_neighborlist_device: Neighbor list device, "gpu" or "cpu".

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
            neighborlist_device = "cuda" if torch.cuda.is_available() else "cpu"
            data_set = build_atomic_data_gpu(
                configs=configs,
                cutoff=float(r_max),
                atomic_numbers=atomic_numbers,
                device=neighborlist_device,
            )
        else:
            raise ValueError("`create_neighborlist_device` must be 'gpu' or 'cpu'.")

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
                
            batch_dict = batch.to_dict()
            output = model(batch_dict, 
                           compute_force =compute_force,
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
    mace_model_path = "potential/macemodel260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_e3nn.model"

    mace_dtype = "float32"
    mace_device = "cuda"

    # --- Load structure ---
    # stru_file = "structure/initstru/sc_222_n_64.vasp"
    # stru_file = "structure/initstru/sc_322_n_96.vasp"
    # stru_file = "structure/initstru/sc_422_n_128.vasp"
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

    # Warmup
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
    n_show = min(5, F_warmup.shape[0])
    print(f"First {n_show} force rows (eV/A):")
    print(F_warmup[:n_show])
    print("Stress vector S (xx, yy, zz, xy, xz, yz):")
    print(S_warmup)
    

    # Warmup
    print("\n[Warmup]")
    atoms.set_positions(positions_init)
    E_warmup, F_warmup, S_warmup = mace_inference(
        atoms=atoms,
        compute_force=True,
        create_neighborlist_device="gpu",
    )
    print(f"Energy: {E_warmup:.8f} eV")
    print(f"Forces shape: {F_warmup.shape}")
    print(f"Stress shape: {S_warmup.shape}")
    n_show = min(5, F_warmup.shape[0])
    print(f"First {n_show} force rows (eV/A):")
    print(F_warmup[:n_show])
    print("Stress vector S (xx, yy, zz, xy, xz, yz):")
    print(S_warmup)

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
#### test on dcu
srun -p newlarge --gres=dcu:1 --ntasks-per-node=1 --cpus-per-task=14 --mem=64G --time=12:00:00 --pty /bin/bash
unset PROMPT_COMMAND 2>/dev/null || true
export PROMPT_COMMAND=

conda activate macepy311-oeq
cd /public/home/zhangqi2025/zqcode/watericeIh-mc-master/source

sed -i 's/\r$//' dcu/dtk_env.sh
chmod +x dcu/dtk_env.sh
source dcu/dtk_env.sh

python potentialmace_oeq.py




#### test on tania
srun -p large_tania --gres=dcu:1 --ntasks-per-node=1 --cpus-per-task=14 --mem=64G --time=12:00:00 --pty /bin/bash
unset PROMPT_COMMAND 2>/dev/null || true
export PROMPT_COMMAND=

conda activate macepy311-oeq
cd /public/home/zhangqi2025/zqcode/watericeIh-mc-master/source

sed -i 's/\r$//' dcu/dtk_env.sh
chmod +x dcu/dtk_env.sh
source dcu/dtk_env.sh

python potentialmace_oeq.py


"""

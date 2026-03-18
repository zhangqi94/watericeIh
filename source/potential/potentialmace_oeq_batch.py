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
    mace_batch_size: int,
    mace_dtype: str = "float32",
    mace_device: str = "d",
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

    ####################################################################################################
    def mace_inference(
        atoms: ase.Atoms,
        atomcoords: np.ndarray,
        compute_force: bool = True,
    ):
        """Run MACE inference for one or many coordinate frames.
        Args:
            atoms: An ASE `Atoms` template (chemical identity, cell, pbc, etc.).
            atomcoords: Coordinates, shape `(N,3)` or `(B,N,3)`.
            compute_stress: If True, compute forces and stresses; else energies only.

        Returns:
            If a single frame is provided:
                (energy: float, forces: (N, 3), stress_vec: (6,))
            If multiple frames are provided:
                (energies: (B,), forces: (B, N, 3), stress_vecs: (B, 6))
        """
        compute_stress = compute_force  # Keep the same switch for forces/stress
        
        # ---------------- Normalize dtype and batch dimension ----------------
        if mace_args.default_type == "float32":
            atomcoords = np.asarray(atomcoords, dtype=np.float32).copy()
        elif mace_args.default_type == "float64":
            atomcoords = np.asarray(atomcoords, dtype=np.float64).copy()
            
        # Add batch dimension if not present
        if atomcoords.ndim == 2:
            atomcoords = atomcoords[None, ...]  # -> (1, N, 3)
            single_frame = True
        elif atomcoords.ndim == 3:
            single_frame = False
        else:
            raise ValueError("`atomcoords` must be (N,3) or (B,N,3).")
            
        total_batch, num_atoms, dim = atomcoords.shape
        atomcoords = atomcoords.reshape(total_batch, num_atoms, dim)
        
        # ---------------- Build a list of ASE Atoms for each frame ----------------
        atoms_list: List[ase.Atoms] = []
        for i in range(total_batch):
            atoms.set_positions(atomcoords[i])
            atoms_list.append(atoms.copy())
            
        # ---------------- Convert ASE Atoms to MACE input format ----------------
        configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
        z_table = utils.AtomicNumberTable([int(z) for z in atomic_numbers])

        data_set = [data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=float(r_max), heads=heads
                    )
                    for config in configs
                    ]
        # Create a data loader for batched processing of input data
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset = data_set,
            batch_size = mace_batch_size,
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
                
                forces = np.split(
                    torch_tools.to_numpy(output["forces"]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )
                forces_collection.append(forces[:-1])  # drop last as its empty

        # ---------------- Aggregate results ----------------
        energies = np.concatenate(energies_list, axis=0).astype(np.float64)
        
        if compute_stress:
            stresses = np.concatenate(stresses_list, axis=0).astype(np.float64)
            # Flatten forces and ensure consistency with input
            forces_list = [forces for forces_list in forces_collection for forces in forces_list]
            forces = np.array(forces_list, dtype=np.float64)
        else:
            forces = np.zeros((total_batch, num_atoms, dim))
        
        # ---------------- Reshape outputs for consistency ----------------
        energies = energies.reshape(total_batch)
        
        if compute_stress:
            stresses = stresses.reshape(total_batch, dim, dim)
        else:
            stresses = np.zeros((total_batch, dim, dim))
        
        # Convert full stress tensors to 6-component vectors (xx, yy, zz, xy, xz, yz)
        stress_vectors = np.stack([stresses[..., 0, 0], 
                                   stresses[..., 1, 1], 
                                   stresses[..., 2, 2], 
                                   stresses[..., 0, 1], 
                                   stresses[..., 0, 2], 
                                   stresses[..., 1, 2]], axis=-1
                                  )
        
        # ---------------- Return single-frame or batched outputs ----------------
        if single_frame:
            return float(energies[0]), forces[0], stress_vectors[0]
        else:
            return energies, forces, stress_vectors

    return mace_inference


####################################################################################################
# Test section
####################################################################################################

if __name__ == "__main__":
    import time

    print("mace version:", mace.__version__)

    ################################################################################################
    # --- Model setup ---
    mace_model_path = "potential/macemodel/mace_iceIh_128x0e128x1o_r5.0_float32_k43_mace314.model"
    mace_dtype = "float32"
    mace_batch_size = 1
    mace_device = "cuda"
    # mace_device = "cpu"

    # Initialize the custom MACE inference function
    mace_inference = initialize_mace_model(
        mace_model_path=mace_model_path,
        mace_batch_size=mace_batch_size,
        mace_dtype=mace_dtype,
        mace_device=mace_device,
    )

    # Also initialize an ASE-compatible calculator for verification
    # mace_calc = make_mace_calculator(
    #     mace_model_path=mace_model_path,
    #     mace_dtype=mace_dtype,
    #     mace_device=mace_device,
    #     enable_oeq=True,
    # )
    
    ################################################################################################
    # --- Load structure ---
    stru_file = "structure/initstru/sc_212_n_32_rho_933.vasp"
    
    atoms = ase.io.read(stru_file)
    # atoms.calc = mace_calc  # attach calculator for ASE evaluation

    positions_init = atoms.get_positions()
    num_atoms, dim = positions_init.shape
    rng = np.random.default_rng(seed=0)

    # Generate a slightly perturbed structure for testing
    atomcoords_batch = positions_init + 0.2 * rng.uniform(size=(1, num_atoms, dim))

    if 1:
        ################################################################################################
        # 1) Single-frame evaluation with forces & stress
        ################################################################################################
        print("\n>>> Single-frame evaluation (with forces & stress)...")

        t0 = time.perf_counter()
        E_custom, F_custom, S_custom = mace_inference(
            atoms=atoms,
            atomcoords=positions_init,
            compute_force=True,
        )
        t1 = time.perf_counter()
        print(f"Custom inference done in {t1 - t0:.3f} s")
        print(f"Energy (custom) [eV]: {E_custom:.8f}")
        print(f"Forces shape: {F_custom.shape}")
        print(f"Stress vector (xx,yy,zz,xy,xz,yz): {S_custom}")

        print("-"*80)
        # ---------- Energy ----------
        print("Energy (eV):")
        print(f"  {E_custom: .16f}")
        print()

        # ---------- Stress ----------
        print("Stress (eV/Å³):")
        try:
            S_flat = S_custom.reshape(-1)
        except:
            S_flat = S_custom

        for i, val in enumerate(S_flat):
            print(f"  component {i:2d}: {val: .16f}")
        print()

        # ---------- Forces ----------
        print("Forces (eV/Å):")
        for i, f in enumerate(F_custom):
            print(f"  atom {i:3d}: {f[0]: .16f}  {f[1]: .16f}  {f[2]: .16f}")
        
        # ---------- Test two-frame consistency ----------
        E1, F1, S1 = mace_inference(atoms=atoms, atomcoords=positions_init, compute_force=True)
        E2, F2, S2 = mace_inference(atoms=atoms, atomcoords=positions_init, compute_force=True)

        print("max |ΔF|:", np.max(np.abs(F1 - F2)))
        print("max |ΔS|:", np.max(np.abs(S1 - S2)))
        print("ΔE:", abs(E1 - E2))
        
        ################################################################################################
        # --- Compare to ASE calculator results ---
        ################################################################################################
        # # Use ASE calculator for reference computation
        # atoms.set_positions(atomcoords_batch[0])
        # E_ase = atoms.get_potential_energy()
        # F_ase = atoms.get_forces()
        # S_ase_full = atoms.get_stress(voigt=False)  # full 3x3 tensor
        # S_ase = np.array([S_ase_full[0, 0], S_ase_full[1, 1], S_ase_full[2, 2],
        #                   S_ase_full[0, 1], S_ase_full[0, 2], S_ase_full[1, 2]])

        # # Compute absolute differences
        # dE = abs(E_custom - E_ase)
        # dF = np.max(np.abs(F_custom - F_ase))
        # dS = np.max(np.abs(S_custom - S_ase))

        # print("\n--- Consistency check (custom vs ASE calculator) ---")
        # print(f"ΔE (abs)  = {dE:.3e} eV")
        # print(f"ΔF (rel)  = {dF:.3e}")
        # print(f"ΔS (rel)  = {dS:.3e}")

    if 0:
        ################################################################################################
        # 2) Timing benchmark: 100 evaluations with forces & stress
        ################################################################################################
        print("\n>>> Timing 100 evaluations (with forces & stress)...")
        t0 = time.perf_counter()
        for _ in range(100):
            atomcoords_batch = positions_init + 0.1 * rng.uniform(size=(1, num_atoms, dim))
            energies, forces, stress_vectors = mace_inference(
                atoms=atoms,
                atomcoords=atomcoords_batch,
                compute_force=True,
            )
        t1 = time.perf_counter()
        print(f"100 evaluations finished in {t1 - t0:.3f} s")
        print("energies shape:", energies.shape)
        print("forces shape:", forces.shape)
        print("stress_vectors shape:", stress_vectors.shape)

        print("\n>>> Timing 100 evaluations (with forces & stress)...")
        t0 = time.perf_counter()
        for _ in range(100):
            atomcoords_batch = positions_init + 0.1 * rng.uniform(size=(1, num_atoms, dim))
            energies, forces, stress_vectors = mace_inference(
                atoms=atoms,
                atomcoords=atomcoords_batch,
                compute_force=True,
            )
        t1 = time.perf_counter()
        print(f"100 evaluations finished in {t1 - t0:.3f} s")
        print("energies shape:", energies.shape)
        print("forces shape:", forces.shape)
        print("stress_vectors shape:", stress_vectors.shape)
        ################################################################################################
        # 3) Evaluation without forces & stress
        ################################################################################################
        print("\n>>> Timing 100 evaluations (energy only)...")

        t0 = time.perf_counter()
        for _ in range(100):
            atomcoords_batch = positions_init + 0.1 * rng.uniform(size=(1, num_atoms, dim))
            energies_ns, forces_ns, stress_vectors_ns = mace_inference(
                atoms=atoms,
                atomcoords=atomcoords_batch,
                compute_force=False,
            )
        t1 = time.perf_counter()
        print(f"100 energy-only evaluation done in {t1 - t0:.3f} s")

        print("energies shape:", energies_ns.shape)
        print("forces (should be zeros) shape:", forces_ns.shape)
        print("stress_vectors (should be zeros) shape:", stress_vectors_ns.shape)

        ################################################################################################
        # 4) Single-frame energy-only test (for scalar output verification)
        ################################################################################################
        print("\n>>> Single-frame energy-only test...")

        single_coords = positions_init + 0.1 * rng.uniform(size=(num_atoms, dim))
        E_ns, F_ns, S_ns = mace_inference(
            atoms=atoms,
            atomcoords=single_coords,
            compute_force=False,
        )

        print(f"Energy (custom, no-stress): {E_ns:.8f}")
        print(f"Forces shape (zeros expected): {F_ns.shape}")
        print(f"Stress vector (zeros expected): {S_ns}")

        ################################################################################################
        # 5) Evaluation without forces & stress
        ################################################################################################
        print("\n>>> Timing 100 evaluations (energy only)...")

        rng = np.random.default_rng(seed=0)
        t0 = time.perf_counter()
        for ii in range(100):
            atomcoords_batch = positions_init + 0.1 * rng.uniform(size=(1, num_atoms, dim))
            energies_ns, forces_ns, stress_vectors_ns = mace_inference(
                atoms=atoms,
                atomcoords=atomcoords_batch,
                compute_force=False,
            )
            print(f"Energy (custom, no-stress): {ii:4d}  {energies_ns[0]:.8f}")
        t1 = time.perf_counter()
        print(f"100 energy-only evaluation done in {t1 - t0:.3f} s")
    

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

#### test on 4090
singularity exec --nv --no-home \
    --bind /home/zq/zqcodeml:/home/zq/zqcodeml \
    /home/zq/zqdata/images/cuda12.8-jax0503-torch280-mace314.sif bash -c \
"
source /jaxtorchmace/bin/activate
cd /home/zq/zqcodeml/watericeIh-mc-master/source
python3 potentialmace_oeq.py
"

#### test on t02
srun -p home --cpus-per-task=16 --mem=32GB --gres=gpu:NV5090:1 -t 0-01:00 --pty /bin/bash 
singularity exec --no-home --nv --bind /home/user_zhangqi/private/homefile/t02codeml:/t02codeml \
    /home/user_zhangqi/private/homefile/images/cuda12.8-jax0503-torch280-mace314.sif bash -c \
"
source /jaxtorchmace/bin/activate
cd /t02codeml/watericeIh-mc-master/source
python3 potentialmace_oeq.py
"

"""
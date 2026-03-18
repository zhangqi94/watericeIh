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
    mace_batch_size: int,
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

    # if mace_args.device == "cuda":
    #     model = run_e3nn_to_cueq(deepcopy(model), device=device)

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
        atomcoords: np.ndarray,
        compute_force: bool = True,
        create_neighborlist_device: str = "gpu",
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
    mace_model_path = "potential/macemodel251125/mace_iceIh_128x0e128x1o_r4.5_float32_seed144_cueq.model"
    
    mace_dtype = "float32"
    mace_batch_size = 1
    mace_device = "cuda"
    # mace_device = "cpu"

    # --- Load structure ---
    stru_file = "structure/initstru/sc_322_n_96.vasp"
    
    atoms = ase.io.read(stru_file)
    positions_init = atoms.get_positions()
    num_atoms, dim = positions_init.shape
    rng = np.random.default_rng(seed=0)


    # ------------------------------------------------------------
    # GPU neighbourlist timing
    # ------------------------------------------------------------
    if 1:
        mace_inference = initialize_mace_model(
            mace_model_path=mace_model_path,
            mace_batch_size=mace_batch_size,
            mace_dtype=mace_dtype,
            mace_device=mace_device,
        )
        
        ## warmup
        E_custom, F_custom, S_custom = mace_inference(
            atoms=atoms,
            atomcoords=positions_init,     # 但用新的坐标
            compute_force=True,
        )
        
        # Number of evaluations (customizable)
        num_iter = 100   # <<< change this to any number you want
        compute_force = False
        
        
        # ============================================================
        # GPU neighbour list timing
        # ============================================================
        print("\n==================== GPU neighbour list ====================\n")

        t0 = time.perf_counter()
        rng = np.random.default_rng(seed=42)

        for i in range(num_iter):
            atomcoords = positions_init + 0.10 * rng.uniform(size=(num_atoms, dim))
            E1, _, _ = mace_inference(
                atoms,
                atomcoords,
                compute_force=compute_force,
                create_neighborlist_device="gpu",
            )

        t1 = time.perf_counter()

        gpu_time = t1 - t0
        gpu_avg = gpu_time / num_iter

        label_width = 45
        print(f"{'[GPU] Total time for ' + str(num_iter) + ' evaluations :':<{label_width}} {gpu_time:.6f} s")
        print(f"{'[GPU] Average time per evaluation :' :<{label_width}} {gpu_avg:.6f} s")
        print(f"{'E (last evaluation) :' :<{label_width}} {E1}\n")


        # ============================================================
        # CPU neighbour list timing
        # ============================================================
        print("\n==================== CPU neighbour list ====================\n")

        t0 = time.perf_counter()
        rng = np.random.default_rng(seed=42)

        for i in range(num_iter):
            atomcoords = positions_init + 0.10 * rng.uniform(size=(num_atoms, dim))
            E1, _, _ = mace_inference(
                atoms,
                atomcoords,
                compute_force=compute_force,
                create_neighborlist_device="cpu",
            )

        t1 = time.perf_counter()

        cpu_time = t1 - t0
        cpu_avg = cpu_time / num_iter
        
        label_width = 45
        print(f"{'[GPU] Total time for ' + str(num_iter) + ' evaluations :':<{label_width}} {cpu_time:.6f} s")
        print(f"{'[GPU] Average time per evaluation :' :<{label_width}} {cpu_avg:.6f} s")
        print(f"{'E (last evaluation) :' :<{label_width}} {E1}\n")


                
                













    ## test for mace_inference and ase atoms speed
    if 0:
        
        mace_inference = initialize_mace_model(
            mace_model_path=mace_model_path,
            mace_batch_size=mace_batch_size,
            mace_dtype=mace_dtype,
            mace_device=mace_device,
        )
        
        mace_calc = make_mace_calculator(
            mace_model_path=mace_model_path,
            mace_dtype=mace_dtype,
            mace_device=mace_device,
            enable_cueq=True,
        )
        
        
        rng = np.random.default_rng(seed=0)
        atoms_ref = ase.io.read(stru_file)       # 原子拓扑
        positions_init = atoms_ref.get_positions()
        num_atoms, dim = positions_init.shape
        num_test = 100
        
        # ---------------------------------------------------
        # 预热阶段：ASE
        # ---------------------------------------------------
        atoms_ref.calc = mace_calc
        E0 = atoms_ref.get_potential_energy()
        print("Warmup ASE energy =", E0)

        # ---------------------------------------------------
        # 预热阶段：mace_inference
        # ---------------------------------------------------
        E_custom0, F_custom0, S_custom0 = mace_inference(
            atoms=atoms_ref, 
            atomcoords=positions_init,
            compute_force=False
        )
        print("Warmup custom energy =", E_custom0)

        # ---------------------------------------------------
        # 测速：ASE
        # ---------------------------------------------------
        rng = np.random.default_rng(seed=0)
        t0 = time.time()
        for _ in range(num_test):
            positions = positions_init + 0.1 * rng.uniform(size=(num_atoms, dim))
            atoms_ref.set_positions(positions)
            E = atoms_ref.get_potential_energy()
        t1 = time.time()
        print(E)
        print(f"ASE get_potential_energy: {(t1 - t0)/num_test:.6f} s/call")

        # ---------------------------------------------------
        # 测速：mace_inference（使用固定拓扑）
        # ---------------------------------------------------
        rng = np.random.default_rng(seed=0)
        t0 = time.time()
        for _ in range(num_test):
            positions = positions_init + 0.1 * rng.uniform(size=(num_atoms, dim))
            E_custom, F_custom, S_custom = mace_inference(
                atoms=atoms_ref,          # 用原子的拓扑
                atomcoords=positions,     # 但用新的坐标
                compute_force=False,
            )
        t1 = time.time()
        print(E_custom)
        print(f"mace_inference: {(t1 - t0)/num_test:.6f} s/call")


    ## test for mace_inference
    if 0:
        # Initialize the custom MACE inference function
        mace_inference = initialize_mace_model(
            mace_model_path=mace_model_path,
            mace_batch_size=mace_batch_size,
            mace_dtype=mace_dtype,
            mace_device=mace_device,
        )
        
        ################################################################################################
        # Generate a slightly perturbed structure for testing
        atomcoords_batch = positions_init + 0.2 * rng.uniform(size=(1, num_atoms, dim))

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
            atomcoords_batch = positions_init + 0.1 * rng.uniform(size=(num_atoms, dim))
            energies_ns, forces_ns, stress_vectors_ns = mace_inference(
                atoms=atoms,
                atomcoords=atomcoords_batch,
                compute_force=False,
            )
            print(f"Energy (custom, no-stress): {ii:4d}  {energies_ns:.8f}")
        t1 = time.perf_counter()
        print(f"100 energy-only evaluation done in {t1 - t0:.3f} s")


    ## test for ase_calc
    if 0:
        # # Also initialize an ASE-compatible calculator for verification
        mace_calc1 = make_mace_calculator(
            mace_model_path=mace_model_path,
            mace_dtype=mace_dtype,
            mace_device=mace_device,
            enable_cueq=False,
        )
        
        atoms1 = ase.io.read(stru_file)
        atoms1.calc = mace_calc1  # attach calculator for ASE evaluation

        mace_calc2 = make_mace_calculator(
            mace_model_path=mace_model_path,
            mace_dtype=mace_dtype,
            mace_device=mace_device,
            enable_cueq=False,
        )
        
        atoms2 = ase.io.read(stru_file)
        atoms2.calc = mace_calc2  # attach calculator for ASE evaluation

        # ---------- Test two-frame consistency ----------
        E1 = atoms1.get_potential_energy()
        F1 = atoms1.get_forces()
        S1 = atoms1.get_stress(voigt=False)  # full 3x3 tensor
        
        E2 = atoms2.get_potential_energy()
        F2 = atoms2.get_forces()
        S2 = atoms2.get_stress(voigt=False)  # full 3x3 tensor
        
        print("max |ΔF|:", np.max(np.abs(F1 - F2)))
        print("max |ΔS|:", np.max(np.abs(S1 - S2)))
        print("ΔE:", abs(E1 - E2))


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
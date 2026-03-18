import os
import sys
import time
from pathlib import Path
import runtools

####################################################################################################
# ======== USER PARAMETERS ========

# Slurm resources
gpu = "NV5090:1"
cpus_per_task = 8
mem_gb = 32

# Run IDs inside one job (e.g., run_5, run_6, ...)
run_ids = [f"{i:04d}" for i in range(1, 9)]
temperatures = [20, 50, 55, 60, 65, 70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5, 90, 92.5, 95, 97.5, 100, 105, 110, 120, 130, 140, 150, 200]

init_stru_path = "source/structure/initstru/sc_533_n_360.json"
mala_width_ref = 0.0075
cell_width_ref = 0.0009
num_blocks = 42000
num_loop_steps = 100
num_cont_steps = 30
p_mala = 0.9
thermal_loop = 800
thermal_cont = 100

#######################

# Output root on t02
out_root = "/t02codeml/watericeIh_data/mcmix_t02_batch_02/"

# -------------------------------
# Monte Carlo parameters (Loop + MALA)
# -------------------------------
# MC parameters (Loop + MALA + CELL)
thermal_loop_force_flip = 0

# Print frequency inside main_mcmix.py
print_interval_loop = 1000
print_interval_mala = 1000

update_mala_mode = "all"
cell_mode = "anisotropic"
pressure_GPa = 0.0
create_neighborlist_device = "gpu"
save_xyz = False

# MACE settings
mace_model_path = "source/potential/macemodel260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_cueq.model"
mace_device = "cuda"
mace_dtype = "float32"

# Project root directory inside the container
project_root = "/t02codeml/watericeIh-mc-master"

# Naming controls
append_timestamp = False
job_prefix = "job_mix"

####################################################################################################
# ======== AUTO-GENERATED NAME TAGS ========
structure_tag = Path(init_stru_path).stem
timestamp = time.strftime("%Y%m%d-%H%M%S") if append_timestamp else ""
p_mala_tag = f"{p_mala:.3f}"

####################################################################################################
# ======== LOOP OVER TEMPERATURES ========
temperatures = [f"{t:.2f}" for t in temperatures]
for T in temperatures:

    # Construct job name
    runs_compact = "-".join(str(r) for r in run_ids)
    base_job = f"{job_prefix}_{structure_tag}_T_{T}_mc_{num_loop_steps}_{num_cont_steps}_pmala_{p_mala_tag}"
    job_name = f"{base_job}_[{runs_compact}]"
    job_file_name = f"{job_name}.slurm"

    # Output directory (inside container)
    out_dir = f"{out_root}/{structure_tag}"
    log_dir = f"{out_dir}"
    base_name = f"{structure_tag}_T_{T}_mc_{num_loop_steps}_{num_cont_steps}_pmala_{p_mala_tag}"

    ################################################################################################
    # ======== PYTHON COMMAND TO EXECUTE INSIDE SINGULARITY ========
    py_lines = [
        f"cd {project_root}",
        f"mkdir -p {out_dir}",
        "echo MC_runs_started",
    ]
        
    # ---- Launch multiple independent runs inside a single job ----
    for rid in run_ids:
        run_tag = f"run_{rid}"

        out_file = f"{base_name}_{run_tag}.json"
        log_file = f"{log_dir}/{base_name}_{run_tag}.log"
        save_file_path = f"{out_dir}/{out_file}"

        py_lines += [
            f"echo Starting_{run_tag}_at_T_{T}K",
            "python3 main_mcmix.py \\",
            f"    --init_stru_path {init_stru_path} \\",
            f"    --save_file_path {save_file_path} \\",
            f"    --save_xyz {save_xyz} \\",
            f"    --mace_model_path {mace_model_path} \\",
            f"    --mace_device {mace_device} \\",
            f"    --mace_dtype {mace_dtype} \\",
            f"    --target_temperature_K {T} \\",
            f"    --num_blocks {num_blocks} \\",
            f"    --num_loop_steps {num_loop_steps} \\",
            f"    --num_cont_steps {num_cont_steps} \\",
            f"    --mala_width_ref {mala_width_ref} \\",
            f"    --thermal_loop_force_flip {thermal_loop_force_flip} \\",
            f"    --thermal_loop {thermal_loop} \\",
            f"    --thermal_cont {thermal_cont} \\",
            f"    --print_interval_loop {print_interval_loop} \\",
            f"    --print_interval_mala {print_interval_mala} \\",
            f"    --update_mala_mode {update_mala_mode} \\",
            f"    --p_mala {p_mala} \\",
            f"    --cell_mode {cell_mode} \\",
            f"    --cell_width_ref {cell_width_ref} \\",
            f"    --pressure_GPa {pressure_GPa} \\",
            f"    --create_neighborlist_device {create_neighborlist_device} \\",
            f"    > {log_file} 2>&1 &",
            "",
        ]

    # Wait for all background jobs (MC + monitor) to finish
    py_lines += [
        "",
        "echo Waiting_for_all_background_jobs_to_finish",
        "wait",
        "echo All_background_jobs_finished",
        "",
    ]

    pyscript = "\n".join(py_lines) + "\n"

    ################################################################################################
    # ======== SLURM JOB HEADER FOR t02 ========
    gpuscript = f"""#SBATCH --partition=home
#SBATCH --exclude=n004
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{gpu}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem_gb}G
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}-%j.out
"""

    ################################################################################################
    # ======== WRITE + SUBMIT JOB (SINGULARITY) ========
    slurm_script = runtools.generate_slurm_script_singularity_withnv(
        gpuscript=gpuscript,
        pyscript=pyscript,
    )

    runtools.write_slurm_script_to_file(slurm_script, job_file_name)
    runtools.submit_slurm_script(job_file_name)

    print(f"[INFO] Submitted T={T} K with runs={run_ids}")
    print(f"[INFO] Job name:   {job_name}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Script:     {Path(job_file_name).resolve()}")
    print("-" * 80)
    time.sleep(0.2)

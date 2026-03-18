import subprocess

# Define the function for submitting a job
def submit_slurm_job(gpu, 
                     x, 
                     r,
                     dtype,
                     seed,
                     train_file, 
                     test_file):
    # Create the Slurm script content
    slurm_script = f"""#!/bin/bash
#SBATCH --partition=home
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{gpu}
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --job-name=mace_iceIh_{x}x0e{x}x1o_r{r}_{dtype}_seed{seed}
#SBATCH --output=job_mace_iceIh_{x}x0e{x}x1o_r{r}_{dtype}_seed{seed}-%j.out

set -euo pipefail

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo "The current job ID is $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST"
echo "Using $SLURM_NTASKS_PER_NODE tasks per node"
echo "A total of $SLURM_NTASKS tasks is used"
echo "List of CUDA devices: $CUDA_VISIBLE_DEVICES"
echo

echo "==== Job started at $(date) ===="
echo

module purge

singularity exec --no-home --nv --bind /home/user_zhangqi/private/homefile/t02codeml:/t02codeml \
    /home/user_zhangqi/private/homefile/images/cuda12.8-jax0503-torch280-mace314.sif bash -c \
"
source /jaxtorchmace/bin/activate

# Environment information
nvcc --version
which python3
python3 --version
pip show torch
nvidia-smi

cd /t02codeml/mace-train-iceIh/

mace_run_train \\
    --name="mace_iceIh_{x}x0e{x}x1o_r{r}_{dtype}_seed{seed}" \\
    --train_file="{train_file}" \\
    --valid_fraction=0.10 \\
    --test_file="{test_file}" \\
    --config_type_weights='{{"Default":1.0}}' \\
    --E0s="average" \\
    --model="MACE" \\
    --hidden_irreps='{x}x0e + {x}x1o' \\
    --correlation=3 \\
    --r_max={r} \\
    --forces_weight=100 \\
    --energy_weight=10 \\
    --energy_key="TotEnergy" \\
    --forces_key="force" \\
    --eval_interval=1 \\
    --max_num_epochs=400 \\
    --scheduler_patience=15 \\
    --patience=30 \\
    --ema \\
    --seed={seed} \\
    --restart_latest \\
    --default_dtype="{dtype}" \\
    --device=cuda \\
    --batch_size=8 \\
    --enable_cueq=True

"

echo
echo "==== Job finished at $(date) ===="
"""

    # Build the script filename
    script_filename = f"job_mace_iceIh_{x}x0e{x}x1o_r{r}_{dtype}_seed{seed}.sh"

    # Write the script to a file
    with open(script_filename, 'w') as file:
        file.write(slurm_script)

    # Submit the Slurm job
    subprocess.run(['sbatch', script_filename], check=True)

####################################################################################################
####################################################################################################

if __name__ == "__main__":
    
    train_file = "data_xyz_260201/watericeIh_train.xyz"
    test_file  = "data_xyz_260201/watericeIh_test.xyz"

    #####################################################################################################
    gpu = "NV5090:1"
    x = "128"
    r = "4.5"
    dtype = "float32"
    seed = "146"
    submit_slurm_job(gpu, x, r, dtype, seed, train_file, test_file)

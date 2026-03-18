import os
import sys
import subprocess
import shlex

####################################################################################################
# -------- helper for shell-quoting --------
def q(x):
    return shlex.quote(str(x))

def write_slurm_script_to_file(script_content, file_name):
    with open(file_name, 'w') as file:
        file.write(script_content)

def submit_slurm_script(file_name):
    result = subprocess.run(['sbatch', file_name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Job submitted successfully: {result.stdout}")
    else:
        print(f"Error in job submission: {result.stderr}")

####################################################################################################
####################################################################################################
#========== singularity ==========
def generate_slurm_script_singularity_withnv(gpuscript, pyscript):
    slurm_script = f"""#!/bin/bash
{gpuscript}

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo


# ================= MONITOR =================
echo "[MONITOR] Started"

(
  while true; do
      echo "===== TIME: $(date '+%Y-%m-%d %H:%M:%S') ====="
      echo "----- GPU -----"
      nvidia-smi
      echo "----- CPU -----"
      top -b -n 1 | head -n 20
      echo "----------------"
      sleep 600
  done
) &

MONITOR_PID=$!
echo "[MONITOR] PID = $MONITOR_PID"
# ============================================


echo ==== Job started at `date` ====
echo

module purge
apptainer exec --no-home --nv --bind /home/user_zhangqi/private/homefile/t02codeml:/t02codeml \\
    /home/user_zhangqi/private/homefile/images/cuda12.8-jax0503-torch280-mace314.sif bash -c \\
"
source /jaxtorchmace/bin/activate

nvcc --version
which python3
python3 --version
pip show torch

{pyscript}
"

# ======== STOP MONITOR ========
echo "[MONITOR] Killing PID $MONITOR_PID"
kill $MONITOR_PID 2>/dev/null
# ==============================

echo
echo ==== Job finished at `date` ====
"""
    return slurm_script

####################################################################################################
####################################################################################################
# #========== singularity ==========
def generate_slurm_script_singularity(gpuscript, pyscript):
    slurm_script = f"""#!/bin/bash
{gpuscript}

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

echo ==== Job started at `date` ====
echo

module purge
singularity exec --no-home --nv --bind /home/user_zhangqi/private/homefile/t02codeml:/t02codeml \\
    /home/user_zhangqi/private/homefile/images/cuda12.8-jax0503-torch280-mace314.sif bash -c \\
"
source /jaxtorchmace/bin/activate

nvcc --version
which python3
python3 --version
pip show torch
nvidia-smi

{pyscript}
"

echo
echo ==== Job finished at `date` ====
"""
    return slurm_script
import subprocess

# 定义提交任务的函数
def submit_slurm_job(structure_name, start, end):
    # 创建 Slurm 脚本内容
    slurm_script = f"""#!/bin/bash
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBATCH --job-name=job_{structure_name}_[{start}_{end}].sh
#SBATCH --output=job_{structure_name}_[{start}_{end}]-%j.out

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

export FI_PROVIDER="tcp"

echo The current job ID is $SLURM_JOB_ID
echo Running on $SLURM_JOB_NUM_NODES nodes: $SLURM_JOB_NODELIST
echo Using $SLURM_NTASKS_PER_NODE tasks per node
echo A total of $SLURM_NTASKS tasks is used
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

echo ==== Job started at `date` ====
echo

ulimit -s unlimited
ulimit -c unlimited
module load pmix/2.2.2
module load parallel_studio/2020.2.254
module load intelmpi/2020.2.254
module load vasp6/6.1

for ii in {{{start}..{end}}}; do
    folder="/home/users/zhangqi/zqcode/watericeIh_vasp/datavasp/{structure_name}/$(printf '%06d' $ii)"
    echo "=========================================================================================="
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"
        cd "$folder"
        srun --mpi=pmi2 vasp_std
        echo "Finished processing folder: $folder"
        echo
    else
        echo "Folder $folder does not exist. Skipping..."
    fi
done

echo
echo ==== Job finished at `date` ====
"""

    # 构建脚本文件名称
    script_filename = f"job_{structure_name}_[{start}_{end}].sh"

    # 将脚本写入文件
    with open(script_filename, 'w') as file:
        file.write(slurm_script)

    # 提交 Slurm 作业
    subprocess.run(['sbatch', script_filename])

####################################################################################################
# 示例：提交一个任务，结构名称为 ice08_n222
if __name__ == "__main__":
    # submit_slurm_job('mc_ff_n032',    1, 3000)
    # submit_slurm_job('mc_ff_n064',    1, 3000)
    # submit_slurm_job('mc_ff_n096',    1, 2000)
    # submit_slurm_job('mc_ff_n096', 2001, 4000)
    # submit_slurm_job('mc_ff_n096', 4001, 6000)
    # submit_slurm_job('mc_ff_n128',    1, 2000)
    # submit_slurm_job('mc_ff_n128', 2001, 4000)
    # submit_slurm_job('mc_ff_n128', 4001, 6000)

    # submit_slurm_job('mc_n016', 1, 6000)
    # submit_slurm_job('mc_n032', 1, 6000)
    # submit_slurm_job('mc_n064', 1, 2000)
    # submit_slurm_job('mc_n064', 2001, 4000)
    # submit_slurm_job('mc_n096', 1, 2000)
    # submit_slurm_job('mc_n096', 2001, 4000)
    # submit_slurm_job('mc_n096', 4001, 6000)
    # submit_slurm_job('mc_n096', 6001, 8000)
    
    # submit_slurm_job('mc_n128', 1, 2000)
    # submit_slurm_job('mc_n128', 2001, 4000)

    # submit_slurm_job('mc_ff_n128', 6001, 8000)
    
"""
cd /home/users/zhangqi/zqcode/watericeIh_vasp
python3 submitjob.py
"""

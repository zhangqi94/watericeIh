# Ab initio simulation of the first-order proton-ordering transition in water ice

This repository accompanies the paper [*Ab initio simulation of the first-order proton-ordering transition in water ice*](https://arxiv.org/abs/2603.09247). It contains the code for ab initio-accurate equilibrium sampling of proton ordering in water ice, combining a MACE potential trained on DFT data with loop updates that preserve the ice rules and continuous atomic-coordinate Monte Carlo updates. The repository includes the full workflow: structure preparation, DFT data extraction, MACE training and validation, finite-temperature Monte Carlo runs, and post-processing.

## What is in this repository

The project is organized around the main working directories that correspond to the research workflow:

- `source/`: core physics and simulation code
- `run/`: Slurm job builders for batch Monte Carlo runs
- `analysis/`: log merging and post-processing utilities
- `dft/`: DFT data preparation, MACE training, and RMSE evaluation
- `results/`: example trained models and processed outputs

In addition, the repository root contains the main executable scripts:

- `main_mcmix.py`: finite-temperature equilibrium sampling with loop and continuous updates

## Environment

The codebase was developed with Python packages listed in `requirements.txt`. Core packages used throughout the workflow include:

- `numpy==1.26.4`
- `ase==3.26.0`
- `torch==2.8.0+cu128`
- `mace-torch==0.3.14` ([MACE GitHub](https://github.com/acesuit/mace))
- `e3nn==0.4.4`
- `cuequivariance==0.6.1`
- `matscipy==1.1.1`
- `scipy==1.16.2`

For the full software stack, see `requirements.txt`.

## Main workflow

### 1. Initial structures and simulation core

The `source/` directory contains the reusable simulation logic:

- `updateloop.py`: proton topology updates with short-loop moves
- `updatemala.py`: MALA updates for atomic coordinates
- `updatecell.py`: cell updates for NPT-style sampling
- `updateblock.py`: block-level orchestration of loop, MALA, and cell moves
- `ckpt.py`: JSON / log / XYZ checkpoint handling
- `crystaltools.py`, `tools.py`, `units.py`, `rotation.py`: utility modules
- `potentialmace_cueq.py`: MACE inference wrapper
- `potential/potential_neighborlist.py`: neighbor-list support
- `structure/initstru/`: prepared initial ice-Ih supercells such as `sc_222_n_64.json`, `sc_422_n_128.json`, `sc_533_n_360.json`


The main production entry point is `main_mcmix.py`, which combines:

- loop updates for hydrogen-bond topology
- MALA coordinate updates
- cell length updates

`main_mcloop.py` is the reduced discrete-only version when only proton topology sampling is needed.

### 2. DFT data preparation and MACE training

The `dft/` directory contains the VASP input files and MACE training scripts used in this project:

- `vasp_inputs/R2SCAN/`: VASP input templates such as `INCAR` and `POTCAR`
- `mace-train-script/submitjob_mace314.py`: Slurm script generator for MACE training
- `rmse_calculate_macepotential.py` and `rmse_plot_macevalues.py`: scripts for MACE validation against DFT reference data

The full DFT training dataset is too large to be included in this repository.

### 3. Monte Carlo production runs

The `run/` directory contains cluster-specific batch launchers:

- `submitjob_mcmix_t02_nv_n128.py`
- `submitjob_mcmix_t02_nv_n360.py`
- `runtools.py`

These scripts generate Slurm jobs that:

- launch multiple independent runs inside one job
- sweep over a temperature list
- run `main_mcmix.py` inside an Apptainer/Singularity container
- save per-run `.json` outputs and `.log` files

### 4. Merging and analysis of Monte Carlo logs

The `analysis/` directory contains post-processing helpers:

- `anatools.py`: shared analysis routines for reading logs, trimming burn-in, computing observables, and rebinning
- `anacolors.py`: plotting color definitions
- `save_mc_merged_data.py`: merges multi-run outputs and exports summary CSV tables

The current analysis script is configured to:

- merge raw multi-run Monte Carlo outputs into a common directory
- discard an initial burn-in segment
- group runs by system size and temperature
- compute summary statistics
- export a combined CSV file

### 5. Example outputs

The `results/` directory stores representative outputs that document the workflow:

- `results/mace_models/`: trained MACE models, training logs, and generated job scripts
- `results/energy_overlap/`: JSON data used for energy-overlap checks between states / temperatures
- `results/final_results/`: merged analysis table

These files are useful as references for expected naming conventions and output formats.

## Typical usage

### Run equilibrium sampling with loop and continuous updates

Example:

```bash
python main_mcmix.py \
  --init_stru_path source/structure/initstru/sc_422_n_128.json \
  --save_file_path /tmp/sc_422_n_128_T_100 \
  --save_xyz False \
  --mace_model_path source/potential/macemodel260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_cueq.model \
  --mace_device cuda \
  --mace_dtype float32 \
  --target_temperature_K 100 \
  --num_blocks 42000 \
  --num_loop_steps 20 \
  --num_cont_steps 25 \
  --mala_width_ref 0.01 \
  --thermal_loop_force_flip 0 \
  --thermal_loop 300 \
  --thermal_cont 100 \
  --print_interval_loop 1000 \
  --print_interval_mala 1000 \
  --update_mala_mode all \
  --p_mala 0.9 \
  --cell_mode anisotropic \
  --cell_width_ref 0.001 \
  --pressure_GPa 0.0 \
  --create_neighborlist_device gpu
```

Outputs are written using the `save_file_path` prefix and typically include:

- a text log for energy and other observables
- a snapshot JSON file
- an optional multi-frame XYZ trajectory, which can be very large for long runs !!!

### Run loop-only Monte Carlo

Example:

```bash
python main_mcloop.py \
  --init_stru_path source/structure/initstru/sc_222_n_64.json \
  --save_file_path /tmp/sc_222_n_64_T_100_loop \
  --mace_model_path source/potential/macemodel260201/mace_iceIh_128x0e128x1o_r4.5_float32_seed146_cueq.model \
  --target_temperature_K 100 \
  --num_blocks 1000 \
  --num_loop_steps 20
```

### Submit production jobs on Slurm

The `run/submitjob_mcmix_t02_nv_n128.py` and `run/submitjob_mcmix_t02_nv_n360.py` scripts are the practical production wrappers. They define:

- system size
- temperature grid
- Monte Carlo lengths
- output directory
- GPU / CPU / memory allocation

Then they generate and submit a Slurm script automatically.

## Notes

- Several scripts contain hard-coded absolute paths for the original cluster environment. Update these paths before running on a new machine.
- The Slurm submission helpers are environment-specific and assume a working Apptainer/Singularity installation.
- The bundled models and example outputs in `results/` and `source/potential/` are useful references for reproducing naming conventions and directory layouts.

## Summary of the repository roles

- Use `source/` for the simulation engine and reusable physics code.
- Use `run/` to launch large temperature sweeps on a cluster.
- Use `analysis/` to merge logs and generate final tables.
- Use `dft/` to prepare DFT data and train or validate the MACE potential.
- Use `results/` as a reference snapshot of trained models and processed outputs.

## Reference

If you use this repository, please cite:

```bibtex
@misc{zhang2026abinitiosimulationfirstorder,
      title={Ab initio simulation of the first-order proton-ordering transition in water ice},
      author={Qi Zhang and Sicong Wan and Lei Wang},
      year={2026},
      eprint={2603.09247},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2603.09247},
}
```

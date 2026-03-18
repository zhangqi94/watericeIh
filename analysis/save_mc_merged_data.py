#!/usr/bin/env python3
from __future__ import annotations

####################################################################################################
# - Load required modules
####################################################################################################
from pathlib import Path
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anatools import analyze_log_summary_multirun_rebin

####################################################################################################
# - Merge settings
####################################################################################################
RAW_ROOT = Path("/home/zq/zqcodeml/watericeIh_data/raw_mcmix")
MERGED_ROOT = Path("/home/zq/zqcodeml/watericeIh_data/merged_mcmix")

####################################################################################################
# - Analysis settings
####################################################################################################
output_csv_name = "/home/zq/zqcodeml/watericeIh_data/df_all_mix/df_all_mix_20260202.csv"

# Burn-in setting
drop = 2000

# Parallel settings
use_parallel = True
# max_workers = None  # None -> use os.cpu_count()
max_workers = 32

# Systems
systems = [
    "sc_222_n_64",
    "sc_322_n_96",
    "sc_422_n_128",
    "sc_533_n_360",
]

# Root directory for analysis
base_root = MERGED_ROOT

# Color map (optional)
cmap = plt.get_cmap("tab10")
system_colors = {sys: cmap(i % cmap.N) for i, sys in enumerate(systems)}

# Default for interactive cells / imports
df_all = None

####################################################################################################
# - Merge helpers
####################################################################################################
def _collect_depth2_dirnames(root: Path) -> list[str]:
    names = set()
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) == 2:
            names.add(path.name)
    return sorted(names)


def _find_dirs_named(root: Path, name: str) -> list[Path]:
    return [p for p in root.rglob(name) if p.is_dir() and p.name == name]


def merge_mcmix() -> None:
    if MERGED_ROOT.exists():
        shutil.rmtree(MERGED_ROOT)
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)

    if not RAW_ROOT.is_dir():
        raise RuntimeError(f"Raw root not found: {RAW_ROOT}")

    dirnames = _collect_depth2_dirnames(RAW_ROOT)
    for name in dirnames:
        target = MERGED_ROOT / name
        print(f">>> Merging {name} -> {target}")
        target.mkdir(parents=True, exist_ok=True)

        for src_dir in _find_dirs_named(RAW_ROOT, name):
            print(f"  from {src_dir}")
            for f in src_dir.iterdir():
                if f.is_file() and f.suffix == ".txt":
                    dst = target / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
        print(">>> Done", name)
        print()


####################################################################################################
# - Analysis helpers
####################################################################################################
def _analyze_one_temperature(args):
    prefix, T, paths, drop_val = args

    try:
        res = analyze_log_summary_multirun_rebin(
            paths,
            drop=drop_val,
            verbose=False,
            use_energy_outlier_detection=False,
            block_size=2048,
            # min_blocks=8,
        )
    except RuntimeError as e:
        # Skip groups with no valid samples after trimming
        if "All logs are empty after trimming" in str(e):
            print(f"[WARN] {prefix}, T={T}: no valid samples after trimming, skipped")
            return None
        raise

    res["t_k"] = T
    res["system"] = prefix
    res["num_runs"] = len(paths)
    return res


def analyze_merged_data() -> pd.DataFrame:
    df_all_list = []

    ################################################################################################
    # - Loop for each system
    ################################################################################################
    fname_re = re.compile(r"_T_([0-9]+(?:\.[0-9]+)?)_mc_")

    for prefix in systems:
        base_dir = base_root / prefix
        if not base_dir.is_dir():
            print(f"[WARN] Directory not found for {prefix}: {base_dir}")
            continue

        # Collect files
        files = sorted(base_dir.glob(f"*_T_*_mc_*.txt"))
        if not files:
            print(f"[WARN] No log files found for {prefix}")
            continue

        # Group by temperature
        files_by_T = {}
        for f in files:
            m = fname_re.search(f.name)
            if not m:
                print(f"[WARN] Skip unmatched filename: {f.name}")
                continue
            T = float(m.group(1))
            files_by_T.setdefault(T, []).append(f)

        results = []
        items = sorted(files_by_T.items())
        for T, paths in items:
            print(f"[INFO] {prefix}, T={T}: {len(paths)} files")

        # Analyze each temperature separately
        if use_parallel and items:
            args = [(prefix, T, paths, drop) for T, paths in items]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_analyze_one_temperature, a) for a in args]
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None:
                        results.append(res)

        if not results:
            print(f"[WARN] No valid results for {prefix}")
            continue

        df_sys = pd.DataFrame(results).sort_values("t_k").reset_index(drop=True)
        df_all_list.append(df_sys)

    ################################################################################################
    # - Merge and save
    ################################################################################################
    if df_all_list:
        df_all_local = pd.concat(df_all_list, ignore_index=True)
    else:
        raise RuntimeError("No valid data found for any system.")

    print("\n========== Summary ==========")
    print(df_all_local)

    df_all_local.to_csv(output_csv_name, index=False)
    print(f"\nSaved to {output_csv_name}\n")
    return df_all_local


if __name__ == "__main__":
    merge_mcmix()
    df_all = analyze_merged_data()

####################################################################################################
# Notes
####################################################################################################
"""
conda activate jax0503-torch280-mace314
cd /home/zq/zqcodeml/watericeIh-mc-master/analysis/
python3 save_mc_merged_data.py
"""

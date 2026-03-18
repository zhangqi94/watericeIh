import torch
import mace
from mace.calculators import MACECalculator
from mace import data
from mace.tools import torch_geometric, torch_tools, utils
# from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from copy import deepcopy

import ase
import argparse
import numpy as np
from typing import Callable, Any, Optional, Tuple, Dict, List

####################################################################################################
####################################################################################################
def convert_mace_model_e3nn_to_cueq(
    mace_model_path: str,
    mace_dtype: str = "float32",
    mace_device: str = "cuda",
    save_converted: bool = True,
) -> torch.nn.Module:
    """
    Convert a pretrained MACE model from E3NN to CUEQ format and optionally save it.

    Args:
        mace_model_path (str): Path to the original MACE .model file.
        mace_dtype (str): Default floating-point precision ("float32" or "float64").
        mace_device (str): Device to load and convert model on ("cuda" or "cpu").
        save_converted (bool): Whether to save the converted model.

    Returns:
        torch.nn.Module: The converted MACE model ready for inference.
    """
    # ----------------------------------------------------------------------------------------------
    # Initialize PyTorch environment
    mace_args = argparse.Namespace(default_type=mace_dtype,
                                   model=mace_model_path,
                                   device=mace_device)

    torch_tools.set_default_dtype(mace_args.default_type)
    device = torch_tools.init_device(mace_args.device)

    # ----------------------------------------------------------------------------------------------
    # Load pretrained model
    print(f"[INFO] Loading MACE model from: {mace_args.model}")
    model = torch.load(f=mace_args.model, map_location=mace_args.device)

    # ----------------------------------------------------------------------------------------------
    # Perform E3NN → CUEQ/OEQ conversion
    if mace_args.device == "cuda":
        print("[INFO] Converting model from E3NN to CUEQ format for CUDA device...")
        from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
        model = run_e3nn_to_cueq(deepcopy(model), device=device)
        conversion_tag = "cueq"
        print("[INFO] Conversion to CUEQ complete.")

    elif mace_args.device == "dcu":
        print("[INFO] Converting model from E3NN to OEQ format for DCU device...")
        from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq
        model = run_e3nn_to_oeq(deepcopy(model), device=device)
        conversion_tag = "oeq"
        print("[INFO] Conversion to OEQ complete.")
    
    else:
        print("[WARN] Device is CPU; conversion may not be necessary or supported.")

    # ----------------------------------------------------------------------------------------------
    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(mace_args.device)

    # ----------------------------------------------------------------------------------------------
    # Save converted model
    if save_converted and conversion_tag is not None:
        converted_model_path = mace_args.model.replace(".model", f"_{conversion_tag}.model")
        torch.save(model, converted_model_path)
        print(f"[INFO] Converted model saved to: {converted_model_path}")
    elif save_converted:
        print("[WARN] No conversion performed — skipping model save.")

    return model

    
####################################################################################################
if __name__ == "__main__":
    import time

    print("mace version:", mace.__version__)

    ################################################################################################
    # --- Model setup ---
    # mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel260201/modles/mace_iceIh_128x0e128x1o_r4.5_float32_seed146.model"
    # mace_dtype = "float32"
    # mace_device = "cuda"
    
    mace_model_path = "/home/zq/zqcodeml/watericeIh-mc-master/source/potential/macemodel260201/modles/mace_iceIh_128x0e128x1o_r4.5_float64_seed142.model"
    mace_dtype = "float64"
    mace_device = "cuda"

    model = convert_mace_model_e3nn_to_cueq(
        mace_model_path=mace_model_path,
        mace_dtype=mace_dtype,
        mace_device=mace_device,
        save_converted=True,
    )

    print("[INFO] Model conversion finished successfully.")
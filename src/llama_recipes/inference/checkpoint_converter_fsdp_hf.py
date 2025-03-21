# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import os
import sys

import fire
import yaml

from llama_recipes.inference.model_utils import load_llama_from_config

from transformers import AutoConfig, AutoTokenizer, MllamaProcessor

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)
from model_checkpointing import load_sharded_model_single_gpu


def main(
    fsdp_checkpoint_path="",  # Path to FSDP Sharded model checkpoints
    consolidated_model_path="",  # Path to save the HF converted model checkpoints
    HF_model_path_or_name="",  # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
):

    try:
        file_name = "train_params.yaml"
        # Combine the directory and file name to create the full path
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        # Open the file
        with open(train_params_path, "r") as file:
            # Load the YAML data
            data = yaml.safe_load(file)

            # Access the 'model_name' field
            HF_model_path_or_name = data.get("model_name")

            print(f"Model name: {HF_model_path_or_name}")
    except FileNotFoundError:
        print(f"The file {train_params_path} does not exist.")
        HF_model_path_or_name = input("Please enter the model name: ")
        print(f"Model name: {HF_model_path_or_name}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # load the HF model definition from config
    model_def = load_llama_from_config(HF_model_path_or_name)
    print("model is loaded from config")
    # load the FSDP sharded checkpoints into the model
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("model is loaded from FSDP checkpoints")
    # loading the tokenizer form the  model_path
    config = AutoConfig.from_pretrained(HF_model_path_or_name)
    # save the processor and config for mllama models
    if config.model_type == "mllama":
        processor = MllamaProcessor.from_pretrained(HF_model_path_or_name)
        processor.save_pretrained(consolidated_model_path)
        print(
            f"HuggingFace mllama processor has been saved in {consolidated_model_path}"
        )
    else:
        # save the tokenizer for llama models
        tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
        tokenizer.save_pretrained(consolidated_model_path)
        print(
            f"HuggingFace llama tokenizer has been saved in {consolidated_model_path}"
        )
    # save the FSDP sharded checkpoints in HF format
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")


if __name__ == "__main__":
    fire.Fire(main)


## using GPU
import os
import sys
import fire
import yaml
import gc
import torch
from llama_recipes.inference.model_utils import load_llama_from_config
from transformers import AutoConfig, AutoTokenizer, MllamaProcessor

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_directory = os.path.dirname(current_directory)
# Append the parent directory to sys.path
sys.path.append(parent_directory)

from model_checkpointing import load_sharded_model_single_gpu

def main(
    fsdp_checkpoint_path="",  # Path to FSDP Sharded model checkpoints
    consolidated_model_path="",  # Path to save the HF converted model checkpoints
    HF_model_path_or_name="",  # Path/ name of the HF model that include config.json and tokenizer_config.json
    max_shard_size="5GB",  # Break the model into smaller shards
    gpu_id=1,  # Which GPU to use (0 or 1 for your 2 A100s)
):
    # Set the GPU to use
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        file_name = "train_params.yaml"
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        with open(train_params_path, "r") as file:
            data = yaml.safe_load(file)
            HF_model_path_or_name = data.get("model_name")
            print(f"Model name: {HF_model_path_or_name}")
    except FileNotFoundError:
        print(f"The file {train_params_path} does not exist.")
        HF_model_path_or_name = input("Please enter the model name: ")
        print(f"Model name: {HF_model_path_or_name}")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Load model definition from config
    model_def = load_llama_from_config(HF_model_path_or_name)
    print("Model is loaded from config")
    
    # Load FSDP sharded checkpoints
    print(f"Loading model from FSDP checkpoints to {device}...")
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    
    # Move model to specified GPU
    model = model.to(device)
    print(f"Model is loaded from FSDP checkpoints to {device}")
    
    # Load tokenizer
    config = AutoConfig.from_pretrained(HF_model_path_or_name)
    
    # Save processor/tokenizer
    if config.model_type == "mllama":
        processor = MllamaProcessor.from_pretrained(HF_model_path_or_name)
        processor.save_pretrained(consolidated_model_path)
        print(f"HuggingFace mllama processor has been saved in {consolidated_model_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
        tokenizer.save_pretrained(consolidated_model_path)
        print(f"HuggingFace llama tokenizer has been saved in {consolidated_model_path}")
    
    # Save model with GPU acceleration and smaller shards
    print(f"Starting to save model in {max_shard_size} shards...")
    try:
        model.save_pretrained(
            consolidated_model_path,
            max_shard_size=max_shard_size,
            safe_serialization=True  # Use safetensors format
        )
        print(f"HuggingFace model checkpoints have been saved in {consolidated_model_path}")
    except Exception as e:
        print(f"Error during model saving: {e}")
        
        # Try alternative method if first attempt fails
        print("Trying progressive saving approach...")
        try:
            # Create directory if it doesn't exist
            os.makedirs(consolidated_model_path, exist_ok=True)
            
            # Save config first
            model.config.save_pretrained(consolidated_model_path)
            
            # Get state dict while keeping on GPU
            print("Getting state dict from model...")
            state_dict = model.state_dict()
            
            # Define custom save function for large models
            from transformers.modeling_utils import (
                shard_checkpoint,
                WEIGHTS_NAME,
                SAFE_WEIGHTS_NAME,
                WEIGHTS_INDEX_NAME,
                SAFE_WEIGHTS_INDEX_NAME,
            )
            from transformers.utils import ContextManagers
            
            # Create shards
            shards, index = shard_checkpoint(
                state_dict, max_shard_size=max_shard_size, weights_name=SAFE_WEIGHTS_NAME
            )
            
            # Save each shard separately
            for shard_file, shard in shards.items():
                print(f"Saving shard {shard_file}...")
                torch.save(
                    shard,
                    os.path.join(consolidated_model_path, shard_file),
                    _use_new_zipfile_serialization=False
                )
            
            # Save the index
            if index is not None:
                print("Saving weights index...")
                with open(os.path.join(consolidated_model_path, WEIGHTS_INDEX_NAME), "w") as f:
                    import json
                    json.dump(index, f)
            
            print(f"Model successfully saved in {consolidated_model_path}")
        except Exception as e2:
            print(f"Alternative saving method failed: {e2}")
            print("Please try with smaller shard sizes or different memory settings.")

if __name__ == "__main__":
    fire.Fire(main)

'''
python src/llama_recipes/inference/checkpoint_converter_fsdp_hf.py   --fsdp_checkpoint_path "/home/a100server/shared_folder/FFT_mnt/FFT_Research/llama-recipes/fft_linkbricks_8b_train_3.12.0_20250312_v1_cpy/fft_linkbricks_8b_train_3.12.0_20250317/"   --consolidated_model_path "/home/a100server/shared_folder/FFT_data/Triton_Inference_Server/TensorRT_FFT_LB_train_3.12.0_20250317/fft_linkbricks_8b_train_3.12.0_20250317"   --gpu_id 1   --max_shard_size "5GB"

'''

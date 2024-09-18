import argparse
import os
import json
from pathlib import Path
import torch
import warnings
import time

from audio_extraction.baseline_features import extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from filter_split_data import *
from slice import *

CHECKPOINT_FILE = "dataset_creation_checkpoint.json"

def setup_device():
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        torch.backends.mps.enable = True
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU for computations")
    
    return device

def create_dataset(opt):
    print("Starting create_dataset function")
    device = setup_device()
    checkpoint = load_checkpoint()
    print(f"Loaded checkpoint: {checkpoint}")

    # split the data according to the splits files
    if not checkpoint.get("split_completed", False):
        print("Creating train / test split")
        split_data(opt.dataset_folder)
        update_checkpoint("split_completed", True)
    else:
        print("Train/test split already completed")

    # slice motions/music into sliding windows to create training dataset
    if not checkpoint.get("train_sliced", False):
        print("Slicing train data")
        slice_aistpp(f"train/motions", f"train/wavs")
        update_checkpoint("train_sliced", True)
    else:
        print("Train data already sliced")

    if not checkpoint.get("test_sliced", False):
        print("Slicing test data")
        slice_aistpp(f"test/motions", f"test/wavs")
        update_checkpoint("test_sliced", True)
    else:
        print("Test data already sliced")

    # process dataset to extract audio features
    if opt.extract_baseline:
        if not checkpoint.get("baseline_extracted", False):
            print("Extracting baseline features")
            baseline_extract("train/wavs_sliced", "train/baseline_feats")
            baseline_extract("test/wavs_sliced", "test/baseline_feats")
            update_checkpoint("baseline_extracted", True)
        else:
            print("Baseline features already extracted")

    if opt.extract_jukebox:
        if not checkpoint.get("jukebox_extracted", False):
            print("Extracting jukebox features")
            print(f"Current device before jukebox_extract: {device}")
            
            if device.type == 'mps':
                print("MPS device type detected. Enabling MPS fallback.")
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                
            # Monkey-patch torch.load to use our device and suppress the warning
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['map_location'] = device
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return original_load(*args, **kwargs)
            torch.load = patched_load
            
            try:
                start_time = time.time()
                print("Starting jukebox extraction for train data")
                jukebox_extract("train/wavs_sliced", "train/jukebox_feats", device=device)
                print("Completed jukebox extraction for train data")
                print("Starting jukebox extraction for test data")
                jukebox_extract("test/wavs_sliced", "test/jukebox_feats", device=device)
                print("Completed jukebox extraction for test data")
                end_time = time.time()
                print(f"Total time for jukebox extraction: {end_time - start_time:.2f} seconds")
            except Exception as e:
                print(f"Error during jukebox_extract: {e}")
                print(f"Current device after error: {device}")
                raise
            finally:
                # Restore original torch.load
                torch.load = original_load
            
            update_checkpoint("jukebox_extracted", True)
        else:
            print("Jukebox features already extracted")

    print("Dataset creation completed!")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

def update_checkpoint(key, value):
    checkpoint = load_checkpoint()
    checkpoint[key] = value
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)
    print(f"Updated checkpoint: {key} = {value}")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    print("Script started")
    opt = parse_opt()
    print(f"Parsed options: {opt}")
    create_dataset(opt)
    print("Script completed")

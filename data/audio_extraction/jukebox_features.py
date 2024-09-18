import os
import argparse
from functools import partial
from pathlib import Path

import jukemirlib
import numpy as np
import torch
from tqdm import tqdm

FPS = 30
LAYER = 66

def print_mps_memory(label):
    if torch.backends.mps.is_available():
        print(f"{label} - MPS memory allocated: {torch.mps.current_allocated_memory() / 1e6:.2f} MB")
        print(f"{label} - MPS memory reserved: {torch.mps.driver_allocated_memory() / 1e6:.2f} MB")

def extract(fpath, skip_completed=True, dest_dir="aist_juke_feats", device=None, print_mps_memory=False):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    os.makedirs(dest_dir, exist_ok=True)
    audio_name = Path(fpath).stem
    save_path = os.path.join(dest_dir, audio_name + ".npy")

    if os.path.exists(save_path) and skip_completed:
        return

    if print_mps_memory:
        print_mps_memory(f"Before loading audio: {audio_name}")
    audio = jukemirlib.load_audio(fpath)
    if print_mps_memory:
        print_mps_memory(f"After loading audio: {audio_name}")

    if print_mps_memory:
        print_mps_memory(f"Before extracting features: {audio_name}")
    reps = jukemirlib.extract(audio, layers=[LAYER], downsample_target_rate=FPS)
    if print_mps_memory:
        print_mps_memory(f"After extracting features: {audio_name}")

    return reps[LAYER], save_path

def extract_folder(src, dest, device=None, print_mps_memory=False):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if print_mps_memory:
        print_mps_memory("Before starting folder extraction")

    fpaths = Path(src).glob("*")
    fpaths = sorted(list(fpaths))
    extract_ = partial(extract, skip_completed=False, dest_dir=dest, device=device, print_mps_memory=print_mps_memory)
    
    for fpath in tqdm(fpaths):
        if print_mps_memory:
            print_mps_memory(f"Before processing file: {fpath.name}")
        rep, path = extract_(fpath)
        if print_mps_memory:
            print_mps_memory(f"After processing file: {fpath.name}")
        
        if print_mps_memory:
            print_mps_memory(f"Before saving: {fpath.name}")
        np.save(path, rep)
        if print_mps_memory:
            print_mps_memory(f"After saving: {fpath.name}")

    if print_mps_memory:
        print_mps_memory("After completing folder extraction")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", help="source path to AIST++ audio files")
    parser.add_argument("--dest", help="dest path to audio features")
    parser.add_argument("--print_mps_memory", action="store_true", help="Print MPS memory usage information")

    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    extract_folder(args.src, args.dest, device, print_mps_memory=args.print_mps_memory)

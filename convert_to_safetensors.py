import argparse
import os
import sys
from typing import Dict, Any

import torch
from safetensors.torch import save_file


def detect_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Heuristically extract a state_dict-like mapping (str -> Tensor)
    from various common .pt formats.
    """
    # Case 1: pure state_dict (mapping: name -> tensor)
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        print("  Detected format: pure state_dict")
        return obj

    # Case 2: container with 'state_dict' key
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        if isinstance(sd, dict):
            print("  Detected format: container with 'state_dict' key")
            return sd
        else:
            raise ValueError("  'state_dict' key exists but is not a dict of tensors.")

    # Case 3: nn.Module
    if hasattr(obj, "state_dict"):
        print("  Detected format: nn.Module (pickled). Extracting state_dict...")
        return obj.state_dict()

    # Fallback: try to guess if there is a nested dict of tensors
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict) and all(isinstance(x, torch.Tensor) for x in v.values()):
                print(f"  Detected nested dict at key '{k}'. Using it as state_dict.")
                return v

    raise ValueError("  Unknown .pt structure. Expected state_dict, {'state_dict': ...}, or nn.Module.")


def convert_file(pt_path: str, overwrite: bool = False) -> None:
    safetensors_path = os.path.splitext(pt_path)[0] + ".safetensors"

    if os.path.exists(safetensors_path) and not overwrite:
        print(f"[SKIP]  {pt_path} -> {safetensors_path} (already exists, use --overwrite to replace)")
        return

    print(f"[LOAD]  {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")

    try:
        state_dict = detect_state_dict(obj)
    except ValueError as e:
        print(f"[FAIL]  {pt_path}: {e}")
        return

    print(f"[SAVE]  {safetensors_path}")
    save_file(state_dict, safetensors_path)
    print(f"[OK]    {pt_path} -> {safetensors_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert all .pt files in a folder to .safetensors."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory to search for .pt files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .safetensors files if they already exist.",
    )
    parser.add_argument(
        "--suffix-filter",
        type=str,
        default="",
        help="Only convert .pt files whose filename contains this substring (optional).",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)

    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for .pt files under: {root_dir}")
    if args.suffix_filter:
        print(f"  Filename filter: must contain '{args.suffix_filter}'")

    num_files = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".pt"):
                continue
            if args.suffix_filter and args.suffix_filter not in fname:
                continue

            pt_path = os.path.join(dirpath, fname)
            num_files += 1
            convert_file(pt_path, overwrite=args.overwrite)

    if num_files == 0:
        print("No .pt files found.")
    else:
        print(f"Done. Processed {num_files} .pt file(s).")


if __name__ == "__main__":
    main()
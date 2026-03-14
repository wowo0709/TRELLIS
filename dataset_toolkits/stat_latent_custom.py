import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute normalization stats from a directory containing instance subfolders with slat.npz files."
    )
    parser.add_argument(
        "slat_root",
        type=str,
        help="Directory like .../02958343 that contains instance folders, each with slat.npz",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/slat.npz",
        help="Glob pattern under slat_root used to find slat.npz files (default: */slat.npz)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="feats",
        help="Key inside each npz file to use for normalization stats (default: feats)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of slat.npz files to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the resulting JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the final JSON payload",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    slat_root = Path(args.slat_root)
    if not slat_root.exists():
        raise FileNotFoundError(f"slat_root does not exist: {slat_root}")
    if not slat_root.is_dir():
        raise NotADirectoryError(f"slat_root is not a directory: {slat_root}")

    files = sorted(slat_root.glob(args.pattern))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern {args.pattern!r} under {slat_root}"
        )

    sum_mean = None
    sum_sq_mean = None
    count = 0

    iterator = files if args.quiet else tqdm(files, desc="Computing normalization")
    for path in iterator:
        data = np.load(path)
        if args.key not in data:
            raise KeyError(f"Key {args.key!r} not found in {path}")
        feats = data[args.key].astype(np.float64)
        if feats.ndim != 2:
            raise ValueError(
                f"Expected 2D array for key {args.key!r} in {path}, got shape {feats.shape}"
            )
        mean = feats.mean(axis=0)
        sq_mean = np.square(feats).mean(axis=0)
        if sum_mean is None:
            sum_mean = mean
            sum_sq_mean = sq_mean
        else:
            sum_mean += mean
            sum_sq_mean += sq_mean
        count += 1

    dataset_mean = sum_mean / count
    variance = np.maximum(sum_sq_mean / count - np.square(dataset_mean), 0.0)
    dataset_std = np.sqrt(variance)

    result = {
        "slat_root": str(slat_root),
        "num_files": count,
        "pattern": args.pattern,
        "key": args.key,
        "normalization": {
            "mean": dataset_mean.tolist(),
            "std": dataset_std.tolist(),
        },
    }

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=4))

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()



"""
python dataset_toolkits/stat_latent_custom.py \
    /root/node1/data3/shape-generation/shapenetv1/trellis_tmlr/slat_uniform_light/dinov2_vitl14_reg_slat_vae_gs_300k_b4x2_ch576_k16_shapenet_car_20260308_074029_step0150000/02958343
"""
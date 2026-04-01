import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from cleanfid import fid
from tqdm import tqdm

CACHE_VERSION = 1


def _iter_rgb_images(root_dir):
    root = Path(root_dir)
    valid_suffixes = {".png", ".jpg", ".jpeg"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_suffixes:
            continue
        yield path


def _looks_like_trellis_results_dir(gen_dir):
    root = Path(gen_dir)
    if not root.is_dir():
        return False
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "00000.png").exists():
            return True
    return False


def _count_top_level_images(gen_dir):
    root = Path(gen_dir)
    valid_suffixes = {".png", ".jpg", ".jpeg"}
    return sum(
        1
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in valid_suffixes
    )


def _get_cache_dir(gen_dir, cache_root):
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "gen_dir": str(Path(gen_dir).resolve()),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return Path(cache_root) / cache_key


def resolve_gen_dir(gen_dir, cache_root, rebuild_cache=False):
    gen_dir = Path(gen_dir)
    if not gen_dir.is_dir():
        raise ValueError(f"gen_dir does not exist or is not a directory: {gen_dir}")

    top_level_images = _count_top_level_images(gen_dir)
    if top_level_images > 0 and not _looks_like_trellis_results_dir(gen_dir):
        print(f"Using flat generated image directory: {gen_dir}")
        return str(gen_dir)

    cache_dir = _get_cache_dir(gen_dir, cache_root)
    flat_dir = cache_dir / "images"
    ready_path = cache_dir / ".ready"
    metadata_path = cache_dir / "metadata.json"

    if rebuild_cache and cache_dir.exists():
        print(f"Rebuilding generated-image cache at {cache_dir} ...")
        shutil.rmtree(cache_dir)

    if ready_path.exists() and flat_dir.exists():
        print(f"Using cached flattened gen_dir: {flat_dir}")
        return str(flat_dir)

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    flat_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(_iter_rgb_images(gen_dir))
    if not image_paths:
        shutil.rmtree(cache_dir)
        raise ValueError(f"No RGB images found under gen_dir: {gen_dir}")

    num_images = 0
    for image_path in tqdm(image_paths, desc="Link generated RGBs", ncols=100):
        rel_name = "__".join(image_path.relative_to(gen_dir).parts)
        link_name = flat_dir / rel_name
        os.symlink(image_path, link_name)
        num_images += 1

    metadata = {
        "gen_dir": str(gen_dir.resolve()),
        "flattened_dir": str(flat_dir),
        "num_images": num_images,
        "cache_version": CACHE_VERSION,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ready_path.touch()

    print(f"Flattened {num_images} generated RGB images from {gen_dir} to {flat_dir}")
    return str(flat_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/root/data/shape-generation/shapenetv1/rendered_shapenet_for_inference_21/02691156/train/rendered_images",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / ".cache" / "evaluate_diffusion"),
        help="Directory for storing flattened generated-image caches.",
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Rebuild the flattened generated-image cache even if it already exists.",
    )
    args = parser.parse_args()

    resolved_gen_dir = resolve_gen_dir(args.gen_dir, args.cache_root, args.rebuild_cache)

    results = dict()
    results["FID"] = fid.compute_fid(args.gt_dir, resolved_gen_dir, mode="clean")
    results["KID"] = fid.compute_kid(args.gt_dir, resolved_gen_dir, mode="clean")
    results["CLIP-FID"] = fid.compute_fid(
        args.gt_dir,
        resolved_gen_dir,
        mode="clean",
        model_name="clip_vit_b_32",
    )

    for k, v in results.items():
        print(f"{k}: {v}")

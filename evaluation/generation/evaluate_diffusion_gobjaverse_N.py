import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from cleanfid import fid
from tqdm import tqdm

CACHE_VERSION = 1


def iter_instance_dirs(gt_dir):
    gt_root = Path(gt_dir)
    for category in os.listdir(gt_root):
        category_path = gt_root / category
        if not category_path.is_dir():
            continue
        for dict_id in os.listdir(category_path):
            dict_path = category_path / dict_id
            if not dict_path.is_dir():
                continue
            for instance_id in os.listdir(dict_path):
                instance_path = dict_path / instance_id
                if not instance_path.is_dir():
                    continue
                yield category, dict_id, instance_id, instance_path


def get_gt_cache_dir(gt_dir, num_instances, cache_root):
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "gt_dir": str(Path(gt_dir).resolve()),
                "N": num_instances,
                "rgb_only": True,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return Path(cache_root) / cache_key


def build_subset_gt_dir(gt_dir, num_instances, cache_root, rebuild_cache=False):
    print(f"Scanning GT instances from {gt_dir} ...")
    instance_dirs = list(tqdm(iter_instance_dirs(gt_dir), desc="Discover instances", ncols=100))
    if not instance_dirs:
        raise ValueError(f"No instance directories found in gt_dir: {gt_dir}")

    if num_instances <= 0:
        raise ValueError("--N must be a positive integer")

    selected_instances = instance_dirs[-num_instances:]
    cache_dir = get_gt_cache_dir(gt_dir, num_instances, cache_root)
    subset_dir = cache_dir / "images"
    ready_path = cache_dir / ".ready"
    metadata_path = cache_dir / "metadata.json"

    if rebuild_cache and cache_dir.exists():
        print(f"Rebuilding cache at {cache_dir} ...")
        shutil.rmtree(cache_dir)

    if ready_path.exists() and subset_dir.exists():
        print(f"Using cached GT subset from {subset_dir}")
        return str(subset_dir)

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    num_images = 0
    with tqdm(selected_instances, desc="Link RGB PNGs", ncols=100) as pbar:
        for category, dict_id, instance_id, instance_path in pbar:
            for png_path in instance_path.rglob("*.png"):
                if not png_path.stem.isdigit():
                    continue
                rel_path = png_path.relative_to(instance_path)
                rel_name = "__".join(rel_path.parts)
                link_name = subset_dir / f"{category}__{dict_id}__{instance_id}__{rel_name}"
                os.symlink(png_path, link_name)
                num_images += 1
            pbar.set_postfix(instances=len(selected_instances), images=num_images)

    if num_images == 0:
        shutil.rmtree(cache_dir)
        raise ValueError(
            f"No RGB PNG images found in the selected {len(selected_instances)} instance directories from {gt_dir}"
        )

    selected_from = selected_instances[0][:3]
    selected_to = selected_instances[-1][:3]
    metadata = {
        "gt_dir": str(Path(gt_dir).resolve()),
        "N": num_instances,
        "num_instances": len(selected_instances),
        "num_images": num_images,
        "selected_from": list(selected_from),
        "selected_to": list(selected_to),
        "subset_dir": str(subset_dir),
        "cache_version": CACHE_VERSION,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    ready_path.touch()

    print(
        f"Cached {len(selected_instances)} instances / {num_images} RGB PNGs at {subset_dir} "
        f"from {selected_from} to {selected_to}"
    )
    return str(subset_dir)


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


def get_gen_cache_dir(gen_dir, cache_root):
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "gen_dir": str(Path(gen_dir).resolve()),
                "rgb_only": True,
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

    cache_dir = get_gen_cache_dir(gen_dir, cache_root)
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
        default="/root/data/shape-generation/gobjaverse/gobjaverse_280k_category/Animals",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--N",
        type=int,
        required=True,
        help="Use only the last N instance directories from gt_dir traversal order.",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / ".cache" / "evaluate_diffusion_gobjaverse_N"),
        help="Directory for storing persistent GT subset caches and flattened generated-image caches.",
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Rebuild the cached GT subset and flattened generated-image cache even if they already exist.",
    )
    args = parser.parse_args()

    subset_gt_dir = build_subset_gt_dir(args.gt_dir, args.N, Path(args.cache_root) / "gt", args.rebuild_cache)
    resolved_gen_dir = resolve_gen_dir(args.gen_dir, Path(args.cache_root) / "gen", args.rebuild_cache)

    results = dict()

    print("Computing FID ...")
    results["FID"] = fid.compute_fid(subset_gt_dir, resolved_gen_dir, mode="clean")

    print("Computing KID ...")
    results["KID"] = fid.compute_kid(subset_gt_dir, resolved_gen_dir, mode="clean")

    print("Computing CLIP-FID ...")
    results["CLIP-FID"] = fid.compute_fid(
        subset_gt_dir, resolved_gen_dir, mode="clean", model_name="clip_vit_b_32"
    )

    for k, v in results.items():
        print(f"{k}: {v}")

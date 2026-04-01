import argparse
import json
import os
import sys
sys.path.append("/root/dev/TRELLIS")
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import utils3d
from tqdm import tqdm

from trellis.pipelines import TrellisTextTo3DPipeline, TrellisTextUncond3DPipeline
from trellis.utils import render_utils

# os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def _deg2rad_if_needed(value: float) -> float:
    return float(np.deg2rad(value) if value > np.pi else value)


def load_cameras_shapenet(camera_root: Path, n_views: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    extrinsics, intrinsics, metadata = [], [], []
    for i in range(n_views):
        stem = f"{i:05d}"
        png_path = camera_root / f"{stem}.png"
        json_path = camera_root / f"{stem}.json"
        if not (_safe_exists(png_path) and _safe_exists(json_path)):
            continue
        with open(json_path, "r") as fp:
            meta = json.load(fp)

        if "transform_matrix" in meta:
            c2w = np.array(meta["transform_matrix"], dtype=np.float32)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w).astype(np.float32)
            fov_x = meta.get("camera_angle_x", np.deg2rad(70.0))
            fov_y = fov_x
        elif all(key in meta for key in ("x", "y", "z", "origin")):
            r1, r2, r3, t = map(
                lambda value: np.array(value, dtype=np.float32)[None, :].T,
                [meta["x"], meta["y"], meta["z"], meta["origin"]],
            )
            c2w = np.vstack([
                np.hstack([r1, r2, r3, t]),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ])
            w2c = np.linalg.inv(c2w).astype(np.float32)
            fov_x = _deg2rad_if_needed(meta.get("x_fov", np.deg2rad(70.0)))
            fov_y = _deg2rad_if_needed(meta.get("y_fov", np.deg2rad(70.0)))
        else:
            raise ValueError(f"Unsupported ShapeNet camera format: {json_path}")

        extrinsics.append(torch.from_numpy(w2c))
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(float(fov_x), dtype=torch.float32),
            torch.tensor(float(fov_y), dtype=torch.float32),
        ))
        metadata.append({
            "view_index": i,
            "camera_json": str(json_path),
            "image_path": str(png_path),
        })
    return extrinsics, intrinsics, metadata


def select_camera_views(
    extrinsics: Sequence[torch.Tensor],
    intrinsics: Sequence[torch.Tensor],
    metadata: Sequence[Dict],
    n_views: int,
    mode: str,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    total = len(extrinsics)
    if total == 0:
        raise ValueError("No cameras were loaded from the source item.")
    if n_views > total:
        raise ValueError(f"Requested n_views={n_views}, but only {total} cameras are available.")

    if mode == "first":
        indices = list(range(n_views))
    elif mode == "last":
        indices = list(range(total - n_views, total))
    elif mode == "uniform":
        if n_views == 1:
            indices = [0]
        else:
            indices = np.linspace(0, total - 1, num=n_views)
            indices = np.round(indices).astype(int).tolist()
    elif mode == "random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=n_views, replace=False).tolist()
    else:
        raise ValueError(f"Unsupported camera_mode: {mode}")

    selected_extrinsics = [extrinsics[i] for i in indices]
    selected_intrinsics = [intrinsics[i] for i in indices]
    selected_metadata = []
    for out_idx, src_idx in enumerate(indices):
        meta = dict(metadata[src_idx])
        meta["source_view_index"] = meta.get("view_index", src_idx)
        meta["selected_order"] = out_idx
        selected_metadata.append(meta)
    return selected_extrinsics, selected_intrinsics, selected_metadata


def validate_camera_batch(
    extrinsics: Sequence[torch.Tensor],
    intrinsics: Sequence[torch.Tensor],
    expected_views: int,
    instance_id: str,
) -> None:
    if len(extrinsics) < expected_views or len(intrinsics) < expected_views:
        raise ValueError(
            f"Camera set for {instance_id} has only {len(extrinsics)} valid views; expected at least {expected_views}."
        )
    for idx, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
        if extr.shape != (4, 4):
            raise ValueError(f"Invalid extrinsics shape at view {idx}: {tuple(extr.shape)}")
        if intr.shape != (3, 3):
            raise ValueError(f"Invalid intrinsics shape at view {idx}: {tuple(intr.shape)}")
        if not torch.isfinite(extr).all():
            raise ValueError(f"Non-finite extrinsics detected at view {idx}")
        if not torch.isfinite(intr).all():
            raise ValueError(f"Non-finite intrinsics detected at view {idx}")


def is_empty_structure_error(exc: BaseException) -> bool:
    message = str(exc)
    return "Empty sparse structure" in message or "empty sparse structure" in message


def resolve_pipeline_type(pipeline_path: Path, requested_type: str) -> str:
    pipeline_json = pipeline_path / "pipeline.json"
    if not pipeline_json.exists():
        return requested_type

    with open(pipeline_json, "r") as fp:
        pipeline_spec = json.load(fp)

    saved_name = pipeline_spec.get("name")
    if saved_name == "TrellisTextTo3DPipeline":
        return "text"
    if saved_name == "TrellisTextUncond3DPipeline":
        return "text-uncond"
    return requested_type


SHAPENET_CATEGORY_TO_SUBDIR = {
    "02958343": "car",
    "03001627": "chair",
    "03790512": "motorbike",
    "04379243": "table",
}


def resolve_caption_subdir(category: str, caption_root: Path, caption_subdir: Optional[str]) -> str:
    if caption_subdir is not None:
        return caption_subdir
    if category in SHAPENET_CATEGORY_TO_SUBDIR:
        return SHAPENET_CATEGORY_TO_SUBDIR[category]
    candidate = caption_root / category
    if candidate.is_dir():
        return category
    return category


def normalize_caption_entry(value, category: str) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None

    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[0], str) and isinstance(value[1], str):
            if value[1] == category and value[0].strip():
                return value[0].strip()
        for item in value:
            if isinstance(item, str) and item.strip() and item != category:
                return item.strip()
        return None

    if isinstance(value, dict):
        for key in ["caption", "text"]:
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        for key in ["captions", "texts"]:
            nested = value.get(key)
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
    return None


def load_caption_map(caption_root: Path, category: str, caption_subdir: Optional[str], caption_file: str) -> Dict[str, str]:
    subdir = resolve_caption_subdir(category, caption_root, caption_subdir)
    caption_path = caption_root / subdir / caption_file
    if not caption_path.exists():
        raise FileNotFoundError(f"Caption file not found: {caption_path}")

    with open(caption_path, "r") as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected caption file to contain a dict, got {type(raw).__name__}: {caption_path}")

    captions = {}
    for instance_id, value in raw.items():
        text = normalize_caption_entry(value, category)
        if text:
            captions[instance_id] = text
    return captions


def load_split_instance_ids(split_root: Path, category: str, split: str) -> List[str]:
    split_path = split_root / category / f"{split}.lst"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    with open(split_path, "r") as fp:
        return [line.strip() for line in fp if line.strip()]


def discover_shapenet_items(points_dir: Path, images_dir: Path, category: str) -> Dict[str, Dict]:
    items = {}
    cat_points_dir = points_dir / category
    cat_images_dir = images_dir / category
    if not cat_points_dir.is_dir():
        raise FileNotFoundError(f"ShapeNet points category dir not found: {cat_points_dir}")
    if not cat_images_dir.is_dir():
        raise FileNotFoundError(f"ShapeNet images category dir not found: {cat_images_dir}")

    for inst_dir in sorted(cat_points_dir.iterdir()):
        if not inst_dir.is_dir():
            continue
        instance_id = inst_dir.name
        points_path = inst_dir / "voxelized_pc.ply"
        camera_root = cat_images_dir / instance_id / "rendered_images"
        if _safe_exists(points_path) and camera_root.is_dir():
            items[instance_id] = {
                "category": category,
                "instance": instance_id,
                "points_path": points_path,
                "camera_root": camera_root,
            }
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ShapeNet TRELLIS text-conditioned samples from pseudo captions across train/val/test splits."
    )
    parser.add_argument("--pipeline_path", type=str, required=True, help="Path to a TRELLIS text pipeline folder or Hugging Face repo.")
    parser.add_argument("--category", type=str, required=True, help="ShapeNet category id, e.g. 03001627 or 02958343.")
    parser.add_argument("--points_dir", type=str, required=True, help="Root directory containing ShapeNet voxelized point directories.")
    parser.add_argument("--images_dir", type=str, required=True, help="Root directory containing ShapeNet rendered images and camera JSONs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where generated renders and metadata will be saved.")
    parser.add_argument("--caption_root", type=str, default="/root/node1/data3/shape-generation/shapenetv1/pseudo_captions", help="Root directory containing pseudo-caption subdirectories.")
    parser.add_argument("--caption_subdir", type=str, default=None, help="Optional caption subdirectory override such as chair or car.")
    parser.add_argument("--caption_file", type=str, default="id_captions.json", help="Caption JSON filename inside the caption subdirectory.")
    parser.add_argument("--split_root", type=str, default="/root/node1/data3/shape-generation/shapenetv1/OccNet-shapenetv1-split", help="Root directory containing OccNet ShapeNet split lists.")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"], help="Which dataset splits to generate.")
    parser.add_argument("--pipeline_type", choices=["text", "text-uncond"], default="text", help="Which TRELLIS text pipeline to use. Use `text` for *_txt_*_cond checkpoints.")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index to use.")
    parser.add_argument("--n_views", type=int, default=20, help="Number of camera views to render per generated sample.")
    parser.add_argument("--camera_mode", choices=["random", "first", "last", "uniform"], default="uniform", help="How to select n_views cameras from each instance camera pool.")
    parser.add_argument("--camera_seed_base", type=int, default=0, help="Base seed used for camera selection when camera_mode=random.")
    parser.add_argument("--seed_base", type=int, default=0, help="Base generation seed; each instance receives a deterministic offset.")
    parser.add_argument("--resolution", type=int, default=512, help="Render resolution.")
    parser.add_argument("--bg_color", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Renderer background color as three floats in [0, 1].")
    parser.add_argument("--near", type=float, default=0.8, help="Renderer near plane.")
    parser.add_argument("--far", type=float, default=1.6, help="Renderer far plane.")
    parser.add_argument("--ssaa", type=int, default=1, help="Renderer SSAA factor for Gaussian rendering.")
    parser.add_argument("--kernel_size", type=float, default=0.1, help="Gaussian renderer kernel size.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum retries when the generated sparse structure is empty.")
    parser.add_argument("--limit_per_split", type=int, default=None, help="Optional cap on how many instances to generate per split.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip an instance if all expected views already exist.")
    parser.add_argument("--save_ply", action="store_true", help="Also save the generated Gaussian as a PLY file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this script, but no CUDA device is available.")
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
    torch.cuda.set_device(device)

    pipeline_path = Path(args.pipeline_path)
    resolved_pipeline_type = resolve_pipeline_type(pipeline_path, args.pipeline_type)
    pipeline_cls = TrellisTextTo3DPipeline if resolved_pipeline_type == "text" else TrellisTextUncond3DPipeline
    pipeline = pipeline_cls.from_pretrained(args.pipeline_path)
    pipeline.to(device)

    points_dir = Path(args.points_dir)
    images_dir = Path(args.images_dir)
    caption_root = Path(args.caption_root)
    split_root = Path(args.split_root)

    items_by_instance = discover_shapenet_items(points_dir, images_dir, args.category)
    caption_map = load_caption_map(caption_root, args.category, args.caption_subdir, args.caption_file)

    manifest = {
        "pipeline_path": args.pipeline_path,
        "category": args.category,
        "points_dir": str(points_dir),
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "caption_root": str(caption_root),
        "caption_subdir": args.caption_subdir,
        "caption_file": args.caption_file,
        "split_root": str(split_root),
        "splits": args.splits,
        "requested_pipeline_type": args.pipeline_type,
        "resolved_pipeline_type": resolved_pipeline_type,
        "gpu": args.gpu,
        "n_views": args.n_views,
        "camera_mode": args.camera_mode,
        "camera_seed_base": args.camera_seed_base,
        "seed_base": args.seed_base,
        "resolution": args.resolution,
        "bg_color": list(map(float, args.bg_color)),
        "near": args.near,
        "far": args.far,
        "ssaa": args.ssaa,
        "kernel_size": args.kernel_size,
        "max_retries": args.max_retries,
        "limit_per_split": args.limit_per_split,
    }
    with open(output_dir / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)

    global_index = 0
    for split in args.splits:
        split_ids = load_split_instance_ids(split_root, args.category, split)
        if args.limit_per_split is not None:
            split_ids = split_ids[:args.limit_per_split]

        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        failed_log_path = split_dir / "failed_samples.jsonl"

        available_ids = [instance_id for instance_id in split_ids if instance_id in items_by_instance and instance_id in caption_map]
        missing_points_or_images = [instance_id for instance_id in split_ids if instance_id not in items_by_instance]
        missing_captions = [instance_id for instance_id in split_ids if instance_id not in caption_map]

        split_summary = {
            "split": split,
            "requested_instances": len(split_ids),
            "available_instances": len(available_ids),
            "missing_points_or_images": len(missing_points_or_images),
            "missing_captions": len(missing_captions),
        }
        with open(split_dir / "split_summary.json", "w") as fp:
            json.dump(split_summary, fp, indent=2)

        progress = tqdm(available_ids, desc=f"{split}: 0/{len(available_ids)}")
        for split_index, instance_id in enumerate(progress):
            item = items_by_instance[instance_id]
            prompt = caption_map[instance_id]
            sample_dir = split_dir / instance_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            progress.set_description(f"{split}: {split_index + 1}/{len(available_ids)}")

            with open(sample_dir / "prompt.txt", "w") as fp:
                fp.write(prompt + "\n")

            expected_pngs = [sample_dir / f"{view_idx:05d}.png" for view_idx in range(args.n_views)]
            if args.skip_existing and all(path.exists() for path in expected_pngs):
                global_index += 1
                continue

            source_extrinsics, source_intrinsics, source_camera_meta = load_cameras_shapenet(item["camera_root"], 24)
            camera_extrinsics, camera_intrinsics, camera_meta = select_camera_views(
                source_extrinsics,
                source_intrinsics,
                source_camera_meta,
                args.n_views,
                args.camera_mode,
                args.camera_seed_base + global_index,
            )
            validate_camera_batch(camera_extrinsics, camera_intrinsics, args.n_views, instance_id)
            camera_extrinsics = [tensor.to(device) for tensor in camera_extrinsics]
            camera_intrinsics = [tensor.to(device) for tensor in camera_intrinsics]

            gaussian = None
            rendered = None
            used_seed = None
            used_attempt = None
            attempted_seeds = []
            last_error = None
            base_seed = args.seed_base + global_index

            for attempt in range(args.max_retries + 1):
                seed = base_seed + attempt
                attempted_seeds.append(seed)
                progress.set_postfix_str(f"attempt {attempt + 1}/{args.max_retries + 1}")
                try:
                    with torch.no_grad():
                        run_kwargs = {
                            "prompt": prompt,
                            "num_samples": 1,
                            "seed": seed,
                            "formats": ["gaussian"],
                        }
                        if resolved_pipeline_type == "text-uncond":
                            run_kwargs["verbose"] = False
                        outputs = pipeline.run(**run_kwargs)
                        gaussian = outputs["gaussian"][0]
                        rendered = render_utils.render_frames(
                            gaussian,
                            camera_extrinsics,
                            camera_intrinsics,
                            options={
                                "resolution": args.resolution,
                                "bg_color": tuple(args.bg_color),
                                "near": args.near,
                                "far": args.far,
                                "ssaa": args.ssaa,
                                "kernel_size": args.kernel_size,
                            },
                            verbose=False,
                        )
                    used_seed = seed
                    used_attempt = attempt
                    break
                except RuntimeError as exc:
                    if not is_empty_structure_error(exc):
                        raise
                    last_error = exc

            progress.set_postfix_str("")

            if rendered is None or gaussian is None:
                failure = {
                    "split": split,
                    "instance_id": instance_id,
                    "global_index": global_index,
                    "prompt": prompt,
                    "base_seed": base_seed,
                    "attempted_seeds": attempted_seeds,
                    "max_retries": args.max_retries,
                    "error": str(last_error) if last_error is not None else "Unknown empty sparse structure failure.",
                }
                with open(failed_log_path, "a") as fp:
                    fp.write(json.dumps(failure) + "\n")
                with open(sample_dir / "failed.json", "w") as fp:
                    json.dump(failure, fp, indent=2)
                global_index += 1
                continue

            colors = rendered.get("color", [])
            if len(colors) != args.n_views:
                raise RuntimeError(f"Rendered {len(colors)} views for instance {instance_id}, expected {args.n_views}.")

            for view_idx, image in enumerate(colors):
                Image.fromarray(image).save(sample_dir / f"{view_idx:05d}.png")

            if args.save_ply:
                gaussian.save_ply(str(sample_dir / "gaussian.ply"))

            sample_meta = {
                "split": split,
                "instance_id": instance_id,
                "global_index": global_index,
                "split_index": split_index,
                "seed": used_seed,
                "base_seed": base_seed,
                "retry_attempt": used_attempt,
                "attempted_seeds": attempted_seeds,
                "prompt": prompt,
                "category": args.category,
                "points_path": str(item["points_path"]),
                "camera_root": str(item["camera_root"]),
                "camera_mode": args.camera_mode,
                "camera_meta": camera_meta,
            }
            with open(sample_dir / "meta.json", "w") as fp:
                json.dump(sample_meta, fp, indent=2)

            global_index += 1


if __name__ == "__main__":
    main()



"""
python evaluation/generation/generate_text_renders_from_captions.py \
  --pipeline_path /path/to/your/text_pipeline \
  --pipeline_type text \
  --category 03001627 \
  --points_dir /path/to/shapenet_points \
  --images_dir /path/to/rendered_shapenet_uniform_light \
  --output_dir /path/to/output/chair \
  --splits train val test \
  --gpu 0


python evaluation/generation/generate_text_renders_from_captions.py \
    --pipeline_path /root/data/shape-generation/TRELLIS/pipelines/trellis_txt_base/car \
    --pipeline_type text \
    --category 02958343 \
    --points_dir /root/node1/data3/shape-generation/shapenetv1/trellis_tmlr/voxelized_pcs_uniform_light \
    --images_dir /root/data/shape-generation/shapenetv1/rendered_shapenet_for_inference_21 \
    --output_dir /root/data/shape-generation/TRELLIS/pipelines/trellis_txt_base/car/results_orbit \
    --splits test \
    --gpu 5 \
    --n_views 20 \
    --camera_mode first
"""
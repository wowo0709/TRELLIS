import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision
from torchmetrics.multimodal import CLIPScore
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP text-image similarity for TRELLIS-generated image folders.")
    parser.add_argument("--pred_dir_path", type=str, required=True, help="TRELLIS generated results root, e.g. results_orbit.")
    parser.add_argument("--gpu_idx", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--splits", nargs="+", default=None, help="Optional split names to evaluate. Defaults to all subdirectories under pred_dir_path.")
    parser.add_argument("--view_idx", type=int, default=None, help="If set, evaluate only this view index per instance.")
    parser.add_argument("--num_views", type=int, default=None, help="If set, evaluate at most this many sorted views per instance.")
    parser.add_argument("--max_instances", type=int, default=None, help="Optional cap on number of instances to evaluate per split.")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model for torchmetrics CLIPScore. This matches evaluate_cond_text_image.py.")
    parser.add_argument("--save_json", type=str, default=None, help="Optional output JSON path. Defaults to <pred_dir_path>/clip_score_summary.json.")
    return parser.parse_args()


def list_splits(pred_dir: Path, splits: Optional[List[str]]) -> List[Path]:
    if splits is not None:
        return [pred_dir / split for split in splits]
    return sorted([path for path in pred_dir.iterdir() if path.is_dir()])


def parse_view_index(path: Path) -> Optional[int]:
    match = re.search(r"(\d{5})(?=\.png$)", path.name)
    return int(match.group(1)) if match else None


def resolve_prompt(instance_dir: Path) -> str:
    prompt_txt = instance_dir / "prompt.txt"
    if prompt_txt.exists():
        return prompt_txt.read_text().strip()

    meta_json = instance_dir / "meta.json"
    if meta_json.exists():
        with open(meta_json, "r") as fp:
            meta = json.load(fp)
        prompt = meta.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt.strip()

    raise FileNotFoundError(f"No prompt source found under {instance_dir}")


def collect_image_paths(instance_dir: Path, view_idx: Optional[int], num_views: Optional[int]) -> List[Path]:
    image_paths = sorted(instance_dir.glob("*.png"))
    if view_idx is not None:
        image_paths = [path for path in image_paths if parse_view_index(path) == view_idx]
    if num_views is not None:
        image_paths = image_paths[:num_views]
    return image_paths


@torch.no_grad()
def score_instance(
    clip_score: CLIPScore,
    image_paths: List[Path],
    prompt: str,
    device: torch.device,
) -> Tuple[float, int]:
    score_sum = 0.0
    count = 0
    for image_path in image_paths:
        image = torchvision.io.read_image(str(image_path)).to(dtype=torch.float32, device=device)
        score = clip_score(image, [prompt])
        score_sum += float(score.item())
        count += 1
    if count == 0:
        raise ValueError("No image views available for scoring.")
    return score_sum / count, count


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred_dir = Path(args.pred_dir_path)
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"pred_dir_path does not exist or is not a directory: {pred_dir}")

    split_dirs = list_splits(pred_dir, args.splits)
    split_dirs = [path for path in split_dirs if path.is_dir()]
    if not split_dirs:
        raise ValueError(f"No split directories found under {pred_dir}")

    clip_score = CLIPScore(model_name_or_path=args.model_name).to(device)

    summary: Dict[str, object] = {
        "pred_dir_path": str(pred_dir),
        "gpu_idx": args.gpu_idx,
        "device": str(device),
        "model_name": args.model_name,
        "view_idx": args.view_idx,
        "num_views": args.num_views,
        "max_instances": args.max_instances,
        "splits": {},
    }

    total_instance_score = 0.0
    total_instances = 0
    total_image_score = 0.0
    total_images = 0

    for split_dir in split_dirs:
        instance_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
        if args.max_instances is not None:
            instance_dirs = instance_dirs[:args.max_instances]

        split_results = []
        split_instance_score = 0.0
        split_image_score = 0.0
        split_image_count = 0

        for instance_dir in tqdm(instance_dirs, desc=f"Scoring {split_dir.name}"):
            try:
                prompt = resolve_prompt(instance_dir)
                image_paths = collect_image_paths(instance_dir, args.view_idx, args.num_views)
                instance_score, image_count = score_instance(clip_score, image_paths, prompt, device)
            except Exception as exc:
                split_results.append({
                    "instance_id": instance_dir.name,
                    "error": str(exc),
                })
                continue

            split_results.append({
                "instance_id": instance_dir.name,
                "prompt": prompt,
                "num_images": image_count,
                "clip_score": instance_score,
            })
            split_instance_score += instance_score
            split_image_score += instance_score * image_count
            split_image_count += image_count
            total_instance_score += instance_score
            total_instances += 1
            total_image_score += instance_score * image_count
            total_images += image_count

        valid_results = [result for result in split_results if "clip_score" in result]
        summary["splits"][split_dir.name] = {
            "num_instances": len(valid_results),
            "num_images": split_image_count,
            "mean_clip_score_per_instance": (split_instance_score / len(valid_results)) if valid_results else None,
            "mean_clip_score_per_image": (split_image_score / split_image_count) if split_image_count > 0 else None,
            "results": split_results,
        }

    summary["overall"] = {
        "num_instances": total_instances,
        "num_images": total_images,
        "mean_clip_score_per_instance": (total_instance_score / total_instances) if total_instances > 0 else None,
        "mean_clip_score_per_image": (total_image_score / total_images) if total_images > 0 else None,
    }

    output_path = Path(args.save_json) if args.save_json else pred_dir / "clip_score_summary.json"
    with open(output_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(json.dumps(summary["overall"], indent=2))
    print(f"Saved summary to: {output_path}")


if __name__ == "__main__":
    main()

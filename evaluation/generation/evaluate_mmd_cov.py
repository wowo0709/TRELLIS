import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from tqdm import tqdm

"""
shapenet
CUDA_VISIBLE_DEVICES=0 python evaluate_mmd_cov.py --gen_dir output/02958343/results_E850/cfg_3.5/plys/ --gt_dir /root/data_yw/data/shape-generation/shapenetv1/ --shapenet --category 02958343 --split test

objaverse
CUDA_VISIBLE_DEVICES=2 python evaluate_mmd_cov.py --gen_dir output/plants_small/results_E450/cfg_3.5/plys/ --gt_dir /root/data_yw/data/shape-generation/gobjaverse/ --objaverse --category Plants --split test

objaverse last N + cache
CUDA_VISIBLE_DEVICES=2 python evaluate_mmd_cov.py --gen_dir output/plants_small/results_E450/cfg_3.5/plys/ --gt_dir /root/data_yw/data/shape-generation/gobjaverse/ --objaverse --category Plants --split test --N 50

3dfront
CUDA_VISIBLE_DEVICES=4 python evaluate_mmd_cov.py --gen_dir /path/to/results --gt_dir /root/node7/data2/shape-generation/3D-FRONT/3D-FRONT-processed_bedroom-noair/bedrooms_without_lamps_full_colored_pcs/images_256_zuniform-h3-r5_noair --3dfront --category bedroom --split train
"""

CACHE_VERSION = 3


def log_status(message):
    print(f"[evaluate_mmd_cov] {message}", flush=True)


def tqdm_kwargs():
    return {"file": sys.stdout, "dynamic_ncols": True, "mininterval": 1.0}


def deterministic_sample_indices(length, num_points, seed_key):
    if length <= num_points:
        return None

    seed = int(hashlib.sha1(seed_key.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.choice(length, num_points, replace=False)



def load_ply_positions(ply_path, num_points=2048):
    """
    PLY 파일에서 가우시안 중심점(x, y, z)을 추출하고 결정적으로 샘플링합니다.
    """
    try:
        ply_path = Path(ply_path)
        plydata = PlyData.read(str(ply_path))
        x = plydata.elements[0].data["x"]
        y = plydata.elements[0].data["y"]
        z = plydata.elements[0].data["z"]

        points = np.stack([x, y, z], axis=1)

        if len(points) < num_points:
            print(f"Error: {ply_path} has only {len(points)} points (< {num_points}). Skipping.")
            return None

        idx = deterministic_sample_indices(len(points), num_points, f"ply::{ply_path.resolve()}::{num_points}")
        if idx is not None:
            points = points[idx]

        return points.astype(np.float32, copy=False)
    except Exception as e:
        print(f"Error loading {ply_path}: {e}")
        return None



def load_npz_positions(npz_path, num_points=2048):
    """
    NPZ 파일에서 GT 포인트 클라우드를 추출하고 결정적으로 샘플링합니다.
    """
    try:
        npz_path = Path(npz_path)
        data = np.load(npz_path)
        if hasattr(data, "files"):
            points = data[data.files[0]]
        else:
            points = data

        if len(points.shape) > 2:
            points = points.reshape(-1, points.shape[-1])

        if points.shape[1] > 3:
            points = points[:, :3]

        if len(points) < num_points:
            print(f"Error: {npz_path} has only {len(points)} points (< {num_points}). Skipping.")
            return None

        idx = deterministic_sample_indices(len(points), num_points, f"npz::{npz_path.resolve()}::{num_points}")
        if idx is not None:
            points = points[idx]

        return points.astype(np.float32, copy=False)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None



def normalize_pc(pc):
    """
    Center를 0으로 맞추고 구(Sphere) 형태로 Normalize
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



def chamfer_distance(pc1, pc2):
    """
    두 점군 간의 Chamfer Distance (PyTorch 이용)
    pc1: [N, 3], pc2: [M, 3] tensors
    """
    dist = torch.cdist(pc1, pc2)
    cd = dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()
    return cd.item()



def get_file_signature(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }



def get_cache_dir(gen_paths, gt_paths, num_points, cache_root, args_signature):
    log_status(f"Building cache key from {len(gen_paths)} generated files and {len(gt_paths)} GT files...")
    cache_key = hashlib.sha1(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "num_points": num_points,
                "args": args_signature,
                "gen_files": [get_file_signature(p) for p in gen_paths],
                "gt_files": [get_file_signature(p) for p in gt_paths],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_dir = Path(cache_root) / cache_key
    log_status(f"Resolved cache directory: {cache_dir}")
    return cache_dir



def maybe_limit_last_n(paths, num_instances, label):
    if num_instances is None:
        return paths
    if num_instances <= 0:
        raise ValueError("--N must be a positive integer")
    if len(paths) < num_instances:
        raise ValueError(f"Requested last {num_instances} {label}, but only found {len(paths)}")
    selected = paths[-num_instances:]
    log_status(f"Using last {len(selected)} {label}: from {selected[0]} to {selected[-1]}")
    return selected



def collect_gen_paths(gen_dir):
    gen_root = Path(gen_dir)
    if not gen_root.is_dir():
        raise ValueError(f"gen_dir does not exist or is not a directory: {gen_dir}")

    flat_plys = sorted(p for p in gen_root.glob('*.ply') if p.is_file())
    if flat_plys:
        log_status(f"Found {len(flat_plys)} top-level PLY files in {gen_root}")
        return [str(p) for p in flat_plys]

    gaussian_plys = sorted(p for p in gen_root.rglob('gaussian.ply') if p.is_file())
    if gaussian_plys:
        log_status(f"Found {len(gaussian_plys)} nested gaussian.ply files in {gen_root}")
        return [str(p) for p in gaussian_plys]

    nested_plys = sorted(p for p in gen_root.rglob('*.ply') if p.is_file())
    if nested_plys:
        log_status(f"Found {len(nested_plys)} nested PLY files in {gen_root}")
        return [str(p) for p in nested_plys]

    return []



def collect_gt_paths(gt_dir, shapenet_split=False, objaverse_split=False, category=None, split="test"):
    if shapenet_split:
        if category is None:
            raise ValueError("--shapenet 사용 시 --category가 반드시 주어져야 합니다. (예: 02691156)")

        split_file = os.path.join(gt_dir, "OccNet-shapenetv1-split", category, f"{split}.lst")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            valid_ids = [line.strip() for line in f.readlines() if line.strip()]

        log_status(f"Loaded {len(valid_ids)} valid IDs from {split} split.")

        search_dir = os.path.join(gt_dir, "colored_pcs_uniform_light", category)
        gt_paths = []
        for obj_id in valid_ids:
            obj_dir = os.path.join(search_dir, obj_id)
            if os.path.exists(obj_dir):
                gt_paths.extend(glob.glob(os.path.join(obj_dir, "*.npz")))
        return gt_paths

    if objaverse_split:
        if category is None:
            raise ValueError("--objaverse 사용 시 --category가 반드시 주어져야 합니다.")

        all_lst_path = os.path.join(gt_dir, "colored_pcs_gobjaverse_280k_category", category, "all.lst")
        if not os.path.exists(all_lst_path):
            raise FileNotFoundError(f"all.lst not found at: {all_lst_path}")

        with open(all_lst_path, "r", encoding="utf-8") as f:
            all_ids = [line.strip() for line in f.readlines() if line.strip()]

        if split == "test":
            valid_ids = all_ids[-50:]
        elif split == "val":
            valid_ids = all_ids[-100:-50]
        elif split == "train":
            valid_ids = all_ids[:-100]
        else:
            raise ValueError(f"Unknown split: {split}")

        blacklist_path = os.path.join(gt_dir, "gobjaverse_280k_category", category, f"blacklist_{category}.txt")
        if os.path.exists(blacklist_path):
            with open(blacklist_path, "r", encoding="utf-8") as f:
                blacklist = set(line.strip() for line in f.readlines() if line.strip())
            valid_ids = [vid for vid in valid_ids if vid not in blacklist]

        log_status(f"Loaded {len(valid_ids)} valid IDs from {split} split for objaverse.")

        gt_paths = []
        for obj_id in valid_ids:
            npz_pattern = os.path.join(
                gt_dir, "colored_pcs_gobjaverse_280k_category", category, obj_id, "colored_pc_*.npz"
            )
            npz_paths = sorted(glob.glob(npz_pattern))
            if npz_paths:
                gt_paths.append(npz_paths[0])
                continue

            ply_path = os.path.join(gt_dir, "gobjaverse_280k_category", category, obj_id, "points3d.ply")
            if os.path.exists(ply_path):
                gt_paths.append(ply_path)
        return gt_paths

    return glob.glob(os.path.join(gt_dir, "*", "*.npz"))


def collect_3dfront_paths(gt_dir, category=None, split="test"):
    gt_root = Path(gt_dir)
    if not gt_root.is_dir():
        raise ValueError(f"gt_dir does not exist or is not a directory: {gt_dir}")

    split_dir = gt_root / split
    search_root = split_dir if split_dir.is_dir() else gt_root

    all_paths = sorted(str(path) for path in search_root.glob("*/colored_pc_*.npz") if path.is_file())
    if not all_paths:
        return []

    blacklist = set()
    for parent in [gt_root, gt_root.parent, gt_root.parent.parent]:
        blacklist_path = parent / "blacklist.lst"
        if blacklist_path.exists():
            with open(blacklist_path, "r", encoding="utf-8") as f:
                blacklist = {line.strip() for line in f.readlines() if line.strip()}
            log_status(f"Loaded {len(blacklist)} blacklisted 3D-FRONT instances from {blacklist_path}")
            break

    filtered_paths = []
    for path_str in all_paths:
        instance_id = Path(path_str).parent.name
        if category and category.lower() not in instance_id.lower():
            continue
        if instance_id in blacklist:
            continue
        filtered_paths.append(path_str)

    if split_dir.is_dir():
        log_status(f"Loaded {len(filtered_paths)} 3D-FRONT GT files from explicit split directory: {split_dir}")
        return filtered_paths

    total = len(filtered_paths)
    if total == 0:
        return []

    test_count = min(50, total)
    val_count = min(50, max(0, total - test_count))
    train_end = total - val_count - test_count

    if split == "train":
        selected = filtered_paths[:train_end]
    elif split == "val":
        selected = filtered_paths[train_end:train_end + val_count]
    elif split == "test":
        selected = filtered_paths[train_end + val_count:]
    else:
        raise ValueError(f"Unknown split: {split}")

    log_status(
        f"Loaded {len(selected)} 3D-FRONT GT files from deterministic {split} split "
        f"(train={train_end}, val={val_count}, test={test_count}, total={total})."
    )
    return selected


def load_or_create_point_cache(paths, loader, num_points, cache_path, label):
    cache_path = Path(cache_path)
    log_status(f"Preparing {label}: {len(paths)} files, target cache={cache_path}")
    if cache_path.exists():
        log_status(f"Using cached {label} point clouds from {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        valid_paths = cached["paths"].tolist()
        valid_points = cached["points"]
        return valid_paths, [valid_points[i] for i in range(len(valid_points))]

    valid_paths = []
    valid_points = []
    for path in tqdm(paths, desc=f"Loading {label}", **tqdm_kwargs()):
        pc = loader(path, num_points)
        if pc is not None:
            valid_paths.append(str(Path(path).resolve()))
            valid_points.append(pc)

    if not valid_points:
        raise ValueError(f"유효한 {label} 데이터가 부족합니다.")

    points_array = np.stack(valid_points, axis=0).astype(np.float32, copy=False)
    np.savez_compressed(cache_path, paths=np.asarray(valid_paths), points=points_array)
    log_status(f"Cached {len(valid_points)} {label} point clouds to {cache_path}")
    return valid_paths, valid_points



def load_or_create_dist_mat(gen_points, gt_points, cache_path, device):
    cache_path = Path(cache_path)
    if cache_path.exists():
        log_status(f"Using cached distance matrix from {cache_path}")
        return np.load(cache_path)

    log_status(f"Converting point clouds to tensors on {device} (gen={len(gen_points)}, gt={len(gt_points)})...")
    gen_tensors = [torch.tensor(pc, dtype=torch.float32, device=device) for pc in gen_points]
    gt_tensors = [torch.tensor(pc, dtype=torch.float32, device=device) for pc in gt_points]

    n_gen = len(gen_tensors)
    n_gt = len(gt_tensors)
    dist_mat = np.zeros((n_gen, n_gt), dtype=np.float32)

    log_status("Computing distance matrix...")
    for i in tqdm(range(n_gen), desc="Gen vs GT", **tqdm_kwargs()):
        for j in range(n_gt):
            with torch.no_grad():
                dist_mat[i, j] = chamfer_distance(gen_tensors[i], gt_tensors[j])

    np.save(cache_path, dist_mat)
    log_status(f"Cached distance matrix to {cache_path}")
    return dist_mat



def evaluate_mmd_cov(
    gen_dir,
    gt_dir,
    num_points=2048,
    shapenet_split=False,
    objaverse_split=False,
    front3d_split=False,
    category=None,
    split="test",
    num_instances=None,
    cache_root=None,
    rebuild_cache=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_status("Starting evaluation")
    log_status(f"Arguments: gen_dir={Path(gen_dir).resolve()}, gt_dir={Path(gt_dir).resolve()}, split={split}, category={category}, num_points={num_points}, N={num_instances}")
    log_status(f"Using device: {device}")

    log_status(f"Scanning generated files under {Path(gen_dir).resolve()}...")
    gen_paths = collect_gen_paths(gen_dir)
    log_status(f"Scanning GT files under {Path(gt_dir).resolve()}...")
    if front3d_split:
        gt_paths = sorted(collect_3dfront_paths(gt_dir, category, split))
    else:
        gt_paths = sorted(collect_gt_paths(gt_dir, shapenet_split, objaverse_split, category, split))

    log_status(f"Discovered {len(gen_paths)} generated files and {len(gt_paths)} GT files before optional --N filtering")

    if len(gen_paths) == 0 or len(gt_paths) == 0:
        raise ValueError(f"데이터가 부족합니다. 생성된 파일 수: {len(gen_paths)}, GT 파일 수: {len(gt_paths)}")

    gen_paths = maybe_limit_last_n(gen_paths, num_instances, "generated files")
    gt_paths = maybe_limit_last_n(gt_paths, num_instances, "GT files")

    log_status(f"Preparing {len(gen_paths)} Gen PLYs and {len(gt_paths)} GT files (NPZ/PLY)...")

    cache_root = Path(cache_root)
    args_signature = {
        "gen_dir": str(Path(gen_dir).resolve()),
        "gt_dir": str(Path(gt_dir).resolve()),
        "shapenet": shapenet_split,
        "objaverse": objaverse_split,
        "3dfront": front3d_split,
        "category": category,
        "split": split,
        "N": num_instances,
    }
    cache_dir = get_cache_dir(gen_paths, gt_paths, num_points, cache_root, args_signature)
    if rebuild_cache and cache_dir.exists():
        log_status(f"Rebuilding cache at {cache_dir} ...")
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    gen_cache_path = cache_dir / "gen_points.npz"
    gt_cache_path = cache_dir / "gt_points.npz"
    log_status("Starting point-cloud loading stage...")
    dist_cache_path = cache_dir / "dist_mat.npy"
    metadata_path = cache_dir / "metadata.json"

    gen_valid_paths, gen_pcs = load_or_create_point_cache(gen_paths, load_ply_positions, num_points, gen_cache_path, "Gen PLYs")
    if objaverse_split:
        gt_loader = load_npz_positions if gt_paths and gt_paths[0].endswith(".npz") else load_ply_positions
        gt_label = "GT NPZs" if gt_loader is load_npz_positions else "GT PLYs"
        gt_valid_paths, gt_pcs = load_or_create_point_cache(gt_paths, gt_loader, num_points, gt_cache_path, gt_label)
    else:
        gt_valid_paths, gt_pcs = load_or_create_point_cache(gt_paths, load_npz_positions, num_points, gt_cache_path, "GT point clouds")

    if len(gen_pcs) == 0 or len(gt_pcs) == 0:
        raise ValueError(
            f"유효한 데이터가 부족합니다. 성공적으로 불러온 생성된 파일 수: {len(gen_pcs)}, GT 파일 수: {len(gt_pcs)}"
        )

    log_status("Starting Chamfer distance stage...")
    dist_mat = load_or_create_dist_mat(gen_pcs, gt_pcs, dist_cache_path, device)

    n_gen = len(gen_pcs)
    n_gt = len(gt_pcs)
    mmd = np.mean(np.min(dist_mat, axis=0))
    closest_gts = np.argmin(dist_mat, axis=1)
    unique_matched_gts = np.unique(closest_gts)
    cov = len(unique_matched_gts) / n_gt

    metadata = {
        "cache_version": CACHE_VERSION,
        "num_points": num_points,
        "device": str(device),
        "num_gen": n_gen,
        "num_gt": n_gt,
        "gen_dir": str(Path(gen_dir).resolve()),
        "gt_dir": str(Path(gt_dir).resolve()),
        "gen_paths": gen_valid_paths,
        "gt_paths": gt_valid_paths,
        "args": args_signature,
        "results": {
            "mmd": float(mmd),
            "cov": float(cov),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n========= [Evaluation Results] =========")
    print(f"MMD (Minimum Matching Distance): {mmd:.6f}", flush=True)
    print(f"COV (Coverage): {cov * 100:.2f} %", flush=True)
    print(f"Cache directory: {cache_dir}", flush=True)
    print("========================================", flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MMD and COV between GT points and Generated Gaussian PLYs")
    parser.add_argument(
        "--gen_dir",
        type=str,
        required=True,
        help="Directory containing generated .ply files or TRELLIS results/<sample>/gaussian.ply files",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Directory containing GT subfolders with .npz files (or base colored_pcs_uniform_light dir)",
    )
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points to sample per shape")

    parser.add_argument("--shapenet", action="store_true", help="Use ShapeNet split lists to filter GT files")
    parser.add_argument("--objaverse", action="store_true", help="Use Objaverse split lists to filter GT files")
    parser.add_argument("--3dfront", dest="front3d", action="store_true", help="Use 3D-FRONT instance directories and split handling")
    parser.add_argument("--category", type=str, default=None, help="Category ID (e.g., 02691156)")
    parser.add_argument("--split", type=str, choices=["train", "test", "val"], default="test", help="Dataset split to evaluate")
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Use only the last N generated files and the last N GT files after split/path ordering.",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / ".cache" / "evaluate_mmd_cov"),
        help="Directory for storing persistent sampled point clouds and distance matrix caches.",
    )
    parser.add_argument(
        "--rebuild_cache",
        action="store_true",
        help="Rebuild caches even if matching cached files already exist.",
    )

    args = parser.parse_args()
    evaluate_mmd_cov(
        args.gen_dir,
        args.gt_dir,
        args.num_points,
        args.shapenet,
        args.objaverse,
        args.front3d,
        args.category,
        args.split,
        args.N,
        args.cache_root,
        args.rebuild_cache,
    )



"""
conda run -n 3dgs-gen-4 python evaluation/diffusion/evaluate_mmd_cov.py   --gen_dir /path/to/generated_plys   --gt_dir /root/data_yw/data/shape-generation/gobjaverse   --objaverse   --category Plants   --split test   --num_points 2048   --N 50
"""

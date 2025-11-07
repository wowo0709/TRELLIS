import os
import argparse
import time
import random
from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d
from tqdm import tqdm

import utils3d


# -------------------------------
# Normalization utilities
# -------------------------------
def _normalize_to_cube(xyz_full: np.ndarray,
                       fit: bool,
                       scale_factor: float,
                       pad: float,
                       eps: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    Decide center and scale so that points land inside [-0.5, 0.5]^3.

    If fit=True:
        - center: bbox center
        - uniform scale so that the longest side becomes (1 - 2*pad)
    If fit=False:
        - center: bbox center
        - scale = user-provided 'scale_factor' (i.e., divide by this)
          (keeps previous CLI behavior)

    Returns:
        center, scale (the divisor)
    """
    mins = xyz_full.min(axis=0)
    maxs = xyz_full.max(axis=0)
    center = (mins + maxs) / 2.0

    if fit:
        side = (maxs - mins).max()
        target_side = max(1.0 - 2.0 * pad, eps)  # avoid 0
        scale = max(side, eps) / target_side     # we divide by 'scale'
    else:
        # keep old behavior: divide by user 'scale_factor'
        if not (scale_factor is not None and scale_factor > 0):
            raise ValueError("When --fit=False you must pass a positive --scale.")
        scale = float(scale_factor)

    return center, scale


# -------------------------------
# Core voxelization (per instance)
# -------------------------------
def _voxelize_pointcloud(
    sha256: str,
    input_path: str,
    output_path: str,
    scale: float,
    num_points: Optional[int] = None,
    structure: str = "shapenet",
    fit: bool = True,
    pad: float = 0.0,
) -> int:
    """
    Normalize + voxelize a colored_pc_*.npz point cloud.

    Args:
        sha256: Unique identifier (instance id; for logging).
        input_path: Path to colored_pc_*.npz.
        output_path: Output PLY path for voxel centers (64^3).
        scale: If fit=False, we divide coordinates by this value.
               If fit=True, this is ignored (kept for backward compatibility).
        num_points: Optional random subsampling count (applied AFTER normalization).
        structure: {"shapenet","gobjaverse","3dfront"} for dataset-specific centering.
        fit: If True, auto-fit into [-0.5,0.5]^3 with uniform scale (keeps relative size).
        pad: Margin inside the cube (0~0.49). With fit=True, longest side becomes (1-2*pad).

    Returns:
        Number of occupied voxels saved.
    """
    pc_dict = np.load(input_path, allow_pickle=True)
    coords_full = pc_dict["coords"].astype(np.float32)

    # Colors
    rgb_full = np.stack(
        [pc_dict["R"], pc_dict["G"], pc_dict["B"]],
        axis=1
    ).astype(np.float32)

    # ----- dataset-specific centering (computed on FULL set) -----
    if structure == "3dfront":
        # correct Y-center: (y_max + y_min) / 2
        y_center = (coords_full[:, 1].max() + coords_full[:, 1].min()) / 2.0
        coords_full = coords_full.copy()
        coords_full[:, 1] -= y_center

    # ----- normalization params from FULL set (no subsample bias) -----
    center, scale_div = _normalize_to_cube(
        xyz_full=coords_full,
        fit=fit,
        scale_factor=scale,
        pad=pad,
    )

    # ----- apply normalization -----
    if fit:
        # bbox 기반 자동 스케일
        xyz_norm = (coords_full - center) / max(scale_div, 1e-12)
    else:
        # 사용자가 전달한 scale 값 기준 (ex. 0.55 → /0.55 * 0.5)
        xyz_norm = (coords_full - center) * (0.5 / scale)

    # Optional subsampling AFTER normalization (keeps geometry scale/center intact)
    if num_points is not None and len(xyz_norm) > num_points:
        idx = np.random.choice(len(xyz_norm), size=num_points, replace=False)
        xyz_in = xyz_norm[idx]
        rgb_in = rgb_full[idx]
    else:
        xyz_in, rgb_in = xyz_norm, rgb_full

    # Only tiny clip to avoid numerical boundary issues (no geometric trimming)
    eps = 1e-6
    xyz_in = np.clip(xyz_in, -0.5 + eps, 0.5 - eps)

    # ----- voxelization -----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_in)
    pcd.colors = o3d.utility.Vector3dVector(rgb_in / 255.0)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    voxels = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    if voxels.size == 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        utils3d.io.write_ply(output_path, np.zeros((0, 3), dtype=np.float32))
        print(f"[Voxelized] {sha256} (fit={fit}, pad={pad:.3f}, scale_div={scale_div:.4f}, voxels=0)")
        return 0

    assert np.all(voxels >= 0) and np.all(voxels < 64), "Out-of-bound voxel indices"
    voxel_centers = (voxels + 0.5) / 64 - 0.5

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils3d.io.write_ply(output_path, voxel_centers.astype(np.float32))

    print(f"[Voxelized] {sha256} (fit={fit}, pad={pad:.3f}, scale_div={scale_div:.4f}, voxels={len(voxel_centers)})")
    return len(voxel_centers)


# ---------------------------------------
# Instance discovery (kept as in your code)
# ---------------------------------------
def _find_npz_file(inst_dir: str) -> Optional[str]:
    """Return a colored_pc_*.npz path if it exists inside inst_dir, else None."""
    candidates = [
        "colored_pc_100000.npz",   # ShapeNet + 3D-FRONT
        "colored_pc_200000.npz",   # GObjaverse (common)
    ]
    for fname in candidates:
        fpath = os.path.join(inst_dir, fname)
        if os.path.exists(fpath):
            return fpath
    for fname in os.listdir(inst_dir):
        if fname.startswith("colored_pc_") and fname.endswith(".npz"):
            return os.path.join(inst_dir, fname)
    return None


def discover_shapenet(input_root: str, category: Optional[str]) -> List[Tuple[str, Optional[str], str, str]]:
    """
    {input_root}/{category}/{instanceID}/colored_pc_*.npz
    Returns: (cat, None, inst_id, npz_path)
    """
    cats = [category] if category else sorted(os.listdir(input_root))
    results = []
    for cat in cats:
        cat_dir = os.path.join(input_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for inst_id in sorted(os.listdir(cat_dir)):
            inst_dir = os.path.join(cat_dir, inst_id)
            if not os.path.isdir(inst_dir):
                continue
            npz = _find_npz_file(inst_dir)
            if npz:
                results.append((cat, None, inst_id, npz))
    return results


def discover_gobjaverse(input_root: str, category: Optional[str], sub_category: Optional[str]) -> List[Tuple[str, Optional[str], str, str]]:
    """
    {input_root}/{category}/{sub_category}/{instanceID}/colored_pc_*.npz
    Returns: (cat, subcat, inst_id, npz_path)
    """
    cats = [category] if category else sorted(os.listdir(input_root))
    results = []
    for cat in cats:
        cat_dir = os.path.join(input_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        subcats = [sub_category] if sub_category else sorted(os.listdir(cat_dir))
        for subcat in subcats:
            sub_dir = os.path.join(cat_dir, subcat)
            if not os.path.isdir(sub_dir):
                continue
            for inst_id in sorted(os.listdir(sub_dir)):
                inst_dir = os.path.join(sub_dir, inst_id)
                if not os.path.isdir(inst_dir):
                    continue
                npz = _find_npz_file(inst_dir)
                if npz:
                    results.append((cat, subcat, inst_id, npz))
    return results


def discover_3dfront(input_root: str) -> List[Tuple[Optional[str], Optional[str], str, str]]:
    """
    {input_root}/{instanceID}/colored_pc_100000.npz
    Returns: (None, None, inst_id, npz_path)
    """
    results = []
    for inst_id in sorted(os.listdir(input_root)):
        inst_dir = os.path.join(input_root, inst_id)
        if not os.path.isdir(inst_dir):
            continue
        npz = _find_npz_file(inst_dir)
        if npz:
            results.append((None, None, inst_id, npz))
    return results


def discover_auto(input_root: str, category: Optional[str], sub_category: Optional[str]) -> Tuple[str, List[Tuple[Optional[str], Optional[str], str, str]]]:
    """Try GObjaverse -> ShapeNet -> 3D-FRONT."""
    gobs = discover_gobjaverse(input_root, category, sub_category)
    if gobs:
        return "gobjaverse", gobs
    shn = discover_shapenet(input_root, category)
    if shn:
        return "shapenet", shn
    fr3d = discover_3dfront(input_root)
    if fr3d:
        return "3dfront", fr3d
    return "shapenet", []


# ----------------------------
# Batch driver with tqdm
# ----------------------------
def batch_voxelize_pointcloud(input_root: str, output_root: str, scale: float,
                              category: Optional[str] = None,
                              sub_category: Optional[str] = None,
                              num_points: Optional[int] = None,
                              structure: str = "auto",
                              fit: bool = True,
                              pad: float = 0.0) -> int:
    """
    Voxelize all pointclouds under input_root, saving to mirrored folder structure in output_root.

    Output:
      - ShapeNet:   {output_root}/{category}/{instanceID}/voxelized_pc.ply
      - GObjaverse: {output_root}/{category}/{sub_category}/{instanceID}/voxelized_pc.ply
      - 3D-FRONT:   {output_root}/{instanceID}/voxelized_pc.ply
    """
    if structure == "auto":
        detected, items = discover_auto(input_root, category, sub_category)
        structure = detected
    elif structure == "shapenet":
        items = discover_shapenet(input_root, category)
    elif structure == "gobjaverse":
        items = discover_gobjaverse(input_root, category, sub_category)
    elif structure == "3dfront":
        items = discover_3dfront(input_root)
    else:
        raise ValueError("--structure must be one of: auto, shapenet, gobjaverse, 3dfront")

    if not items:
        print("No colored_pc_*.npz files found. Check --input_root and filters.")
        return 0

    tasks = []
    for cat, subcat, inst_id, npz_path in items:
        if structure == "shapenet":
            out_ply = os.path.join(output_root, cat, inst_id, "voxelized_pc.ply")
        elif structure == "gobjaverse":
            out_ply = os.path.join(output_root, cat, subcat, inst_id, "voxelized_pc.ply")
        else:  # 3dfront
            out_ply = os.path.join(output_root, inst_id, "voxelized_pc.ply")

        if os.path.exists(out_ply):
            # already processed
            continue
        tasks.append((cat, subcat, inst_id, npz_path, out_ply))

    total = len(tasks)
    if total == 0:
        print("All outputs already exist. Nothing to do.")
        return 0

    print(f"Structure: {structure} | To process: {total} instances "
          f"(category={category or 'ALL'}{', sub_category='+sub_category if sub_category else ''})")

    start_time = time.time()
    processed = 0

    with tqdm(total=total, unit="file", ncols=100, desc="Voxelizing") as pbar:
        for cat, subcat, inst_id, input_npz, out_ply in tasks:
            try:
                _voxelize_pointcloud(
                    sha256=inst_id,
                    input_path=input_npz,
                    output_path=out_ply,
                    scale=scale,
                    num_points=num_points,
                    structure=structure,
                    fit=fit,
                    pad=pad,
                )
                processed += 1
            except Exception as e:
                print(f"[Error] {input_npz}: {e}")
            pbar.update(1)

    elapsed = time.time() - start_time
    avg_time = elapsed / processed if processed > 0 else 0.0
    print(f"\n✅ Done. {processed} processed.")
    print(f"⏱️  Total: {elapsed/60:.2f} min | Avg per file: {avg_time:.2f} s")
    return processed


# -------------
# CLI
# -------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch voxelize point clouds with proper normalization (center+uniform scale).")
    parser.add_argument("--input_root", type=str, required=True,
                        help=("Root of input pointcloud trees.\n"
                              "  ShapeNet:   {cat}/{instance}/colored_pc_100000.npz\n"
                              "  GObjaverse: {cat}/{subcat}/{instance}/colored_pc_200000.npz\n"
                              "  3D-FRONT:   {instance}/colored_pc_100000.npz"))
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root to save outputs (voxelized_pc.ply), structure mirrored from input.")
    # GObjaverse: 0.45 / ShapeNet: 0.55 / 3D-FRONT (Bedroom): 3.0
    parser.add_argument("--scale", type=float, required=True,
                        help=("Manual scale divisor used when --fit=False.\n"
                              "When --fit=True (default) this is ignored."))
    parser.add_argument("--structure", type=str, default="auto",
                        choices=["auto", "shapenet", "gobjaverse", "3dfront"],
                        help="Directory structure. 'auto' tries GObjaverse, then ShapeNet, then 3D-FRONT.")
    parser.add_argument("--category", type=str, default=None,
                        help="(ShapeNet/GObjaverse) Process only this category (optional).")
    parser.add_argument("--sub_category", type=str, default=None,
                        help="(GObjaverse) Process only this sub_category within --category (optional).")
    parser.add_argument("--num_points", type=int, default=None,
                        help="Optional random subsample size (e.g., 100000).")
    # New but optional; default keeps your desired behavior (auto fit with no margin).
    parser.add_argument("--fit", type=lambda x: str(x).lower() in ["1","true","t","yes","y"], default=False,
                        help="If True, auto-fit bbox to [-0.5,0.5]^3 with uniform scaling (keeps object proportions).")
    parser.add_argument("--pad", type=float, default=0.0,
                        help="Padding margin inside the cube when --fit=True. 0.02 → object fits within [-0.48,0.48].")
    args = parser.parse_args()

    batch_voxelize_pointcloud(
        input_root=args.input_root,
        output_root=args.output_root,
        scale=args.scale,
        category=args.category,
        sub_category=args.sub_category,
        num_points=args.num_points,
        structure=args.structure,
        fit=args.fit,
        pad=args.pad,
    )


"""
(주어진 pointcloud마다 다르게 스케일링)
python voxelize_pointcloud.py \
  --input_root /data/ShapeNet_pcs \
  --output_root /data/ShapeNet_vox64 \
  --structure shapenet \
  --category chair \
  --num_points 100000 \
  --scale 1.0 \
  --fit True \
  --pad 0.02

(주어진 pointcloud마다 동일하게 스케일링)
python voxelize_pointcloud.py \
  --input_root /data/GObjaverse_pcs \
  --output_root /data/GObjaverse_vox64 \
  --structure gobjaverse \
  --category furniture \
  --sub_category chair \
  --num_points 200000 \
  --scale 0.55 \
  --fit False
"""
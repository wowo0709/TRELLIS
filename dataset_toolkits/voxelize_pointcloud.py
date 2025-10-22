import os
import argparse
import time
import random
from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d
from tqdm import tqdm

import utils3d


# ----------------------------
# Core voxelization
# ----------------------------
def _voxelize_pointcloud(
    sha256: str,
    input_path: str,
    output_path: str,
    scale: float,
    num_points: Optional[int] = None,
    structure: str = "shapenet",
) -> int:
    """
    Voxelize .npz point cloud using a manually provided scale value.

    Args:
        sha256: Unique identifier (used for logging).
        input_path: Path to colored_pc_*.npz
        output_path: Output .ply path
        scale: User-provided scale (e.g., 0.45, 0.55, 3.0, 6.0)
        num_points: Optional subsample count
        structure: one of {"shapenet","gobjaverse","3dfront"}

    Returns:
        Number of occupied voxels saved.
    """
    pc_dict = np.load(input_path, allow_pickle=True)
    coords_full = pc_dict["coords"].astype(np.float32)

    # Colors
    rgb = np.hstack([
        pc_dict["R"][:, None],
        pc_dict["G"][:, None],
        pc_dict["B"][:, None]
    ]).astype(np.float32)

    # === Centering logic by dataset ===
    if structure == "3dfront":
        # Center only in Y using half-range, as per your snippet
        y_min, y_max = coords_full[:, 1].min(), coords_full[:, 1].max()
        h_centering = (y_max - y_min) / 2.0
        xyz_full = np.stack([
            coords_full[:, 0],
            coords_full[:, 1] - h_centering,
            coords_full[:, 2]
        ], axis=1).astype(np.float32)

        # Optional subsampling (after computing centering from the full set)
        if num_points is not None and len(xyz_full) > num_points:
            idx = random.sample(range(len(xyz_full)), num_points)
            xyz_in, rgb = xyz_full[idx], rgb[idx]
        else:
            xyz_in = xyz_full

        # Scale & clip to [-0.5, 0.5]^3
        xyz_in = xyz_in / scale
        xyz_in = np.clip(xyz_in, -0.5 + 1e-6, 0.5 - 1e-6)

    else:
        # ShapeNet / GObjaverse: center by bbox center in (x,y,z), then scale
        xyz_in = coords_full.copy()

        # Optional subsampling (centering computed on the sampled set implicitly)
        if num_points is not None and len(xyz_in) > num_points:
            idx = random.sample(range(len(xyz_in)), num_points)
            xyz_in, rgb = xyz_in[idx], rgb[idx]

        # Center by bbox center
        center = (xyz_in.max(axis=0) + xyz_in.min(axis=0)) / 2.0
        xyz_in = (xyz_in - center) / scale
        xyz_in = np.clip(xyz_in, -0.5 + 1e-6, 0.5 - 1e-6)

    # Build O3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_in)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)

    # Voxelize into a 64^3 cube centered at origin
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    voxels = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    if voxels.size == 0:
        # still write an empty ply to mark done
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        utils3d.io.write_ply(output_path, np.zeros((0, 3), dtype=np.float32))
        print(f"[Voxelized] {sha256} (scale={scale:.3f}, voxels=0)")
        return 0

    assert np.all(voxels >= 0) and np.all(voxels < 64), "Out-of-bound voxel indices"
    voxel_centers = (voxels + 0.5) / 64 - 0.5

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    utils3d.io.write_ply(output_path, voxel_centers.astype(np.float32))

    print(f"[Voxelized] {sha256} (scale={scale:.3f}, voxels={len(voxel_centers)})")
    return len(voxel_centers)


# ---------------------------------------
# Instance discovery (ShapeNet/GObjaverse/3D-FRONT)
# ---------------------------------------
def _find_npz_file(inst_dir: str) -> Optional[str]:
    """
    Return a colored_pc_*.npz path if it exists inside inst_dir, else None.
    """
    candidates = [
        "colored_pc_100000.npz",   # ShapeNet + 3D-FRONT
        "colored_pc_200000.npz",   # GObjaverse common
    ]
    for fname in candidates:
        fpath = os.path.join(inst_dir, fname)
        if os.path.exists(fpath):
            return fpath
    # fallback: any colored_pc_*.npz
    for fname in os.listdir(inst_dir):
        if fname.startswith("colored_pc_") and fname.endswith(".npz"):
            return os.path.join(inst_dir, fname)
    return None


def discover_shapenet(input_root: str, category: Optional[str]) -> List[Tuple[str, Optional[str], str, str]]:
    """
    Discover ShapeNet layout:
        {input_root}/{category}/{instanceID}/colored_pc_*.npz

    Returns: list of tuples (cat, subcat=None, instance_id, input_npz_path)
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
    Discover GObjaverse layout:
        {input_root}/{category}/{sub_category}/{instanceID}/colored_pc_*.npz

    Returns: list of tuples (cat, subcat, instance_id, input_npz_path)
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
    Discover 3D-FRONT layout:
        {input_root}/{instanceID}/colored_pc_100000.npz

    Returns: list of tuples (cat=None, subcat=None, instance_id, input_npz_path)
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
    """
    Try to detect whether input_root is GObjaverse, ShapeNet, or 3D-FRONT.
    Priority: GObjaverse -> ShapeNet -> 3D-FRONT
    """
    gobs = discover_gobjaverse(input_root, category, sub_category)
    if gobs:
        return "gobjaverse", gobs
    shn = discover_shapenet(input_root, category)
    if shn:
        return "shapenet", shn
    fr3d = discover_3dfront(input_root)
    if fr3d:
        return "3dfront", fr3d
    return "shapenet", []  # default if nothing found (will print 0)


# ----------------------------
# Batch driver with ETA / tqdm
# ----------------------------
def batch_voxelize_pointcloud(input_root: str, output_root: str, scale: float,
                              category: Optional[str] = None,
                              sub_category: Optional[str] = None,
                              num_points: Optional[int] = None,
                              structure: str = "auto") -> int:
    """
    Voxelize all pointclouds under input_root, saving to mirrored folder structure in output_root.

    - ShapeNet output:   {output_root}/{category}/{instanceID}/voxelized_pc.ply
    - GObjaverse output: {output_root}/{category}/{sub_category}/{instanceID}/voxelized_pc.ply
    - 3D-FRONT output:   {output_root}/{instanceID}/voxelized_pc.ply
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

    # Build list of tasks with output paths
    tasks = []
    for cat, subcat, inst_id, npz_path in items:
        if structure == "shapenet":
            out_ply = os.path.join(output_root, cat, inst_id, "voxelized_pc.ply")
        elif structure == "gobjaverse":
            out_ply = os.path.join(output_root, cat, subcat, inst_id, "voxelized_pc.ply")
        else:  # 3dfront
            out_ply = os.path.join(output_root, inst_id, "voxelized_pc.ply")

        if os.path.exists(out_ply):
            # already done
            continue
        tasks.append((cat, subcat, inst_id, npz_path, out_ply))

    total = len(tasks)
    if total == 0:
        print("All outputs already exist. Nothing to do.")
        return 0

    print(f"Structure: {structure}  |  To process: {total} instances "
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
                )
                processed += 1
            except Exception as e:
                print(f"[Error] {input_npz}: {e}")
            pbar.update(1)
            # ETA (optional, tqdm already shows an ETA)
            elapsed = time.time() - start_time
            per = elapsed / max(processed, 1)
            eta = per * (total - processed)
            pbar.set_postfix({"avg_s": f"{per:.2f}", "ETA_min": f"{eta/60:.1f}"})

    elapsed = time.time() - start_time
    avg_time = elapsed / processed if processed > 0 else 0.0
    print(f"\n✅ Done. {processed} processed.")
    print(f"⏱️  Total: {elapsed/60:.2f} min  |  Avg per file: {avg_time:.2f} s")
    return processed


# -------------
# CLI Entrypoint
# -------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch voxelize point clouds (ShapeNet + GObjaverse + 3D-FRONT) with a given scale.")
    parser.add_argument("--input_root", type=str, required=True,
                        help=("Root of input pointcloud trees.\n"
                              "  ShapeNet:   {cat}/{instance}/colored_pc_100000.npz\n"
                              "  GObjaverse: {cat}/{subcat}/{instance}/colored_pc_200000.npz\n"
                              "  3D-FRONT:   {instance}/colored_pc_100000.npz"))
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root to save outputs (voxelized_pc.ply), structure mirrored from input.")
    parser.add_argument("--scale", type=float, required=True,
                        help=("Manual scale (e.g., 0.45, 0.55, 3.0, 6.0).\n"
                              "We divide by this to fit into [-0.5, 0.5]^3.\n"
                              "NOTE: For 3D-FRONT we first center Y by half-range, then scale."))
    parser.add_argument("--structure", type=str, default="auto",
                        choices=["auto", "shapenet", "gobjaverse", "3dfront"],
                        help="Directory structure. 'auto' tries GObjaverse, then ShapeNet, then 3D-FRONT.")
    parser.add_argument("--category", type=str, default=None,
                        help="(ShapeNet/GObjaverse) Process only this category (optional).")
    parser.add_argument("--sub_category", type=str, default=None,
                        help="(GObjaverse) Process only this sub_category within --category (optional).")
    parser.add_argument("--num_points", type=int, default=None,
                        help="Optional random subsample size (e.g., 100000).")
    args = parser.parse_args()

    batch_voxelize_pointcloud(
        input_root=args.input_root,
        output_root=args.output_root,
        scale=args.scale,
        category=args.category,
        sub_category=args.sub_category,
        num_points=args.num_points,
        structure=args.structure,
    )
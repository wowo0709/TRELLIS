import os
import argparse
import time
from typing import List, Tuple, Optional

import numpy as np
import open3d as o3d
from tqdm import tqdm

import utils3d


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
    resolution: int = 64,
) -> int:
    """
    Voxelize pointcloud WITHOUT scaling/centering.
    World grid = [-scale, scale]^3 with given resolution.
    """
    if scale <= 0:
        raise ValueError("--scale must be positive.")
    if resolution <= 0:
        raise ValueError("--resolution must be positive.")

    pc = np.load(input_path, allow_pickle=True)
    pts_in = pc["coords"].astype(np.float32)

    # Colors (optional)
    if all(k in pc for k in ("R", "G", "B")):
        cols_in = np.stack([pc["R"], pc["G"], pc["B"]], axis=1).astype(np.float32)
    else:
        cols_in = np.zeros((pts_in.shape[0], 3), dtype=np.float32)

    # ⭐ 3D-FRONT일 때: h_centering 계산 + y좌표 translation + 기록
    h_centering = 0.0
    if structure.lower() == "3dfront":
        y_min = pts_in[:, 1].min()
        y_max = pts_in[:, 1].max()
        h_centering = (y_max - y_min) / 2.0
        pts_in[:, 1] -= h_centering
        print(f"[3D-FRONT] h_centering={h_centering:.6f} applied for {sha256}")

    # Subsample before voxelization
    if num_points is not None and len(pts_in) > num_points:
        idx = np.random.choice(len(pts_in), size=num_points, replace=False)
        pts_in = pts_in[idx]
        cols_in = cols_in[idx]

    # ✅ CLIP points to cube boundaries
    pts_in = np.clip(pts_in, -scale + 1e-6, scale - 1e-6)

    # Build O3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_in)
    pcd.colors = o3d.utility.Vector3dVector(cols_in / 255.0)

    # ✅ Voxelization
    voxel_size = (2.0 * scale) / resolution
    minb = (-scale, -scale, -scale)
    maxb = ( scale,  scale,  scale)

    vg = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=voxel_size,
        min_bound=minb,
        max_bound=maxb,
    )

    voxels = vg.get_voxels()
    if len(voxels) == 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        utils3d.io.write_ply(output_path, np.zeros((0, 3), dtype=np.float32))
        print(f"[Voxelized] {sha256} (voxels=0)")
        return 0

    grid_idx = np.array([v.grid_index for v in voxels], dtype=np.int64)
    grid_idx = np.clip(grid_idx, 0, resolution - 1)
    centers = np.array(minb)[None, :] + (grid_idx + 0.5) * voxel_size

    # ✅ 저장 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ✅ voxelized_pc.ply 저장
    utils3d.io.write_ply(output_path, centers.astype(np.float32))
    print(f"[Voxelized] {sha256} (voxels={len(centers)})")

    # ⭐ 3D-FRONT일 때 h_centering.txt 파일 생성
    if structure.lower() == "3dfront":
        txt_path = os.path.join(os.path.dirname(output_path), "h_centering.txt")
        with open(txt_path, "w") as f:
            f.write(f"{h_centering:.8f}\n")
        # print(f"[Saved] h_centering.txt → {txt_path}")

    return len(centers)


# ---------------------------------------
# Instance discovery (unchanged)
# ---------------------------------------
def _find_npz_file(inst_dir: str) -> Optional[str]:
    candidates = ["colored_pc_100000.npz", "colored_pc_200000.npz"]
    for fname in candidates:
        fpath = os.path.join(inst_dir, fname)
        if os.path.exists(fpath):
            return fpath
    for fname in os.listdir(inst_dir):
        if fname.startswith("colored_pc_") and fname.endswith(".npz"):
            return os.path.join(inst_dir, fname)
    return None


from typing import List, Tuple, Optional

def discover_shapenet(input_root: str, category: Optional[str]) -> List[Tuple[str, Optional[str], str, str]]:
    results = []
    cats = [category] if category else sorted(os.listdir(input_root))
    for cat in cats:
        cat_dir = os.path.join(input_root, cat)
        if not os.path.isdir(cat_dir): continue
        for inst_id in sorted(os.listdir(cat_dir)):
            inst_dir = os.path.join(cat_dir, inst_id)
            if not os.path.isdir(inst_dir): continue
            npz = _find_npz_file(inst_dir)
            if npz: results.append((cat, None, inst_id, npz))
    return results

def discover_gobjaverse(input_root: str, category: Optional[str], sub_category: Optional[str]) -> List[Tuple[str, Optional[str], str, str]]:
    results = []
    cats = [category] if category else sorted(os.listdir(input_root))
    for cat in cats:
        cat_dir = os.path.join(input_root, cat)
        if not os.path.isdir(cat_dir): continue
        subs = [sub_category] if sub_category else sorted(os.listdir(cat_dir))
        for sub in subs:
            sub_dir = os.path.join(cat_dir, sub)
            if not os.path.isdir(sub_dir): continue
            for inst_id in sorted(os.listdir(sub_dir)):
                inst_dir = os.path.join(sub_dir, inst_id)
                if not os.path.isdir(inst_dir): continue
                npz = _find_npz_file(inst_dir)
                if npz: results.append((cat, sub, inst_id, npz))
    return results

def discover_3dfront(input_root: str) -> List[Tuple[Optional[str], Optional[str], str, str]]:
    results = []
    for inst_id in sorted(os.listdir(input_root)):
        inst_dir = os.path.join(input_root, inst_id)
        if not os.path.isdir(inst_dir): continue
        npz = _find_npz_file(inst_dir)
        if npz: results.append((None, None, inst_id, npz))
    return results

def discover_auto(input_root: str, category: Optional[str], sub_category: Optional[str]):
    gobs = discover_gobjaverse(input_root, category, sub_category)
    if gobs: return "gobjaverse", gobs
    shn = discover_shapenet(input_root, category)
    if shn: return "shapenet", shn
    fr3d = discover_3dfront(input_root)
    if fr3d: return "3dfront", fr3d
    return "shapenet", []


# ----------------------------
# Batch driver with tqdm
# ----------------------------
def batch_voxelize_pointcloud(input_root: str, output_root: str, scale: float,
                              category: Optional[str] = None,
                              sub_category: Optional[str] = None,
                              num_points: Optional[int] = None,
                              structure: str = "auto", 
                              resolution: int = 64) -> int:
    """
    Voxelize all pointclouds under input_root WITHOUT normalization.
    Grid: 64^3 over world cube [-scale, scale]^3. Save centers in world coords.
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
        else:
            out_ply = os.path.join(output_root, inst_id, "voxelized_pc.ply")

        if os.path.exists(out_ply):
            continue
        tasks.append((cat, subcat, inst_id, npz_path, out_ply))

    total = len(tasks)
    if total == 0:
        print("All outputs already exist. Nothing to do.")
        return 0

    from tqdm import tqdm
    print(f"Structure: {structure} | To process: {total} instances "
          f"(category={category or 'ALL'}{', sub_category='+sub_category if sub_category else ''})")

    processed = 0
    start = time.time()
    with tqdm(total=total, unit="file", ncols=100, desc="Voxelizing (world-bounded)") as pbar:
        for cat, subcat, inst_id, npz, out_ply in tasks:
            try:
                _voxelize_pointcloud(
                    sha256=inst_id,
                    input_path=npz,
                    output_path=out_ply,
                    scale=scale,
                    num_points=num_points,
                    structure=structure,
                )
                processed += 1
            except Exception as e:
                print(f"[Error] {npz}: {e}")
            pbar.update(1)

    elapsed = time.time() - start
    print(f"\n✅ Done. {processed} processed.  ⏱ {elapsed/60:.2f} min")
    return processed


# -------------
# CLI
# -------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxelize point clouds WITHOUT normalization. Grid is [-scale, scale]^3 @ 64^3.")
    parser.add_argument("--input_root", type=str, required=True,
                        help=("Root of input pointcloud trees.\n"
                              "  ShapeNet:   {cat}/{instance}/colored_pc_*.npz\n"
                              "  GObjaverse: {cat}/{subcat}/{instance}/colored_pc_*.npz\n"
                              "  3D-FRONT:   {instance}/colored_pc_*.npz"))
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root to save outputs (voxelized_pc.ply), structure mirrored from input.")
    parser.add_argument("--scale", type=float, required=True,
                        help="Half-side of the world cube. Grid bounds are [-scale, scale]^3.")
    parser.add_argument("--structure", type=str, default="auto",
                        choices=["auto", "shapenet", "gobjaverse", "3dfront"],
                        help="Directory structure. 'auto' tries GObjaverse, then ShapeNet, then 3D-FRONT.")
    parser.add_argument("--category", type=str, default=None,
                        help="(ShapeNet/GObjaverse) Process only this category (optional).")
    parser.add_argument("--sub_category", type=str, default=None,
                        help="(GObjaverse) Process only this sub_category within --category (optional).")
    parser.add_argument("--num_points", type=int, default=None,
                        help="Optional random subsample size before voxelization (e.g., 100000).")
    parser.add_argument("--resolution", type=int, default=64,
                    help="Voxel grid resolution (default: 64).")
    args = parser.parse_args()

    batch_voxelize_pointcloud(
        input_root=args.input_root,
        output_root=args.output_root,
        scale=args.scale,
        category=args.category,
        sub_category=args.sub_category,
        num_points=args.num_points,
        structure=args.structure,
        resolution=args.resolution,
    )



"""
python voxelize_pointcloud.py \
  --input_root /data/ShapeNet_pcs \
  --output_root /data/ShapeNet_vox64_world \
  --structure shapenet \
  --category chair \
  --scale 0.55 \
  --resolution 64
"""
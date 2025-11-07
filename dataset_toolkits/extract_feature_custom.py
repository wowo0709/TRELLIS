import os
import sys
import json
import time
import argparse
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import utils3d

torch.set_grad_enabled(False)

# -----------------------
# Small utilities
# -----------------------
def _deg2rad_if_needed(val):
    return np.deg2rad(val) if val > np.pi else val

def _safe_exists(p):  # avoid exceptions in discovery loops
    try:
        return os.path.exists(p)
    except Exception:
        return False

# -----------------------
# Rotation helper (3D-FRONT)
# -----------------------
def rotation_from_forward_vec(forward_vec, up_axis='Y', inplane_rot=None):
    """
    forward_vec: np.ndarray, shape (N, 3). Right-handed convention.
    Returns rotation matrices of shape (N, 3, 3) with columns [right, up, -forward].
    """
    fwd = np.array(forward_vec, dtype=np.float64)
    fwd_norm = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)

    if up_axis.upper() == 'Y':
        up = np.array([0.0, 1.0, 0.0])
    elif up_axis.upper() == 'Z':
        up = np.array([0.0, 0.0, 1.0])
    elif up_axis.upper() == 'X':
        up = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError("Invalid up_axis. Choose from 'X', 'Y', or 'Z'.")

    right = np.cross(fwd_norm, up)       # right-hand
    right /= np.linalg.norm(right, axis=1, keepdims=True)

    up_true = np.cross(right, fwd_norm)  # right-hand
    up_true /= np.linalg.norm(up_true, axis=1, keepdims=True)

    R = np.stack((right, up_true, -fwd_norm), axis=-1)  # (N,3,3)

    if inplane_rot is not None:
        c, s = np.cos(inplane_rot), np.sin(inplane_rot)
        Rz = np.array([[c, -s, 0],
                       [s,  c, 0],
                       [0,  0, 1]])
        R = R @ Rz
    return R

# -----------------------
# Dataset discovery
# -----------------------
def detect_dataset(args):
    """
    Try to infer dataset type from provided roots.
    Priority:
      1) 3D-FRONT if labels_dir has {inst}/boxes.npz
      2) GObjaverse if images_dir has nested per-view JSON folders
      3) ShapeNet otherwise
    """
    if args.dataset != "auto":
        return args.dataset

    # 3D-FRONT hint: labels_dir/<inst>/boxes.npz exists for some inst
    if args.labels_dir:
        for d in os.listdir(args.labels_dir):
            if _safe_exists(os.path.join(args.labels_dir, d, "boxes.npz")):
                return "3dfront"

    # GObjaverse hint: images_dir/<cat>/<sub>/<inst>/<inst>/campos_512_v4/00000/00000.json
    # (We only peek shallowly)
    for cat in os.listdir(args.images_dir):
        cat_dir = os.path.join(args.images_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for sub in os.listdir(cat_dir):
            sub_dir = os.path.join(cat_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            for inst in os.listdir(sub_dir):
                vroot = os.path.join(sub_dir, inst, inst, "campos_512_v4", "00000", "00000.json")
                if _safe_exists(vroot):
                    return "gobjaverse"

    # Otherwise assume ShapeNet
    return "shapenet"

def list_3dfront(points_dir, images_dir, raw_points_dir, labels_dir, instance=None):
    """
    Required:
      points_dir/<inst>/voxelized_pc.ply
      images_dir/<inst>/<0000..0039>_colors.png
      raw_points_dir/<inst>/colored_pc_100000.npz
      labels_dir/<inst>/boxes.npz
    """
    items = []
    ids = [instance] if instance else sorted(os.listdir(points_dir))
    for inst in ids:
        ply = os.path.join(points_dir, inst, "voxelized_pc.ply")
        img_root = os.path.join(images_dir, inst)
        raw_npz = os.path.join(raw_points_dir, inst, "colored_pc_100000.npz")
        cam_npz = os.path.join(labels_dir, inst, "boxes.npz")
        if all(map(_safe_exists, [ply, img_root, raw_npz, cam_npz])) and os.path.isdir(img_root):
            items.append((inst, ply, img_root, raw_npz, cam_npz))
    return items

def list_gobjaverse(points_dir, images_dir, category=None, sub_category=None, instance=None):
    """
    points_dir/<cat>/<sub>/<inst>/voxelized_pc.ply
    images_dir/<cat>/<sub>/<inst>/<inst>/campos_512_v4/<00000>/<00000.json & .png>
    """
    items = []
    cats = [category] if category else sorted(os.listdir(points_dir))
    for cat in cats:
        cdir = os.path.join(points_dir, cat)
        if not os.path.isdir(cdir):
            continue
        subs = [sub_category] if sub_category else sorted(os.listdir(cdir))
        for sub in subs:
            sdir = os.path.join(cdir, sub)
            if not os.path.isdir(sdir):
                continue
            insts = [instance] if instance else sorted(os.listdir(sdir))
            for inst in insts:
                ply = os.path.join(sdir, inst, "voxelized_pc.ply")
                img_root = os.path.join(images_dir, cat, sub, inst, inst, "campos_512_v4")
                if _safe_exists(ply) and os.path.isdir(img_root):
                    items.append((cat, sub, inst, ply, img_root))
    return items

def list_shapenet(points_dir, images_dir, category=None, instance=None):
    """
    points_dir/<cat>/<inst>/voxelized_pc.ply
    images_dir/<cat>/<inst>/rendered_images/<00000.png and 00000.json>
    """
    items = []
    cats = [category] if category else sorted(os.listdir(points_dir))
    for cat in cats:
        cdir = os.path.join(points_dir, cat)
        if not os.path.isdir(cdir):
            continue
        insts = [instance] if instance else sorted(os.listdir(cdir))
        for inst in insts:
            ply = os.path.join(cdir, inst, "voxelized_pc.ply")
            img_root = os.path.join(images_dir, cat, inst, "rendered_images")
            if _safe_exists(ply) and os.path.isdir(img_root):
                items.append((cat, inst, ply, img_root))
    return items

# -----------------------
# Per-dataset view loaders
# -----------------------
def load_views_3dfront(images_root, raw_npz_path, camera_npz_path,
                       num_views=40, resize_to=518, fov_deg=70.0):
    """
    Build (image, extrinsics (World->Cam), normalized intrinsics) for 3D-FRONT.
    Apply SAME Y-centering used in voxelization (h_centering).
    """
    pc = np.load(raw_npz_path, allow_pickle=True)
    y_min, y_max = pc["coords"][:, 1].min(), pc["coords"][:, 1].max()
    h_centering = (y_max - y_min) / 2.0

    cam = np.load(camera_npz_path, allow_pickle=True)
    if "camera_coords" not in cam or "target_coords" not in cam:
        raise KeyError(f"{camera_npz_path} missing 'camera_coords'/'target_coords'")
    cam_coords_all = cam["camera_coords"]
    tgt_coords_all = cam["target_coords"]

    views = []
    for i in range(num_views):  # 0000..0039
        stem = f"{i:04d}"
        png_path = os.path.join(images_root, f"{stem}_colors.png")
        if not _safe_exists(png_path):
            continue

        # image
        img = Image.open(png_path).convert("RGB")
        img = img.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
        img = (np.array(img).astype(np.float32) / 255.0)
        image = torch.from_numpy(img).permute(2, 0, 1).float()

        # camera with Y-centering
        target = tgt_coords_all[i]
        camera = cam_coords_all[i]
        target = np.array([target[0], target[1] - h_centering, target[2]], dtype=np.float32)
        camera = np.array([camera[0], camera[1] - h_centering, camera[2]], dtype=np.float32)
        forward = target - camera

        R = -rotation_from_forward_vec(forward[None, ...])[0]  # sign matches your snippet
        t = camera
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        w2c = np.linalg.inv(c2w).astype(np.float32)

        extrinsics = torch.from_numpy(w2c)
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(np.deg2rad(fov_deg), dtype=torch.float32),
            torch.tensor(np.deg2rad(fov_deg), dtype=torch.float32)
        )
        views.append({"image": image, "extrinsics": extrinsics, "intrinsics": intrinsics})
    return views, float(h_centering)

def load_views_gobjaverse(images_root, num_views=40, resize_to=518):
    """
    Each view at: {images_root}/{00000}/{00000}.png & {00000}.json
    JSON contains x,y,z (basis), origin (translation), x_fov, y_fov.
    """
    views = []
    for i in range(num_views):
        stem5 = f"{i:05d}"
        vdir = os.path.join(images_root, stem5)
        png_path = os.path.join(vdir, f"{stem5}.png")
        json_path = os.path.join(vdir, f"{stem5}.json")
        if not (_safe_exists(png_path) and _safe_exists(json_path)):
            continue

        rgba = np.array(Image.open(png_path))
        rgb = Image.fromarray(rgba[:, :, :3])
        rgb = rgb.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
        rgb = (np.array(rgb).astype(np.float32) / 255.0)
        image = torch.from_numpy(rgb).permute(2, 0, 1).float()

        with open(json_path, "r") as f:
            cam = json.load(f)
        r1, r2, r3, t = map(lambda x: np.array(x, dtype=np.float32)[None, :].T,
                            [cam["x"], cam["y"], cam["z"], cam["origin"]])
        c2w = np.vstack([np.hstack([r1, r2, r3, t]),
                         np.array([0., 0., 0., 1.], dtype=np.float32)])
        w2c = np.linalg.inv(c2w).astype(np.float32)
        extrinsics = torch.from_numpy(w2c)

        fov_x = _deg2rad_if_needed(cam.get("x_fov", np.deg2rad(70.)))
        fov_y = _deg2rad_if_needed(cam.get("y_fov", np.deg2rad(70.)))
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(fov_x, dtype=torch.float32),
            torch.tensor(fov_y, dtype=torch.float32)
        )
        views.append({"image": image, "extrinsics": extrinsics, "intrinsics": intrinsics})
    return views

def load_views_shapenet(images_root, num_views=24, resize_to=518):
    """
    Expect per-view files: {images_root}/{00000}.png and {00000}.json
    JSON can be either:
      - NeRF-style: { "transform_matrix": 4x4, "camera_angle_x": float }
      - Basis-style: { "x","y","z","origin", optionally "x_fov","y_fov" }
    Fallback FOV = 70 deg if not provided.
    """
    views = []
    for i in range(num_views):
        stem = f"{i:05d}"
        png_path = os.path.join(images_root, f"{stem}.png")
        json_path = os.path.join(images_root, f"{stem}.json")
        if not (_safe_exists(png_path) and _safe_exists(json_path)):
            continue

        img = Image.open(png_path).convert("RGB")
        img = img.resize((resize_to, resize_to), Image.Resampling.LANCZOS)
        img = (np.array(img).astype(np.float32) / 255.0)
        image = torch.from_numpy(img).permute(2, 0, 1).float()

        with open(json_path, "r") as f:
            meta = json.load(f)

        if "transform_matrix" in meta:  # NeRF style
            c2w = np.array(meta["transform_matrix"], dtype=np.float32)
            c2w[:3, 1:3] *= -1  # usual NeRF axis adjust
            w2c = np.linalg.inv(c2w).astype(np.float32)
            fov_x = meta.get("camera_angle_x", np.deg2rad(70.0))
            fov_y = fov_x
        elif all(k in meta for k in ("x", "y", "z", "origin")):  # basis style
            r1, r2, r3, t = map(lambda x: np.array(x, dtype=np.float32)[None, :].T,
                                [meta["x"], meta["y"], meta["z"], meta["origin"]])
            c2w = np.vstack([np.hstack([r1, r2, r3, t]),
                             np.array([0., 0., 0., 1.], dtype=np.float32)])
            w2c = np.linalg.inv(c2w).astype(np.float32)
            fov_x = _deg2rad_if_needed(meta.get("x_fov", np.deg2rad(70.0)))
            fov_y = _deg2rad_if_needed(meta.get("y_fov", np.deg2rad(70.0)))
        else:
            # Fallback: identity pose with 70 deg FOV
            w2c = np.eye(4, dtype=np.float32)
            fov_x = fov_y = np.deg2rad(70.0)

        extrinsics = torch.from_numpy(w2c)
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(fov_x, dtype=torch.float32),
            torch.tensor(fov_y, dtype=torch.float32)
        )
        views.append({"image": image, "extrinsics": extrinsics, "intrinsics": intrinsics})
    return views

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Unified DINO feature extraction for 3D-FRONT, ShapeNet, GObjaverse.")
    # IO roots
    parser.add_argument("--points_dir", type=str, required=True, help="Root of voxel points")
    parser.add_argument("--images_dir", type=str, required=True, help="Root of images")
    parser.add_argument("--output_root", type=str, required=True, help="Where to save features")
    # 3D-FRONT specifics
    parser.add_argument("--labels_dir", type=str, default=None, help="3D-FRONT labels root (contains boxes.npz)")
    parser.add_argument("--raw_points_dir", type=str, default=None, help="3D-FRONT raw PCs root (for h_centering)")
    parser.add_argument("--fov3dfront_deg", type=float, default=70.0, help="3D-FRONT symmetric FOV (deg)")
    # dataset selection
    parser.add_argument("--dataset", type=str, default="auto",
                        choices=["auto", "3dfront", "shapenet", "gobjaverse"])
    # optional filters
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--sub_category", type=str, default=None)
    parser.add_argument("--instance", type=str, default=None)
    # model/runtime
    parser.add_argument("--feat_model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_views", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        print(f"✅ Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available — running on CPU.")

    # DINO model + normalization
    dinov2 = torch.hub.load("facebookresearch/dinov2", args.feat_model).eval().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    n_patch = 518 // 14  # for ViT-L/14

    # dataset choice
    dataset = detect_dataset(args)
    print(f"📦 Dataset: {dataset}")

    # discover instances
    if dataset == "3dfront":
        if not (args.labels_dir and args.raw_points_dir):
            raise ValueError("3D-FRONT requires --labels_dir and --raw_points_dir.")
        items = list_3dfront(args.points_dir, args.images_dir, args.raw_points_dir, args.labels_dir, instance=args.instance)
        if not items:
            print("No 3D-FRONT instances found.")
            return
    elif dataset == "gobjaverse":
        items = list_gobjaverse(args.points_dir, args.images_dir,
                                category=args.category, sub_category=args.sub_category, instance=args.instance)
        if not items:
            print("No GObjaverse instances found (check category/sub_category/instance).")
            return
    else:  # shapenet
        items = list_shapenet(args.points_dir, args.images_dir, category=args.category, instance=args.instance)
        if not items:
            print("No ShapeNet instances found (check category/instance).")
            return

    # output root
    out_feat_root = os.path.join(args.output_root, args.feat_model)
    os.makedirs(out_feat_root, exist_ok=True)

    # queues
    load_q = Queue(maxsize=8)
    start = time.time()

    # dataset-specific loader wrappers
    def _load_views_and_positions(entry):
        if dataset == "3dfront":
            inst, ply, img_root, raw_npz, cam_npz = entry
            views, h_centering = load_views_3dfront(img_root, raw_npz, cam_npz,
                                                    num_views=args.num_views, resize_to=518, fov_deg=args.fov3dfront_deg)
            pos = utils3d.io.read_ply(ply)[0].astype(np.float32)
            meta = {"h_centering": float(h_centering)}
            return (inst,), views, pos, meta
        elif dataset == "gobjaverse":
            cat, sub, inst, ply, img_root = entry
            views = load_views_gobjaverse(img_root, num_views=args.num_views, resize_to=518)
            pos = utils3d.io.read_ply(ply)[0].astype(np.float32)
            meta = {}
            return (cat, sub, inst), views, pos, meta
        else:  # shapenet
            cat, inst, ply, img_root = entry
            views = load_views_shapenet(img_root, num_views=min(args.num_views, 40), resize_to=518)
            pos = utils3d.io.read_ply(ply)[0].astype(np.float32)
            meta = {}
            return (cat, inst), views, pos, meta

    def loader(entry):
        try:
            key_tuple, views, positions, meta = _load_views_and_positions(entry)  # ← meta 포함
            load_q.put((key_tuple, views, positions, meta))
        except Exception as e:
            print(f"[LoadError] {entry}: {e}")

    def saver(key_tuple, pack, meta):
        if dataset == "3dfront":
            inst, = key_tuple
            out_dir = os.path.join(out_feat_root, inst)
        elif dataset == "gobjaverse":
            cat, sub, inst = key_tuple
            out_dir = os.path.join(out_feat_root, cat, sub, inst)
        else:  # shapenet
            cat, inst = key_tuple
            out_dir = os.path.join(out_feat_root, cat, inst)
        os.makedirs(out_dir, exist_ok=True)

        feat_path = os.path.join(out_dir, "features.npz")

        np.savez_compressed(feat_path, **pack)

        # 3D-FRONT에 한해 h_centering.txt 기록
        if "h_centering" in meta:
            hc_path = os.path.join(out_dir, "h_centering.txt")
            with open(hc_path, "w") as f:
                f.write(f"{meta['h_centering']:.8f}\n")

    with ThreadPoolExecutor(max_workers=args.workers) as loader_pool, \
         ThreadPoolExecutor(max_workers=args.workers) as saver_pool:

        for entry in items:
            loader_pool.submit(loader, entry)

        pbar = tqdm(total=len(items), desc="Extracting DINO features", ncols=100)

        for done in range(len(items)):
            key_tuple, views, positions, meta = load_q.get()  # ← meta 포함

            if len(views) == 0:
                print(f"[Skip] No valid views for {key_tuple}")
                pbar.update(1)
                continue

            positions_t = torch.from_numpy(positions).float().to(device)
            indices = ((positions_t + 0.5) * 64).long()
            assert torch.all(indices >= 0) and torch.all(indices < 64), "Voxel indices out-of-bounds"

            pack = {"indices": indices.detach().cpu().numpy().astype(np.uint8)}
            patch_list, uv_list = [], []

            # batch over views
            for s in range(0, len(views), args.batch_size):
                batch = views[s:s + args.batch_size]
                bs = len(batch)
                imgs = torch.stack([normalize(v["image"]) for v in batch]).to(device)
                exts = torch.stack([v["extrinsics"] for v in batch]).to(device)
                ints = torch.stack([v["intrinsics"] for v in batch]).to(device)

                with torch.no_grad():
                    feats = dinov2(imgs, is_training=True)

                # tokens → (B, 1024, H, W)
                # (ViT-L/14: 1024 channels; n_patch = 518//14)
                patch = feats["x_prenorm"][:, dinov2.num_register_tokens + 1:]
                patch = patch.permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)

                # project to normalized coords [0,1] → [-1,1] for grid_sample
                uv = utils3d.torch.project_cv(positions_t, exts, ints)[0] * 2 - 1

                patch_list.append(patch)
                uv_list.append(uv)

            patchtokens = torch.cat(patch_list, dim=0)           # (V, 1024, H, W)
            uv = torch.cat(uv_list, dim=0).to(patchtokens.dtype)  # (V, N, 2)

            sampled = F.grid_sample(
                patchtokens,
                uv.unsqueeze(1),  # (V,1,N,2)
                mode="bilinear",
                align_corners=False,
            ).squeeze(2).permute(0, 2, 1).cpu().numpy()  # (V, N, 1024)

            pack["patchtokens"] = np.mean(sampled, axis=0).astype(np.float16)
            saver_pool.submit(saver, key_tuple, pack, meta)

            # ETA
            elapsed = time.time() - start
            eta = elapsed / (done + 1) * (len(items) - (done + 1))
            pbar.set_postfix({"ETA": f"{eta/60:.1f} min"})
            pbar.update(1)

        saver_pool.shutdown(wait=True)
        pbar.close()

    total = time.time() - start
    print(f"\n✅ Done. Extracted DINO features for {len(items)} instances in {total/60:.2f} min.")


if __name__ == "__main__":
    main()


"""
python extract_feature_custom.py \
  --points_dir /data/ShapeNet_vox64 \
  --images_dir /data/ShapeNet_renderings \
  --output_root /data/ShapeNet_dino_feats \
  --dataset shapenet \
  --category chair \
  --gpu 0 \
  --batch_size 16 \
  --num_views 24 \
  --workers 8
"""
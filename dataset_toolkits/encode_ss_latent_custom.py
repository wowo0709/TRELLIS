import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch
import numpy as np
from tqdm import tqdm

import utils3d
import trellis.models as models

torch.set_grad_enabled(False)


# ------------------------------
# Utility: Read voxel centers
# ------------------------------
def get_voxels_from_ply(ply_path: str, resolution: int) -> torch.Tensor:
    """
    Read voxel centers (in [-0.5,0.5]^3) from PLY and build a dense occupancy grid.

    Returns:
        ss: LongTensor of shape (1, R, R, R) with 1 at occupied cells.
    """
    pos = utils3d.io.read_ply(ply_path)[0].astype(np.float32)
    pos = np.clip(pos, -0.5 + 1e-6, 0.5 - 1e-6)

    coords = ((torch.tensor(pos) + 0.5) * resolution).long()
    coords = torch.clamp(coords, 0, resolution - 1)

    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


# ------------------------------
# Encoder loader
# ------------------------------
def load_encoder(opt):
    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval()
        print(f"Loaded pretrained encoder: {latent_name}")
    else:
        latent_name = f'{opt.enc_model}_{opt.ckpt}'
        cfg_path = os.path.join(opt.model_root, opt.enc_model, 'config.json')
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
        enc_cfg = cfg["models"]["encoder"]
        encoder = getattr(models, enc_cfg["name"])(**enc_cfg["args"]).eval()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        state = torch.load(ckpt_path, map_location='cpu')
        encoder.load_state_dict(state, strict=False)
        print(f"Loaded custom encoder checkpoint: {ckpt_path}")
    return encoder, latent_name


# ------------------------------
# Structure detection
# ------------------------------
def detect_structure(points_dir: str) -> str:
    """
    Auto-detect dataset structure based on directory depth/content.
      - 3D-FRONT: points_dir/<inst>/voxelized_pc.ply
      - ShapeNet: points_dir/<cat>/<inst>/voxelized_pc.ply
      - GObjaverse: points_dir/<cat>/<subcat>/<inst>/voxelized_pc.ply
    """
    entries = [d for d in os.listdir(points_dir) if os.path.isdir(os.path.join(points_dir, d))]
    if not entries:
        return "unknown"

    # Check 3D-FRONT (instance dirs directly under points_dir)
    for inst in entries:
        if os.path.exists(os.path.join(points_dir, inst, "voxelized_pc.ply")):
            return "3dfront"

    # Check ShapeNet vs GObjaverse
    for cat in entries:
        cat_dir = os.path.join(points_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for sub in os.listdir(cat_dir):
            sub_dir = os.path.join(cat_dir, sub)
            if not os.path.isdir(sub_dir):
                continue

            # ShapeNet if voxelized at this depth
            if os.path.exists(os.path.join(sub_dir, "voxelized_pc.ply")):
                return "shapenet"

            # GObjaverse if one deeper level has voxelized_pc.ply
            for inst in os.listdir(sub_dir):
                inst_dir = os.path.join(sub_dir, inst)
                if os.path.exists(os.path.join(inst_dir, "voxelized_pc.ply")):
                    return "gobjaverse"

    return "unknown"


# ------------------------------
# Structure-specific listing
# ------------------------------
def list_instances_3dfront(points_dir, only_instance=None):
    """
    3D-FRONT:
      points_dir/<inst>/voxelized_pc.ply
    """
    pairs = []
    inst_ids = [only_instance] if only_instance else sorted(os.listdir(points_dir))
    for inst in inst_ids:
        inst_dir = os.path.join(points_dir, inst)
        if not os.path.isdir(inst_dir):
            continue
        ply_path = os.path.join(inst_dir, "voxelized_pc.ply")
        if os.path.exists(ply_path):
            # use cat=None, subcat=None for consistent tuple shape
            pairs.append((None, None, inst, ply_path))
    return pairs


def list_instances_shapenet(points_dir, only_category=None, only_instance=None):
    """
    ShapeNet:
      points_dir/<cat>/<inst>/voxelized_pc.ply
    """
    pairs = []
    categories = [only_category] if only_category else sorted(os.listdir(points_dir))
    for cat in categories:
        cat_dir = os.path.join(points_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        inst_ids = [only_instance] if only_instance else sorted(os.listdir(cat_dir))
        for inst in inst_ids:
            inst_dir = os.path.join(cat_dir, inst)
            if not os.path.isdir(inst_dir):
                continue
            ply_path = os.path.join(inst_dir, "voxelized_pc.ply")
            if os.path.exists(ply_path):
                pairs.append((cat, None, inst, ply_path))
    return pairs


def list_instances_gobjaverse(points_dir, only_category=None, only_sub_category=None, only_instance=None):
    """
    GObjaverse:
      points_dir/<cat>/<subcat>/<inst>/voxelized_pc.ply
    """
    pairs = []
    cats = [only_category] if only_category else sorted(os.listdir(points_dir))
    for cat in cats:
        cat_dir = os.path.join(points_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        subcats = [only_sub_category] if only_sub_category else sorted(os.listdir(cat_dir))
        for subcat in subcats:
            sub_dir = os.path.join(cat_dir, subcat)
            if not os.path.isdir(sub_dir):
                continue
            insts = [only_instance] if only_instance else sorted(os.listdir(sub_dir))
            for inst in insts:
                inst_dir = os.path.join(sub_dir, inst)
                if not os.path.isdir(inst_dir):
                    continue
                ply_path = os.path.join(inst_dir, "voxelized_pc.ply")
                if os.path.exists(ply_path):
                    pairs.append((cat, subcat, inst, ply_path))
    return pairs


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Encode sparse structure latents (3D-FRONT / ShapeNet / GObjaverse).")
    parser.add_argument("--points_dir", type=str, required=True,
                        help="Input root containing voxelized_pc.ply files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output root for ss_latents/")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Voxel grid resolution (default=64)")

    # model loading
    parser.add_argument("--enc_pretrained", type=str,
                        default="microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16")
    parser.add_argument("--model_root", type=str, default="results")
    parser.add_argument("--enc_model", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)

    # device & selection
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--structure", type=str, choices=["auto", "3dfront", "shapenet", "gobjaverse"],
                        default="auto", help="Dataset structure type")
    parser.add_argument("--category", type=str, default=None, help="(ShapeNet/GObjaverse) category filter")
    parser.add_argument("--sub_category", type=str, default=None, help="(GObjaverse) sub-category filter")
    parser.add_argument("--instance", type=str, default=None, help="Specific instance id")
    parser.add_argument("--workers", type=int, default=16)
    opt = parser.parse_args()

    # Device setup
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        device = torch.device(f"cuda:{opt.gpu}")
        print(f"✅ Using GPU {opt.gpu}: {torch.cuda.get_device_name(opt.gpu)}")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available — running on CPU.")

    # Model
    encoder, latent_name = load_encoder(opt)
    encoder = encoder.to(device).eval()

    # Detect structure
    if opt.structure == "auto":
        opt.structure = detect_structure(opt.points_dir)
        print(f"🔍 Auto-detected structure: {opt.structure}")

    if opt.structure == "unknown":
        print("❌ Could not detect dataset structure. Please specify --structure manually.")
        return

    # List instances
    if opt.structure == "3dfront":
        items = list_instances_3dfront(opt.points_dir, opt.instance)
    elif opt.structure == "shapenet":
        items = list_instances_shapenet(opt.points_dir, opt.category, opt.instance)
    else:  # gobjaverse
        items = list_instances_gobjaverse(opt.points_dir, opt.category, opt.sub_category, opt.instance)

    if len(items) == 0:
        print("No voxelized_pc.ply files found. Check paths and structure/filters.")
        return
    print(f"Found {len(items)} instances to encode.")

    # Output root
    save_root = os.path.join(opt.output_dir, latent_name)
    os.makedirs(save_root, exist_ok=True)

    # Threads & queue
    load_q = Queue(maxsize=8)
    start = time.time()

    def loader(cat, subcat, inst, ply_path):
        try:
            ss = get_voxels_from_ply(ply_path, opt.resolution)[None].float()
            load_q.put((cat, subcat, inst, ss), block=True)
        except Exception as e:
            print(f"[LoadError] {cat or '-'} / {subcat or '-'} / {inst}: {e}")

    def saver(cat, subcat, inst, pack):
        if opt.structure == "3dfront":
            out_dir = os.path.join(save_root, inst)
        elif opt.structure == "shapenet":
            out_dir = os.path.join(save_root, cat, inst)
        else:  # gobjaverse
            out_dir = os.path.join(save_root, cat, subcat, inst)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "ss_latent.npz"), **pack)

    with ThreadPoolExecutor(max_workers=opt.workers) as loader_pool, \
         ThreadPoolExecutor(max_workers=opt.workers) as saver_pool:

        for item in items:
            loader_pool.submit(loader, *item)

        pbar = tqdm(total=len(items), desc="Encoding SS latents", ncols=100)
        for done in range(len(items)):
            cat, subcat, inst, ss = load_q.get(block=True)
            ss = ss.to(device).float()

            with torch.no_grad():
                latent = encoder(ss, sample_posterior=False)

            # encoder may return tuple; use first as mean
            latent_mean = latent[0] if isinstance(latent, (tuple, list)) else latent
            assert torch.isfinite(latent_mean).all(), f"Non-finite latent for {cat}/{subcat}/{inst}"

            pack = {"mean": latent_mean.squeeze(0).cpu().numpy().astype(np.float16)}
            saver_pool.submit(saver, cat, subcat, inst, pack)

            elapsed = time.time() - start
            eta = elapsed / (done + 1) * (len(items) - (done + 1))
            pbar.set_postfix({"ETA": f"{eta/60:.1f} min"})
            pbar.update(1)

        saver_pool.shutdown(wait=True)
        pbar.close()

    total = time.time() - start
    print(f"\n✅ Done. Encoded {len(items)} instances in {total/60:.2f} min.")


if __name__ == "__main__":
    main()


"""
python encode_ss_latent_custom.py \
  --points_dir /data/shapenet_vox64 \
  --output_dir /data/ss_latents_out \
  --structure shapenet \
  --category chair \
  --resolution 64 \
  --gpu 0

python dataset_toolkits/encode_ss_latent_custom.py \
    --points_dir /root/node1/data3/shape-generation/3D-FRONT/3D-FRONT-processed_bedroom-noair/trellis_new/bedrooms_without_lamps_full_voxelized_pcs/images_256_zuniform-h3-r5_noair \
    --output_dir /root/node1/data3/shape-generation/3D-FRONT/3D-FRONT-processed_bedroom-noair/trellis_custom/bedrooms_without_lamps_full_ss_latent/images_256_zuniform-h3-r5_noair \
    --model_root /root/node1/data3/shape-generation/TRELLIS/outputs/trellis_ss_vae \
    --enc_model ss_vae_300k_b4x2_3dfront_bedrooms_20251112_213954 \
    --ckpt step0300000 \
    --structure 3dfront \
    --resolution 64 \
    --gpu 0
"""
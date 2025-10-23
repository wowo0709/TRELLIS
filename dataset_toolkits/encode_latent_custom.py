import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis.models as models
import trellis.modules.sparse as sp

torch.set_grad_enabled(False)


# -------------------------------
# Structure detection & discovery
# -------------------------------
def detect_structure(root_dir: str) -> str:
    """
    Detect dataset tree from a features root.
      - 3D-FRONT: {inst}/features.npz
      - ShapeNet: {cat}/{inst}/features.npz
      - GObjaverse: {cat}/{subcat}/{inst}/features.npz
    """
    entries = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    if not entries:
        return "unknown"

    # 3D-FRONT: features directly under instance dirs
    for inst in entries:
        if os.path.exists(os.path.join(root_dir, inst, "features.npz")):
            return "3dfront"

    # ShapeNet vs GObjaverse
    for cat in entries:
        cat_dir = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for sub in os.listdir(cat_dir):
            sub_dir = os.path.join(cat_dir, sub)
            if not os.path.isdir(sub_dir):
                continue

            # ShapeNet: {cat}/{inst}/features.npz
            if os.path.exists(os.path.join(sub_dir, "features.npz")):
                return "shapenet"

            # GObjaverse: {cat}/{subcat}/{inst}/features.npz
            for inst in os.listdir(sub_dir):
                inst_dir = os.path.join(sub_dir, inst)
                if os.path.exists(os.path.join(inst_dir, "features.npz")):
                    return "gobjaverse"

    return "unknown"


def list_instances_3dfront(root_dir, instance=None):
    """
    3D-FRONT: {root_dir}/{instance}/features.npz
    Returns: [(None, None, inst, feat_path), ...]
    """
    items = []
    insts = [instance] if instance else sorted(os.listdir(root_dir))
    for inst in insts:
        inst_dir = os.path.join(root_dir, inst)
        if not os.path.isdir(inst_dir):
            continue
        feat = os.path.join(inst_dir, "features.npz")
        if os.path.exists(feat):
            items.append((None, None, inst, feat))
    return items


def list_instances_shapenet(root_dir, category=None, instance=None):
    """
    ShapeNet: {root_dir}/{category}/{instance}/features.npz
    Returns: [(cat, None, inst, feat_path), ...]
    """
    items = []
    cats = [category] if category else sorted(os.listdir(root_dir))
    for cat in cats:
        cat_dir = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        insts = [instance] if instance else sorted(os.listdir(cat_dir))
        for inst in insts:
            inst_dir = os.path.join(cat_dir, inst)
            if not os.path.isdir(inst_dir):
                continue
            feat = os.path.join(inst_dir, "features.npz")
            if os.path.exists(feat):
                items.append((cat, None, inst, feat))
    return items


def list_instances_gobjaverse(root_dir, category=None, sub_category=None, instance=None):
    """
    GObjaverse: {root_dir}/{category}/{sub_category}/{instance}/features.npz
    Returns: [(cat, subcat, inst, feat_path), ...]
    """
    items = []
    cats = [category] if category else sorted(os.listdir(root_dir))
    for cat in cats:
        cat_dir = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        subs = [sub_category] if sub_category else sorted(os.listdir(cat_dir))
        for sub in subs:
            sub_dir = os.path.join(cat_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            insts = [instance] if instance else sorted(os.listdir(sub_dir))
            for inst in insts:
                inst_dir = os.path.join(sub_dir, inst)
                if not os.path.isdir(inst_dir):
                    continue
                feat = os.path.join(inst_dir, "features.npz")
                if os.path.exists(feat):
                    items.append((cat, sub, inst, feat))
    return items


# -------------------------------
# Encoder loader
# -------------------------------
def load_encoder(opt):
    """Load the sparse latent encoder."""
    if opt.enc_model is None:
        latent_name = f"{opt.feat_model}_{opt.enc_pretrained.split('/')[-1]}"
        encoder = models.from_pretrained(opt.enc_pretrained).eval()
    else:
        latent_name = f"{opt.feat_model}_{opt.enc_model}_{opt.ckpt}"
        cfg_path = os.path.join(opt.model_root, opt.enc_model, "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        enc_cfg = cfg["models"]["encoder"]
        encoder = getattr(models, enc_cfg["name"])(**enc_cfg["args"]).eval()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, "ckpts", f"encoder_{opt.ckpt}.pt")
        encoder.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
        print(f"Loaded custom encoder from {ckpt_path}")
    return encoder, latent_name


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build sparse latents (3D-FRONT + ShapeNet + GObjaverse) from precomputed DINO features.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Root of DINO features "
                             "(3D-FRONT: inst/features.npz; "
                             "ShapeNet: cat/inst/features.npz; "
                             "GObjaverse: cat/subcat/inst/features.npz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root to save latents (e.g., slat_uniform_light/)")
    parser.add_argument("--structure", type=str, default="auto", choices=["auto", "3dfront", "shapenet", "gobjaverse"],
                        help="Dataset structure (auto-detect by default)")
    parser.add_argument("--feat_model", type=str, default="dinov2_vitl14_reg",
                        help="Feature extractor model name (for folder naming)")
    parser.add_argument("--enc_pretrained", type=str,
                        default="microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16",
                        help="Pretrained encoder path or identifier")
    parser.add_argument("--model_root", type=str, default="results",
                        help="Root of models (when using --enc_model/--ckpt)")
    parser.add_argument("--enc_model", type=str, default=None,
                        help="Custom encoder experiment name")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint tag for custom encoder")
    parser.add_argument("--category", type=str, default=None,
                        help="(ShapeNet/GObjaverse) Specific category to process (optional)")
    parser.add_argument("--sub_category", type=str, default=None,
                        help="(GObjaverse) Specific sub-category (optional)")
    parser.add_argument("--instance", type=str, default=None,
                        help="(3D-FRONT/ShapeNet/GObjaverse) Specific instance to process (optional)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of loader/saver threads (default: 16)")
    opt = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        device = torch.device(f"cuda:{opt.gpu}")
        print(f"✅ Using GPU {opt.gpu}: {torch.cuda.get_device_name(opt.gpu)}")
    else:
        device = torch.device("cpu")
        print("⚠️ CUDA not available. Running on CPU.")

    # Encoder
    encoder, latent_name = load_encoder(opt)
    encoder = encoder.to(device)

    # Output root
    out_root = os.path.join(opt.output_dir, latent_name)
    os.makedirs(out_root, exist_ok=True)

    # Detect structure
    structure = opt.structure
    if structure == "auto":
        structure = detect_structure(opt.input_dir)
        print(f"🔍 Auto-detected structure: {structure}")
    if structure == "unknown":
        print("❌ Could not auto-detect dataset structure. Specify --structure {3dfront|shapenet|gobjaverse}.")
        return

    # Gather instances
    if structure == "3dfront":
        instances = list_instances_3dfront(opt.input_dir, instance=opt.instance)
    elif structure == "shapenet":
        instances = list_instances_shapenet(opt.input_dir, category=opt.category, instance=opt.instance)
    else:  # gobjaverse
        instances = list_instances_gobjaverse(opt.input_dir, category=opt.category,
                                             sub_category=opt.sub_category, instance=opt.instance)

    if not instances:
        print("No features found. Check your --input_dir/filters/structure.")
        return

    print(f"Found {len(instances)} instances to process.")
    load_queue = Queue(maxsize=8)

    def loader(cat, subcat, inst, feat_path):
        try:
            feats_npz = np.load(feat_path)
            load_queue.put((cat, subcat, inst, feats_npz))
        except Exception as e:
            print(f"[LoadError] {cat or ''}/{subcat or ''}/{inst}: {e}")

    def saver(cat, subcat, inst, pack):
        # Save with mirrored structure
        if structure == "3dfront":
            out_dir = os.path.join(out_root, inst)
        elif structure == "shapenet":
            out_dir = os.path.join(out_root, cat, inst)
        else:  # gobjaverse
            out_dir = os.path.join(out_root, cat, subcat, inst)
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "slat.npz"), **pack)

    start_time = time.time()
    eta_tqdm = tqdm(range(len(instances)), desc="Encoding SLATs", ncols=100)

    try:
        with ThreadPoolExecutor(max_workers=opt.workers) as loader_pool, \
             ThreadPoolExecutor(max_workers=opt.workers) as saver_pool:

            # Schedule loads
            for cat, subcat, inst, feat_path in instances:
                loader_pool.submit(loader, cat, subcat, inst, feat_path)

            # Consume queue and encode
            for i in range(len(instances)):
                cat, subcat, inst, feats_npz = load_queue.get()

                feats = torch.from_numpy(feats_npz["patchtokens"]).float()
                coords = torch.cat([
                    torch.zeros(feats.shape[0], 1).int(),  # batch index
                    torch.from_numpy(feats_npz["indices"]).int()
                ], dim=1)

                sparse = sp.SparseTensor(feats=feats, coords=coords).to(device)
                with torch.no_grad():
                    latent = encoder(sparse, sample_posterior=False)

                latent_feats = latent.feats.detach().cpu().numpy().astype(np.float32)
                latent_coords = latent.coords[:, 1:].detach().cpu().numpy().astype(np.uint8)
                pack = {"feats": latent_feats, "coords": latent_coords}
                saver_pool.submit(saver, cat, subcat, inst, pack)

                # ETA
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                eta = avg_time * (len(instances) - (i + 1))
                eta_tqdm.set_postfix({"ETA": f"{eta/60:.1f} min"})
                eta_tqdm.update(1)

            saver_pool.shutdown(wait=True)

    except KeyboardInterrupt:
        print("\n⛔ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
    finally:
        eta_tqdm.close()

    total_time = time.time() - start_time
    print(f"\n✅ Done! Processed {len(instances)} instances in {total_time/60:.2f} min.")


if __name__ == "__main__":
    main()
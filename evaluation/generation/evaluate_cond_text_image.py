import csv
import os
import json
import argparse
from pathlib import Path

import torch
import torchvision
# from torchmetrics.functional.multimodal import clip_score
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

""" text2shape evaluation metrics
python evaluation/diffusion/evaluate_cond_text_image.py \
    --mode text2shape \
    --gpu_idx 0 \
    --pred_dir_path /root/node2/data2/trip2gs/ckpts/triplane_diffusion/chair-b2x4-lr1e-4_occ-300k-query-30k-detach_32x32x4-kl1e-6_voxel64-upsamplenet4_chamfer100.0_imglattv1e-4_2024-12-26_241226_2222_gaussian_vae_config_best_ep00290000_1000k-b16x4-lr2e-5_kar_cond-text_2025-02-18/250218_0123_triplane_diffusion_config/generated_samples_ema_320000_white_bg_text_step40_res224 \
    --prompts_json /root/node2/data3/shape-generation/shapenetv1/neurips_condgen/pseudo_captions/chair/id_captions.json \
    --prompts_csv ./data/text/captions.tablechair.csv \
    --num_views 20
    # --view_idx 18
"""

""" image2shape evaluation metrics
python evaluation.py \
    --mode image2shape \
    --gpu_idx 0 \
    --pred_dir_path /root/dataset_sj/3DGSGen_Comp/image2shape/splatter-image/chair \
    --gt_dir_path /root/data/3DGSGen_Comp/gt_images_224_20/03001627 \
    --num_views 20
"""


def image2shape_metrics(pred, target, device="cuda"):
    assert pred.min() >= 0 and pred.max() <= 1, f"pred.min(): {pred.min()}, pred.max(): {pred.max()}"
    assert target.min() >= 0 and target.max() <= 1, f"target.min(): {target.min()}, target.max(): {target.max()}"
    
    cal_psnr = PeakSignalNoiseRatio(data_range=1.).to(device)
    cal_ssim = StructuralSimilarityIndexMeasure(data_range=1.).to(device)
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    
    val_psnr = cal_psnr(pred, target)
    val_ssim = cal_ssim(pred, target)
    
    # LPIPS expects inputs in [-1, 1]
    pred_lpips = pred * 2 - 1
    target_lpips = target * 2 - 1
    val_lpips = cal_lpips(pred_lpips, target_lpips)
    return val_psnr, val_ssim, val_lpips


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["text2shape", "image2shape"])
    parser.add_argument("--gpu_idx", type=str, required=True)
    parser.add_argument("--pred_dir_path", type=str, required=True)
    
    # text2shape
    parser.add_argument("--view_idx", type=int, default=None) # 18
    parser.add_argument("--prompts_json", type=str, default=None) # /root/data/3DGSGen_Comp/neurips_condgen/pseudo_captions/chair/id_captions.json
    parser.add_argument("--prompts_csv", type=str, default=None) # ./data/text/captions.tablechair.csv
    
    # image2shape
    parser.add_argument("--num_views", type=int, default=20)
    parser.add_argument("--gt_dir_path", type=str, default=None) # /root/dataset_sj/3DGSGen_Comp/image2shape/chair_gt_images
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert (args.mode == "text2shape" and (args.prompts_json is not None or args.prompts_csv is not None)) or ((args.mode == "image2shape" and args.gt_dir_path is not None))
    
    if args.mode == "text2shape":
        prompts = {}
        if args.prompts_json is not None:
            all_data = json.load(open(args.prompts_json, "r"))
            for uid in all_data:
                prompts.update({uid: all_data[uid][0]})
        elif args.prompts_csv is not None:
            with open(args.prompts_csv, mode='r', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model_id = row["modelId"]
                    caption = row["description"]
                    prompts.update({model_id: caption})
        else:
            raise NotImplementedError("Prompt data should be passed.")
        pred_paths = [p for p in Path(args.pred_dir_path).iterdir()]
        # num_samples = len(pred_paths)
        print(f"Number of images: ", len(pred_paths))
        num_samples = 0
        clip_sim = 0.
        clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch32").to(device)
        for p in tqdm(pred_paths):
            filename = str(p.stem)
            if not str(p).endswith("png"):
                continue
            # if num_samples >= 10:
            #     break
            try:
                instance_id = filename.split("_")[0]
                img_batch = []
                text = prompts[instance_id]
                if args.view_idx is not None and args.num_views is None:
                    if not os.path.splitext(filename)[0].endswith(str(args.view_idx)):
                        continue
                    img = torchvision.io.read_image(str(p)).to(dtype=torch.float32, device=device)
                    clip_sim_inst = clip_score(img, [text])
                elif args.num_views is not None and args.view_idx is None:
                    clip_sim_inst = 0.
                    img = torchvision.io.read_image(str(p)).to(dtype=torch.float32, device=device)
                    clip_sim_inst += clip_score(img, [text])
                else:
                    raise ValueError(f"Either view_idx or num_views should be given. args.view_idx: {args.view_idx}, args.num_views: {args.num_views}")
            except Exception as e:
                print(e)
                continue
            # clip_sim_inst = 0.
            # for i in range(args.num_views):
            #     img = torchvision.io.read_image(str(p / f"{i}.png")).to(dtype=torch.float32, device=device)
            #     clip_sim_inst += clip_score(img, [text], model_name_or_path="openai/clip-vit-base-patch32") / args.num_views
            print(f" => clip similarity of {instance_id}: {clip_sim_inst}")
            clip_sim += clip_sim_inst.item()
            num_samples += 1
        print(f"Computed samples: {num_samples}")
        print(f"CLIP-Similarity: {(clip_sim / num_samples):.4f}%")
    else:
        psnr, ssim, lpips = 0., 0., 0.
        pred_paths = [p for p in Path(args.pred_dir_path).iterdir()]
        print(f"Number of images (containing {args.num_views} views per object): ", len(pred_paths))
        num_samples = 0
        pred_batch, gt_batch = [], []
        for p in tqdm(pred_paths):
            filename = str(p.stem)
            if not str(p).endswith("png"):
                continue
            pred_batch = []
            gt_batch = []
            instance_id = filename.split("_")[0]
            i = int(os.path.splitext(filename)[0].split("_")[-1][4:])
            pred_img = torchvision.io.read_image(str(p)).to(dtype=torch.float32, device=device)
            gt_img = torchvision.io.read_image(str(Path(args.gt_dir_path) / instance_id / f"{i:05d}.png")).to(dtype=torch.float32, device=device)
            pred_img = pred_img / 255. # range: [0, 1]
            gt_img = gt_img / 255.
            # pred_batch.append(pred_img)
            # gt_batch.append(gt_img)
            # pred_imgs = torch.stack(pred_batch)
            # gt_imgs = torch.stack(gt_batch)
            num_samples += 1
            # print(num_samples)
            pred_batch.append(pred_img)
            gt_batch.append(gt_img)
            if (num_samples+1) % args.num_views == 0:
                pred_imgs = torch.stack(pred_batch)
                gt_imgs = torch.stack(gt_batch)
                vals = image2shape_metrics(pred_imgs, gt_imgs)
                psnr += vals[0].item()
                ssim += vals[1].item()
                lpips += vals[2].item()
                pred_batch, gt_batch = [], []
        print(f"PSNR: {(psnr/(num_samples/args.num_views)):.4f}, SSIM: {(ssim/(num_samples/args.num_views)):.4f}, LPIPS: {(lpips/(num_samples/args.num_views)):.4f}")
    
    
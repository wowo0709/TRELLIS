import argparse
from cleanfid import fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir", 
        type=str, 
        default="/root/data/shape-generation/shapenetv1/rendered_shapenet_for_inference/03001627/test/rendered_images"
    )
    parser.add_argument(
        "--gen_dir", 
        type=str, 
        required=True
    )
    args = parser.parse_args()

    results = dict()
    results["FID"] = fid.compute_fid(args.gt_dir, args.gen_dir, mode="clean")
    results["KID"] = fid.compute_kid(args.gt_dir, args.gen_dir, mode="clean")
    results["CLIP-FID"] = fid.compute_fid(args.gt_dir, args.gen_dir, mode="clean", model_name="clip_vit_b_32")

    for k, v in results.items():
        print(f"{k}: {v}")
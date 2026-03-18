import argparse
import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
import numpy as np
import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the TRELLIS text-to-3D variant pipeline.')
    parser.add_argument('--model', type=str, default='microsoft/TRELLIS-text-xlarge', help='Pipeline checkpoint path or Hugging Face repo.')
    parser.add_argument('--mesh', type=str, default='assets/T.ply', help='Base mesh path for variant generation.')
    parser.add_argument('--prompt', type=str, default='Rugged, metallic texture with orange and white paint finish, suggesting a durable, industrial feel.', help='Text prompt used to guide the variant generation.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--scale', type=float, default=0.5, help='Half-extent of voxelization bounds used for the input mesh.')
    parser.add_argument('--output', type=str, default='sample_variant.mp4', help='Output video path.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    base_mesh = o3d.io.read_triangle_mesh(args.mesh)
    outputs = pipeline.run_variant(
        base_mesh,
        args.prompt,
        seed=args.seed,
        scale=args.scale,
    )

    video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
    video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
    video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
    imageio.mimsave(args.output, video, fps=30)


if __name__ == '__main__':
    main()

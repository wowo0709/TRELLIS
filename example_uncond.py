import argparse
import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
import open3d as o3d
from trellis.pipelines import TrellisTextUncond3DPipeline, TrellisUncond3DPipeline
from trellis.utils import render_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the TRELLIS unconditioned pipelines.')
    parser.add_argument('--pipeline', choices=['uncond', 'text-uncond'], default='uncond', help='Which pipeline to run.')
    parser.add_argument('--model', type=str, default='/root/node1/data3/shape-generation/TRELLIS/pipelines/trellis_uncond_base/bedrooms', help='Pipeline checkpoint path or Hugging Face repo.')
    parser.add_argument('--prompt', type=str, default='null', help='Prompt for the text-unconditioned pipeline.')
    parser.add_argument('--mesh', type=str, default=None, help='Optional base mesh path. If set, run variant generation.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--scale', type=float, default=0.5, help='Half-extent of voxelization bounds used for variant generation.')
    parser.add_argument('--output-prefix', type=str, default='sample', help='Prefix for output files.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline_cls = TrellisTextUncond3DPipeline if args.pipeline == 'text-uncond' else TrellisUncond3DPipeline
    pipeline = pipeline_cls.from_pretrained(args.model)
    pipeline.cuda()

    if args.mesh:
        base_mesh = o3d.io.read_triangle_mesh(args.mesh)
        if args.pipeline == 'text-uncond':
            outputs = pipeline.run_variant(
                base_mesh,
                args.prompt,
                num_samples=1,
                seed=args.seed,
                scale=args.scale,
                formats=['gaussian'],
            )
        else:
            outputs = pipeline.run_variant(
                base_mesh,
                num_samples=1,
                seed=args.seed,
                scale=args.scale,
                formats=['gaussian'],
            )
    else:
        if args.pipeline == 'text-uncond':
            outputs = pipeline.run(
                prompt=args.prompt,
                num_samples=1,
                seed=args.seed,
                formats=['gaussian'],
            )
        else:
            outputs = pipeline.run(
                num_samples=1,
                seed=args.seed,
                formats=['gaussian'],
            )

    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f'{args.output_prefix}_gs_{args.seed}.mp4', video, fps=30, format='FFMPEG')
    outputs['gaussian'][0].save_ply(f'{args.output_prefix}_{args.seed}.ply')


if __name__ == '__main__':
    main()

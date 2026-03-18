import argparse
import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the TRELLIS text-to-3D pipeline.')
    parser.add_argument('--model', type=str, default='microsoft/TRELLIS-text-xlarge', help='Pipeline checkpoint path or Hugging Face repo.')
    parser.add_argument('--prompt', type=str, default='A chair looking like a avocado.', help='Text prompt to generate from.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--variant-mesh', type=str, default=None, help='Optional base mesh path. If set, run variant generation instead of text-only generation.')
    parser.add_argument('--scale', type=float, default=0.5, help='Half-extent of voxelization bounds for variant generation.')
    parser.add_argument('--output-prefix', type=str, default='sample', help='Prefix for output files.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()

    if args.variant_mesh:
        base_mesh = o3d.io.read_triangle_mesh(args.variant_mesh)
        outputs = pipeline.run_variant(
            base_mesh,
            args.prompt,
            seed=args.seed,
            scale=args.scale,
        )
    else:
        outputs = pipeline.run(
            args.prompt,
            seed=args.seed,
        )

    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f'{args.output_prefix}_gs.mp4', video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(f'{args.output_prefix}_rf.mp4', video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f'{args.output_prefix}_mesh.mp4', video, fps=30)

    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,
        texture_size=1024,
    )
    glb.export(f'{args.output_prefix}.glb')
    outputs['gaussian'][0].save_ply(f'{args.output_prefix}.ply')


if __name__ == '__main__':
    main()

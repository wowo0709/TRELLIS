import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import argparse
from easydict import EasyDict as edict
import imageio
import json
from PIL import Image
from trellis import datasets
from trellis.pipelines import TrellisUncond3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.

def main(args):
    pipeline = TrellisUncond3DPipeline.from_pretrained(args.root_dir)
    pipeline.cuda()

    # cfg = 
    dataset = getattr(datasets, cfg.dataset.name)(
            cfg.dataset.data_path, 
            cfg.dataset.type, 
            cfg.dataset.category, 
            **cfg.dataset.args
        )

    # Run the pipeline
    outputs = pipeline.run(
        num_samples=1,
        seed=0,
        formats=['gaussian'],
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave("sample_gs.mp4", video, fps=30, format='FFMPEG') # apt-get install ffmpeg && pip install imageio[ffmpeg]

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply("sample.ply")

    output = outputs['gaussian'][0]
    print(type(output)) # <class 'trellis.representations.gaussian.gaussian_model.Gaussian'>




if __name__ == "__main__":
    # Arguments and config
    parser = argparse.ArgumentParser()
    ## config
    parser.add_argument('--root_dir', type=str, required=True, help='Path to pipeline directory')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save rendered outputs')
    opt = parser.parse_args()
    config = json.load(open(opt.config, 'r'))

    cfg = edict()

    main(opt)

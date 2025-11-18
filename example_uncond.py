import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisUncond3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisUncond3DPipeline.from_pretrained(
    "/root/node15/data/shape-generation/TRELLIS/pipelines/trellis_uncond_base/airplane"
)
pipeline.cuda()

# Run the pipeline
outputs = pipeline.run(
    num_samples=10,
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

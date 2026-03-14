from typing import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

from ....utils import dist_utils


class ImageConditionedMixin:
    """
    Mixin for image-conditioned models.
    
    Args:
        image_cond_model: The image conditioning model.
    """
    def __init__(
        self,
        *args,
        image_cond_model: str = 'dinov2_vitl14_reg',
        log_cond_images_to_wandb: bool = False,
        cond_image_log_num_samples: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_cond_model_name = image_cond_model
        self.image_cond_model = None     # the model is init lazily
        self.log_cond_images_to_wandb = log_cond_images_to_wandb
        self.cond_image_log_num_samples = max(int(cond_image_log_num_samples), 1)
        self._cond_image_logged_step = -1
        
    @staticmethod
    def prepare_for_training(image_cond_model: str, **kwargs):
        """
        Prepare for training.
        """
        if hasattr(super(ImageConditionedMixin, ImageConditionedMixin), 'prepare_for_training'):
            super(ImageConditionedMixin, ImageConditionedMixin).prepare_for_training(**kwargs)
        # download the model
        torch.hub.load('facebookresearch/dinov2', image_cond_model, pretrained=True)
        
    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model.
        """
        with dist_utils.local_master_first():
            dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
    
    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim in [4, 5], "Image tensor should be (B, C, H, W) or (B, V, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).cuda()
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        if self.image_cond_model is None:
            self._init_image_cond_model()

        multi_view = isinstance(image, torch.Tensor) and image.ndim == 5
        if multi_view:
            b, v, c, h, w = image.shape
            image = image.reshape(b * v, c, h, w)

        image = self.image_cond_model['transform'](image).cuda()
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])

        if multi_view:
            _, t, d = patchtokens.shape
            patchtokens = patchtokens.reshape(b, v * t, d)
        return patchtokens
        
    def get_image_wandb_payload(self, cond) -> Optional[Dict[str, torch.Tensor]]:
        should_log_images = (
            self.log_cond_images_to_wandb
            and self.is_master
            and self.wandb_config.get("use_wandb", False)
            and ((self.step + 1) % self.i_log == 0)
            and self._cond_image_logged_step != self.step
        )
        if not should_log_images or not isinstance(cond, torch.Tensor):
            return None
        if cond.ndim not in [4, 5]:
            return None

        self._cond_image_logged_step = self.step
        num_samples = min(self.cond_image_log_num_samples, cond.shape[0])
        img_dict = {}
        if cond.ndim == 4:
            for i in range(num_samples):
                img_dict[f'cond/image_{i:02d}'] = cond[i:i+1].detach()
        else:
            for i in range(num_samples):
                strip = torch.cat([view for view in cond[i]], dim=2).unsqueeze(0).detach()
                img_dict[f'cond/image_{i:02d}'] = strip
        return img_dict

    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_image(cond)
        kwargs['neg_cond'] = torch.zeros_like(cond)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

    def vis_cond(self, cond, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {'image': {'value': cond, 'type': 'image'}}

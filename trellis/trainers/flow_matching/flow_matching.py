from typing import *
import copy
import io
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
from PIL import Image

from ..basic import BasicTrainer
from ...pipelines import samplers 
from ...utils.general_utils import dict_reduce
from .mixins.classifier_free_guidance import ClassifierFreeGuidanceMixin
from .mixins.text_conditioned import TextConditionedMixin
from .mixins.image_conditioned import ImageConditionedMixin


class FlowMatchingTrainer(BasicTrainer):
    """
    Trainer for diffusion model with flow matching objective.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
    """
    def __init__(
        self,
        *args,
        t_schedule: dict = {
            'name': 'logitNormal',
            'args': {
                'mean': 0.0,
                'std': 1.0,
            }
        },
        sigma_min: float = 1e-5,
        log_voxel_to_wandb: bool = False,
        voxel_log_num_images: int = 1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min
        self.log_voxel_to_wandb = log_voxel_to_wandb
        self.voxel_log_num_images = max(int(voxel_log_num_images), 1)
        self._voxel_logged_step = -1
        self._voxel_log_warned = False

    @torch.no_grad()
    def _voxel_plot_tensor(self, occ_grid: torch.Tensor, title: str = "") -> torch.Tensor:
        """
        Render a boolean voxel grid [D, H, W] to a 2D RGB image tensor [1, 3, H, W]
        using matplotlib's 3D voxel plot.
        """
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        occ = occ_grid.detach().cpu().numpy().astype(bool)
        fig = plt.figure(figsize=(5, 5), dpi=120)
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(
            occ,
            facecolors='#1f77b4',
            edgecolor=None,
            linewidth=0.0,
            shade=True,
        )
        if title:
            ax.set_title(title, fontsize=10)
        ax.view_init(elev=25, azim=-55)

        # Keep axis visible like a standard 3D plot.
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, 3, H, W]
        return img_t

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            t: The [N] tensor of diffusion steps [0-1].
            noise: If specified, use this noise instead of generating new noise.

        Returns:
            x_t, the noisy version of x_0 under timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t

    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Get original image from noisy version under timestep t.
        """
        assert noise.shape == x_t.shape, "noise must have same shape as x_t"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape) - 1)])
        x_0 = (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * noise) / (1 - t)
        return x_0

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the velocity of the diffusion process at time t.
        """
        return (1 - self.sigma_min) * noise - x_0

    def get_cond(self, cond: Optional = None, **kwargs):
        """
        Get the conditioning data.
        """
        return cond
    
    def get_inference_cond(self, cond: Optional = None, **kwargs):
        """
        Get the conditioning data for inference.
        """
        return {'cond': cond, **kwargs}

    def get_sampler(self, **kwargs) -> samplers.FlowEulerSampler:
        """
        Get the sampler for the diffusion process.
        """
        return samplers.FlowEulerSampler(self.sigma_min)
    
    def vis_cond(self, **kwargs):
        """
        Visualize the conditioning data.
        """
        return {}

    def sample_t(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps.
        """
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size) * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    def training_losses(
        self,
        x_0: torch.Tensor,
        cond=None,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for a single timestep.

        Args:
            x_0: The [N x C x ...] tensor of noiseless inputs.
            cond: The [N x ...] tensor of additional conditions.
            kwargs: Additional arguments to pass to the backbone.

        Returns:
            a dict with the key "loss" containing a tensor of shape [N].
            may also contain other keys for different terms.
        """
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        
        pred = self.training_models['denoiser'](x_t, t * 1000, cond, **kwargs)
        assert pred.shape == noise.shape == x_0.shape
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"]

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item()
            for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"bin_{i}"] = {"mse": mse_per_instance[time_bin == i].mean()}

        img_dict = None
        should_log_voxel = (
            self.log_voxel_to_wandb
            and self.is_master
            and self.wandb_config.get("use_wandb", False)
            and ((self.step + 1) % self.i_log == 0)
            and self._voxel_logged_step != self.step
        )
        if should_log_voxel:
            try:
                num_images = min(self.voxel_log_num_images, x_0.shape[0])
                x0_gt = x_0[:num_images].detach()
                x0_pred = ((1 - self.sigma_min) * noise[:num_images] - pred[:num_images]).detach()

                if not hasattr(self.dataset, 'decode_latent'):
                    raise RuntimeError("Dataset does not provide decode_latent for voxel logging.")

                # Decode once for both gt/pred to avoid loading decoder twice.
                x_cat = torch.cat([x0_gt, x0_pred], dim=0)
                x_dec = self.dataset.decode_latent(x_cat, batch_size=min(4, x_cat.shape[0]))
                x_dec = x_dec.detach()
                x_dec_gt = x_dec[:num_images]
                x_dec_pred = x_dec[num_images:]

                img_dict = {}
                for i in range(num_images):
                    gt_occ = x_dec_gt[i, 0] > 0
                    pred_occ = x_dec_pred[i, 0] > 0
                    img_dict[f'voxel/gt_{i:02d}'] = self._voxel_plot_tensor(gt_occ, title=f'GT {i}')
                    img_dict[f'voxel/pred_{i:02d}'] = self._voxel_plot_tensor(pred_occ, title=f'Pred {i}')
                self._voxel_logged_step = self.step
            except Exception as e:
                if not self._voxel_log_warned:
                    print(f'\nWarning: voxel WandB logging failed once and will be skipped. ({e})')
                    self._voxel_log_warned = True
                self.log_voxel_to_wandb = False

        if img_dict is not None:
            return terms, {}, img_dict
        return terms, {}
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        sampler = self.get_sampler()
        sample_gt = []
        sample = []
        cond_vis = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            data = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            noise = torch.randn_like(data['x_0'])
            sample_gt.append(data['x_0'])
            cond_vis.append(self.vis_cond(**data))
            del data['x_0']
            args = self.get_inference_cond(**data)
            res = sampler.sample(
                self.models['denoiser'],
                noise=noise,
                **args,
                steps=50, 
                # cfg_strength=3.0, 
                verbose=verbose,
            )
            sample.append(res.samples)

        sample_gt = torch.cat(sample_gt, dim=0)
        sample = torch.cat(sample, dim=0)
        sample_dict = {
            'sample_gt': {'value': sample_gt, 'type': 'sample'},
            'sample': {'value': sample, 'type': 'sample'},
        }
        sample_dict.update(dict_reduce(cond_vis, None, {
            'value': lambda x: torch.cat(x, dim=0),
            'type': lambda x: x[0],
        }))
        
        return sample_dict

    
class FlowMatchingCFGTrainer(ClassifierFreeGuidanceMixin, FlowMatchingTrainer):
    """
    Trainer for diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
    """
    pass


class TextConditionedFlowMatchingCFGTrainer(TextConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for text-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        text_cond_model(str): Text conditioning model.
    """
    pass


class ImageConditionedFlowMatchingCFGTrainer(ImageConditionedMixin, FlowMatchingCFGTrainer):
    """
    Trainer for image-conditioned diffusion model with flow matching objective and classifier-free guidance.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.

        t_schedule (dict): Time schedule for flow matching.
        sigma_min (float): Minimum noise level.
        p_uncond (float): Probability of dropping conditions.
        image_cond_model (str): Image conditioning model.
    """
    pass

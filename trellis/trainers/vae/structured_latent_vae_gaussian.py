from typing import *
import copy
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as edict
import utils3d.torch

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image

from ..basic import BasicTrainer
from ...representations import Gaussian
from ...renderers import GaussianRenderer
from ...modules.sparse import SparseTensor
from ...utils.loss_utils import l1_loss, l2_loss, ssim, lpips


class SLatVaeGaussianTrainer(BasicTrainer):
    """
    Trainer for structured latent VAE.
    
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
        
        loss_type (str): Loss type. Can be 'l1', 'l2'
        lambda_ssim (float): SSIM loss weight.
        lambda_lpips (float): LPIPS loss weight.
        lambda_kl (float): KL loss weight.
        regularizations (dict): Regularization config.
    """
    
    def __init__(
        self,
        *args,
        loss_type: str = 'l1',
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.2,
        lambda_kl: float = 1e-6,
        regularizations: Dict = {},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        
        self.lambda_kl = lambda_kl
        self.regularizations = regularizations
        
        self._init_renderer()

        # --- reconstruction metrics ---
        # 이미지가 [0,1] 범위라고 가정 (training에서 data_range=1.0 쓰는 것과 일치)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(self.device)
        
    def _init_renderer(self):
        rendering_options = {"near" : 0.8,
                             "far" : 1.6,
                             "bg_color" : 'random'}
        self.renderer = GaussianRenderer(rendering_options)
        self.renderer.pipe.kernel_size = self.models['decoder'].rep_config['2d_filter_kernel_size']
        
    def _render_batch(self, reps: List[Gaussian], extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """
        Render a batch of representations.

        Args:
            reps: The dictionary of lists of representations.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
        """
        ret = None
        for i, representation in enumerate(reps):
            render_pack = self.renderer.render(representation, extrinsics[i], intrinsics[i])
            if ret is None:
                ret = {k: [] for k in list(render_pack.keys()) + ['bg_color']}
            for k, v in render_pack.items():
                ret[k].append(v)
            ret['bg_color'].append(self.renderer.bg_color)
        for k, v in ret.items():
            ret[k] = torch.stack(v, dim=0) 
        return ret
        
    @torch.no_grad()
    def _get_status(self, z: SparseTensor, reps: List[Gaussian]) -> Dict:
        xyz = torch.cat([g.get_xyz for g in reps], dim=0)
        xyz_base = (z.coords[:, 1:].float() + 0.5) / self.models['decoder'].resolution - 0.5
        offset = xyz - xyz_base.unsqueeze(1).expand(-1, self.models['decoder'].rep_config['num_gaussians'], -1).reshape(-1, 3)
        status = {
            'xyz': xyz,
            'offset': offset,
            'scale': torch.cat([g.get_scaling for g in reps], dim=0),
            'opacity': torch.cat([g.get_opacity for g in reps], dim=0),
        }

        for k in list(status.keys()):
            status[k] = {
                'mean': status[k].mean().item(),
                'max': status[k].max().item(),
                'min': status[k].min().item(),
            }
            
        return status
    
    def _get_regularization_loss(self, reps: List[Gaussian]) -> Tuple[torch.Tensor, Dict]:
        loss = 0.0
        terms = {}
        if 'lambda_vol' in self.regularizations:
            scales = torch.cat([g.get_scaling for g in reps], dim=0)   # [N x 3]
            volume = torch.prod(scales, dim=1)  # [N]
            terms[f'reg_vol'] = volume.mean()
            loss = loss + self.regularizations['lambda_vol'] * terms[f'reg_vol']
        if 'lambda_opacity' in self.regularizations:
            opacity = torch.cat([g.get_opacity for g in reps], dim=0)
            terms[f'reg_opacity'] = (opacity - 1).pow(2).mean()
            loss = loss + self.regularizations['lambda_opacity'] * terms[f'reg_opacity']
        return loss, terms
    
    def training_losses(
        self,
        feats: SparseTensor,
        image: torch.Tensor,
        alpha: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        return_aux: bool = True,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            feats: The [N x * x C] sparse tensor of features.
            image: The [N x 3 x H x W] tensor of images.
            alpha: The [N x H x W] tensor of alpha channels.
            extrinsics: The [N x 4 x 4] tensor of extrinsics.
            intrinsics: The [N x 3 x 3] tensor of intrinsics.
            return_aux: Whether to return auxiliary information.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        z, mean, logvar = self.training_models['encoder'](feats, sample_posterior=True, return_raw=True)
        reps = self.training_models['decoder'](z)
        self.renderer.rendering_options.resolution = image.shape[-1]
        render_results = self._render_batch(reps, extrinsics, intrinsics)     
        
        terms = edict(loss = 0.0, rec = 0.0)
        
        rec_image = render_results['color']
        gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * render_results['bg_color'][..., None, None]
                
        if self.loss_type == 'l1':
            terms["l1"] = l1_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l1"]
        elif self.loss_type == 'l2':
            terms["l2"] = l2_loss(rec_image, gt_image)
            terms["rec"] = terms["rec"] + terms["l2"]
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        if self.lambda_ssim > 0:
            terms["ssim"] = 1 - ssim(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_ssim * terms["ssim"]
        if self.lambda_lpips > 0:
            terms["lpips"] = lpips(rec_image, gt_image)
            terms["rec"] = terms["rec"] + self.lambda_lpips * terms["lpips"]
        terms["loss"] = terms["loss"] + terms["rec"]

        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lambda_kl * terms["kl"]
        
        reg_loss, reg_terms = self._get_regularization_loss(reps)
        terms.update(reg_terms)
        terms["loss"] = terms["loss"] + reg_loss
        
        status = self._get_status(z, reps)
        
        if return_aux:
            return terms, status, {'rec_image': rec_image, 'gt_image': gt_image}       
        return terms, status
    
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
        ret_dict = {}
        gt_images = []
        exts = []
        ints = []
        reps = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() for k, v in data.items()}
            gt_images.append(args['image'] * args['alpha'][:, None])
            exts.append(args['extrinsics'])
            ints.append(args['intrinsics'])
            z = self.models['encoder'](args['feats'], sample_posterior=True, return_raw=False)
            reps.extend(self.models['decoder'](z))
        gt_images = torch.cat(gt_images, dim=0)
        ret_dict.update({f'gt_image': {'value': gt_images, 'type': 'image'}})

        # render single view
        exts = torch.cat(exts, dim=0)
        ints = torch.cat(ints, dim=0)
        self.renderer.rendering_options.bg_color = (0, 0, 0)
        self.renderer.rendering_options.resolution = gt_images.shape[-1]
        render_results = self._render_batch(reps, exts, ints)
        ret_dict.update({f'rec_image': {'value': render_results['color'], 'type': 'image'}})

        # render multiview
        # self.renderer.rendering_options.resolution = 512
        ## Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        ## render each view
        miltiview_images = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            extrinsics = extrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            intrinsics = intrinsics.unsqueeze(0).expand(num_samples, -1, -1)
            render_results = self._render_batch(reps, extrinsics, intrinsics)
            miltiview_images.append(render_results['color'])

        ## Concatenate views
        miltiview_images = torch.cat([
            torch.cat(miltiview_images[:2], dim=-2),
            torch.cat(miltiview_images[2:], dim=-2),
        ], dim=-1)
        ret_dict.update({f'miltiview_image': {'value': miltiview_images, 'type': 'image'}})

        self.renderer.rendering_options.bg_color = 'random'
                                    
        return ret_dict
    
    @torch.no_grad()
    def evaluate_reconstruction(
        self,
        verbose: bool = True,
        num_samples: Optional[int] = None,
        to_save: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        현재 self.dataset (예: test set)에 대해 VAE 재구성 품질을 평가.

        - PSNR, SSIM, LPIPS를 전체(또는 num_samples 개) 이미지에 대해 계산.
        - self.dataset 이 이미 test set으로 세팅되어 있고,
        prepare_dataloader() 가 호출되어 있다고 가정하지 않고,
        여기서 batch_size=1인 local dataloader를 새로 만든다.
        - image가 [B, 3, H, W] 라고 가정 (multi-view는 별도 함수에서 처리하는 게 안전).

        Args:
            use_iterator: (호환성용, 현재 구현에서는 사용하지 않음)
            verbose: tqdm 표시 여부.
            num_samples: 평가할 최대 이미지 개수. None이면 전체 데이터 사용.
            to_save: True면 GT/RECON 이미지를 {instance_id}_{gt|recon}.png 로 저장.
            output_dir: to_save=True일 때 이미지를 저장할 디렉토리 경로.

        Returns:
            {
                'psnr': float,
                'ssim': float,
                'lpips': float,
                'num_samples': int,  # 실제 metric에 사용된 이미지 개수
                'num_batches': int,
            }
        """
        if to_save and output_dir is None:
            raise ValueError("to_save=True 이면 output_dir을 꼭 지정해줘야 합니다.")

        if to_save:
            os.makedirs(output_dir, exist_ok=True)

        # metric state 초기화
        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.lpips_metric.reset()

        # eval mode
        self.models['encoder'].eval()
        self.models['decoder'].eval()

        num_used_images = 0   # 실제 metric에 들어간 이미지 수 (batch_size=1이면 == num_batches)
        num_batches = 0

        # 평가용 dataloader (batch_size=1, shuffle=False)
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        if num_samples is None:
            total_batches = len(dataloader)
        else:
            total_batches = min(len(dataloader), num_samples)

        if verbose:
            batch_iter = tqdm(
                range(total_batches),
                desc="Eval reconstruction",
                total=total_batches,
            )
        else:
            batch_iter = range(total_batches)

        dataloader_iter = iter(dataloader)

        for batch_idx in batch_iter:
            data = next(dataloader_iter)   # batch_size=1이므로 OK

            # batch_size=1로 만들었으니 여기서 B=1일 것.
            # 그래도 일반성을 위해 B를 추적.
            # GPU로 이동 (SparseTensor는 .to 사용, 나머지는 .to(device))
            batch = {}
            for k, v in data.items():
                if k == 'instance_id':
                    # 문자열 / 리스트일 수 있으니 그대로 둔다
                    batch[k] = v
                else:
                    # Tensor 또는 SparseTensor (SparseTensor도 .to 지원)
                    if hasattr(v, 'to'):
                        batch[k] = v.to(self.device)
                    else:
                        batch[k] = v

            feats = batch['feats']                 # SparseTensor
            image = batch['image']                 # [B, 3, H, W], [0,1]
            alpha = batch['alpha']                 # [B, H, W]
            extrinsics = batch['extrinsics']       # [B, 4, 4]
            intrinsics = batch['intrinsics']       # [B, 3, 3]

            B = image.shape[0]

            # num_samples 제한 고려: 이 batch에서 실제 사용할 개수 계산
            remain = None if num_samples is None else max(num_samples - num_used_images, 0)
            if remain is not None and remain == 0:
                break
            if remain is not None and remain < B:
                # 안전하게 앞에서부터 remain개만 사용 (지금은 B=1이라 사실상 영향 없음)
                image = image[:remain]
                alpha = alpha[:remain]
                extrinsics = extrinsics[:remain]
                intrinsics = intrinsics[:remain]
                # feats도 필요하다면 잘라야 하는데, SparseTensor라면
                # 보통 coords/feats에 instance별 layout 정보가 있으니
                # 여기서는 batch_size=1 가정하므로 슬라이스는 생략.

                B = remain

            # --- VAE 인퍼런스 (posterior mean 사용) ---
            z = self.models['encoder'](
                feats,
                sample_posterior=False,
                return_raw=False,
            )
            reps = self.models['decoder'](z)

            # --- 3DGS 렌더 ---
            self.renderer.rendering_options.resolution = image.shape[-1]
            render_results = self._render_batch(reps, extrinsics, intrinsics)

            rec_image = render_results['color']              # [B, 3, H, W]
            bg_color = render_results['bg_color'][..., None, None]  # [B, 3, 1, 1]

            # training_losses의 정의를 그대로 따라 GT 구성
            gt_image = image * alpha[:, None] + (1 - alpha[:, None]) * bg_color

            rec_image = rec_image.clamp(0.0, 1.0)
            gt_image = gt_image.clamp(0.0, 1.0)

            # --- metric 업데이트 ---
            self.psnr_metric.update(rec_image, gt_image)
            self.ssim_metric.update(rec_image, gt_image)
            self.lpips_metric.update(rec_image, gt_image)

            # --- 이미지 저장 옵션 ---
            if to_save:
                # instance_id 가져오기 (있으면 그걸 쓰고, 없으면 index 기반 이름)
                if 'instance_id' in batch:
                    ids = batch['instance_id']
                    # collate_fn에 따라 형태가 달라질 수 있어서 다 처리
                    if isinstance(ids, (list, tuple)):
                        instance_ids = ids
                    else:
                        # tensor or 단일 문자열일 가능성
                        try:
                            # tensor일 경우
                            if hasattr(ids, 'tolist'):
                                instance_ids = ids.tolist()
                            else:
                                instance_ids = [ids]
                        except Exception:
                            instance_ids = [str(ids)]
                else:
                    # fallback: dataloader 상의 index로 이름 부여
                    instance_ids = [f"{batch_idx:06d}"]

                # B와 instance_ids 길이를 맞춰서 저장
                for i in range(B):
                    if i < len(instance_ids):
                        inst_id = instance_ids[i]
                    else:
                        inst_id = f"{batch_idx:06d}_{i}"

                    # [3, H, W] 한 장씩 꺼내서 저장
                    gt_img_i = gt_image[i].detach().cpu()
                    rec_img_i = rec_image[i].detach().cpu()

                    gt_path = os.path.join(output_dir, f"{inst_id}_gt.png")
                    rec_path = os.path.join(output_dir, f"{inst_id}_recon.png")

                    save_image(gt_img_i, gt_path)
                    save_image(rec_img_i, rec_path)

            num_used_images += B
            num_batches += 1

            if num_samples is not None and num_used_images >= num_samples:
                break

        # compute 최종 값
        psnr = self.psnr_metric.compute().item()
        ssim = self.ssim_metric.compute().item()
        lpips = self.lpips_metric.compute().item()

        return {
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "num_samples": num_used_images,
            "num_batches": num_batches,
        }
import argparse
import json
import os
import sys
sys.path.append("/root/dev/TRELLIS")
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import utils3d
from tqdm import tqdm

from trellis.pipelines import TrellisTextUncond3DPipeline
from trellis.utils import render_utils

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def _deg2rad_if_needed(value: float) -> float:
    return float(np.deg2rad(value) if value > np.pi else value)


def rotation_from_forward_vec(
    forward_vec: np.ndarray,
    up_axis: str = 'Y',
    inplane_rot: float = None,
) -> np.ndarray:
    """Match TRELLIS/3DGS-Gen 3D-FRONT camera rotation construction."""
    forward = np.array(forward_vec, dtype=np.float64)
    forward = forward / np.linalg.norm(forward, axis=1, keepdims=True)

    if up_axis.upper() == 'Y':
        up = np.array([0.0, 1.0, 0.0])
    elif up_axis.upper() == 'Z':
        up = np.array([0.0, 0.0, 1.0])
    elif up_axis.upper() == 'X':
        up = np.array([1.0, 0.0, 0.0])
    else:
        raise ValueError("Invalid up_axis. Choose from 'X', 'Y', or 'Z'.")

    right = np.cross(forward, up)
    right /= np.linalg.norm(right, axis=1, keepdims=True)

    up_true = np.cross(right, forward)
    up_true /= np.linalg.norm(up_true, axis=1, keepdims=True)

    rotation = np.stack((right, up_true, -forward), axis=-1)

    if inplane_rot is not None:
        c, s = np.cos(inplane_rot), np.sin(inplane_rot)
        rot_z = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ])
        rotation = rotation @ rot_z
    return rotation


def list_shapenet(points_dir: Path, images_dir: Path, category: str = None, instance: str = None) -> List[Dict]:
    items = []
    categories = [category] if category else sorted(p.name for p in points_dir.iterdir() if p.is_dir())
    for cat in categories:
        cat_dir = points_dir / cat
        if not cat_dir.is_dir():
            continue
        instances = [instance] if instance else sorted(p.name for p in cat_dir.iterdir() if p.is_dir())
        for inst in instances:
            ply = cat_dir / inst / 'voxelized_pc.ply'
            image_root = images_dir / cat / inst / 'rendered_images'
            if _safe_exists(ply) and image_root.is_dir():
                items.append({
                    'dataset': 'shapenet',
                    'category': cat,
                    'instance': inst,
                    'points_path': ply,
                    'camera_root': image_root,
                })
    return items


def list_gobjaverse(
    points_dir: Path,
    images_dir: Path,
    category: str = None,
    sub_category: str = None,
    instance: str = None,
) -> List[Dict]:
    items = []
    categories = [category] if category else sorted(p.name for p in points_dir.iterdir() if p.is_dir())
    for cat in categories:
        cat_dir = points_dir / cat
        if not cat_dir.is_dir():
            continue
        sub_categories = [sub_category] if sub_category else sorted(p.name for p in cat_dir.iterdir() if p.is_dir())
        for sub in sub_categories:
            sub_dir = cat_dir / sub
            if not sub_dir.is_dir():
                continue
            instances = [instance] if instance else sorted(p.name for p in sub_dir.iterdir() if p.is_dir())
            for inst in instances:
                ply = sub_dir / inst / 'voxelized_pc.ply'
                image_root = images_dir / cat / sub / inst / inst / 'campos_512_v4'
                if _safe_exists(ply) and image_root.is_dir():
                    items.append({
                        'dataset': 'gobjaverse',
                        'category': cat,
                        'sub_category': sub,
                        'instance': inst,
                        'points_path': ply,
                        'camera_root': image_root,
                    })
    return items


def list_3dfront(points_dir: Path, images_dir: Path, labels_dir: Path, instance: str = None) -> List[Dict]:
    items = []
    instances = [instance] if instance else sorted(p.name for p in points_dir.iterdir() if p.is_dir())
    for inst in instances:
        point_dir = points_dir / inst
        ply = point_dir / 'voxelized_pc.ply'
        h_centering = point_dir / 'h_centering.txt'
        image_root = images_dir / inst
        camera_npz = labels_dir / inst / 'boxes.npz'
        if all(_safe_exists(p) for p in [ply, h_centering, camera_npz]) and image_root.is_dir():
            items.append({
                'dataset': '3dfront',
                'instance': inst,
                'points_path': ply,
                'camera_root': image_root,
                'camera_npz': camera_npz,
                'h_centering_path': h_centering,
            })
    return items


def load_cameras_shapenet(camera_root: Path, n_views: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    extrinsics, intrinsics, metadata = [], [], []
    for i in range(n_views):
        stem = f'{i:05d}'
        png_path = camera_root / f'{stem}.png'
        json_path = camera_root / f'{stem}.json'
        if not (_safe_exists(png_path) and _safe_exists(json_path)):
            continue
        with open(json_path, 'r') as fp:
            meta = json.load(fp)

        if 'transform_matrix' in meta:
            c2w = np.array(meta['transform_matrix'], dtype=np.float32)
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w).astype(np.float32)
            fov_x = meta.get('camera_angle_x', np.deg2rad(70.0))
            fov_y = fov_x
        elif all(key in meta for key in ('x', 'y', 'z', 'origin')):
            r1, r2, r3, t = map(
                lambda value: np.array(value, dtype=np.float32)[None, :].T,
                [meta['x'], meta['y'], meta['z'], meta['origin']],
            )
            c2w = np.vstack([
                np.hstack([r1, r2, r3, t]),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ])
            w2c = np.linalg.inv(c2w).astype(np.float32)
            fov_x = _deg2rad_if_needed(meta.get('x_fov', np.deg2rad(70.0)))
            fov_y = _deg2rad_if_needed(meta.get('y_fov', np.deg2rad(70.0)))
        else:
            raise ValueError(f'Unsupported ShapeNet camera format: {json_path}')

        extrinsics.append(torch.from_numpy(w2c))
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(float(fov_x), dtype=torch.float32),
            torch.tensor(float(fov_y), dtype=torch.float32),
        ))
        metadata.append({'view_index': i, 'camera_json': str(json_path), 'image_path': str(png_path)})
    return extrinsics, intrinsics, metadata


def load_cameras_gobjaverse(camera_root: Path, n_views: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    extrinsics, intrinsics, metadata = [], [], []
    for i in range(n_views):
        stem = f'{i:05d}'
        view_dir = camera_root / stem
        png_path = view_dir / f'{stem}.png'
        json_path = view_dir / f'{stem}.json'
        if not (_safe_exists(png_path) and _safe_exists(json_path)):
            continue
        with open(json_path, 'r') as fp:
            meta = json.load(fp)

        r1, r2, r3, t = map(
            lambda value: np.array(value, dtype=np.float32)[None, :].T,
            [meta['x'], meta['y'], meta['z'], meta['origin']],
        )
        c2w = np.vstack([
            np.hstack([r1, r2, r3, t]),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ])
        w2c = np.linalg.inv(c2w).astype(np.float32)
        fov_x = _deg2rad_if_needed(meta.get('x_fov', np.deg2rad(70.0)))
        fov_y = _deg2rad_if_needed(meta.get('y_fov', np.deg2rad(70.0)))

        extrinsics.append(torch.from_numpy(w2c))
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(float(fov_x), dtype=torch.float32),
            torch.tensor(float(fov_y), dtype=torch.float32),
        ))
        metadata.append({'view_index': i, 'camera_json': str(json_path), 'image_path': str(png_path)})
    return extrinsics, intrinsics, metadata


def load_cameras_3dfront(
    camera_root: Path,
    camera_npz_path: Path,
    h_centering_path: Path,
    n_views: int,
    fov_deg: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    with open(h_centering_path, 'r') as fp:
        h_centering = float(fp.readline().strip())

    cam = np.load(camera_npz_path, allow_pickle=True)
    if 'camera_coords' not in cam or 'target_coords' not in cam:
        raise KeyError(f"{camera_npz_path} missing 'camera_coords'/'target_coords'")

    camera_coords_all = cam['camera_coords']
    target_coords_all = cam['target_coords']
    extrinsics, intrinsics, metadata = [], [], []

    for i in range(n_views):
        png_path = camera_root / f'{i:04d}_colors.png'
        if not _safe_exists(png_path):
            continue

        target = target_coords_all[i]
        camera = camera_coords_all[i]
        target = np.array([target[0], target[1] - h_centering, target[2]], dtype=np.float32)
        camera = np.array([camera[0], camera[1] - h_centering, camera[2]], dtype=np.float32)
        forward = target - camera
        rotation = -rotation_from_forward_vec(forward[None, ...])[0]

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = camera
        w2c = np.linalg.inv(c2w).astype(np.float32)

        extrinsics.append(torch.from_numpy(w2c))
        intrinsics.append(utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(np.deg2rad(fov_deg), dtype=torch.float32),
            torch.tensor(np.deg2rad(fov_deg), dtype=torch.float32),
        ))
        metadata.append({
            'view_index': i,
            'image_path': str(png_path),
            'camera_npz': str(camera_npz_path),
            'h_centering': h_centering,
        })
    return extrinsics, intrinsics, metadata


def load_item_cameras(item: Dict, n_views: int, fov3dfront_deg: float) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    dataset = item['dataset']
    if dataset == 'shapenet':
        return load_cameras_shapenet(item['camera_root'], n_views)
    if dataset == 'gobjaverse':
        return load_cameras_gobjaverse(item['camera_root'], n_views)
    if dataset == '3dfront':
        return load_cameras_3dfront(item['camera_root'], item['camera_npz'], item['h_centering_path'], n_views, fov3dfront_deg)
    raise ValueError(f'Unsupported dataset: {dataset}')




def get_camera_pool_limit(dataset: str) -> int:
    if dataset == 'shapenet':
        return 24
    if dataset in ('gobjaverse', '3dfront'):
        return 40
    raise ValueError(f'Unsupported dataset: {dataset}')


def select_camera_views(
    extrinsics: Sequence[torch.Tensor],
    intrinsics: Sequence[torch.Tensor],
    metadata: Sequence[Dict],
    n_views: int,
    mode: str,
    seed: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Dict]]:
    total = len(extrinsics)
    if total == 0:
        raise ValueError('No cameras were loaded from the source item.')
    if n_views > total:
        raise ValueError(f'Requested n_views={n_views}, but only {total} cameras are available.')

    if mode == 'first':
        indices = list(range(n_views))
    elif mode == 'last':
        indices = list(range(total - n_views, total))
    elif mode == 'uniform':
        if n_views == 1:
            indices = [0]
        else:
            indices = np.linspace(0, total - 1, num=n_views)
            indices = np.round(indices).astype(int).tolist()
    elif mode == 'random':
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=n_views, replace=False).tolist()
    else:
        raise ValueError(f'Unsupported camera_mode: {mode}')

    selected_extrinsics = [extrinsics[i] for i in indices]
    selected_intrinsics = [intrinsics[i] for i in indices]
    selected_metadata = []
    for out_idx, src_idx in enumerate(indices):
        meta = dict(metadata[src_idx])
        meta['source_view_index'] = meta.get('view_index', src_idx)
        meta['selected_order'] = out_idx
        selected_metadata.append(meta)
    return selected_extrinsics, selected_intrinsics, selected_metadata

def validate_camera_batch(extrinsics: Sequence[torch.Tensor], intrinsics: Sequence[torch.Tensor], expected_views: int, item: Dict) -> None:
    if len(extrinsics) < expected_views or len(intrinsics) < expected_views:
        raise ValueError(
            f"Camera set for {item['instance']} has only {len(extrinsics)} valid views; expected at least {expected_views}."
        )
    for idx, (extr, intr) in enumerate(zip(extrinsics, intrinsics)):
        if extr.shape != (4, 4):
            raise ValueError(f'Invalid extrinsics shape at view {idx}: {tuple(extr.shape)}')
        if intr.shape != (3, 3):
            raise ValueError(f'Invalid intrinsics shape at view {idx}: {tuple(intr.shape)}')
        if not torch.isfinite(extr).all():
            raise ValueError(f'Non-finite extrinsics detected at view {idx}')
        if not torch.isfinite(intr).all():
            raise ValueError(f'Non-finite intrinsics detected at view {idx}')


def discover_items(args: argparse.Namespace) -> List[Dict]:
    points_dir = Path(args.points_dir)
    images_dir = Path(args.images_dir)
    dataset = args.dataset.lower()
    if dataset == 'shapenet':
        items = list_shapenet(points_dir, images_dir, category=args.category, instance=args.instance)
    elif dataset == 'gobjaverse':
        items = list_gobjaverse(points_dir, images_dir, category=args.category, sub_category=args.sub_category, instance=args.instance)
    elif dataset == '3dfront':
        if not args.labels_dir:
            raise ValueError('--labels_dir is required for 3dfront')
        items = list_3dfront(points_dir, images_dir, Path(args.labels_dir), instance=args.instance)
    else:
        raise ValueError(f'Unsupported dataset: {args.dataset}')

    if len(items) == 0:
        raise ValueError('No dataset items found with the provided roots/options.')
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate TRELLIS text-unconditioned 3D samples and render them with dataset cameras.')
    parser.add_argument('--pipeline_path', type=str, required=True, help='Path to a TRELLIS text-unconditioned pipeline folder or Hugging Face repo.')
    parser.add_argument('--gpu', type=int, default=None, help='CUDA device index to use. Defaults to the current CUDA default device.')
    parser.add_argument('--dataset', choices=['shapenet', 'gobjaverse', '3dfront'], required=True, help='Dataset camera convention to use.')
    parser.add_argument('--points_dir', type=str, required=True, help='Root directory used to discover valid dataset instances and metadata.')
    parser.add_argument('--images_dir', type=str, required=True, help='Root directory containing rendered images and camera files.')
    parser.add_argument('--labels_dir', type=str, default=None, help='3D-FRONT labels root containing boxes.npz; required only for 3dfront.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where rendered PNGs and metadata will be saved.')
    parser.add_argument('--prompt', type=str, default='null', help='Prompt passed to TrellisTextUncond3DPipeline.run().')
    parser.add_argument('--n_items', type=int, default=2500, help='Number of generated samples to render.')
    parser.add_argument('--n_views', type=int, default=20, help='Number of dataset camera views to use per sample.')
    parser.add_argument('--start_index', type=int, default=0, help='Starting output sample index.')
    parser.add_argument('--seed_base', type=int, default=0, help='Base seed; sample i uses seed_base + i.')
    parser.add_argument('--camera_source_index', type=int, default=0, help='Index of the dataset instance used to source the camera set.')
    parser.add_argument('--camera_mode', choices=['random', 'first', 'last', 'uniform'], default='first', help='How to select n_views cameras from the source item camera pool.')
    parser.add_argument('--camera_seed', type=int, default=0, help='Seed used when camera_mode=random.')
    parser.add_argument('--category', type=str, default=None, help='Optional category filter for ShapeNet or GObjaverse.')
    parser.add_argument('--sub_category', type=str, default=None, help='Optional sub-category filter for GObjaverse.')
    parser.add_argument('--instance', type=str, default=None, help='Optional single instance filter.')
    parser.add_argument('--resolution', type=int, default=512, help='Render resolution.')
    parser.add_argument('--bg_color', type=float, nargs=3, default=(1.0, 1.0, 1.0), help='Renderer background color as three floats in [0, 1].')
    parser.add_argument('--near', type=float, default=0.8, help='Renderer near plane.')
    parser.add_argument('--far', type=float, default=1.6, help='Renderer far plane.')
    parser.add_argument('--ssaa', type=int, default=1, help='Renderer SSAA factor for Gaussian rendering.')
    parser.add_argument('--kernel_size', type=float, default=0.1, help='Gaussian renderer kernel size.')
    parser.add_argument('--fov3dfront_deg', type=float, default=70.0, help='Symmetric FOV used for 3D-FRONT camera loading.')
    parser.add_argument('--skip_existing', action='store_true', help='Skip a sample if all expected view PNGs already exist.')
    parser.add_argument('--save_ply', action='store_true', help='Also save the generated Gaussian as a PLY file for each sample.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to run this script, but no CUDA device is available.')
    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cuda')
    torch.cuda.set_device(device)

    pipeline = TrellisTextUncond3DPipeline.from_pretrained(args.pipeline_path)
    pipeline.to(device)

    items = discover_items(args)
    if args.camera_source_index < 0 or args.camera_source_index >= len(items):
        raise ValueError(
            f'camera_source_index={args.camera_source_index} is out of range for {len(items)} discovered items.'
        )
    camera_source_item = items[args.camera_source_index]
    source_extrinsics, source_intrinsics, source_camera_meta = load_item_cameras(
        camera_source_item,
        get_camera_pool_limit(args.dataset),
        args.fov3dfront_deg,
    )
    camera_extrinsics, camera_intrinsics, camera_meta = select_camera_views(
        source_extrinsics,
        source_intrinsics,
        source_camera_meta,
        args.n_views,
        args.camera_mode,
        args.camera_seed,
    )
    validate_camera_batch(camera_extrinsics, camera_intrinsics, args.n_views, camera_source_item)
    camera_extrinsics = [tensor.to(device) for tensor in camera_extrinsics]
    camera_intrinsics = [tensor.to(device) for tensor in camera_intrinsics]

    manifest = {
        'pipeline_path': args.pipeline_path,
        'gpu': args.gpu,
        'dataset': args.dataset,
        'prompt': args.prompt,
        'n_items': args.n_items,
        'n_views': args.n_views,
        'start_index': args.start_index,
        'seed_base': args.seed_base,
        'camera_source_index': args.camera_source_index,
        'camera_mode': args.camera_mode,
        'camera_seed': args.camera_seed,
        'camera_source_item': {key: str(value) for key, value in camera_source_item.items()},
        'resolution': args.resolution,
        'bg_color': list(map(float, args.bg_color)),
        'near': args.near,
        'far': args.far,
        'ssaa': args.ssaa,
        'kernel_size': args.kernel_size,
        'fov3dfront_deg': args.fov3dfront_deg,
    }
    with open(output_dir / 'manifest.json', 'w') as fp:
        json.dump(manifest, fp, indent=2)

    progress = tqdm(range(args.n_items), desc=f'Sample {args.start_index}/{args.start_index + args.n_items}')
    for sample_offset in progress:
        sample_idx = args.start_index + sample_offset
        seed = args.seed_base + sample_offset
        progress.set_description(f'Sample {sample_idx + 1}/{args.start_index + args.n_items}')
        sample_dir = output_dir / f'{sample_idx:05d}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        expected_pngs = [sample_dir / f'{view_idx:05d}.png' for view_idx in range(args.n_views)]
        if args.skip_existing and all(path.exists() for path in expected_pngs):
            continue

        with torch.no_grad():
            outputs = pipeline.run(
                prompt=args.prompt,
                num_samples=1,
                seed=seed,
                formats=['gaussian'],
                verbose=False,
            )
            gaussian = outputs['gaussian'][0]
            rendered = render_utils.render_frames(
                gaussian,
                camera_extrinsics,
                camera_intrinsics,
                options={
                    'resolution': args.resolution,
                    'bg_color': tuple(args.bg_color),
                    'near': args.near,
                    'far': args.far,
                    'ssaa': args.ssaa,
                    'kernel_size': args.kernel_size,
                },
                verbose=False,
            )

        colors = rendered.get('color', [])
        if len(colors) != args.n_views:
            raise RuntimeError(f'Rendered {len(colors)} views for sample {sample_idx}, expected {args.n_views}.')

        for view_idx, image in enumerate(colors):
            Image.fromarray(image).save(sample_dir / f'{view_idx:05d}.png')

        if args.save_ply:
            gaussian.save_ply(str(sample_dir / 'gaussian.ply'))

        sample_meta = {
            'sample_index': sample_idx,
            'seed': seed,
            'prompt': args.prompt,
            'camera_source_item': {key: str(value) for key, value in camera_source_item.items()},
            'camera_mode': args.camera_mode,
            'camera_meta': camera_meta[:args.n_views],
        }
        with open(sample_dir / 'meta.json', 'w') as fp:
            json.dump(sample_meta, fp, indent=2)


if __name__ == '__main__':
    main()




"""
[shapenet]
cd /root/dev/TRELLIS
python evaluation/generation/generate_text_uncond_renders.py \
  --pipeline_path /path/to/trellis_text_uncond_pipeline \
  --dataset shapenet \
  --points_dir /path/to/shapenet_vox_points \
  --images_dir /path/to/shapenet_images \
  --output_dir /path/to/output \
  --prompt "null" \
  --n_items 2500 \
  --n_views 20 \
  --camera_source_index 0 \
  --camera_mode uniform \
  --save_ply \
  --gpu 0

[gobjaverse]
cd /root/dev/TRELLIS
python evaluation/generation/generate_text_uncond_renders.py \
  --pipeline_path /path/to/trellis_text_uncond_pipeline \
  --dataset gobjaverse \
  --points_dir /path/to/gobjaverse_vox_points \
  --images_dir /path/to/gobjaverse_images \
  --output_dir /path/to/output \
  --prompt "null" \
  --n_items 2500 \
  --n_views 20 \
  --camera_source_index 0 \
  --camera_mode uniform \
  --save_ply \
  --gpu 0

[3dfront]
cd /root/dev/TRELLIS
python evaluation/generation/generate_text_uncond_renders.py \
  --pipeline_path /path/to/trellis_text_uncond_pipeline \
  --dataset 3dfront \
  --points_dir /path/to/3dfront_vox_points \
  --images_dir /path/to/3dfront_images \
  --labels_dir /path/to/3dfront_labels \
  --output_dir /path/to/output \
  --prompt "null" \
  --n_items 2500 \
  --n_views 20 \
  --camera_source_index 0 \
  --camera_mode last \
  --gpu 0 \
  --resolution 256 \
  --save_ply \
  --fov3dfront_deg 70
"""
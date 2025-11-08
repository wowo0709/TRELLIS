import json
import os
from PIL import Image
from typing import Optional

import numpy as np
import pandas as pd
import torch
import utils3d.torch

from ..modules.sparse.basic import SparseTensor
from .components import CustomStandardDatasetBase



def rotation_from_forward_vec(
        forward_vec: np.ndarray, 
        up_axis: str = 'Y', 
        inplane_rot: Optional[float] = None,
    ) -> np.ndarray:
        """ Returns a camera rotation matrix for the given forward vector and up axis using NumPy
        :param forward_vec: The forward vector which specifies the direction the camera should look.
        :param up_axis: The up axis, usually Y.
        :param inplane_rot: The in-plane rotation in radians. If None is given, the in-plane rotation is determined only
                            based on the up vector.
        :return: The corresponding rotation matrix.
        """
        # Normalize the forward vector
        forward_vector = np.array(forward_vec, dtype=np.float64)
        forward_vector_norm = forward_vector / np.linalg.norm(forward_vector, axis=1, keepdims=True)

        # forward_vec = forward_vec / np.linalg.norm(forward_vec)

        # Define the up vector
        if up_axis.upper() == 'Y':
            up_vec = np.array([0.0, 1.0, 0.0])
        elif up_axis.upper() == 'Z':
            up_vec = np.array([0.0, 0.0, 1.0])
        elif up_axis.upper() == 'X':
            up_vec = np.array([1.0, 0.0, 0.0])
        else:
            raise ValueError("Invalid up_axis. Choose from 'X', 'Y', or 'Z'.")

        # Compute the right vector (cross product of forward and up)
        right_vec = np.cross(forward_vector_norm, up_vec)   # right-hand
        # right_vec = np.cross(up_vec, forward_vector_norm) # left-hand
        right_vec /= np.linalg.norm(right_vec, axis=1, keepdims=True)

        # Recompute the true up vector (orthogonal to forward and right)
        up_vec = np.cross(right_vec, forward_vector_norm) # right-hand
        # up_vec = np.cross(forward_vector_norm, right_vec) # left-hand
        up_vec /= np.linalg.norm(up_vec, axis=1, keepdims=True)

        # # Recompute the right vector
        # right_vec = np.cross(forward_vector_norm, up_vec)
        # right_vec /= np.linalg.norm(right_vec, axis=1, keepdims=True)

        # Construct the rotation matrix (columns represent right, up, forward)
        rotation_matrix = np.stack((right_vec, up_vec, -forward_vector_norm), axis=-1)
        # rotation_matrix = np.stack((right_vec, up_vec, -forward_vector_norm), axis=1)

        # Apply in-plane rotation if specified
        if inplane_rot is not None:
            inplane_rotation = np.array([
                [np.cos(inplane_rot), -np.sin(inplane_rot), 0],
                [np.sin(inplane_rot),  np.cos(inplane_rot), 0],
                [0,                   0,                   1]
            ])
            rotation_matrix = rotation_matrix @ inplane_rotation

        return rotation_matrix


class CustomSparseFeat2Render(CustomStandardDatasetBase):
    """
    SparseFeat2Render dataset.
    
    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        model (str): model name
        resolution (int): resolution of the data
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        data_type: str, 
        data_category: str,
        image_size: int,
        n_views: int,
        model: str = 'dinov2_vitl14_reg',
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        cam_path_3dfront: str = None,
    ):
        self.image_size = image_size
        self.n_views = n_views
        self.model = model
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        self.cam_path_3dfront = cam_path_3dfront
        
        super().__init__(roots, data_type, data_category)

        # SLAT VAE use rendering & dino_feature data
        assert len(self.roots) % 2 == 0

        assert self.data_type in ["shapenet", "gobjaverse", "3dfront"]
        
        if self.data_type == "3dfront":
            assert self.cam_path_3dfront is not None

        self.roots[1::2] = list(map(lambda x: os.path.join(x, self.model), self.roots[1::2]))
        self.images_root = self.roots[::2]
        self.features_root = self.roots[1::2]
        for images_root, features_root in zip(self.images_root, self.features_root):
            if self.data_type == "shapenet":
                data_root = os.path.join(features_root, self.data_category)
                for instanceID in os.listdir(data_root):
                    if not os.path.exists(os.path.join(data_root, instanceID, "features.npz")):
                        continue
                    self.instances.append(
                        (
                            [
                                os.path.join(images_root, self.data_category), 
                                os.path.join(features_root, self.data_category)
                            ], 
                            instanceID
                        )
                    )
            elif self.data_type == "gobjaverse":
                data_root = os.path.join(features_root, self.data_category)
                for sub_category in os.listdir(data_root):
                    for instanceID in os.listdir(os.path.join(data_root, sub_category)):
                        if not os.path.exists(os.path.join(data_root, sub_category, instanceID, "features.npz")):
                            continue
                        self.instances.append(
                            (
                                [
                                    os.path.join(images_root, self.data_category, sub_category), 
                                    os.path.join(features_root, self.data_category, sub_category)
                                ], 
                                instanceID
                            )
                        )
            elif self.data_type == "3dfront":
                data_root = features_root
                for instanceID in os.listdir(data_root):
                    if not os.path.exists(os.path.join(data_root, instanceID, "features.npz")):
                        continue
                    self.instances.append(
                        (
                            [
                                images_root, 
                                features_root,
                            ], 
                            instanceID
                        )
                    )

    # Renderings
    def _get_image(self, root, instance, kwargs: Optional[dict] = None):
        # with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
        #     metadata = json.load(f)
        # n_views = len(metadata['frames'])
        # view = np.random.randint(n_views)
        # metadata = metadata['frames'][view]
        # fov = metadata['camera_angle_x']
        # intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        # c2w = torch.tensor(metadata['transform_matrix'])
        # c2w[:3, 1:3] *= -1
        # extrinsics = torch.inverse(c2w)
        view = np.random.randint(self.n_views)
        if self.data_type == "shapenet" or self.data_type == "gobjaverse":
            camera_path = os.path.join(root, instance, "rendered_images", f"{view:05d}.json") if self.data_type == "shapenet" \
                                    else os.path.join(root, instance, instance, "campos_512_v4", f"{view:05d}", f"{view:05d}.json")
            image_path = os.path.join(root, instance, "rendered_images", f"{view:05d}.png") if self.data_type == "shapenet" \
                                    else os.path.join(root, instance, instance, "campos_512_v4", f"{view:05d}", f"{view:05d}.png")

            with open(camera_path) as f:
                metadata = json.load(f)
            r1, r2, r3, t = map(
                lambda x: np.array(x, dtype=np.float32)[None, :].T, 
                [metadata["x"], metadata["y"], metadata["z"], metadata["origin"]]
            )
            c2w = np.vstack([np.hstack([r1, r2, r3, t]), np.array([0., 0., 0., 1.], dtype=np.float32)])
            w2c = np.linalg.inv(c2w).astype(np.float32)
            fov_x = metadata["x_fov"]
            fov_y = metadata["y_fov"]
        elif self.data_type == "3dfront":
            assert 'h_centering' in kwargs, "Key 'h_centering' should exist for 3dfront dataset"
            h_centering = kwargs['h_centering']

            camera_path = os.path.join(self.cam_path_3dfront, instance, "boxes.npz")
            image_path = os.path.join(root, instance, f"{view:04d}_colors.png")

            metadata = np.load(camera_path, allow_pickle=True)
            target_coords = metadata["target_coords"][view]
            camera_coords = metadata["camera_coords"][view]
            target_coords = np.array([target_coords[0], target_coords[1] - h_centering, target_coords[2]])
            camera_coords = np.array([camera_coords[0], camera_coords[1] - h_centering, camera_coords[2]])
            forward_vec = target_coords - camera_coords

            rotation_matrix = -rotation_from_forward_vec(forward_vec[None, ...])[0]
            translation_vector = camera_coords
            c2w = np.eye(4)
            c2w[:3, :3] = rotation_matrix
            c2w[:3, 3] = translation_vector
            w2c = np.linalg.inv(c2w)
            fov_x = fov_y = np.deg2rad(70.0)

        extrinsics = torch.from_numpy(w2c).to(torch.float32)
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(fov_x, dtype=torch.float32), 
            torch.tensor(fov_y, dtype=torch.float32)
        )
        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }
    
    # DINO feature
    def _get_feat(self, root, instance, kwargs: Optional[dict] = None):
        DATA_RESOLUTION = 64
        feats_path = os.path.join(root, instance, "features.npz")
        feats = np.load(feats_path, allow_pickle=True)
        coords = torch.tensor(feats['indices']).int()
        feats = torch.tensor(feats['patchtokens']).float()
        
        if self.resolution != DATA_RESOLUTION:
            factor = DATA_RESOLUTION // self.resolution
            coords = coords // factor
            coords, idx = coords.unique(return_inverse=True, dim=0)
            feats = torch.scatter_reduce(
                torch.zeros(coords.shape[0], feats.shape[1], device=feats.device),
                dim=0,
                index=idx.unsqueeze(-1).expand(-1, feats.shape[1]),
                src=feats,
                reduce='mean'
            )
        
        return {
            'coords': coords,
            'feats': feats,
        }

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        return sample['image']

    @staticmethod
    def collate_fn(batch):
        pack = {}
        coords = []
        for i, b in enumerate(batch):
            coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
        coords = torch.cat(coords)
        feats = torch.cat([b['feats'] for b in batch])
        pack['feats'] = SparseTensor(
            coords=coords,
            feats=feats,
        )
        
        pack['image'] = torch.stack([b['image'] for b in batch])
        pack['alpha'] = torch.stack([b['alpha'] for b in batch])
        pack['extrinsics'] = torch.stack([b['extrinsics'] for b in batch])
        pack['intrinsics'] = torch.stack([b['intrinsics'] for b in batch])

        return pack

    def get_instance(self, root, instance):
        meta = {}
        if self.data_type == "3dfront":
            hc_path = os.path.join(root[1], instance, "h_centering.txt")
            if not os.path.exists(hc_path):
                raise FileNotFoundError(f"h_centering.txt not found for 3D-FRONT instance {instance}: {hc_path}")
            with open(hc_path, "r") as f:
                raw_val = f.read().strip()
            try:
                meta['h_centering'] = float(raw_val)
            except ValueError:
                raise ValueError(f"Invalid h_centering value in {hc_path}: {raw_val!r}")
        image = self._get_image(root[0], instance, meta)
        feat = self._get_feat(root[1], instance, meta)
        return {
            **image,
            **feat,
        }

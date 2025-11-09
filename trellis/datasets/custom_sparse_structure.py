import os
import json
from typing import Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import utils3d
from .components import CustomStandardDatasetBase
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer



class CustomSparseStructure(CustomStandardDatasetBase):
    """
    Sparse structure dataset

    Args:
        roots (str): path to the dataset
        resolution (int): resolution of the voxel grid
        min_aesthetic_score (float): minimum aesthetic score of the instances to be included in the dataset
    """

    def __init__(self,
        roots,
        data_type,
        data_category,
        resolution: int = 64,
        min_aesthetic_score: float = 5.0,
        aabb: list = [-0.5, -0.5, -0.5, 1.0, 1.0, 1.0]
    ):
        self.resolution = resolution
        self.min_aesthetic_score = min_aesthetic_score
        self.aabb = aabb
        self.value_range = (0, 1)

        super().__init__(roots, data_type, data_category)

        assert self.data_type in ["shapenet", "gobjaverse", "3dfront"]

        for root in self.roots:
            if self.data_type == "shapenet":
                data_root = os.path.join(root, self.data_category)
                for instanceID in os.listdir(data_root):
                    self.instances.append((data_root, instanceID))
            elif self.data_type == "gobjaverse":
                data_root = os.path.join(root, self.data_category)
                for sub_category in os.listdir(data_root):
                    for instanceID in os.listdir(os.path.join(data_root, sub_category)):
                        self.instances.append((os.path.join(data_root, sub_category), instanceID))
            elif self.data_type == "3dfront":
                data_root = root
                for instanceID in os.listdir(data_root):
                    self.instances.append((data_root, instanceID))

    def get_instance(self, root, instance):
        position = utils3d.io.read_ply(os.path.join(root, instance, 'voxelized_pc.ply'))[0]

        pos = torch.tensor(position, dtype=torch.float32)

        # aabb = [xmin, ymin, zmin, size_x, size_y, size_z]
        min_xyz = torch.tensor(self.aabb[:3], dtype=torch.float32)      # 예: [-0.55, -0.55, -0.55]
        size_xyz = torch.tensor(self.aabb[3:], dtype=torch.float32)     # 예: [1.10, 1.10, 1.10]

        # 월드 좌표 -> [0, resolution) 인덱스로 선형 매핑
        coords = ((pos - min_xyz) * self.resolution / size_xyz).long()

        # 안전하게 0 ~ resolution-1로 클램프
        coords = coords.clamp(0, self.resolution - 1)

        ss = torch.zeros(1, self.resolution, self.resolution, self.resolution, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        return {'ss': ss}

    @torch.no_grad()
    def visualize_sample(self, ss: Union[torch.Tensor, dict]):
        ss = ss if isinstance(ss, torch.Tensor) else ss['ss']
        
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4
        renderer.pipe.primitive = 'voxel'
        
        # Build camera
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Build each representation
        ss = ss.cuda()
        for i in range(ss.shape[0]):
            representation = Octree(
                depth=10,
                aabb=self.aabb,
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            coords = torch.nonzero(ss[i, 0], as_tuple=False)
            representation.position = coords.float() / self.resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(self.resolution)), dtype=torch.uint8, device='cuda')

            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)
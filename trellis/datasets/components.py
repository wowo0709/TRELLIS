from typing import *
from abc import abstractmethod
import os
import json
import warnings
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(',')
        self.instances = []
        self.metadata = pd.DataFrame()
        
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))
            self._stats[key]['Total'] = len(metadata)
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])
            
    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        pass
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class CustomStandardDatasetBase(Dataset):
    """
    Base class for standard datasets.

    Args:
        roots (str): paths to the dataset
    """

    def __init__(self,
        roots: str,
        data_type: str,
        data_category: str,
    ):
        super().__init__()
        self.roots = roots.split(',') if isinstance(roots, str) else roots
        self.data_type = data_type
        self.data_category = data_category
        self.instances = []
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        pass
        
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Data Type: {self.data_type}')
        lines.append(f'  - Data category: {self.data_category}')
        lines.append(f'  - Total instances: {len(self)}')
        # lines.append(f'  - Sources:')
        # for key, stats in self._stats.items():
        #     lines.append(f'    - {key}:')
        #     for k, v in stats.items():
        #         lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class TextConditionedMixin:
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])
        pack['cond'] = text
        return pack


class CustomTextConditionedMixin:
    def __init__(self, roots, data_type, data_category, **kwargs):
        caption_root = kwargs.pop('caption_root', None)
        caption_file = kwargs.pop('caption_file', 'id_captions.json')
        caption_subdir = kwargs.pop('caption_subdir', None)
        allow_empty_caption = kwargs.pop('allow_empty_caption', False)
        super().__init__(roots, data_type, data_category, **kwargs)
        self.caption_root = caption_root
        self.caption_file = caption_file
        self.caption_subdir = caption_subdir
        self.allow_empty_caption = allow_empty_caption
        self.captions = {}
        if self.caption_root is None:
            for instance in self.instances:
                sha256 = instance[1]
                # Preserve the previous behavior for existing configs.
                self.captions[sha256] = ['']
            return

        caption_map = self._load_custom_captions()
        missing = 0
        for _, instance in self.instances:
            captions = caption_map.get(instance)
            if captions:
                self.captions[instance] = captions
            else:
                missing += 1
                self.captions[instance] = ['']

        if missing > 0:
            message = (
                f'CustomTextConditionedMixin: missing captions for {missing}/{len(self.instances)} '
                f'instances under {self.caption_root}.'
            )
            if self.allow_empty_caption:
                warnings.warn(message + ' Falling back to empty captions for missing entries.')
            else:
                raise ValueError(message)

    _SHAPENET_CATEGORY_TO_SUBDIR = {
        '02958343': 'car',
        '03001627': 'chair',
        '03790512': 'motorbike',
        '04379243': 'table',
    }

    def _resolve_caption_subdir(self) -> str:
        if self.caption_subdir is not None:
            return self.caption_subdir
        if self.data_type == 'shapenet':
            if self.data_category in self._SHAPENET_CATEGORY_TO_SUBDIR:
                return self._SHAPENET_CATEGORY_TO_SUBDIR[self.data_category]
            if os.path.isdir(os.path.join(self.caption_root, self.data_category)):
                return self.data_category
        return self.data_category

    def _normalize_caption_entry(self, value: Any) -> Optional[List[str]]:
        captions: List[str] = []

        if isinstance(value, str):
            captions = [value]
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], str):
                # Pseudo-caption format: ["caption text", "03001627"]
                if len(value) >= 2 and isinstance(value[1], str) and value[1] == self.data_category:
                    captions = [value[0]]
                else:
                    captions = [v for v in value if isinstance(v, str)]
        elif isinstance(value, dict):
            for key in ['captions', 'caption', 'texts', 'text']:
                nested = value.get(key)
                if isinstance(nested, str):
                    captions = [nested]
                    break
                if isinstance(nested, list):
                    captions = [v for v in nested if isinstance(v, str)]
                    break

        captions = [caption.strip() for caption in captions if isinstance(caption, str) and caption.strip()]
        return captions if captions else None

    def _load_custom_captions(self) -> Dict[str, List[str]]:
        subdir = self._resolve_caption_subdir()
        caption_path = os.path.join(self.caption_root, subdir, self.caption_file)
        if not os.path.exists(caption_path):
            raise FileNotFoundError(f'Caption file not found: {caption_path}')

        with open(caption_path, 'r') as f:
            raw = json.load(f)

        if not isinstance(raw, dict):
            raise ValueError(f'Expected caption file to contain a dict, got {type(raw).__name__}: {caption_path}')

        normalized = {}
        for instance_id, value in raw.items():
            captions = self._normalize_caption_entry(value)
            if captions:
                normalized[instance_id] = captions
        return normalized
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]
        stats['With captions'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])
        pack['cond'] = text
        return pack
    
    
class ImageConditionedMixin:
    def __init__(self, roots, *, image_size=518, **kwargs):
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)
       
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)

        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image
       
        return pack


class CustomImageConditionedMixin:
    def __init__(
        self,
        roots,
        data_type,
        data_category,
        *,
        image_size=518,
        image_root: Optional[str] = None,
        image_folder: str = 'rendered_images',
        condition_view_indices: Optional[List[int]] = None,
        num_condition_views: Optional[int] = None,
        **kwargs,
    ):
        self.image_size = image_size
        self.image_root = image_root
        self.image_folder = image_folder
        self.condition_view_indices = list(condition_view_indices) if condition_view_indices is not None else None
        self.num_condition_views = num_condition_views
        super().__init__(roots, data_type, data_category, **kwargs)
    
    def filter_metadata(self, metadata):
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]
        stats['Cond rendered'] = len(metadata)
        return metadata, stats

    def _resolve_condition_view_pool(self, available_views: List[int]) -> List[int]:
        if self.condition_view_indices is None:
            return available_views
        pool = [v for v in self.condition_view_indices if v in available_views]
        if self.num_condition_views is not None:
            pool = pool[:self.num_condition_views]
        if len(pool) == 0:
            raise ValueError('No valid condition views remain after applying condition_view_indices/num_condition_views.')
        return pool

    def _load_and_process_rgba(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        return image * alpha.unsqueeze(0)
    
    def get_instance(self, root, instance):
        pack = super().get_instance(root, instance)

        if self.image_root is None:
            image_root = os.path.join(root, 'renders_cond', instance)
            with open(os.path.join(image_root, 'transforms.json')) as f:
                metadata = json.load(f)
            n_views = len(metadata['frames'])
            view = np.random.randint(n_views)
            metadata = metadata['frames'][view]
            image_path = os.path.join(image_root, metadata['file_path'])
            image = self._load_and_process_rgba(image_path)
            pack['cond'] = image
            return pack

        image_dir = os.path.join(self.image_root, self.data_category, instance, self.image_folder)
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f'Image directory not found: {image_dir}')

        available_views = sorted(
            int(p.stem)
            for p in Path(image_dir).glob('*.png')
            if p.stem.isdigit()
        )
        if len(available_views) == 0:
            raise ValueError(f'No PNG views found in {image_dir}')

        condition_pool = self._resolve_condition_view_pool(available_views)
        view = int(np.random.choice(condition_pool))
        image_path = os.path.join(image_dir, f'{view:05d}.png')
        image = self._load_and_process_rgba(image_path)
        pack['cond'] = image
        pack['cond_view'] = view
        return pack
    
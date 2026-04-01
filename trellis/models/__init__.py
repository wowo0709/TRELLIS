import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'ElasticSLatEncoder': 'structured_latent_vae',
    'ElasticSLatGaussianDecoder': 'structured_latent_vae',
    'ElasticSLatRadianceFieldDecoder': 'structured_latent_vae',
    'ElasticSLatMeshDecoder': 'structured_latent_vae',
    
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: local config/model files should be f'{path}.json' plus either f'{path}.safetensors' or f'{path}.pt'.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    import torch
    from safetensors.torch import load_file

    config_path = f"{path}.json"
    safetensors_path = f"{path}.safetensors"
    pt_path = f"{path}.pt"
    is_local = os.path.exists(config_path) and (os.path.exists(safetensors_path) or os.path.exists(pt_path))

    if is_local:
        config_file = config_path
        model_file = safetensors_path if os.path.exists(safetensors_path) else pt_path
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        if len(path_parts) < 3 or path_parts[0] == '' or path_parts[1] == '':
            raise ValueError(
                f"Invalid Hugging Face model path: '{path}'. "
                "Expected format: '<org>/<repo>/<model_name>'."
            )
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)

    if model_file.endswith('.safetensors'):
        state_dict = load_file(model_file)
    else:
        try:
            state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(model_file, map_location='cpu')
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()

    model.load_state_dict(state_dict)

    return model 


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import (
        SparseStructureEncoder, 
        SparseStructureDecoder,
    )
    
    from .sparse_structure_flow import SparseStructureFlowModel
    
    from .structured_latent_vae import (
        SLatEncoder,
        SLatGaussianDecoder,
        SLatRadianceFieldDecoder,
        SLatMeshDecoder,
        ElasticSLatEncoder,
        ElasticSLatGaussianDecoder,
        ElasticSLatRadianceFieldDecoder,
        ElasticSLatMeshDecoder,
    )
    
    from .structured_latent_flow import (
        SLatFlowModel,
        ElasticSLatFlowModel,
    )

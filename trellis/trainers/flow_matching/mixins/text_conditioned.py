from typing import *
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
import torch
from transformers import AutoTokenizer, CLIPTextModel

from ....utils import dist_utils


class TextConditionedMixin:
    """
    Mixin for text-conditioned models.
    
    Args:
        text_cond_model: The text conditioning model.
    """
    def __init__(
        self,
        *args,
        text_cond_model: str = 'openai/clip-vit-large-patch14',
        log_text_to_wandb: bool = False,
        text_log_num_samples: int = 4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.text_cond_model_name = text_cond_model
        self.text_cond_model = None     # the model is init lazily
        self.log_text_to_wandb = log_text_to_wandb
        self.text_log_num_samples = max(int(text_log_num_samples), 1)
        self._text_logged_step = -1
        
    def _init_text_cond_model(self):
        """
        Initialize the text conditioning model.
        """
        # load model
        with dist_utils.local_master_first():
            model = CLIPTextModel.from_pretrained(self.text_cond_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.text_cond_model_name)
        model.eval()
        model = model.cuda()
        self.text_cond_model = {
            'model': model,
            'tokenizer': tokenizer,
        }
        self.text_cond_model['null_cond'] = self.encode_text([''])
        
    @torch.no_grad()
    def encode_text(self, text: List[str]) -> torch.Tensor:
        """
        Encode the text.
        """
        assert isinstance(text, list) and isinstance(text[0], str), "TextConditionedMixin only supports list of strings as cond"
        if self.text_cond_model is None:
            self._init_text_cond_model()
        encoding = self.text_cond_model['tokenizer'](text, max_length=77, padding='max_length', truncation=True, return_tensors='pt')
        tokens = encoding['input_ids'].cuda()
        embeddings = self.text_cond_model['model'](input_ids=tokens).last_hidden_state
        
        return embeddings
        
    def get_text_wandb_payload(self, cond) -> Optional[Dict[str, List[str]]]:
        """
        Build a lightweight text payload for WandB logging.
        """
        should_log_text = (
            self.log_text_to_wandb
            and self.is_master
            and self.wandb_config.get("use_wandb", False)
            and ((self.step + 1) % self.i_log == 0)
            and self._text_logged_step != self.step
        )
        if not should_log_text:
            return None
        if not isinstance(cond, list) or len(cond) == 0 or not isinstance(cond[0], str):
            return None

        self._text_logged_step = self.step
        return {
            'captions': cond[:self.text_log_num_samples]
        }
        
    def get_cond(self, cond, **kwargs):
        """
        Get the conditioning data.
        """
        cond = self.encode_text(cond)
        kwargs['neg_cond'] = self.text_cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_cond(cond, **kwargs)
        return cond
    
    def get_inference_cond(self, cond, **kwargs):
        """
        Get the conditioning data for inference.
        """
        cond = self.encode_text(cond)
        kwargs['neg_cond'] = self.text_cond_model['null_cond'].repeat(cond.shape[0], 1, 1)
        cond = super().get_inference_cond(cond, **kwargs)
        return cond

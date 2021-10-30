from typing import Callable, Union, Optional
import pathlib
import logging

import torch
from torch import Tensor
import torch.nn as nn

logger = logging.getLogger(__name__)


class AuemBaseModel(nn.Module):
    sample_rate = 22050
    embedding_size = 4096
    scene_embedding_size = embedding_size
    timestamp_embedding_size = embedding_size
    seed = 11
    epsilon = 1e-6

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        raise NotImplementedError("Must implement as a child class.")

    @property
    def n_fft(self) -> int:
        raise NotImplementedError()

    @property
    def min_frame_size(self) -> int:
        return self.n_fft

    def save(self, save_path: Union[pathlib.Path, str],
             optimizer: Optional[torch.nn.Module] = None,
             init_kwargs: Optional[dict] = None,
             **metadata):
        """Save a model to a .pth file, but *including* the necessary
        pieces to re-load the model later.
        """
        save_dict = {
            "class_name": self.__class__.__name__,
            "model_state_dict": self.state_dict(),
            **metadata
        }
        if optimizer is not None:
            save_dict.update(optimizer_state_dict=optimizer.state_dict())
        if init_kwargs is not None:
            save_dict.update(init_kwargs=init_kwargs)

        torch.save(save_dict, save_path)

    @staticmethod
    def load(path: Union[pathlib.Path, str], clsname: Optional[str]) -> nn.Module:
        """Load a model from a saved model file."""
        checkpoint = torch.load(path)

        from .factory import create_model

        if "model_state_dict" in checkpoint and "class_name" in checkpoint:
            model_cls = checkpoint.get("class_name", clsname)
            model_kwargs = checkpoint.get("init_kwargs", {})

            model = create_model(model_cls, **model_kwargs)
            model.load_state_dict(checkpoint["model_state_dict"])
        
        else:
            if clsname is None:
                logger.error("class_name not found in model, and no clsname provided.\n"
                             "please provide a classname to load the model.")
                return None
            
            model = create_model(clsname)
            assert model is not None
            model.load_state_dict(checkpoint, strict=False)

        return model

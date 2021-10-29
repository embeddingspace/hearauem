from torch import Tensor
import torch.nn as nn


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

"""
Model loading for an EfficientNet on a Mel Spectrogram.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnAudio import Spectrogram
from efficientnet_pytorch import EfficientNet


class MelEfficientNet(nn.Module):
    embedding_size = 1280

    def __init__(self, drop_connect_rate=0.1,
                 compress_mel: bool = True,
                 epsilon: float = 1e-5,
                 sample_rate: float = 22050,
                 **kwargs
                 ):
        super().__init__()
        
        self.compress_mel = compress_mel
        self.epsilon = epsilon

        self.mel = Spectrogram.MelSpectrogram(
            sr=sample_rate,
            **kwargs
        )
        self.efficientnet = EfficientNet.from_name(
            "efficientnet-b0",
            include_top=False,
            drop_connect_rate=drop_connect_rate,
            in_channels=1
        )

    def forward(self, x):
        S = self.mel(x)
    
        if self.compress_mel:
            S = torch.log(S + self.epsilon)

        #### TODO Include Batch norm?
        s_x = self.efficientnet(S.unsqueeze(1))

        y = s_x.squeeze(3).squeeze(2)
        return y

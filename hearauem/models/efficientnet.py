"""
Model loading for an EfficientNet on a Mel Spectrogram.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnAudio import Spectrogram
from efficientnet_pytorch import EfficientNet

from .base import AuemBaseModel
from .factory import register_model


@register_model
class MelEfficientNet(AuemBaseModel):
    embedding_size = 1280
    win_length = 400
    hop_length = 160
    n_fft = 2048
    n_mels = 192
    fmin = 60
    fmax = 10000.0

    def __init__(self, drop_connect_rate=0.1,
                 compress_mel: bool = True,
                 sample_rate: float = 22050,
                 **kwargs
                 ):
        super().__init__()
        # kwargs.setdefault("n_mels", 192)
        assert self.sample_rate == sample_rate
        
        self.compress_mel = compress_mel

        self.mel = Spectrogram.MelSpectrogram(
            sr=self.sample_rate,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
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

    @property
    def min_frame_size(self) -> int:
        return self.mel.n_fft * 8

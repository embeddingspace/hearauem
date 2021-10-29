"""
Model loading for Team Auem's HEAR 2021 submission.

Much of this borrowed from:
https://github.com/neuralaudio/hear-baseline/blob/main/hearbaseline/naive.py
"""
from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from hearbaseline.util import frame_audio

from .models.base import AuemBaseModel

# Default hop_size in milliseconds
TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250

# Number of frames to batch process for timestamp embeddings
BATCH_SIZE = 512


def load_model(model_file_path: str = "") -> nn.Module:
    """
    Returns a torch.nn.Module that produces embeddings for audio.

    Parameters
    ----------
    model_file_path

    Returns
    -------
    model
    """
    model = AuemBaseModel()
    if model_file_path != "":
        loaded_model = torch.load(model_file_path)
        if not isinstance(loaded_model, OrderedDict):
            raise TypeError(
                "Loaded model must be a state dict of type OrderedDict."
                f"Recieved {type(loaded_model)}"
            )

        model.load_state_dict(loaded_model)

    return model


def get_scene_embeddings(audio: Tensor, model: nn.Module,
                         hop_size: float = TIMESTAMP_HOP_SIZE) -> Tuple[Tensor, Tensor]:
    """
    This function returns embeddings at regular intervals centered at timestamps.
    Both the embeddings and the corresponding timestamps (in milliseconds) are returned.
    
    Parameters
    ----------

    Returns
    -------
    embeddings
        A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
    
    timestamps
        Centered timestamps in milliseconds corresponding to each embedding in the output.
        Shape: (n_sounds, num_samples).
    """
    # Assert audio is of correct shape
    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )
    
    # Make sure the correct model type was passed in
    if not isinstance(model, AuemBaseModel):
        raise ValueError(
            f"Model must be an instance of {AuemBaseModel.__name__}"
        )

    # Send the model to the same device that the audio tensor is on.
    model = model.to(audio.device)

    # Split the input audio signals into frames and then flatten to create a tensor
    # of audio frames that can be batch processed.
    frames, timestamps = frame_audio(
        audio,
        frame_size=model.min_frame_size,
        hop_size=hop_size,
        sample_rate=AuemBaseModel.sample_rate,
    )
    audio_batches, num_frames, frame_size = frames.shape
    frames = frames.flatten(end_dim=1)

    # We're using a DataLoader to help with batching of frames
    dataset = torch.utils.data.TensorDataset(frames)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    model.eval()
    with torch.no_grad():
        embeddings_list = [model(batch[0]) for batch in loader]

    # Concatenate mini-batches back together and unflatten the frames
    # to reconstruct the audio batches
    embeddings = torch.cat(embeddings_list, dim=0)
    embeddings = embeddings.unflatten(0, (audio_batches, num_frames))

    return embeddings, timestamps


def get_timestamp_embeddings(audio: Tensor, model: nn.Module):
    embeddings, _ = get_timestamp_embeddings(audio, model, hop_size=SCENE_HOP_SIZE)
    embeddings = torch.mean(embeddings, dim=1)
    return embeddings


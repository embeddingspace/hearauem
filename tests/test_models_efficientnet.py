import torch
import hearauem.models.efficientnet as M

import pytest

from hearauem.interface import get_scene_embeddings, get_timestamp_embeddings


def test_efficientnet():
    model = M.MelEfficientNet()
    assert model.sample_rate == 22050
    assert model.embedding_size == 1280

    samples = torch.rand(4, model.sample_rate * 3) * 2 - 1

    with torch.no_grad():
        output = model(samples)
    assert output.shape[0] == samples.shape[0]
    assert output.shape[1] == model.embedding_size


class TestEmbeddings:
    @pytest.fixture(scope="session")
    def model(self):
        model = M.MelEfficientNet()
        assert model.sample_rate == 22050
        assert model.embedding_size == 1280
        return model

    @pytest.fixture(params=[1, 5, 10])
    def n_seconds(self, request):
        return request.param

    def test_efficientnet_with_time_embeddings(self, model, n_seconds):
        batch_size = 4
        samples = torch.rand(batch_size, model.sample_rate * n_seconds) * 2 - 1

        embeddings, timestamps = get_scene_embeddings(samples, model)

        assert len(embeddings.shape) == 3
        assert embeddings.shape[0] == batch_size
        assert embeddings.shape[2] == model.embedding_size

        assert len(timestamps.shape) == 2
        assert timestamps.shape[0] == batch_size
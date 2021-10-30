import torch
import hearauem.models.efficientnet as M

import pytest

from hearauem.interface import load_model, get_scene_embeddings, get_timestamp_embeddings


def test_efficientnet():
    model = M.MelEfficientNet()
    assert model.sample_rate == 22050
    assert model.embedding_size == 1280

    samples = torch.rand(4, model.sample_rate * 3) * 2 - 1

    with torch.no_grad():
        output = model(samples)
    assert output.shape[0] == samples.shape[0]
    assert output.shape[1] == model.embedding_size


def test_saving_and_loading(tmp_path):
    save_path = tmp_path / "model.pth"
    model = M.MelEfficientNet()

    model.save(save_path)

    test_model = load_model(save_path)
    assert isinstance(test_model, torch.nn.Module)


def test_loading_from_name(tmp_path):
    save_path = tmp_path / "model.pth"
    model = M.MelEfficientNet()

    torch.save(model.state_dict(), save_path)

    test_model = load_model(save_path, "MelEfficientNet")
    assert test_model is not None
    assert isinstance(test_model, torch.nn.Module)


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

    def test_time_embeddings(self, model, n_seconds):
        batch_size = 4
        samples = torch.rand(batch_size, model.sample_rate * n_seconds) * 2 - 1

        embeddings, timestamps = get_timestamp_embeddings(samples, model)

        assert len(embeddings.shape) == 3
        assert embeddings.shape[0] == batch_size
        assert embeddings.shape[2] == model.embedding_size

        assert len(timestamps.shape) == 2
        assert timestamps.shape[0] == batch_size

    def test_get_scene_embeddings(self, model, n_seconds):
        batch_size = 4
        samples = torch.rand(batch_size, model.sample_rate * n_seconds) * 2 - 1

        embeddings = get_scene_embeddings(samples, model)
        assert len(embeddings.shape) == 2
        assert embeddings.shape[0] == samples.shape[0]

import torch
import hearauem.models.efficientnet as M


def test_efficientnet():
    sample_rate = 22050

    model = M.MelEfficientNet()
    samples = torch.rand(4, sample_rate * 3)

    with torch.no_grad():
        output = model(samples)
    assert output.shape[0] == samples.shape[0]
    assert output.shape[1] == model.embedding_size

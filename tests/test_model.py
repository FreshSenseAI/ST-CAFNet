import torch

from stcafnet.model import STCAFNet


def test_forward_and_loss():
    model = STCAFNet(pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224), torch.randn(2, 120, 10))
        loss = model.uncertainty_weighted_loss(
            output.predictions, torch.randn(2, 3)
        )
    assert output.predictions.shape == (2, 3)
    assert output.gate.shape == (2, 512)
    assert torch.isfinite(loss)


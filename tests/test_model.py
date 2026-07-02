import torch

from stcafnet.model import STCAFNet


def test_forward_and_loss():
    model = STCAFNet(pretrained=False)
    model.eval()
    with torch.no_grad():
        images = torch.randn(2, 3, 224, 224)
        enose = torch.randn(2, 120, 10)
        visual_global, visual_tokens = model.visual(images)
        odor_global, odor_tokens = model.olfactory(enose)
        output = model(images, enose)
        loss = model.uncertainty_weighted_loss(
            output.predictions, torch.randn(2, 3)
        )
    assert visual_global.shape == (2, 512)
    assert visual_tokens.shape == (2, 49, 512)
    assert odor_global.shape == (2, 512)
    assert odor_tokens.shape == (2, 30, 512)
    assert output.predictions.shape == (2, 3)
    assert output.gate.shape == (2, 512)
    assert torch.isfinite(loss)

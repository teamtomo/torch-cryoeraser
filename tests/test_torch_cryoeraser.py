import torch
import torch.nn.functional as F

from torch_cryoeraser.erase import erase_region_2d


def test_erase_2d():
    """Smoke test checking that the function runs and produces output."""
    image = torch.ones((28, 28))
    mask = F.pad(torch.ones((14, 14)), pad=(7, 7, 7, 7))
    inpainted = erase_region_2d(
        image=image,
        mask=mask,
        background_intensity_model_resolution=(5, 5),
        background_intensity_model_samples=200,
    )
    assert inpainted.shape == image.shape


def test_erase_batched_2d():
    """Smoke test checkingthat the function runs and produces output."""
    image = torch.ones((2, 28, 28))
    mask = F.pad(torch.ones((2, 14, 14)), pad=(7, 7, 7, 7))
    inpainted = erase_region_2d(
        image=image,
        mask=mask,
        background_intensity_model_resolution=(5, 5),
        background_intensity_model_samples=200,
    )


assert inpainted.shape == image.shape

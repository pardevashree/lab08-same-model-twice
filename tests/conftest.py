import pytest
import torch

from bikes_data import make_rentals
from prepare import split_frame


@pytest.fixture(scope="session")
def splits():
    return split_frame(make_rentals())


@pytest.fixture()
def small_batch():
    torch.manual_seed(0)
    return torch.randn(32, 8), torch.rand(32) * 20

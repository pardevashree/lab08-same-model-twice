import pytest
import torch
from torch import nn

from model import BikeNet


@pytest.mark.parametrize("hidden", [4, 32, 128])
def test_network_returns_one_value_per_row(hidden):
    model = BikeNet(n_inputs=8, hidden=hidden)
    assert model(torch.zeros(5, 8)).shape == (5,)


def test_one_step_changes_every_layer(small_batch):
    X, y = small_batch
    model = BikeNet()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = nn.MSELoss()(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for name, value in model.state_dict().items():
        assert not torch.equal(before[name], value), name

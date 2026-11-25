import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import BikeNet
from train import fit, run_epoch


def _loader(small_batch, batch_size=32):
    return DataLoader(TensorDataset(*small_batch), batch_size=batch_size)


def test_loop_can_memorize_one_batch(small_batch):
    loader = _loader(small_batch)
    model = BikeNet(hidden=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    first, _ = run_epoch(model, loader, loss_fn, optimizer)
    for _ in range(300):
        last, _ = run_epoch(model, loader, loss_fn, optimizer)
    assert last < 0.05 * first


def test_measuring_epoch_leaves_weights_unchanged(small_batch):
    loader = _loader(small_batch)
    model = BikeNet()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    run_epoch(model, loader, nn.MSELoss())
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value), name


def test_fit_logs_every_epoch_and_restores_best_weights(small_batch):
    loader = _loader(small_batch, batch_size=16)
    model = BikeNet(hidden=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    history = fit(model, loader, loader, nn.MSELoss(), optimizer, epochs=5)
    assert len(history) == 5
    assert set(history[0]) == {"epoch", "train_loss", "val_loss", "val_mae"}
    best = min(row["val_loss"] for row in history)
    val_loss, _ = run_epoch(model, loader, nn.MSELoss())
    assert val_loss == pytest.approx(best)

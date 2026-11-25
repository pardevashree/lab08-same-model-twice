import numpy as np
import torch

from bikes_data import TARGET, make_rentals
from prepare import BikeDataset, make_loaders, make_preprocessor


def _dataset(train):
    pre = make_preprocessor().fit(train)
    return BikeDataset(pre.transform(train), train[TARGET].to_numpy())


def test_dataset_returns_float32_tensors(splits):
    train = splits[0]
    ds = _dataset(train)
    x0, y0 = ds[0]
    assert len(ds) == len(train)
    assert x0.dtype == torch.float32 and y0.dtype == torch.float32
    assert x0.shape == (8,) and y0.shape == ()


def test_splits_have_right_sizes_and_do_not_overlap(splits):
    train, val, test = splits
    assert (len(train), len(val), len(test)) == (3600, 1200, 1200)
    assert len(train) + len(val) + len(test) == len(make_rentals())
    assert not set(train.index) & set(val.index)
    assert not set(train.index) & set(test.index)
    assert not set(val.index) & set(test.index)


def test_scaler_is_fitted_on_train_only(splits):
    train, val, _ = splits
    pre = make_preprocessor().fit(train)
    train_mean = pre.transform(train)[:, :4].mean(axis=0)
    val_mean = pre.transform(val)[:, :4].mean(axis=0)
    assert np.allclose(train_mean, 0, atol=1e-6)
    assert not np.allclose(val_mean, 0, atol=1e-6)


def test_train_loader_shuffles_and_is_repeatable(splits):
    ds = _dataset(splits[0])

    def first_targets(seed):
        loader = make_loaders(ds, ds, ds, batch_size=32, seed=seed)[0]
        return next(iter(loader))[1]

    a, b, c = first_targets(1), first_targets(1), first_targets(2)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert not torch.equal(a, ds.y[:32])

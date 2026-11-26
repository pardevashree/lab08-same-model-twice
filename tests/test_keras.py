"""Tests for the Keras version and the weight copy."""

import keras
import numpy as np
import pytest
import torch

from keras_model import (
    build_keras_net,
    copy_weights_to_keras,
    max_prediction_gap,
    torch_param_count,
)
from model import BikeNet
from prepare import BikeDataset
from train import prepare_data
from train_keras import run_keras_experiment

TOLERANCE = 1e-5


def make_rows(n=64, n_inputs=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_inputs)).astype("float32")
    y = (X[:, 0] * 2 + X[:, 1]).astype("float32")
    return X, y


def make_data(seed=0):
    X, y = make_rows(120, seed=seed)
    return (
        {"train": X[:80], "val": X[80:100], "test": X[100:]},
        {"train": y[:80], "val": y[80:100], "test": y[100:]},
    )


def dense_layers(net):
    return [layer for layer in net.layers if isinstance(layer, keras.layers.Dense)]


def test_keras_returns_one_value_per_row():
    X, _ = make_rows(10)
    net = build_keras_net(8, 16)
    assert net.predict(X, verbose=0).shape == (10, 1)


@pytest.mark.parametrize("hidden", [1, 4, 16, 32, 128])
def test_param_count_matches_pytorch(hidden):
    keras_count = build_keras_net(8, hidden).count_params()
    torch_count = torch_param_count(BikeNet(8, hidden))
    assert keras_count == torch_count


def test_weight_copy_gives_same_predictions():
    torch.manual_seed(1)
    tnet = BikeNet(8, 16)
    knet = build_keras_net(8, 16)
    copy_weights_to_keras(tnet, knet)
    X, _ = make_rows(50)
    assert max_prediction_gap(tnet, knet, X) < TOLERANCE


def test_missing_transpose_would_be_caught():
    # The 16 -> 16 layer is square, so a kernel that is not flipped gives
    # no shape error. This test shows our inputs would notice it.
    torch.manual_seed(1)
    tnet = BikeNet(8, 16)
    knet = build_keras_net(8, 16)
    copy_weights_to_keras(tnet, knet)
    middle = dense_layers(knet)[1]
    kernel, bias = middle.get_weights()
    middle.set_weights([kernel.T, bias])
    X, _ = make_rows(50)
    assert max_prediction_gap(tnet, knet, X) > 1e-3


def test_keras_can_memorize_one_batch():
    keras.utils.set_random_seed(0)
    X, y = make_rows(16)
    net = build_keras_net(8, 64)
    net.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
    net.fit(X, y, epochs=500, batch_size=16, verbose=0)
    assert net.evaluate(X, y, verbose=0) < 0.05


def test_saved_model_reloads_with_same_predictions(tmp_path):
    keras.utils.set_random_seed(0)
    net = build_keras_net(8, 16)
    X, _ = make_rows(20)
    path = tmp_path / "net.keras"
    net.save(path)
    loaded = keras.models.load_model(path)
    np.testing.assert_allclose(
        net.predict(X, verbose=0), loaded.predict(X, verbose=0), atol=TOLERANCE
    )


def test_best_weights_are_restored():
    data = make_data()
    model, history, _ = run_keras_experiment(
        data, hidden=16, lr=0.05, epochs=15, seed=3
    )
    X, y = data
    val_loss = model.evaluate(X["val"], y["val"], verbose=0)[0]
    assert val_loss == pytest.approx(min(history["val_loss"]), rel=1e-3)


def test_same_seed_repeats_a_short_run():
    data = make_data()
    _, first, _ = run_keras_experiment(data, hidden=8, epochs=3, seed=5)
    _, second, _ = run_keras_experiment(data, hidden=8, epochs=3, seed=5)
    np.testing.assert_allclose(first["val_loss"], second["val_loss"], rtol=1e-5)


def test_keras_and_pytorch_use_the_same_rows():
    X, y = prepare_data()
    for key in ("train", "val", "test"):
        dataset = BikeDataset(X[key], y[key])
        np.testing.assert_allclose(
            dataset.X.numpy(), np.asarray(X[key], dtype="float32")
        )
        np.testing.assert_allclose(
            dataset.y.numpy(), np.asarray(y[key], dtype="float32")
        )

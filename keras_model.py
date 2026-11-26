"""Keras version of BikeNet, plus the weight copy from PyTorch."""

import keras
import numpy as np
import torch
import torch.nn as nn
from keras import layers


def build_keras_net(n_inputs=8, hidden=32):
    """Same architecture as BikeNet: two hidden ReLU layers, one linear output."""
    return keras.Sequential(
        [
            keras.Input(shape=(n_inputs,)),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(hidden, activation="relu"),
            layers.Dense(1),
        ]
    )


def torch_param_count(torch_net):
    return sum(p.numel() for p in torch_net.parameters())


def copy_weights_to_keras(torch_net, keras_net):
    """Copy every PyTorch Linear layer into the matching Keras Dense layer.

    PyTorch weight is (outputs, inputs). Keras kernel is (inputs, outputs).
    So the weight is transposed. The bias is copied as it is.
    """
    linears = [m for m in torch_net.layers if isinstance(m, nn.Linear)]
    denses = [layer for layer in keras_net.layers if isinstance(layer, layers.Dense)]
    if len(linears) != len(denses):
        raise ValueError("The two networks have a different number of layers")
    for linear, dense in zip(linears, denses, strict=True):
        kernel = linear.weight.detach().numpy().T
        bias = linear.bias.detach().numpy()
        dense.set_weights([kernel, bias])


def predict_torch(torch_net, x):
    torch_net.eval()
    with torch.no_grad():
        out = torch_net(torch.as_tensor(x, dtype=torch.float32))
    return out.numpy()


def predict_keras(keras_net, x):
    # predict returns (N, 1). Flatten it to (N,) like the PyTorch output.
    return keras_net.predict(x, verbose=0).reshape(-1)


def max_prediction_gap(torch_net, keras_net, x):
    """Largest absolute difference between the two networks' predictions."""
    gap = np.abs(predict_torch(torch_net, x) - predict_keras(keras_net, x))
    return float(gap.max())

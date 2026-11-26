"""Keras training: compile, fit and callbacks, with the same settings as PyTorch."""

import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras_model import build_keras_net
from train import EPOCHS, REPORTS, SEED, prepare_data

BATCH_SIZE = 64
LR = 1e-3


def run_keras_experiment(
    data,
    hidden=32,
    lr=LR,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    seed=SEED,
    checkpoint_path=None,
):
    """Train the Keras network. Returns model, history dict and seconds taken."""
    X, y = data
    keras.utils.set_random_seed(seed)
    model = build_keras_net(X["train"].shape[1], hidden)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    # patience = epochs, so the run never stops early (same length as PyTorch)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=epochs, restore_best_weights=True
        )
    ]
    if checkpoint_path is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_loss", save_best_only=True
            )
        )
    start = time.perf_counter()
    history = model.fit(
        np.asarray(X["train"], dtype="float32"),
        np.asarray(y["train"], dtype="float32"),
        validation_data=(
            np.asarray(X["val"], dtype="float32"),
            np.asarray(y["val"], dtype="float32"),
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    seconds = time.perf_counter() - start
    return model, history.history, seconds


def best_epoch(history):
    """Epoch number (starting at 1) with the lowest validation loss."""
    return int(np.argmin(history["val_loss"])) + 1


def test_mae(model, data):
    """MAE on the test rows. Call this once per final model."""
    X, y = data
    pred = model.predict(np.asarray(X["test"], dtype="float32"), verbose=0)
    return float(np.mean(np.abs(pred.reshape(-1) - np.asarray(y["test"]))))


def save_history(history, path):
    frame = pd.DataFrame(
        {
            "epoch": range(1, len(history["loss"]) + 1),
            "train_loss": history["loss"],
            "val_loss": history["val_loss"],
            "val_mae": history["val_mae"],
        }
    )
    frame.to_csv(path, index=False)


def save_curves(history, path):
    fig, ax = plt.subplots()
    ax.plot(history["loss"], label="train loss")
    ax.plot(history["val_loss"], label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.set_title("Keras main run")
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    REPORTS.mkdir(exist_ok=True)
    data = prepare_data()
    model, history, seconds = run_keras_experiment(
        data, checkpoint_path=str(REPORTS / "best_keras.keras")
    )
    save_history(history, REPORTS / "history_keras.csv")
    save_curves(history, REPORTS / "learning_curves_keras.png")
    print(f"best epoch {best_epoch(history)}, seconds {seconds:.1f}")
    print(f"val MAE {min(history['val_mae']):.3f}")
    print(f"test MAE {test_mae(model, data):.3f}")


if __name__ == "__main__":
    main()

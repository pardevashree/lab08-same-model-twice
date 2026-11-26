"""Train the main configuration in PyTorch and Keras on several seeds."""

import time

import pandas as pd
import torch.nn as nn

from train import REPORTS, RUNS, prepare_data, run_epoch, run_experiment
from train_keras import best_epoch, run_keras_experiment, test_mae

SEEDS = [42, 43, 44]


def run_torch(data, seed):
    start = time.perf_counter()
    model, history, test_loader = run_experiment(data, seed=seed, **RUNS["main"])
    seconds = time.perf_counter() - start
    df = pd.DataFrame(history)
    best = df.loc[df["val_loss"].idxmin()]
    # the test rows are scored once, for the final model
    _, mae = run_epoch(model, test_loader, nn.MSELoss())
    return {
        "framework": "pytorch",
        "seed": seed,
        "best_epoch": int(best["epoch"]),
        "val_mae": float(best["val_mae"]),
        "test_mae": float(mae),
        "seconds": seconds,
    }


def run_keras(data, seed):
    start = time.perf_counter()
    model, history, _ = run_keras_experiment(data, seed=seed, **RUNS["main"])
    seconds = time.perf_counter() - start
    best = best_epoch(history)
    return {
        "framework": "keras",
        "seed": seed,
        "best_epoch": best,
        "val_mae": float(history["val_mae"][best - 1]),
        "test_mae": test_mae(model, data),
        "seconds": seconds,
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    data = prepare_data()
    rows = []
    for seed in SEEDS:
        rows.append(run_torch(data, seed))
        rows.append(run_keras(data, seed))
        print(rows[-2:])
    table = pd.DataFrame(rows).round({"val_mae": 3, "test_mae": 3, "seconds": 1})
    table.to_csv(REPORTS / "comparison.csv", index=False)
    print(table.to_string(index=False))
    means = table.groupby("framework")[["val_mae", "test_mae", "seconds"]].mean()
    print(means.round(3))


if __name__ == "__main__":
    main()

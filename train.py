import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from torch import nn

from bikes_data import TARGET, make_rentals
from model import BikeNet
from prepare import BikeDataset, make_loaders, make_preprocessor, split_frame

SEED = 42
EPOCHS = 100
REPORTS = Path("reports")

# main run first, then one change at a time
RUNS = {
    "main": {"hidden": 32, "lr": 1e-3},
    "lr_too_small": {"hidden": 32, "lr": 1e-5},
    "lr_too_large": {"hidden": 32, "lr": 0.5},
    "huge_network": {"hidden": 512, "lr": 1e-3},
}


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_abs, n = 0.0, 0.0, 0

    with torch.set_grad_enabled(training):
        for X, y in loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_abs += (pred - y).abs().sum().item()
            n += len(y)

    return total_loss / n, total_abs / n


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs):
    history = []
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)  # put back the best epoch's weights
    return history


def prepare_data():
    df = make_rentals()
    train, val, test = split_frame(df)
    pre = make_preprocessor()
    pre.fit(train)  # fitted on train rows only
    X = {
        "train": pre.transform(train),
        "val": pre.transform(val),
        "test": pre.transform(test),
    }
    y = {
        "train": train[TARGET].to_numpy(),
        "val": val[TARGET].to_numpy(),
        "test": test[TARGET].to_numpy(),
    }
    return X, y


def run_experiment(data, hidden=32, lr=1e-3, batch_size=64, epochs=EPOCHS):
    X, y = data
    torch.manual_seed(SEED)
    datasets = [BikeDataset(X[k], y[k]) for k in ("train", "val", "test")]
    train_loader, val_loader, test_loader = make_loaders(
        *datasets, batch_size=batch_size, seed=SEED
    )
    model = BikeNet(n_inputs=X["train"].shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = fit(model, train_loader, val_loader, nn.MSELoss(), optimizer, epochs)
    return model, history, test_loader


def plot_curves(history, path, title):
    df = pd.DataFrame(history)
    best = df.loc[df["val_loss"].idxmin()]
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train loss")
    plt.plot(df["epoch"], df["val_loss"], label="validation loss")
    plt.axvline(
        best["epoch"],
        linestyle="--",
        color="gray",
        label=f"best epoch ({int(best['epoch'])})",
    )
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss (MSE, log scale)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    REPORTS.mkdir(exist_ok=True)
    data = prepare_data()
    X, y = data
    metrics = {}

    baselines = {"mean": DummyRegressor(strategy="mean"), "linear": LinearRegression()}
    for name, base in baselines.items():
        base.fit(X["train"], y["train"])
        metrics[name] = {
            "val_mae": float(mean_absolute_error(y["val"], base.predict(X["val"]))),
            "test_mae": float(mean_absolute_error(y["test"], base.predict(X["test"]))),
        }

    rows, results = [], {}
    for name, settings in RUNS.items():
        model, history, test_loader = run_experiment(data, **settings)
        df = pd.DataFrame(history)
        best = df.loc[df["val_loss"].idxmin()]
        rows.append(
            {
                "run": name,
                **settings,
                "best_epoch": int(best["epoch"]),
                "best_val_mae": round(float(best["val_mae"]), 3),
                "final_train_loss": round(float(df["train_loss"].iloc[-1]), 3),
                "final_val_loss": round(float(df["val_loss"].iloc[-1]), 3),
            }
        )
        plot_curves(history, REPORTS / f"curves_{name}.png", name)
        results[name] = (model, test_loader, df)

    # the final network is chosen on validation MAE only
    final = min(rows, key=lambda r: r["best_val_mae"])["run"]
    final_model, final_test_loader, final_df = results[final]
    plot_curves(final_df, REPORTS / "learning_curves.png", f"final network: {final}")
    final_df.to_csv(REPORTS / "history.csv", index=False)
    pd.DataFrame(rows).to_csv(REPORTS / "experiments.csv", index=False)

    # the test rows are used once, for the chosen network only
    test_loss, test_mae = run_epoch(final_model, final_test_loader, nn.MSELoss())
    best_row = final_df.loc[final_df["val_loss"].idxmin()]
    metrics["network"] = {
        "run": final,
        "val_mae": float(best_row["val_mae"]),
        "test_mae": float(test_mae),
        "test_loss": float(test_loss),
    }
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

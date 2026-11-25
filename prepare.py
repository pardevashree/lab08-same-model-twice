import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 42


# The split
def split_frame(df, seed=SEED):
    # splitting df into 60% train,40% rest
    train, rest = train_test_split(df, test_size=0.4, random_state=seed)

    # splitting remaining df equally into validation and test,20% each
    val, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, val, test


# Scaler and encoder (fitted on train only)
NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph"]
BINARY = ["is_weekend"]
CATEGORY = ["station_type"]


def make_preprocessor():
    return ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                NUMERIC,
            ),  # transforming NUMERIC columns into  standard scale
            ("bin", "passthrough", BINARY),  # keeping BINARY column as it is
            # converting CATEGORY column into three numeric columns using oneHotEncoding
            ("cat", OneHotEncoder(sparse_output=False), CATEGORY),
        ]
    )


# The Dataset class
class BikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# The loaders
def make_loaders(train_ds, val_ds, test_ds, batch_size=64, seed=SEED):
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

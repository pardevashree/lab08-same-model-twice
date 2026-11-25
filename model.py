import torch.nn as nn


class BikeNet(nn.Module):
    def __init__(self, n_inputs=8, hidden=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)

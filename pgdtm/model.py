import torch
import torch.nn as nn
import torch.nn.functional as F


class Compact1DCNN(nn.Module):
    """
    Table 3: 2 conv layers (64/16), kernel=5, maxpool=8, LeakyReLU, dropout=0.3, Adam lr=1e-4. :contentReference[oaicite:23]{index=23}
    Output uses NBF: eta * sigmoid(x). :contentReference[oaicite:24]{index=24}
    """

    def __init__(self, in_channels: int, points: int = 2560, eta: float = 0.05, dropout: float = 0.3):
        super().__init__()
        self.eta = float(eta)

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=8)

        self.conv2 = nn.Conv1d(64, 16, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=8)

        # compute flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, points)
            z = self.pool1(F.leaky_relu(self.conv1(dummy), negative_slope=0.01))
            z = self.pool2(F.leaky_relu(self.conv2(z), negative_slope=0.01))
            flat_dim = z.numel()

        # paper text says "three FC layers" but Table 3 lists 64/1;
        # we implement 2 FC layers consistent with Table 3 by default.
        self.fc1 = nn.Linear(flat_dim, 64)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.leaky_relu(self.conv1(x), negative_slope=0.01))
        x = self.pool2(F.leaky_relu(self.conv2(x), negative_slope=0.01))
        x = x.flatten(1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.drop(x)
        x = self.fc2(x)
        # NBF(·) = eta / (1 + e^{-x}) 
        return self.eta * torch.sigmoid(x)
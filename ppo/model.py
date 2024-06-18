import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    3-layer CNN to encode image
    """
    def __init__(self):
        super(CNN, self).__init__()
        # assume input image with shape (3, 64, 64)
        _CHANNELS = 16
        self.CHANNELS = _CHANNELS

        self.model = nn.Sequential(
            nn.Conv2d(3, _CHANNELS, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(_CHANNELS, _CHANNELS, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(_CHANNELS, _CHANNELS, kernel_size=3, stride=2, padding=1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)
    

class MLP(nn.Module):
    """
    2-layer MLP for PPO implementation
    """
    def __init__(self, in_features, hidden_size, out_features):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features)
        )

    def forward(self, x):
        return self.mlp(x)
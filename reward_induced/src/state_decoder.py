import torch
import torch.nn as nn
from math import log2


class StateDecoder(nn.Module):
    """
    Decode reward-induced representation to image
    """
    def __init__(self, in_size, out_size):
        super(StateDecoder, self).__init__()
        HIDDEN_SIZE = 128
        self.out_size = out_size

        self.layers = nn.Sequential(
            nn.Linear(in_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, out_size*out_size),
            nn.Tanh(),
        )
        
    def forward(self, x):
        B = x.shape[0]
        L = x.shape[1]
        x = self.layers(x)
        x = x.reshape(B, L, 1, self.out_size, self.out_size)
        return x.expand(B, L, 3, self.out_size, self.out_size)

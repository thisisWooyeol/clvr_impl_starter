import torch
import torch.nn as nn
from math import log2

from sprites_datagen.rewards import *

# Reward classes
R_CLASSES_ALL = [AgentXReward().name, AgentYReward().name, TargetXReward().name, TargetYReward().name,
             VertPosReward().name, HorPosReward().name]  
R_CLASSES_BASE = [AgentXReward().name, AgentYReward().name, TargetXReward().name, TargetYReward().name,]


class RewardPredictorModel(nn.Module):
    def __init__(self, image_shape, n_frames, T_future):
        super(RewardPredictorModel, self).__init__()
        self.n_frames = n_frames
        self.T_future = T_future
        REPR_SIZE = 64

        self.image_encoder = ImageEncoder(image_shape, REPR_SIZE)
        self.predictor_lstm = PredictorLSTM(input_size=REPR_SIZE, hidden_size=REPR_SIZE)
        self.reward_head_mlp = nn.ModuleDict({ r: MLP(REPR_SIZE, 1) for r in R_CLASSES_ALL })

    def to(self, *args, **kwargs):
        super(RewardPredictorModel, self).to(*args, **kwargs)
        self.reward_head_mlp.to(*args, **kwargs)
        return self

    def forward(self, frames, reward_type_list=None):
        """
        Args:
            frames: (n_frames + T_future, 3, H, W)
            reward_type_list: list of reward type to predict
        Returns:
            reward_aggregated: dict of reward type to predicted reward
        """
        assert frames.shape[1] == self.n_frames + self.T_future, \
                f'Expected {self.n_frames + self.T_future} size of timesteps, got: {frames.shape[1]}'

        if reward_type_list is None:
            reward_type_list = R_CLASSES_BASE
        else :
            assert all(r in R_CLASSES_ALL for r in reward_type_list)

        # encode conditioning frames & map to 64-dim observation space
        # (B, n_frames + T_future, 3, H, W) -> (B, n_frames + T_future, 64)
        representations = self.image_encoder(frames)

        # predict conditional future representations p(z_t|z_{t-1}, z_{t-2}, ...)
        # n_frames-th repr encodes the 1st future frame
        # future_repr.shape == (B, T_future, 64)
        future_repr, _ = self.predictor_lstm(representations)
        future_repr = future_repr[:, -1-self.T_future:-1]

        # hidden state to reward
        reward_aggregated = {}
        for r in reward_type_list:
            reward_pred = self.reward_head_mlp[r](future_repr)
            reward_aggregated[r] = reward_pred.squeeze(-1)

        return future_repr, reward_aggregated


class ImageEncoder(nn.Module):
    def __init__(self, image_shape: tuple, out_features: int):
        super(ImageEncoder, self).__init__()
        # assume square image and H, W should be power of 2
        assert image_shape[-1] == image_shape[-2]
        self.level = int(log2(image_shape[-1]))
        _channels = 4
        
        _convs = [
            nn.Conv2d(3, _channels, kernel_size=3, stride=2, padding=1)
        ]
        for i in range(self.level-1):
            _convs.append(nn.ReLU())
            _convs.append(
                nn.Conv2d(_channels, _channels * 2, kernel_size=3, stride=2, padding=1))
            _channels *= 2
        self.convs = nn.Sequential(*_convs)

        self.projection = nn.Linear(_channels, out_features)
    
    def forward(self, x):
        # x: (B, L, C_in, H, W)
        B = x.shape[0]
        L = x.shape[1]
        x = x.view(B*L, *x.shape[2:])

        x = self.convs(x)        
        assert x.shape == torch.Size([B*L, 2 ** self.level * 2, 1, 1]), \
                f'Expected shape: [{B}, {L}, {2 ** (1+self.level)}, 1, 1], got: {x.shape}'
        # will output (L, out_features)
        return self.projection(x.view(B, L, -1))


class PredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.h0 = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.c0 = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x: (B, L, input_size)
        h0 = self.h0.expand(1, x.shape[0], -1).contiguous()
        c0 = self.c0.expand(1, x.shape[0], -1).contiguous()
        output, hc = self.lstm(x, (h0, c0))
        return output, hc


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        HIDDEN_SIZE = 32
        self.mlp = nn.Sequential(
            nn.Linear(in_features, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, out_features)
        )

    def forward(self, x):
        return self.mlp(x)

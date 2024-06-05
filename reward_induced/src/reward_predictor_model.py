import torch
import torch.nn as nn
from math import log2

from sprites_datagen.rewards import *

# VertPosReward alias to AgentYReward, HorPosReward alias to AgentXReward
R_CLASSES_ALL = [AgentXReward().name, AgentYReward().name, TargetXReward().name, TargetYReward().name,
             VertPosReward().name, HorPosReward().name]  
R_CLASSES_BASE = [AgentXReward().name, AgentYReward().name, TargetXReward().name, TargetYReward().name,]


class RewardPredictorModel(nn.Module):
    def __init__(self, image_shape, n_frames, T_future):
        super(RewardPredictorModel, self).__init__()
        self.n_frames = n_frames
        self.T_future = T_future
        REPR_SIZE = 64

        self.image_encoder = ImageEncoder(image_shape)
        self.encoder_mlp = MLP(2 ** (1+self.image_encoder.level), REPR_SIZE)
        self.predictor_lstm = nn.LSTM(input_size=REPR_SIZE, hidden_size=REPR_SIZE)
        self.reward_head_mlp = nn.ModuleDict({ r: MLP(REPR_SIZE, 1) for r in R_CLASSES_ALL })

    def to(self, *args, **kwargs):
        super(RewardPredictorModel, self).to(*args, **kwargs)
        self.reward_head_mlp.to(*args, **kwargs)
        return self

    def forward(self, frames, reward_type_list=None):
        # assume input shape is (n_frames + T_future, 3, H, W)
        assert frames.shape[0] == self.n_frames + self.T_future, \
                f'Expected {self.n_frames + self.T_future} size of timesteps, got: {frames.shape[0]}'

        if reward_type_list is None:
            reward_type_list = R_CLASSES_BASE
        else :
            assert all(r in R_CLASSES_ALL for r in reward_type_list)

        # encode each frame & map to 64-dim observation space
        # (n_frames + T_future, 3, H, W) -> (n_frames + T_future, 64)
        representations = self.image_encoder(frames)
        representations = self.encoder_mlp(representations)

        # predict conditional future representations p(z_t|z_{t-1}, z_{t-2}, ...)
        # future_repr.shape == (T_future, 64)
        output, _ = self.predictor_lstm(representations)
        future_repr = output[-self.T_future:, :]

        # hidden state to reward
        reward_aggregated = {}
        for r in reward_type_list:
            reward_pred = self.reward_head_mlp[r](future_repr)
            reward_aggregated[r] = reward_pred.squeeze(-1)

        return reward_aggregated


class ImageEncoder(nn.Module):
    def __init__(self, image_shape: tuple):
        super(ImageEncoder, self).__init__()
        # assume square image and H, W should be power of 2
        assert image_shape[-1] == image_shape[-2]
        self.level = int(log2(image_shape[-1]))
        _channels = 4
        
        self.layers = nn.ModuleList([
            nn.Conv2d(3, _channels, kernel_size=3, stride=2, padding=1)
        ])
        for i in range(self.level-1):
            self.layers.append(nn.ReLU())
            self.layers.append(
                nn.Conv2d(_channels, _channels * 2, kernel_size=3, stride=2, padding=1))
            _channels *= 2
    
    def forward(self, x):
        # x: (L, C_in, H, W), L is the number of frames acts as batch size
        for layer in self.layers:
            x = layer(x)
        
        assert x.shape == torch.Size([x.shape[0], 2 ** self.level * 2, 1, 1]), \
                f'Expected shape: [L, {2 ** (1+self.level)}, 1, 1], got: {x.shape}'
        # will output (L, C_out)
        return torch.flatten(x, start_dim=1)


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        HIDDEN_SIZE = 32
        self.fc1 = nn.Linear(in_features, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
    
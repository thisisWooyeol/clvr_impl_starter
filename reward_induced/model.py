import torch
import torch.nn as nn
from math import log2


class RewardPredictorModel(nn.Module):
    def __init__(self, image_shape, n_frames, T_future):
        super(RewardPredictorModel, self).__init__()
        self.n_frames = n_frames
        self.T_future = T_future

        self.image_encoder = ImageEncoder(image_shape)
        self.encoder_mlp = MLP(2 ** (1+self.image_encoder.level) * self.n_frames, 64)
        self.predictor_lstm = PredictorLSTM(input_size=128, hidden_size=32)
        self.reward_head_mlp = MLP(32, 1)

    def forward(self, conditioning_frames, future_frames):
        # assume input shape is (B, n_frames, 3, 64, 64) and (B, T_future, 3, 64, 64)
        assert conditioning_frames.shape[1] == self.n_frames
        assert future_frames.shape[1] == self.T_future

        # encode each frame
        cond_features = self.image_encoder(conditioning_frames)
        future_features = self.image_encoder(future_frames)

        # map to 64-dim observation space
        cond_features = torch.flatten(cond_features, start_dim=1)
        _hidden_states = self.encoder_mlp(cond_features)
        h0, c0 = torch.chunk(_hidden_states.unsqueeze(0), chunks=2, dim=2)

        # predict future
        output = self.predictor_lstm(future_features, (h0, c0))

        # hidden state to reward
        reward_pred = []
        for i in range(self.T_future):
            reward_pred.append(self.reward_head_mlp(output[:, i, :]))
        reward_pred = torch.stack(reward_pred, dim=1)

        return reward_pred.squeeze()


class ImageEncoder(nn.Module):
    def __init__(self, image_shape: tuple):
        super(ImageEncoder, self).__init__()
        self.level = int(log2(image_shape[-1]))
        _channels = 4
        
        self.convs = nn.ModuleList([
            nn.Conv2d(3, _channels, kernel_size=3, stride=2, padding=1)
        ])
        for i in range(self.level-1):
            self.convs.append(
                nn.Conv2d(_channels, _channels * 2, kernel_size=3, stride=2, padding=1))
            _channels *= 2
    
    def forward(self, x):
        # x: (B, L, C, H, W)
        output = []
        for b in range(x.shape[0]):
            y = x[b]
            for conv in self.convs:
                y = torch.relu(conv(y))
            output.append(y)

        output = torch.stack(output, dim=0)
        assert output.shape == torch.Size([x.shape[0], x.shape[1], 2 ** self.level * 2, 1, 1]), f'Expected shape: [B, L, 128, 1, 1], got: {output.shape}'

        return torch.flatten(output, start_dim=2)


class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class PredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, hidden_states: tuple):
        x, _ = self.lstm(x, hidden_states)
        return x

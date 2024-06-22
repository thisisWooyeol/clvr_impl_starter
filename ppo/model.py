import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


class MLPActorCritic(nn.Module):
    """
    Actor-Critic agent with 2-layer MLPs
    """
    def __init__(self, repr_dim, hidden_size, action_dim, action_std=0.5, encoder=None):
        super(MLPActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(repr_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(repr_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.encoder = encoder

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std**2).to(device)

    def get_action(self, obs):
        if self.encoder is not None:
            obs = self.encoder(obs)
        action_mean = self.actor(obs)
        cov_mat = torch.diag(self.action_var).unsqueeze(0)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(obs)

        return action.detach(), action_logprob.detach(), state_value.detach()

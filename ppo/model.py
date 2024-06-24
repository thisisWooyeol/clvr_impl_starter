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
        # assume input image with shape (B, L, 3, 64, 64)
        _CHANNELS = 16
        self.CHANNELS = _CHANNELS

        self.model = nn.Sequential(
            nn.Conv2d(3, _CHANNELS, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(_CHANNELS, _CHANNELS, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(_CHANNELS, _CHANNELS, kernel_size=3, stride=2),
            nn.Flatten()
        )

    def forward(self, x):
        # x: (B, L, 3, 64, 64)
        B = x.shape[0]
        L = x.shape[1]
        x = x.view(B*L, 3, 64, 64)

        return self.model(x).view(B, L, -1)
    

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
        """
        Get action from the policy network given observation
        Usage: Rollout phase

        Args:
            obs: observation of shape (num_envs, H, W)
            action: action of shape (num_envs, action_dim)
        """
        if self.encoder is not None:
            # unsqueeze as encoder expects (B, L, C, H, W)
            obs = torch.stack([obs] * 3, dim=-3)
            obs = self.encoder(obs.unsqueeze(dim=1))
            # squeeze as encoder returns (B, L, C)
            obs = obs.squeeze(dim=1)
        action_mean = self.actor(obs)
        cov_mat = torch.diag(self.action_var).unsqueeze(0)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(obs)

        return action.detach(), action_logprob.detach(), state_value.squeeze(-1).detach()

    def evaluate(self, obs, action):
        """
        Evaluate actions given observation
        Usage: Training phase

        Args:
            obs: batched observation of shape (B, num_envs, H, W)
            action: batched action of shape (B, num_envs, action_dim)
        """
        if self.encoder is not None:
            obs = torch.stack([obs] * 3, dim=-3)
            obs = self.encoder(obs)
        action_mean = self.actor(obs)
        cov_mat = torch.diag(self.action_var).unsqueeze(0)
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(obs)

        return action_logprob, state_value.squeeze(-1), dist_entropy
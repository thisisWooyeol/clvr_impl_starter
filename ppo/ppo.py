import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

import sprites_env
from typing import Union

from .model import CNN, MLP, MLPActorCritic
from reward_induced.src.reward_predictor import RewardPredictor, ImageEncoder


class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_shape, action_shape, device):
        self.buffer_len = n_steps // n_envs
        self.current_step = 0
        self.obs = torch.zeros((self.buffer_len + 1, n_envs, *obs_shape), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.buffer_len, n_envs, *action_shape), dtype=torch.float32).to(device)
        self.rewards = np.zeros((self.buffer_len, n_envs), dtype=np.float32)
        self.terminals = np.zeros((self.buffer_len, n_envs), dtype=np.float32)

    def append(self, obs, action, reward, terminal):
        if self.current_step == self.buffer_len:
            self._append_last_obs(obs)
        else:
            self.obs[self.current_step] = obs
            self.actions[self.current_step] = action
            self.rewards[self.current_step] = reward
            self.terminals[self.current_step] = terminal
            self.current_step += 1

    def _append_last_obs(self, obs):
        self.obs[-1] = obs

    def reset(self):
        self.current_step = 0
        self.obs = torch.zeros_like(self.obs)
        self.actions = torch.zeros_like(self.actions)
        self.rewards = np.zeros_like(self.rewards)
        self.terminals = np.zeros_like(self.terminals)


class PPO(nn.Module):
    """
    PPO agent with multiple baseline support

    type of baseline:
        - cnn                           : 3-layer CNN to encode image
        - image-scratch                 : encoder from RewardPredictor but initialized randomly
        - image-reconstruction          : encoder from image reconstructin task (freeze encoder)
        - image-reconstruction-finetune : encoder from image reconstruction task (finetune encoder)
        - reward-prediction             : encoder from reward prediction task (freeze encoder)
        - reward-prediction-finetune    : encoder from reward prediction task (finetune encoder)
        - oracle                        : state representation is given by the environment

        representations are fed to the 2-layer MLPs, 
        - only CNN baseline has 64 hidden states 
        - others have 32 hidden states

    Args:
        (args are referred from the Stable Baselines PPO implementation)

        policy: str,                            type of policy network (baseline)
        env: str,                               the environment to learn from
        num_envs: int,                          number of environments to run in parallel
        learning_rate: float,                   learning rate
        n_steps: int,                           number of steps to run for each environment per update
        batch_size: int,                        minibatch size
        gamma: float,                           discount factor
        clip_range: float,                      clip range from (0, 1)
        tensorboard_log: str | None,            the log location for tensorboard (if None, no logging)
        verbose: int,                           the verbosity level: 0 none, 1 training information, 2 debug
        seed: int | None,                       Seed for the pseudo-random generators
        device: device | str,                   Device (cpu, cuda, ...) on which the code should be run. 
    """
    def __init__(
            self,
            policy_type: str,
            env: str,
            num_envs: int = 2,
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            gamma: float = 0.99,
            clip_range: float = 0.2,
            tensorboard_log: str = None,
            verbose: int = 0,
            seed: int = None,
            device: Union[torch.device, str] = 'auto',
    ):
        super(PPO, self).__init__()
        # make gym environment
        self._policy_env_sanity_check(policy_type, env)
        self.envs = gym.make_vec(env, num_envs=num_envs)

        # By default, use image shape (3, 64, 64), and 64 hidden states
        self.image_shape = (3, 64, 64)
        self.hidden_size = 32

        _encoder = self._setup_encoder(policy_type)
        self.policy = self._setup_actor_critic(policy_type, encoder=_encoder)

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_range = clip_range
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.seed = seed
        self.device = device if device != 'auto' else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def _policy_env_sanity_check(policy, env):
        if policy == 'oracle':
            if 'State' not in env:
                raise ValueError(f'Invalid policy-environment combination: {policy}-{env}')
        elif 'State' in env:
            raise ValueError(f'Invalid policy-environment combination: {policy}-{env}')

    def _setup_encoder(self, policy_type):
        if policy_type == 'cnn':
            return CNN()
        elif policy_type == 'image-scratch':
            return ImageEncoder(self.image_shape, self.hidden_size)
        elif policy_type == 'image-reconstruction' \
            or policy_type == 'image-reconstruction-finetune':
            return ImageEncoder(self.image_shape, self.hidden_size)
            # TODO: load encoder from image reconstruction task
        elif policy_type == 'reward-prediction' \
            or policy_type == 'reward-prediction-finetune':
            _reward_predictor = RewardPredictor(self.image_shape, 1, 29)
            # TODO: use general method to load encoder from reward prediction task
            _reward_predictor.load_state_dict(torch.load('reward_induced/models/encoder/encoder_dist0_final.pt'))
            _image_encoder = _reward_predictor.image_encoder
            if policy_type == 'reward-prediction':
                _image_encoder.requires_grad_(False)
            return _image_encoder
        elif policy_type == 'oracle':
            return None
        else:
            raise ValueError(f'Invalid policy: {policy_type}')
        
    def _setup_actor_critic(self, policy_type, encoder):
        action_dim = self.envs.single_action_space.shape[0]
        if policy_type == 'cnn':
            repr_dim = (self.image_shape[-1] // 8 - 1) ** 2 * self.encoder.CHANNELS
            return MLPActorCritic(repr_dim, self.hidden_size * 2, action_dim, encoder=encoder)
        elif policy_type == 'image-scratch' \
            or policy_type == 'image-reconstruction' \
            or policy_type == 'image-reconstruction-finetune' \
            or policy_type == 'reward-prediction' \
            or policy_type == 'reward-prediction-finetune':
            return MLPActorCritic(64, self.hidden_size, action_dim, encoder=encoder)
        elif policy_type == 'oracle':
            repr_dim = self.envs.single_observation_space.shape[0]
            return MLPActorCritic(repr_dim, self.hidden_size, action_dim, encoder=encoder)
        

    def learn(
            self, 
            total_timesteps: int, 
            log_interval: int,
            tb_log_name: str = 'PPO',
            reset_num_timesteps: bool = True,
            progress_bar: bool = False
    ):
        self.policy.to(self.device)
        buffer = RolloutBuffer(self.n_steps, self.envs.num_envs, (4,), self.envs.single_action_space.shape, self.device)

        obs, _ = self.envs.reset(seed=self.seed)

        for update in range(total_timesteps // self.n_steps):
            # Rollout phase
            for _ in range(self.n_steps // self.envs.num_envs):
                obs = torch.FloatTensor(obs).to(self.device)
                action, action_logprob, state_value = self.policy.get_action(obs)

                new_obs, reward, terminated, _, _ = self.envs.step(action.cpu().numpy())
                buffer.append(obs, action, reward, terminated)

                obs = new_obs
                
            # Last observation
            buffer.append(torch.FloatTensor(obs), None, None, None)

            print(f"update #{update + 1}: ", buffer.obs[0], buffer.obs.shape, buffer.actions[0], buffer.actions.shape, buffer.terminals)
            buffer.reset()

        self.envs.close()



if __name__ == '__main__':
    ppo = PPO('oracle', 'SpritesState-v0', num_envs=4)
    ppo.learn(total_timesteps=9192, log_interval=100)

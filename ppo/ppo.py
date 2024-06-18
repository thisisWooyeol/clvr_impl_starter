import gym.vector
import torch
import torch.nn as nn

import gym
import sprites_env
from typing import Any, Dict, Union

from .model import CNN, MLP
from reward_induced.src.reward_predictor import RewardPredictor, ImageEncoder


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
            policy: str,
            env: str,
            num_envs: int = 1,
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
        self._policy_env_sanity_check(policy, env)
        # self.env = gym.vector.make(env, num_envs=num_envs)
        self.env = gym.make(env)

        # By default, use image shape (3, 64, 64), and 64 hidden states
        self.image_shape = (3, 64, 64)
        self.hidden_size = 32

        self.policy = policy
        self.encoder = self._setup_encoder(policy)
        self.actor, self.critic = self._setup_actor_critic(policy)


        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.clip_range = clip_range
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.device = device
    
    @staticmethod
    def _policy_env_sanity_check(policy, env):
        if policy == 'oracle':
            if 'State' not in env:
                raise ValueError(f'Invalid policy-environment combination: {policy}-{env}')
        elif 'State' in env:
            raise ValueError(f'Invalid policy-environment combination: {policy}-{env}')

    def _setup_encoder(self, policy):
        if policy == 'cnn':
            return CNN()
        elif policy == 'image-scratch':
            return ImageEncoder(self.image_shape, self.hidden_size)
        elif policy == 'image-reconstruction' \
            or policy == 'image-reconstruction-finetune':
            return ImageEncoder(self.image_shape, self.hidden_size)
            # TODO: load encoder from image reconstruction task
        elif policy == 'reward-prediction' \
            or policy == 'reward-prediction-finetune':
            _reward_predictor = RewardPredictor(self.image_shape, 1, 29)
            # TODO: use general method to load encoder from reward prediction task
            _reward_predictor.load_state_dict(torch.load('reward_induced/models/encoder/encoder_dist0_final.pt'))
            _image_encoder = _reward_predictor.image_encoder
            if policy == 'reward-prediction':
                _image_encoder.requires_grad_(False)
            return _image_encoder
        elif policy == 'oracle':
            return None
        else:
            raise ValueError(f'Invalid policy: {policy}')
        
    def _setup_actor_critic(self, policy):
        if policy == 'cnn':
            in_features = (self.image_shape[-1] // 8 - 1) ** 2 * self.encoder.CHANNELS
            actor = MLP(in_features, self.hidden_size * 2, self.env.action_space.shape[0])
            critic = MLP(in_features, self.hidden_size * 2, 1)
            return actor, critic
        elif policy == 'image-scratch' \
            or policy == 'image-reconstruction' \
            or policy == 'image-reconstruction-finetune' \
            or policy == 'reward-prediction' \
            or policy == 'reward-prediction-finetune':
            in_features = self.hidden_size
            actor = MLP(in_features, self.hidden_size, self.env.action_space.shape[0])
            critic = MLP(in_features, self.hidden_size, 1)
            return actor, critic
        elif policy == 'oracle':
            in_features = self.env.observation_space.shape[0]
            actor = MLP(in_features, self.hidden_size, self.env.action_space.shape[0])
            critic = MLP(in_features, self.hidden_size, 1)
            return actor, critic
        

    def learn(
            self, 
            total_timesteps: int, 
            log_interval: int,
            tb_log_name: str = 'PPO',
            reset_num_timesteps: bool = True,
            progress_bar: bool = False
    ):
        obs = self.env.reset()

        for t in range(total_timesteps):
            action = self.env.action_space.sample() # debug code
            obs, reward, done, info = self.env.step(action)

            if t % log_interval == 0:
                print(f't: {t}, obs: {obs}, reward: {reward}, done: {done}, info: {info}')

        self.env.close()



if __name__ == '__main__':
    ppo = PPO('oracle', 'SpritesState-v0')
    ppo.learn(total_timesteps=1000, log_interval=100)

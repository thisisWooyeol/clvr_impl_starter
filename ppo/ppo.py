import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

import sprites_env
from typing import Union
import logging

from .model import CNN, MLPActorCritic
from reward_induced.src.reward_predictor import RewardPredictor, ImageEncoder


class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_shape, action_shape, device):
        """
        Length of each buffer:
            obs, state_values, terminals: n_steps + 1
            actions, action_logprobs, rewards: n_steps
        """
        self.n_steps = n_steps
        self.current_step = 0
        self.device = device
        self.obs = torch.zeros((self.n_steps + 1, n_envs, *obs_shape), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.n_steps, n_envs, *action_shape), dtype=torch.float32).to(device)
        self.action_logprobs = torch.zeros((self.n_steps, n_envs), dtype=torch.float32).to(device)
        self.state_values = torch.zeros((self.n_steps + 1, n_envs), dtype=torch.float32).to(device)
        self.rewards = np.zeros((self.n_steps, n_envs), dtype=np.float32)
        self.terminals = np.zeros((self.n_steps + 1, n_envs), dtype=np.float32)

    def append(self, obs, action, action_logprob, state_value, reward, terminal):
        if self.current_step == self.n_steps:
            self._append_last_obs(obs, state_value, terminal)
        else:
            self.obs[self.current_step].copy_(obs)
            self.actions[self.current_step].copy_(action)
            self.action_logprobs[self.current_step].copy_(action_logprob)
            self.state_values[self.current_step].copy_(state_value)
            self.rewards[self.current_step] = reward
            self.terminals[self.current_step] = terminal
            self.current_step += 1

    def _append_last_obs(self, obs, state_value, terminal):
        self.obs[-1].copy_(obs)
        self.state_values[-1].copy_(state_value)
        self.terminals[-1] = terminal

    def reset(self):
        self.current_step = 0
        self.obs = torch.zeros_like(self.obs)
        self.actions = torch.zeros_like(self.actions)
        self.action_logprobs = torch.zeros_like(self.action_logprobs)
        self.state_values = torch.zeros_like(self.state_values)
        self.rewards = np.zeros_like(self.rewards)
        self.terminals = np.zeros_like(self.terminals)

    def compute_returns(self, gamma, gae_lambda):
        # Use GAE to compute advantages
        rewards = torch.from_numpy(self.rewards).to(self.device)
        terminals = torch.from_numpy(self.terminals).to(self.device)    
        next_state_values = self.state_values[1:]
        curr_state_values = self.state_values[:-1]

        # TD-errors per each step
        deltas = rewards + gamma * next_state_values * (1 - terminals[1:]) - curr_state_values

        gae = torch.zeros_like(terminals).to(self.device)
        for t in reversed(range(self.n_steps)):
            gae[t] = deltas[t] + gamma * gae_lambda * (1 - terminals[t+1]) * gae[t+1]
        gae = gae[:-1]  # last gae value: used for for-loop computation, remove it
        returns = gae + curr_state_values

        return returns, gae


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
        env_name: str,                          the environment to learn from
        num_envs: int,                          number of environments to run in parallel
        learning_rate: float,                   learning rate
        n_steps: int,                           number of steps to run for each environment per update
        batch_size: int,                        minibatch size
        n_epochs: int,                          number of epochs to update the policy
        gamma: float,                           discount factor
        gae_lambda: float,                      factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range: float,                      clip range from (0, 1)
        normalize_advantage: bool,              normalize advantage
        ent_coef: float,                        entropy coefficient
        vf_coef: float,                         value function coefficient
        tensorboard_log: str | None,            the log location for tensorboard (if None, no logging)
        verbose: int,                           the verbosity level: 0 warning, 1 info, 2 debug
        seed: int | None,                       Seed for the pseudo-random generators
        device: device | str,                   Device (cpu, cuda, ...) on which the code should be run. 
    """
    def __init__(
            self,
            policy_type: str,
            env_name: str,
            num_envs: int = 2,
            learning_rate: float = 3e-4,
            n_steps: int = 2048,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            normalize_advantage: bool = True,
            ent_coef: float = 0.01,
            vf_coef: float = 0.5,
            tensorboard_log: str = None,
            verbose: int = 0,
            seed: int = None,
            device: Union[torch.device, str] = 'auto',
    ):
        super(PPO, self).__init__()
        # make gym environment
        self._policy_env_sanity_check(policy_type, env_name)
        self.env_name = env_name  # only use for logging purposes
        self.envs = gym.make_vec(env_name, num_envs=num_envs)

        # By default, use image shape (3, 64, 64), and 32 hidden states
        self.image_shape = (3, 64, 64)
        self.hidden_size = 32

        self.policy_type = policy_type  # only use for logging purposes
        _encoder = self._setup_encoder(policy_type)
        self.policy = self._setup_actor_critic(policy_type, encoder=_encoder)

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.seed = seed
        self.device = device if device != 'auto' else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @staticmethod
    def _policy_env_sanity_check(policy, env_name):
        if policy == 'oracle':
            if 'State' not in env_name:
                raise ValueError(f'Invalid policy-environment combination: {policy}-{env_name}')
        elif 'State' in env_name:
            raise ValueError(f'Invalid policy-environment combination: {policy}-{env_name}')

    def _setup_encoder(self, policy_type):
        if policy_type == 'cnn':
            return CNN()
        elif policy_type == 'image-scratch':
            return ImageEncoder(self.image_shape, self.hidden_size)
        elif policy_type == 'image-reconstruction' \
            or policy_type == 'image-reconstruction-finetune':
            image_encoder = ImageEncoder(self.image_shape, self.hidden_size)
            # TODO: load encoder from image reconstruction task
            if policy_type == 'image-reconstruction':
                image_encoder.requires_grad_(False)
            return image_encoder
        elif policy_type == 'reward-prediction' \
            or policy_type == 'reward-prediction-finetune':
            _reward_predictor = RewardPredictor(self.image_shape, 1, 29)
            _reward_predictor.load_state_dict(torch.load(f'reward_induced/models/encoder/encoder_dist{self.env_name[-1]}_final.pt'))
            image_encoder = _reward_predictor.image_encoder
            if policy_type == 'reward-prediction':
                image_encoder.requires_grad_(False)
            return image_encoder
        elif policy_type == 'oracle':
            return None
        else:
            raise ValueError(f'Invalid policy: {policy_type}')
        
    def _setup_actor_critic(self, policy_type, encoder):
        action_dim = self.envs.single_action_space.shape[0]
        if policy_type == 'cnn':
            repr_dim = ((self.image_shape[-1] // 8 - 1) ** 2) * encoder.CHANNELS
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
        
    def collect_rollouts(self, buffer: RolloutBuffer):
        obs, _ = self.envs.reset(seed=self.seed)
        terminated = np.zeros(self.envs.num_envs)
        for _ in range(self.n_steps):
            obs = torch.FloatTensor(obs).to(self.device)
            with torch.no_grad():
                action, action_logprob, state_value = self.policy.get_action(obs)

            next_obs, reward, next_terminated, _, _ = self.envs.step(action.cpu().numpy())
            buffer.append(obs, action, action_logprob, state_value, reward, terminated)

            obs = next_obs
            terminated = next_terminated
            
        # Last observation
        obs = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            _, _, state_value = self.policy.get_action(obs)
        buffer.append(obs, None, None, state_value, None, terminated)

    def validate(self, num_episodes: int = 10):
        """
        Compute `num_episodes` * `num_envs` returns for the current policy
        """
        self.policy.to(self.device)
        returns = []
        for _ in range(num_episodes):
            obs, _ = self.envs.reset()
            total_reward = 0
            done = np.asarray([False] * self.envs.num_envs)
            while not done.all():  # is only working for SpritesEnv
                obs = torch.FloatTensor(obs).to(self.device)
                with torch.no_grad():
                    action, _, _ = self.policy.get_action(obs)
                obs, reward, done, _, _ = self.envs.step(action.cpu().numpy())
                total_reward += reward
            returns.append(total_reward)
        self.envs.close()
        returns = np.concatenate(returns)

        return returns
        
    def update(self, buffer: RolloutBuffer, logger: logging.Logger):
        returns, advantages = buffer.compute_returns(self.gamma, self.gae_lambda)
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
            returns = advantages + buffer.state_values[:-1]
        logger.debug(f'advantages: {advantages}')
        logger.debug(f'returns: {returns}')


        for _ in range(self.n_epochs):
            shuffled_indices = torch.randperm(self.n_steps)
            for indices in range(0, self.n_steps, self.batch_size):
                batch_indices = shuffled_indices[indices:indices + self.batch_size]
                obs_batch = buffer.obs[batch_indices]
                action_batch = buffer.actions[batch_indices]
                old_action_logprobs = buffer.action_logprobs[batch_indices]
                advs_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]

                # Evaluate old actions
                action_logprobs, state_values, dist_entropy = self.policy.evaluate(obs_batch, action_batch)

                # Compute the ratio between old and new actions
                ratios = torch.exp(action_logprobs - old_action_logprobs)

                # Compute surrogate loss
                surr1 = ratios * advs_batch
                surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advs_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                critic_loss = (state_values - returns_batch).pow(2).mean()

                # Compute total loss
                loss = actor_loss + self.ent_coef * critic_loss - self.ent_coef * dist_entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Log
        logger.debug(f'    actor loss:      {actor_loss.item()}')
        logger.debug(f'    critic loss:     {critic_loss.item()}')
        logger.debug(f'    entropy:         {dist_entropy.mean().item()}')

    def learn(
            self, 
            total_timesteps: int, 
            log_interval: int,
            tb_log_name: str = 'PPO',
            reset_num_timesteps: bool = True,
            progress_bar: bool = False
    ):
        self.policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        buffer = RolloutBuffer(self.n_steps, self.envs.num_envs, self.envs.single_observation_space.shape, self.envs.single_action_space.shape, self.device)

        logger = _setup_logger(f'ppo/logs/{self.policy_type}/{self.policy_type}-{self.env_name}.log', self.verbose)
        logger.info(f"""[INFO] Learning Configuration:
            Policy
                - Policy type:      {self.policy_type}
                - Policy network:   {self.policy}
            Environment
                - Env name:         {self.env_name}
                - Num envs:         {self.envs.num_envs}
                - Spec:             {self.envs.spec}
            Rollout
                - Rollout Steps:    {self.n_steps}
            Training
                - Total timesteps:  {total_timesteps}
                - N epochs:         {self.n_epochs}
                - Batch size:       {self.batch_size}
                - gamma (discount): {self.gamma}
                - GAE lambda:       {self.gae_lambda}
                - clip range:       {self.clip_range}
                - Normalize adv:    {self.normalize_advantage}
                - Entropy coef:     {self.ent_coef}
                - Value coef:       {self.vf_coef}
                - Optimizer:        Adam
                - Learning rate:    {self.learning_rate}
            Device
                - Device:           {self.device}
            Verbosity
                - Verbose level:    {self.verbose}
                - Log interval:     {log_interval}
        """)


        for update in range(total_timesteps // (self.n_steps * self.envs.num_envs)):
            logger.info(f'[INFO] Update {update} ({update * self.n_steps * self.envs.num_envs} / {total_timesteps} steps)')
            self.collect_rollouts(buffer)
            self.update(buffer, logger)
            buffer.reset()

            if self.verbose > 0:
                returns = self.validate()
                logger.info(f'    (Validation) Returns: {np.mean(returns)} +/- {np.std(returns)}')
        logger.info(f'[INFO] Training finished ({total_timesteps} steps)')

        self.envs.close()
        self.save(f'ppo/models/{self.policy_type}/{self.policy_type}-{self.env_name}-{total_timesteps}steps.pt')
        logger.info(f'[INFO] Model saved at ppo/models/{self.policy_type}/{self.policy_type}-{self.env_name}-{total_timesteps}steps.pt')

    def save(self, path):
        torch.save(self.policy.state_dict(), path)



def _setup_logger(log_file_path, verbose):
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(log_file_path, mode='w')
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    if verbose == 2:
        logger.setLevel(logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    return logger


if __name__ == '__main__':
    ppo = PPO('cnn', 'Sprites-v0', num_envs=4, verbose=1)
    ppo.learn(total_timesteps=2**22, log_interval=10)

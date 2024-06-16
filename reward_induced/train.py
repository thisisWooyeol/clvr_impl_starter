import torch
import torch.nn as nn

from general_utils import AttrDict
from reward_induced.src.reward_predictor import RewardPredictor
from reward_induced.src.state_decoder import StateDecoder
from sprites_datagen.moving_sprites import MovingSpriteDataset
import sprites_datagen.rewards as rewards_module

import logging
import matplotlib.pyplot as plt


def evaluate_encoder(
        shapes_per_traj,
        batch_size=128,
        n_frames=1,
        T_future=29,
        rewards = ['AgentXReward', 'AgentYReward', 'TargetXReward', 'TargetYReward'],
        model_save_prefix='reward_induced/models/encoder/encoder',
        log_file_prefix='reward_induced/logs/evaluate_encoder'):
    env_mode = f'dist{shapes_per_traj - 2}' if shapes_per_traj > 2 else rewards[0] # HorPosReward or VertPosReward case
    log_file_path = log_file_prefix + f'_{env_mode}.log'
    model_path = model_save_prefix + f'_{env_mode}_final.pt'

    logger = _setup_logger(log_file_path)
    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=shapes_per_traj,
        rewards=[ getattr(rewards_module, r) for r in rewards ],
        batch_size=batch_size,
    )
    dataset = MovingSpriteDataset(spec)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size) # use 128 in default for averaging
    rewards = [r.NAME for r in spec.rewards]

    model = RewardPredictor((3,64,64), n_frames, T_future, reward_type_list=rewards)
    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    logger.info(f"[INFO] Loaded Reward Predictor from {model_path}")
    logger.info(f"""[INFO] Evaluating Reward Predictor with the following configuration:
        - Input shape: (3,64,64)
        - Number of frames: {n_frames}
        - Number of future frames: {T_future}
        - Reward types: {rewards}
        - Shapes per trajectory: {shapes_per_traj}
        - Batch size: {batch_size}
        - Model loaded from: {model_path}""")
    
    data = next(iter(dataloader))
    frames = data['images'].to(device)
    true_rewards = torch.stack([data['rewards'][r] for r in rewards])[:,:,-T_future:].to(device)
    with torch.no_grad():
        _, pred_rewards = model(frames)
        pred_rewards = torch.stack([pred_rewards[r] for r in rewards])

    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(pred_rewards, true_rewards)

    logger.info(f"True rewards from 0th traj: {true_rewards[:,0,:]}")
    logger.info(f"Predicted rewards from 0th traj: {pred_rewards[:,0,:]}")
    logger.info(f"""Losses:
        - AgentXReward:  {torch.mean(loss[0]).item()}
        - AgentYReward:  {torch.mean(loss[1]).item()}
        - TargetXReward: {torch.mean(loss[2]).item()}
        - TargetYReward: {torch.mean(loss[3]).item()}""")
    logger.info(f"Evaluation complete.")   



def train_encoder(
        shapes_per_traj,
        batch_size,
        n_frames=1,
        T_future=29,
        rewards = ['AgentXReward', 'AgentYReward', 'TargetXReward', 'TargetYReward'],
        n_iters=1000,
        save_every=100,
        log_file_prefix='reward_induced/logs/train_encoder',
        model_save_prefix='reward_induced/models/encoder/encoder',
        plot_save_prefix='reward_induced/logs/encoder'):
    env_mode = f'dist{shapes_per_traj - 2}' if shapes_per_traj > 2 else rewards[0] # HorPosReward or VertPosReward case
    log_file_path = log_file_prefix + f'_{env_mode}.log'
    model_save_prefix = model_save_prefix + f'_{env_mode}'
    plot_save_path = plot_save_prefix + f'_{env_mode}.png'

    logger = _setup_logger(log_file_path)

    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=shapes_per_traj,
        rewards=[ getattr(rewards_module, r) for r in rewards ],
        batch_size=batch_size,
    )
    dataset = MovingSpriteDataset(spec)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    rewards = [r.NAME for r in spec.rewards]

    model = RewardPredictor((3,64,64), n_frames, T_future, reward_type_list=rewards)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"[INFO] Training Reward Predictor on device: {device}")
    logger.info(f"""[INFO] Training with the following configuration:
        - Input shape: (3,64,64)
        - Number of frames: {n_frames}
        - Number of future frames: {T_future}
        - Reward types: {rewards}
        - Number of iterations: {n_iters}
        - Shapes per trajectory: {shapes_per_traj}
        - Batch size: {batch_size}""")


    losses = []
    for i in range(n_iters):
        # get data
        data = next(iter(dataloader))
        frames = data['images'].to(device)
        true_rewards = torch.stack([data['rewards'][r] for r in rewards])[:,:,-T_future:].to(device)

        # forward pass
        _, pred_rewards = model(frames)
        pred_rewards = torch.stack([pred_rewards[r] for r in rewards])

        # compute loss for each reward type
        loss = loss_fn(pred_rewards, true_rewards)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log message
        print(f"[{i+1}/{n_iters}] Loss: {loss.item()}", end='\r')
        if (i+1) % save_every == 0:
            logger.info(f"[{i+1}/{n_iters}] Loss: {loss.item()}")

        # save intermediate model
        if (i+1) % (save_every * 4) == 0:
            torch.save(model.state_dict(), f'{model_save_prefix}_{i+1}.pt')
            logger.info(f"Intermediate model save at {model_save_prefix}_{i+1}.pt")


    # plot losses and save
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Reward Predictor Loss')
    plt.savefig(plot_save_path)
    logger.info(f"Loss plot saved at {plot_save_path}")

    torch.save(model.state_dict(), f'{model_save_prefix}_final.pt')
    logger.info(f"Final model saved at {model_save_prefix}_final.pt")
    logger.info(f"Training complete.")


def _setup_logger(log_file_path):
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler(log_file_path, mode='w')
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.INFO)
    return logger
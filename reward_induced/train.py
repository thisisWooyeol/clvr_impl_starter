import torch
import torch.nn as nn
import numpy as np

import cv2
import imageio
import matplotlib.pyplot as plt

from general_utils import AttrDict, make_image_seq_strip
from reward_induced.src.reward_predictor import RewardPredictor
from reward_induced.src.state_decoder import StateDecoder
from sprites_datagen.moving_sprites import MovingSpriteDataset
import sprites_datagen.rewards as rewards_module
from reward_induced.utils import _setup_logger



def visualize_decoder(
        shapes_per_traj,
        n_frames=1,
        T_future=29,
        rewards = ['HorPosReward'],  # or ['VertPosReward']
        encoder_prefix='reward_induced/models/encoder/encoder',
        decoder_prefix='reward_induced/models/decoder/decoder',
        log_file_prefix='reward_induced/logs/decoder/visualize_decoder',
        img_save_prefix='reward_induced/logs/decoder/'):
    # Consider HorPosReward or VertPosReward case
    env_mode = f'dist{shapes_per_traj - 2}' if shapes_per_traj >= 2 else rewards[0]
    encoder_path = encoder_prefix + f'_{env_mode}_final.pt'
    decoder_path = decoder_prefix + f'_{env_mode}_final.pt'
    log_file_path = log_file_prefix + f'_{env_mode}.log'
    img_save_prefix += env_mode

    logger = _setup_logger(log_file_path)

    spec = AttrDict(
        resolution=64,
        max_seq_len=30,
        max_speed=0.05,
        obj_size=0.2,
        shapes_per_traj=shapes_per_traj,
        rewards=[ getattr(rewards_module, r) for r in rewards ],
        batch_size=1,
    )
    dataset = MovingSpriteDataset(spec)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    rewards = [r.NAME for r in spec.rewards]

    encoder = RewardPredictor((3,64,64), n_frames, T_future, reward_type_list=rewards)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder = StateDecoder(in_size=64, out_size=64)
    decoder.load_state_dict(torch.load(decoder_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)
    logger.info(f"[INFO] Visualizing State Decoder on device: {device}")
    logger.info(f"""[INFO] Visualizing with the following configuration:
        - Input shape: (3,64,64)
        - Number of frames: {n_frames}
        - Number of future frames: {T_future}
        - Reward types: {rewards}
        - Shapes per trajectory: {shapes_per_traj}
        - Batch size: 1
        - Encoder loaded from: {encoder_path}
        - Decoder loaded from: {decoder_path}""")


    data = next(iter(dataloader))
    frames = data['images'].to(device)

    # Save ground truth sequence as seq image and gif
    true_frames = ( data['images'].numpy() + 1 ) * (255.0 / 2)
    true_seq_frames = make_image_seq_strip([true_frames], sep_val=255.0).astype(np.uint8)
    cv2.imwrite(f'{img_save_prefix}_true_seq.png', true_seq_frames[0].transpose(1,2,0))
    logger.info(f"True sequence img saved at {img_save_prefix}_true_seq.png")

    with imageio.get_writer(f'{img_save_prefix}_true_seq.gif', mode='I', duration=1.0) as writer:
        for frame in true_frames[0]:
            writer.append_data(frame.astype(np.uint8))
    logger.info(f"True sequence gif saved at {img_save_prefix}_true_seq.gif")

    # Reconstruct frames
    with torch.no_grad():
        recon_frames = decoder(encoder(frames)[0])
    recon_frames = ( recon_frames.cpu().numpy() + 1 ) * (255.0 / 2)

    recon_seq_frames = make_image_seq_strip([recon_frames], sep_val=255.0).astype(np.uint8)
    cv2.imwrite(f'{img_save_prefix}_recon_seq.png', recon_seq_frames[0].transpose(1,2,0))
    logger.info(f"Reconstructed sequence img saved at {img_save_prefix}_recon_seq.png")

    with imageio.get_writer(f'{img_save_prefix}_recon_seq.gif', mode='I', duration=1.0) as writer:
        for frame in recon_frames[0]:
            writer.append_data(frame.astype(np.uint8))
    logger.info(f"Reconstructed sequence gif saved at {img_save_prefix}_recon_seq.gif")
    
    logger.info(f"Visualization complete.")



def train_decoder(
        shapes_per_traj,
        batch_size,
        n_frames=1,
        T_future=29,
        rewards = ['HorPosReward'],  # or ['VertPosReward']
        n_iters=1000,
        save_every=100,
        encoder_prefix='reward_induced/models/encoder/encoder',
        log_file_prefix='reward_induced/logs/decoder/train_decoder',
        model_save_prefix='reward_induced/models/decoder/decoder',
        plot_save_prefix='reward_induced/logs/decoder/decoder'):
    # Consider HorPosReward or VertPosReward case
    env_mode = f'dist{shapes_per_traj - 2}' if shapes_per_traj >= 2 else rewards[0]
    encoder_path = encoder_prefix + f'_{env_mode}_final.pt'
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

    encoder = RewardPredictor((3,64,64), n_frames, T_future, reward_type_list=rewards)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder = StateDecoder(in_size=64, out_size=64)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.RAdam(decoder.parameters(), lr=1e-3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)
    logger.info(f"[INFO] Training State Decoder on device: {device}")
    logger.info(f"""[INFO] Training with the following configuration:
        - Input shape: (3,64,64)
        - Number of frames: {n_frames}
        - Number of future frames: {T_future}
        - Reward types: {rewards}
        - Number of iterations: {n_iters}
        - Shapes per trajectory: {shapes_per_traj}
        - Batch size: {batch_size}
        - Encoder loaded from: {encoder_path}""")


    losses = []
    for i in range(n_iters):
        # get data
        data = next(iter(dataloader))
        frames = data['images'].to(device)

        # lstm encoded future repr & decode to image
        with torch.no_grad():
            reprs = encoder(frames)[0].detach()
        recon_frames = decoder(reprs)
        
        # compute loss
        loss = loss_fn(recon_frames, frames)
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
            torch.save(decoder.state_dict(), f'{model_save_prefix}_{i+1}.pt')
            logger.info(f"Intermediate model save at {model_save_prefix}_{i+1}.pt")


    # plot losses and save
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('State Decoder Loss')
    plt.savefig(plot_save_path)
    logger.info(f"Loss plot saved at {plot_save_path}")

    torch.save(decoder.state_dict(), f'{model_save_prefix}_final.pt')
    logger.info(f"Final model saved at {model_save_prefix}_final.pt")
    logger.info(f"Training complete.")



def evaluate_encoder(
        shapes_per_traj,
        batch_size=128,
        n_frames=1,
        T_future=29,
        rewards = ['AgentXReward', 'AgentYReward', 'TargetXReward', 'TargetYReward'],
        model_save_prefix='reward_induced/models/encoder/encoder',
        log_file_prefix='reward_induced/logs/encoder/evaluate_encoder'):
    # Consider HorPosReward or VertPosReward case
    env_mode = f'dist{shapes_per_traj - 2}' if shapes_per_traj >=2 else rewards[0] 
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size) 
    rewards = [r.NAME for r in spec.rewards]

    model = RewardPredictor((3,64,64), n_frames, T_future, reward_type_list=rewards)
    model.load_state_dict(torch.load(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
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
    logger.info("Losses:")
    for i, r in enumerate(rewards):
        logger.info(f"    - {r}\t: {torch.mean(loss[i]).item()}")
    logger.info(f"Evaluation complete.")   



def train_encoder(
        shapes_per_traj,
        batch_size,
        n_frames=1,
        T_future=29,
        rewards = ['AgentXReward', 'AgentYReward', 'TargetXReward', 'TargetYReward'],
        n_iters=1000,
        save_every=100,
        model_save_prefix='reward_induced/models/encoder/encoder',
        log_file_prefix='reward_induced/logs/encoder/train_encoder',
        plot_save_prefix='reward_induced/logs/encoder/encoder'):
    # Consider HorPosReward or VertPosReward case
    env_mode = f'dist{shapes_per_traj - 2}' if shapes_per_traj >= 2 else rewards[0]
    model_save_prefix = model_save_prefix + f'_{env_mode}'
    log_file_path = log_file_prefix + f'_{env_mode}.log'
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

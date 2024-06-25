import torch
import torch.nn as nn
import numpy as np

import cv2
import imageio

from general_utils import AttrDict, make_image_seq_strip
from reward_induced.src.reward_predictor import RewardPredictor
from reward_induced.src.state_decoder import StateDecoder
from sprites_datagen.moving_sprites import MovingSpriteDataset
import sprites_datagen.rewards as rewards_module
from .utils import _setup_logger


def ablation_decoder(
        n_test_cases=3,
        n_frames=1,
        T_future=29,
        rewards = ['HorPosReward'],  # or ['VertPosReward']
        encoder_prefix='reward_induced/models/encoder/encoder',
        decoder_prefix='reward_induced/models/decoder/decoder',
        log_file_prefix='reward_induced/logs/ablation/ablation_decoder',
        img_save_prefix='reward_induced/logs/ablation/'):
    env_mode = rewards[0]
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
        shapes_per_traj=1,
        rewards=[ getattr(rewards_module, r) for r in rewards ],
        batch_size=1,
    )
    rewards = [r.NAME for r in spec.rewards]

    encoder = RewardPredictor((3,64,64), n_frames, T_future, reward_type_list=rewards)
    encoder.load_state_dict(torch.load(encoder_path))
    decoder = StateDecoder(in_size=64, out_size=64)
    decoder.load_state_dict(torch.load(decoder_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)
    logger.info(f"[INFO] Ablation test: Is the velocity information reconstructed by the decoder?")
    logger.info(f"""[INFO] Ablation test with the following configuration:
        - Input shape: (3,64,64)
        - Number of frames: {n_frames}
        - Number of future frames: {T_future}
        - Reward types: {rewards}
        - Batch size: 1
        - Encoder loaded from: {encoder_path}
        - Decoder loaded from: {decoder_path}
        - Number of test cases: {n_test_cases}""")


    # make n_test_cases with different velocity
    init_pos, vel = np.asarray([0.2, 0.2]), np.asarray([0.025, 0.02])
    for i in range(n_test_cases):
        init_from = np.append(init_pos, i * vel)
        dataset = MovingSpriteDataset(spec, init_from=init_from)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

        data = next(iter(dataloader))
        frames = data['images'].to(device)

        # Save ground truth images as gif
        true_frames = ( data['images'].numpy() + 1 ) * (255.0 / 2)
        # true_seq_frames = make_image_seq_strip([true_frames], sep_val=255.0).astype(np.uint8)
        # cv2.imwrite(f'{img_save_prefix}_vel{i*vel}_true_seq.png', true_seq_frames[0].transpose(1,2,0))
        # logger.info(f"True sequence img saved at {img_save_prefix}_vel{i*vel}_true_seq.png")

        with imageio.get_writer(f'{img_save_prefix}_vel{i*vel}_true_seq.gif', mode='I', duration=1.0) as writer:
            for frame in true_frames[0]:
                writer.append_data(frame.astype(np.uint8))
        logger.info(f"True sequence gif saved at {img_save_prefix}_vel{i*vel}_true_seq.gif")

        # Reconstruct frames
        with torch.no_grad():
            recon_frames = decoder(encoder(frames)[0])
        recon_frames = ( recon_frames.cpu().numpy() + 1 ) * (255.0 / 2)

        # pred_seq_frames = make_image_seq_strip([recon_frames], sep_val=255.0).astype(np.uint8)
        # cv2.imwrite(f'{img_save_prefix}_vel{i*vel}_recon_seq.png', pred_seq_frames[0].transpose(1,2,0))
        # logger.info(f"Reconstructed sequence img saved at {img_save_prefix}_vel{i*vel}_recon_seq.png")

        with imageio.get_writer(f'{img_save_prefix}_vel{i*vel}_recon_seq.gif', mode='I', duration=1.0) as writer:
            for frame in recon_frames[0]:
                writer.append_data(frame.astype(np.uint8))
        logger.info(f"Reconstructed sequence gif saved at {img_save_prefix}_vel{i*vel}_recon_seq.gif")

    logger.info(f"[INFO] Ablation test finished.")
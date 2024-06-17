import fire
from reward_induced.train import (
    visualize_decoder,
    train_decoder,
    evaluate_encoder,
    train_encoder,
)
from reward_induced.ablation import ablation_decoder

if __name__ == '__main__':
    fire.Fire()
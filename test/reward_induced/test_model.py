import torch
import torch.nn as nn

import reward_induced.models.reward_predictor_model as reward_predictor_model


def test_RewardPredictorModel():
    image_shape = (3, 64, 64)
    n_frames = 3
    T_future = 5
    predictor = reward_predictor_model.RewardPredictorModel(image_shape, n_frames, T_future)

    x = torch.randn(2, 3, 3, 64, 64)
    y = torch.randn(2, 5, 3, 64, 64)
    z = predictor(x, y, reward_type=['agent_x', 'agent_y', 'target_x'])
    print(f'[test_RewardPredictorModel] predictor output z: {z}')
    assert z.keys() == {'agent_x', 'agent_y', 'target_x'}
    assert all (output.shape == torch.Size([2, 5]) for output in z.values()), \
            f'Expected shape: [2, 5], got: {[output.shape for output in z.values()]}'


def test_ImageEncoder():
    image_shape = (3, 64, 64)
    encoder = reward_predictor_model.ImageEncoder(image_shape)

    assert encoder.level == 6
    assert encoder.layers[0].in_channels == 3
    assert encoder.layers[10].out_channels == 2 ** 6 * 2  #(=128)
    
    x = torch.randn(2, 3, 3, 64, 64)
    y = encoder(x)
    assert y.shape == torch.Size([2, 3, 128]), f'Expected shape: [2, 3, 128], got: {y.shape}'


if __name__ == "__main__":
    test_ImageEncoder()
    print("[1] reward_induced.model.ImageEncoder passed\n")
    test_RewardPredictorModel()
    print("[2] reward_induced.model.RewardPredictorModel passed\n")
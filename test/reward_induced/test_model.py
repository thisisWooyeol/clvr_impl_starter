import torch
import torch.nn as nn

from reward_induced.src.reward_predictor_model import RewardPredictorModel, ImageEncoder
from reward_induced.src.image_decoder import ImageDecoder



def test_RewardPredictorModel():
    image_shape = (3, 64, 64)
    n_frames = 3
    T_future = 5
    predictor = RewardPredictorModel(image_shape, n_frames, T_future)

    x = torch.randn(2, 8, 3, 64, 64)
    repr, z = predictor(x, reward_type_list=['agent_x', 'agent_y', 'target_x'])
    print(f'[test_RewardPredictorModel] predictor output z: {z}')
    assert z.keys() == {'agent_x', 'agent_y', 'target_x'}
    assert all (output.shape == torch.Size([2, 5]) for output in z.values()), \
            f'Expected shape: [2, 5], got: {[output.shape for output in z.values()]}'
    assert repr.shape == torch.Size([2, 5, 64]), f'Expected shape: [2, 5, 64], got: {repr.shape}'


def test_ImageEncoder():
    image_shape = (3, 64, 64)
    encoder = ImageEncoder(image_shape, out_features=64)

    assert encoder.level == 6
    assert encoder.convs[0].in_channels == 3
    assert encoder.convs[10].out_channels == 2 ** 6 * 2  #(=128)
    
    x = torch.randn(5, 3, 3, 64, 64)
    y = encoder(x)
    assert y.shape == torch.Size([5, 3, 64]), f'Expected shape: [5, 3, 64], got: {y.shape}'


def test_ImageDecoder():
    image_shape = (3, 64, 64)
    decoder = ImageDecoder(image_shape)

    assert decoder.level == 6
    assert decoder.layers[0].in_channels == 128
    assert decoder.layers[10].out_channels == 3

    x = torch.randn(3, 128)
    y = decoder(x)
    assert y.shape == torch.Size([3, 3, 64, 64]), f'Expected shape: [3, 3, 64, 64], got: {y.shape}'


if __name__ == "__main__":
    test_ImageEncoder()
    print("[1] reward_induced.model.ImageEncoder passed\n")
    test_RewardPredictorModel()
    print("[2] reward_induced.model.RewardPredictorModel passed\n")
    test_ImageDecoder()
    print("[3] reward_induced.model.ImageDecoder passed\n")
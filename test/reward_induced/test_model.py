import torch

from reward_induced.src.reward_predictor import RewardPredictor, ImageEncoder
from reward_induced.src.image_decoder import StateDecoder



def test_RewardPredictorModel():
    image_shape = (3, 64, 64)
    n_frames = 3
    T_future = 5
    predictor = RewardPredictor(image_shape, n_frames, T_future, reward_type_list=['agent_x', 'agent_y', 'target_x'])

    x = torch.randn(2, 8, 3, 64, 64)
    repr, z = predictor(x)
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


def test_StateDecoder():
    # image_shape = (3, 64, 64)
    decoder = StateDecoder(in_size=64, out_size=64)

    x = torch.randn(5, 3, 64)
    y = decoder(x)
    assert y.shape == torch.Size([5, 3, 3, 64, 64]), f'Expected shape: [5, 3, 3, 64, 64], got: {y.shape}'


if __name__ == "__main__":
    test_ImageEncoder()
    print("[1] reward_induced.model.ImageEncoder passed\n")
    test_RewardPredictorModel()
    print("[2] reward_induced.model.RewardPredictorModel passed\n")
    test_StateDecoder()
    print("[3] reward_induced.model.StateDecoder passed\n")
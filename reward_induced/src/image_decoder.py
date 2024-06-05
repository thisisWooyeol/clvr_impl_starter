import torch
import torch.nn as nn
from math import log2


class ImageDecoder(nn.Module):
    """
    Decode reward-induced representation to image
    """
    def __init__(self, image_shape: tuple):
        super(ImageDecoder, self).__init__()
        # assume square image and H, W should be power of 2
        assert image_shape[-1] == image_shape[-2]
        self.level = int(log2(image_shape[-1]))
        _channels = 2 ** self.level * 2
        
        self.layers = nn.ModuleList()
        for i in range(self.level-1):
            self.layers.append(
                nn.ConvTranspose2d(_channels, _channels // 2, kernel_size=3, 
                                   stride=2, padding=1, output_padding=1))
            self.layers.append(nn.LeakyReLU(0.2))
            _channels //= 2
        self.layers.append(nn.ConvTranspose2d(_channels, image_shape[0], kernel_size=3, 
                                    stride=2, padding=1, output_padding=1))
        self.layers.append(nn.Tanh())
        
    def forward(self, x):
        # x: (L, C)
        x = x.unsqueeze(-1).unsqueeze(-1) # (L, C) -> (L, C, 1, 1)
        for layer in self.layers:
            x = layer(x)

        assert x.shape == torch.Size([x.shape[0], 3, 64, 64]), \
                f'Expected shape: [L, 3, H, W], got: {x.shape}'
        return x
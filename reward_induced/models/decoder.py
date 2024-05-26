import torch
import torch.nn as nn
from math import log2


class ImageDecoder(nn.Module):
    """
    Decode reward-induced representation to image
    """
    def __init__(self, image_shape: tuple):
        super(ImageDecoder, self).__init__()
        self.level = int(log2(image_shape[-1]))
        _channels = 2 ** self.level * 2
        
        self.layers = nn.ModuleList()
        for i in range(self.level-1):
            self.layers.append(
                nn.ConvTranspose2d(_channels, _channels // 2, kernel_size=3, 
                                   stride=2, padding=1, output_padding=1))
            self.layers.append(nn.LeakyReLU(0.2))
            _channels //= 2
        self.layers.append(nn.ConvTranspose2d(_channels, 3, kernel_size=3, 
                                    stride=2, padding=1, output_padding=1))
        self.layers.append(nn.Tanh())
        
    def forward(self, x):
        # x: (B, L, C)
        output = []
        for b in range(x.shape[0]):
            y = x[b].unsqueeze(-1).unsqueeze(-1) # (L, C) -> (L, C, 1, 1)
            for layer in self.layers:
                y = layer(y)
            output.append(y)
        
        output = torch.stack(output, dim=0)
        assert output.shape == torch.Size([x.shape[0], x.shape[1], 3, 64, 64]), \
                f'Expected shape: [B, L, 3, 64, 64], got: {output.shape}'
        
        return output
import torch
import torch.nn as nn
from math import log2


class ImageDecoder(nn.Module):
    """
    Decode reward-induced representation to image
    """
    def __init__(self, in_size, out_size):
        super(ImageDecoder, self).__init__()
        # assume square image and H, W should be power of 2
        # assert image_shape[-1] == image_shape[-2]
        # self.level = int(log2(image_shape[-1]))
        # _channels = 2 ** self.level
        
        # _layers = []
        # for _ in range(self.level-1):
        #     _layers.append(
        #         nn.ConvTranspose2d(_channels, _channels // 2, kernel_size=3, 
        #                            stride=2, padding=1, output_padding=1))
        #     _layers.append(nn.LeakyReLU(0.2))
        #     _channels //= 2
        # _layers.append(nn.ConvTranspose2d(_channels, _channels, kernel_size=3, 
        #                             stride=2, padding=1, output_padding=1))
        # _layers.append(nn.LeakyReLU(0.2))
        # _layers.append(nn.ConvTranspose2d(_channels, 3, kernel_size=3, 
        #                             stride=1, padding=1))        
        # self.convs = nn.Sequential(*_layers)
        HIDDEN_SIZE = 32
        self.out_size = out_size

        self.layers = nn.Sequential(
            nn.Linear(in_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, out_size*out_size)
        )
        
    def forward(self, x):
        # # x: (B, L, C) 
        # B = x.shape[0]
        # L = x.shape[1]
        # x = x.reshape(B*L, x.shape[2], 1, 1) # (B, L, C) -> (B*L, C, 1, 1)
        
        # y = self.convs(x)
        # y = y.reshape(B, L, 3, y.shape[-1], y.shape[-1])
        # return y

        B = x.shape[0]
        L = x.shape[1]
        x = self.layers(x)
        x = x.reshape(B, L, 1, self.out_size, self.out_size)
        return x.expand(B, L, 3, self.out_size, self.out_size) - 1
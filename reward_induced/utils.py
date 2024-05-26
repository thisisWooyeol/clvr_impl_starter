import numpy as np
import torch

from sprites_datagen.moving_sprites import TemplateMovingSpritesGenerator


def generate_batch(dataset, batch_size, itr):
    batched_data = None
    for i in range(batch_size):
        data = dataset[itr*batch_size+i]
        if batched_data is None:
            batched_data = {'images': [data['images']], 
                            'rewards': {k: [v] for k, v in data['rewards'].items()}}
        else:
            batched_data['images'].append(data['images'])
            for k, v in data['rewards'].items():
                batched_data['rewards'][k].append(v)
    
    images = np.stack(batched_data['images'], axis=0)
    images = torch.tensor(images, dtype=torch.float32)
    rewards = {k: np.stack(v, axis=0) for k, v in batched_data['rewards'].items()}

    return images, rewards


def generate_batch_gen(generator: TemplateMovingSpritesGenerator, batch_size):
    batched_data = None
    for i in range(batch_size):
        data = generator.gen_trajectory()
        if batched_data is None:
            batched_data = {'images': [data.images[:, None].repeat(3, axis=1).astype(np.float32) / (255./2) - 1.0], 
                            'rewards': {k: [v] for k, v in data.rewards.items()}}
        else:
            batched_data['images'].append(data.images[:, None].repeat(3, axis=1).astype(np.float32) / (255./2) - 1.0)
            for k, v in data.rewards.items():
                batched_data['rewards'][k].append(v)

    images = np.stack(batched_data['images'], axis=0)
    images = torch.tensor(images, dtype=torch.float32)
    rewards = {k: np.stack(v, axis=0) for k, v in batched_data['rewards'].items()}

    return images, rewards
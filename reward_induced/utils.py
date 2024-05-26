import numpy as np
import torch

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
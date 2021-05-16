import config

import torch 
import torch.nn as nn 
import numpy as np


class dataloader(torch.utils.data.Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = data[1]
        image = np.array(data[0])

        if self.transforms:
            image = self.transforms(image=image)['image']

        image = torch.tensor(image, dtype=torch.float).unsqueeze(0).permute(0, 3, 1, 2)
        image = image.unfold(2, config.patch_size, config.patch_size).unfold(3, config.patch_size, config.patch_size)
        image = image.permute(0, 2, 3, 1, 4, 5)
        image = image.reshape(
            image.shape[0],
            image.shape[1],
            image.shape[2],
            image.shape[3]*image.shape[4]*image.shape[5]
        )
        image = image.view(image.shape[0], -1, image.shape[-1]) 
        image = image.view(-1, image.shape[-1])
        
        
        return {
            'patches' : image,
            'label' : torch.tensor(label, dtype=torch.long),
        }

      

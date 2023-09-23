import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root

        self.transform = transform

        self.filenames = os.listdir(root)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, self.filenames[idx])).convert("RGB")
        # image = Image.open(os.path.join(self.root_dir, self.filenames[0])).convert("RGB")
        image = np.array(image).astype(np.float32) * (1. / 255.0)
        image = image[None].transpose(0, 3, 1, 2)

        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return image.squeeze(0)

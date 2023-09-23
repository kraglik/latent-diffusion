import torch
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, path=None):
        self.dataset = dataset
        self.cache = {}
        self.path = path

        if self.path is not None:
            try:
                self.cache = torch.load(self.path)
            except FileNotFoundError:
                pass

    def __getitem__(self, index: int):
        if index not in self.cache:
            self.cache[index] = self.dataset[index]

        return self.cache[index]

    def __len__(self):
        return len(self.dataset)

    def __del__(self):
        if self.path is not None:
            torch.save(self.cache, self.path)

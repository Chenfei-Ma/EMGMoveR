import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubsetDataset(Dataset):
    def __init__(self, base_dataset, movement_lb, percentage):
        self.base_dataset = base_dataset
        selected_indices = []
        for lb in movement_lb:
            selected_indices.append(np.where(base_dataset.win_lb == np.array(lb))[0][int(percentage[0]*len(np.where(base_dataset.win_lb == np.array(lb))[0])):int(percentage[1]*len(np.where(base_dataset.win_lb == np.array(lb))[0]))])
        self.selected_indices = np.concatenate(selected_indices)

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        base_idx = self.selected_indices[idx]
        return self.base_dataset[base_idx]


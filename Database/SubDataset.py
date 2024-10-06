import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SubsetDataset(Dataset):
    def __init__(self, base_dataset, movement_lb, percentage):

        self.features = base_dataset.features
        self.win_lb = base_dataset.win_lb
        self.win_idx = base_dataset.win_idx
        self.win_id = base_dataset.win_id

        selected_indices = []
        if percentage[0] < percentage[1] <= 1:
            for lb in movement_lb:
                selected_indices.append(np.where(self.win_lb == np.array(lb))[0][int(percentage[0]*len(np.where(self.win_lb == np.array(lb))[0])):int(percentage[1]*len(np.where(self.win_lb == np.array(lb))[0]))])
        else:
            for lb in movement_lb:
                selected_indices.append(np.where(self.win_lb == np.array(lb))[0][int(percentage[0]):int(percentage[1])])

        selected_indices = np.concatenate(selected_indices)
        self.win_idx = self.win_idx[selected_indices]
        self.win_id = [self.win_id[i] for i in selected_indices]
        self.win_lb = [self.win_lb[i] for i in selected_indices]

    def __len__(self):
        return len(self.win_idx)

    def __getitem__(self, index):
        return self.features[..., self.win_idx[index][0]:self.win_idx[index][1]], np.squeeze(self.win_lb[index]), np.squeeze(self.win_id[index])



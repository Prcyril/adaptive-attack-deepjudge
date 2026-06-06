import torch
import numpy as np


class Subset(torch.utils.data.Dataset):
    def __init__(self, dset, subset_ratio, rand_subset_seed=1292):
        super().__init__()
        self.dset = dset
        total_len = len(self.dset)

        rand_idxes = np.random.RandomState(seed=rand_subset_seed).permutation(total_len)

        # only use a subset of indexes
        subset_len = int(total_len * subset_ratio)
        self.idxes = rand_idxes[: subset_len]

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, index):
        # get the original index corresponding to the full dataset
        dest_idx = self.idxes[index]
        return self.dset[dest_idx]


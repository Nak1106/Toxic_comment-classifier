import random
from typing import List

import numpy as np
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """
    Dataset that yields pairs of texts and a binary label.

    label = 1 if both are from the same group (both toxic_any=1 or both 0)
    label = 0 if they are from different groups.
    """

    def __init__(self, texts: List[str], toxic_any: np.ndarray, num_pairs: int = 100000):
        self.texts = texts
        self.toxic_any = toxic_any.astype("int32")
        self.num_pairs = num_pairs

        self.toxic_idx = np.where(self.toxic_any == 1)[0].tolist()
        self.clean_idx = np.where(self.toxic_any == 0)[0].tolist()

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # Half positive, half negative on expectation
        if random.random() < 0.5:
            # positive pair
            if random.random() < 0.5 and len(self.toxic_idx) >= 2:
                i1, i2 = random.sample(self.toxic_idx, 2)
            else:
                i1, i2 = random.sample(self.clean_idx, 2)
            label = 1
        else:
            # negative pair: one toxic, one clean
            i1 = random.choice(self.toxic_idx)
            i2 = random.choice(self.clean_idx)
            label = 0

        return self.texts[i1], self.texts[i2], label


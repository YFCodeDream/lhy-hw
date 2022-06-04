import math
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LibriPhoneDataset(Dataset):
    def __init__(self, ids, padded_sequences, sequences_len, padded_labels):
        self.ids = ids
        self.padded_sequences = torch.FloatTensor(padded_sequences)
        self.sequences_len = sequences_len
        self.padded_labels = torch.Tensor(padded_labels)

    def __getitem__(self, index):
        return self.ids[index], self.padded_sequences[index], self.sequences_len[index], self.padded_labels[index]

    def __len__(self):
        return len(self.ids)


class LibriPhoneDataLoader(DataLoader):
    def __init__(self, **phone_loader_kwargs):
        pt_dir = str(phone_loader_kwargs.pop('pt_dir'))
        mode = str(phone_loader_kwargs.pop('mode'))

        pt_path = os.path.join(pt_dir, f'{mode}.pt')
        print(f'load data from {pt_path}')

        with open(pt_path, 'rb') as f:
            obj = pickle.load(f)

            self.ids = obj['ids']
            padded_sequences = obj['padded_sequences']
            sequences_len = obj['sequences_len']
            labels = obj['labels']

            max_seq_len = max(sequences_len)
            for i, label in enumerate(labels):
                labels[i].extend([np.inf for _ in range(max_seq_len - len(label))])
                # label.extend([np.inf for _ in range(max_seq_len - len(label))])

            self.dataset = LibriPhoneDataset(self.ids, padded_sequences, sequences_len, labels)

        super(LibriPhoneDataLoader, self).__init__(self.dataset, **phone_loader_kwargs)

    def __len__(self):
        return math.ceil(len(self.ids) / self.batch_size)

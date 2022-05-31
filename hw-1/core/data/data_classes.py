import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class COVID19Dataset(Dataset):
    def __init__(self,
                 covid_features,
                 covid_labels,
                 select_features=None,
                 select_features_model=None):
        self.covid_features = np.array(covid_features)
        self.covid_labels = covid_labels
        self.select_features = select_features
        self.select_features_model = select_features_model

        if select_features is not None and select_features_model is not None:
            self.covid_features = self.select_features_model.transform(self.covid_features)

        self.covid_features = torch.from_numpy(self.covid_features).float()
        if self.covid_labels is not None:
            self.covid_labels = torch.from_numpy(np.array(self.covid_labels)).float()

        self.input_dim = self.covid_features.shape[1]

    def __getitem__(self, index):
        if self.covid_labels is None:
            return self.covid_features[index]
        else:
            return self.covid_features[index], self.covid_labels[index]

    def __len__(self):
        return len(self.covid_features)


class COVID19Dataloader(DataLoader):
    def __init__(self, **covid_loader_kwargs):
        self.covid_data = covid_loader_kwargs.pop('covid_data')

        drop_id = covid_loader_kwargs.pop('drop_id')
        start_col = 1 if drop_id else 0
        covid_features = self.covid_data[:, start_col: -1]
        covid_labels = None

        mode = covid_loader_kwargs.pop('mode')
        if mode != 'test':
            covid_labels = self.covid_data[:, -1]
            assert len(covid_features) == len(covid_labels)
        else:
            covid_features = self.covid_data[:, start_col:]

        self.covid_features = covid_features
        self.covid_labels = covid_labels

        self.select_features = covid_loader_kwargs.pop('select_features')
        self.select_features_model = covid_loader_kwargs.pop('select_features_model') if \
            'select_features_model' in covid_loader_kwargs else None

        self.dataset = COVID19Dataset(covid_features, covid_labels, self.select_features, self.select_features_model)

        self.batch_size = covid_loader_kwargs.get('batch_size')

        super().__init__(self.dataset, **covid_loader_kwargs)

    def __len__(self):
        return math.ceil(len(self.covid_data) / self.batch_size)

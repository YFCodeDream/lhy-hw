import argparse
import os
import pickle

import pandas as pd
import torch.cuda
from sklearn.model_selection import train_test_split

from core.data.data_classes import COVID19Dataloader
from core.data.data_utils import mic_select_features, manual_select_features
from core.model.model import COVID19Net
from train import trainer
from util.randomizer import same_seed


def main(args):
    same_seed(args.seed)
    trainval_data = pd.read_csv('./dataset/covid.train.csv').values
    train_data, valid_data = train_test_split(trainval_data, test_size=args.valid_ratio, random_state=args.seed)

    loader_kwargs = {
        'drop_id': True,
        # 'select_features': None,
        'select_features': 'mic',
        'mic_percentile': 50,
        # 'select_features': 'manual',
        # 'mic_percentile': [0, 1, 2, 3, 4...],
        'shuffle': True,
        'batch_size': args.batch_size
    }

    if loader_kwargs.get('select_features') is not None:
        select_features_model = get_select_features_model(loader_kwargs, trainval_data)
        loader_kwargs.update({
            'select_features_model': select_features_model
        })
        with open(os.path.join(args.save_dir, 'select_features_model.pkl'), 'wb') as f:
            pickle.dump(select_features_model, f)

    train_loader_kwargs = loader_kwargs.copy()
    train_loader_kwargs.update({
        'covid_data': train_data,
        'mode': 'train'
    })

    valid_loader_kwargs = loader_kwargs.copy()
    valid_loader_kwargs.update({
        'covid_data': valid_data,
        'mode': 'valid'
    })

    train_loader = COVID19Dataloader(**train_loader_kwargs)
    valid_loader = COVID19Dataloader(**valid_loader_kwargs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = COVID19Net(train_loader.dataset.input_dim).to(device)

    trainer(train_loader, valid_loader, model, args, device)


def get_select_features_model(loader_kwargs, trainval_data):
    select_features_model = None
    start_col = 1 if loader_kwargs.get('drop_id') else 0
    trainval_covid_features = trainval_data[:, start_col: -1]
    trainval_covid_labels = trainval_data[:, -1]
    if loader_kwargs.get('select_features') == 'mic':
        select_features_model = mic_select_features(trainval_covid_features,
                                                    trainval_covid_labels,
                                                    loader_kwargs.pop('mic_percentile'))
    elif loader_kwargs.get('select_features') == 'manual':
        select_features_model = manual_select_features(loader_kwargs.pop('manual_indices'))
    return select_features_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--seed', default=666)
    parser.add_argument('--valid_ratio', default=0.2)
    parser.add_argument('--lr', default=4e-5)
    parser.add_argument('--lr_half_life', default=50000)
    parser.add_argument('--num_epoch', default=5000)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--early_stopping', action='store_false')
    parser.add_argument('--early_stopping_limit', default=400)
    parser.add_argument('--validate', action='store_false')
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if not args.restore:
        os.mkdir(args.save_dir)
    else:
        assert os.path.isdir(args.save_dir)
    main(args)

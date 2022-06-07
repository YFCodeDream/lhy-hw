import argparse
import os

import torch

from core.data.data_classes import LibriPhoneDataLoader
from core.model.net import LibriPhoneNet
from train import trainer
from util.randomizer import same_seed


# noinspection PyShadowingNames
def main(args):
    same_seed(args.seed)

    train_loader_kwargs = {
        'pt_dir': args.pt_dir,
        'mode': 'train',
        'batch_size': args.batch_size,
        'shuffle': True
    }

    valid_loader_kwargs = {
        'pt_dir': args.pt_dir,
        'mode': 'val',
        'batch_size': args.batch_size,
        'shuffle': True
    }

    train_loader = LibriPhoneDataLoader(**train_loader_kwargs)
    valid_loader = LibriPhoneDataLoader(**valid_loader_kwargs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_kwargs = {
        'feature_encode_mode': 'bi_gru',
        'bi_gru_output_dim': 100,
        'dropout_prob': 0.25,
        'device': device
    }

    model = LibriPhoneNet(39, **model_kwargs).to(device)

    trainer(train_loader, valid_loader, model, args, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--pt_dir', default='./dataset/generated/')
    parser.add_argument('--seed', default=666)
    parser.add_argument('--lr', default=4e-4)
    parser.add_argument('--lr_half_life', default=20000)
    parser.add_argument('--label_smoothing', default=0.12)
    parser.add_argument('--num_epoch', default=2000)
    parser.add_argument('--batch_size', default=32)
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

import argparse
import os

import torch

from core.data.data_classes import FoodDataLoader
from train import trainer
from util.randomizer import same_seed

from core.model import AlexNet, MobileNetV1, VGG16


def main(config):
    same_seed(config.seed)

    loader_kwargs = {'dataset_filename_json': config.dataset_filename_json,
                     'batch_size': config.batch_size,
                     'mode': 'train'}
    train_loader = FoodDataLoader(**loader_kwargs)

    probe_loader = FoodDataLoader(**loader_kwargs)
    tfs_images, _ = next(iter(probe_loader))
    image_shape = (tfs_images.size(-2), tfs_images.size(-1))
    # print(image_shape)

    loader_kwargs['mode'] = 'valid'
    valid_loader = FoodDataLoader(**loader_kwargs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MobileNetV1.MobileNetV1(num_classes=11, image_shape=image_shape).to(device)
    # model = AlexNet.AlexNet(num_classes=11, image_shape=image_shape).to(device)
    # model = VGG16.VGG16Net(num_classes=11, image_shape=image_shape).to(device)

    trainer(train_loader, valid_loader, model, config, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='./ckpt/')
    parser.add_argument('--dataset_filename_json', default='./dataset/generated/dataset_filename.json')
    parser.add_argument('--seed', default=1012)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--lr', default=4e-4)
    parser.add_argument('--lr_half_life', default=50000)
    parser.add_argument('--num_epoch', default=1000)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--early_stopping_limit', default=250)
    parser.add_argument('--restore', action='store_true')
    config = parser.parse_args()
    if not config.restore:
        os.mkdir(config.save_dir)
    else:
        assert os.path.isdir(config.save_dir)
    main(config)

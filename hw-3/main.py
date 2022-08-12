import argparse
import os

import torch

from core.data.data_classes import FoodDataLoader
from core.model.MobileNet import mobile_net_v2, mobile_net_v3_large
from core.model.ResNet import resnet_18
from train import trainer
from util.randomizer import same_seed


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

    pretrain_kwargs = {
        'pretrain_weight_save_dir': config.pretrain_weight_save_dir,
        'device': device,
        'pretrain_frozen': False
    }

    # functional API for pretrain model (recommend)

    model = mobile_net_v3_large(num_classes=11, pretrain=config.use_pretrain, **pretrain_kwargs)
    # model = mobile_net_v2(num_classes=11, pretrain=config.use_pretrain, **pretrain_kwargs)
    # model = resnet_18(num_classes=11, pretrain=config.use_pretrain, **pretrain_kwargs)

    # manual define pretrain model

    # model = get_pretrain_model(MobileNet.MobileNetV2, 11,
    #                            MobileNet.pretrain_weights.get(MobileNet.MobileNetV2.model_name),
    #                            config.pretrain_weight_save_dir, device, pretrain_frozen=False)

    # model = get_pretrain_model(ResNet.ResNet50, 11, ResNet.pretrain_weights.get(ResNet.ResNet50.model_name),
    #                            config.pretrain_weight_save_dir, device, pretrain_frozen=False)

    # without pretrain & class introduction & without image shape

    # model = ResNet.ResNeXt50(num_classes=11).to(device)
    # model = ResNet.ResNet50(num_classes=11).to(device)
    # model = ResNet.ResNet18(num_classes=11).to(device)

    # without pretrain & class introduction

    # model = GoogLeNet.GoogLeNet(num_classes=11, image_shape=image_shape).to(device)

    # functional API can be used

    # model = google_net(num_classes=11, **pretrain_kwargs)

    # model = MobileNet.MobileNetV1(num_classes=11, image_shape=image_shape).to(device)
    # model = AlexNet.AlexNet(num_classes=11, image_shape=image_shape).to(device)
    # model = VGG16.VGG16Net(num_classes=11, image_shape=image_shape).to(device)

    trainer(train_loader, valid_loader, model, config, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='./ckpt/')
    parser.add_argument('--dataset_filename_json', default='./dataset/generated/dataset_filename.json')
    parser.add_argument('--pretrain_weight_save_dir', default='./pretrain/')
    parser.add_argument('--use_pretrain', action='store_true')
    parser.add_argument('--seed', default=1012)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--lr', default=4e-4)
    parser.add_argument('--lr_half_life', default=50000)
    parser.add_argument('--num_epoch', default=5)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--early_stopping_limit', default=250)
    parser.add_argument('--restore', action='store_true')
    config = parser.parse_args()
    if not config.restore:
        os.mkdir(config.save_dir)
    else:
        assert os.path.isdir(config.save_dir)
    main(config)

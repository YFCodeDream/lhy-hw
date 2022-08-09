import os.path
import urllib.request
from collections import OrderedDict

import torch
from torch import nn


def get_pretrain_model(model_class, num_classes,
                       pretrain_weight_url, pretrain_weight_save_dir, device, pretrain_frozen=False):
    if not os.path.exists(pretrain_weight_save_dir):
        os.mkdir(pretrain_weight_save_dir)

    assert os.path.isdir(pretrain_weight_save_dir)

    model = model_class(num_classes=1000)  # imagenet_1k

    pretrain_weight_filename = os.path.join(pretrain_weight_save_dir, model_class.model_name)

    if not os.path.exists(pretrain_weight_filename):
        print(f'pretrain weight for model: {model_class.model_name}, doesn\'t exist, start downloading...')
        urllib.request.urlretrieve(pretrain_weight_url, pretrain_weight_filename)
    else:
        print(f'pretrain weight for model: {model_class.model_name} exists')

    transfer_weights = \
        load_official_pretrain_weights_to_custom_model(model, torch.load(pretrain_weight_filename, map_location=device))

    unexpected_keys, missing_keys = model.load_state_dict(transfer_weights, strict=False)
    print(f'unexpected keys: {unexpected_keys}')
    print(f'missing keys: {missing_keys}')

    if pretrain_frozen:
        for param in model.parameters():
            param.requires_grad = False

    last_fc_in_features = getattr(model, model_class.model_name).fc.in_features
    model.fc = nn.Linear(last_fc_in_features, num_classes)

    model.to(device)

    return model


def load_official_pretrain_weights_to_custom_model(custom_model, official_pretrain_weights):
    """
    https://blog.csdn.net/qq_57886603/article/details/121293540
    :param custom_model:
    :param official_pretrain_weights:
    :return:
    """
    custom_model_modules = [module_name for module_name in custom_model.state_dict()]
    transfer_weights = OrderedDict()

    for idx, (official_param_name, official_param) in enumerate(official_pretrain_weights.items()):
        if custom_model.state_dict()[custom_model_modules[idx]].numel() == official_param.numel():
            print(f'transfer official param: {official_param_name}, to custom model param: {custom_model_modules[idx]}')
            transfer_weights[custom_model_modules[idx]] = official_param

    return transfer_weights

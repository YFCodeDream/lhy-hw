import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from core.model.GoogLeNet import GoogLeNet
from validate import validater


def trainer(train_loader, valid_loader, model, config, device):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(os.path.join(config.save_dir, 'trainer.log.txt'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    logging.info(model)

    for config_key, config_value in vars(config).items():
        logging.info(config_key + ': ' + str(config_value))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1 / config.lr_half_life))

    start_epoch = 0
    if config.restore:
        ckpt_path = os.path.join(config.save_dir, 'model.pt')
        ckpt_state_dict = torch.load(ckpt_path)
        start_epoch = ckpt_state_dict['epoch'] + 1
        model.load_state_dict(ckpt_state_dict['model_state_dict'])
        optimizer.load_state_dict(ckpt_state_dict['optimizer_state_dict'])

    highest_valid_acc, early_stopping_epoch_cnt = -np.inf, 0

    logging.info('start training')

    all_train_loss, all_valid_loss, all_valid_acc = (list() for _ in range(3))

    for epoch in range(start_epoch, config.num_epoch):
        model.train()
        train_loss = list()
        train_loader_processor = tqdm(train_loader, desc='processing batch data: ')
        for batch_index, (tfs_images, labels) in enumerate(train_loader_processor):
            tfs_images = tfs_images.to(device)
            labels = labels.to(device)
            logits = model(tfs_images)

            if isinstance(model, GoogLeNet):
                logits, aux_1_res, aux_2_res = logits
                loss_1 = loss_fn(logits, labels)
                loss_2 = loss_fn(aux_1_res, labels)
                loss_3 = loss_fn(aux_2_res, labels)
                loss = loss_1 + 0.3 * loss_2 + 0.3 * loss_3
            else:
                loss = loss_fn(logits, labels)

            train_loss.append(loss.item())
            train_loader_processor.set_description(f'processing batch data, index: {batch_index + 1}')
            train_loader_processor.set_postfix({'current batch loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        logging.info(f'epoch [{epoch + 1} / {config.num_epoch}]: '
                     f'current epoch mean train loss: {np.mean(train_loss):.4f}')

        all_train_loss.append(np.mean(train_loss))

        if config.validate:
            valid_loss, valid_acc = validater(valid_loader, model, loss_fn, device)

            logging.info(f'validation result: mean valid loss: {valid_loss:.4f}, valid accuracy: {valid_acc:.4f}')

            all_valid_loss.append(valid_loss)
            all_valid_acc.append(valid_acc.item())

            if valid_acc > highest_valid_acc:
                highest_valid_acc = valid_acc
                cur_state_dict = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict()
                }

                torch.save(cur_state_dict, os.path.join(config.save_dir, 'model.pt'))
                logging.info(f'saving current checkpoint to: {os.path.join(config.save_dir, "model.pt")}')
                early_stopping_epoch_cnt = 0
            else:
                early_stopping_epoch_cnt += 1

            if early_stopping_epoch_cnt >= config.early_stopping_limit:
                logging.info('model is not improving, so we halt the training session')
                return
        else:
            cur_state_dict = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()
            }

            torch.save(cur_state_dict, os.path.join(config.save_dir, 'model.pt'))

    sns.set_theme(style='darkgrid')

    loss_curve_data = pd.DataFrame({
        'epoch': list(range(start_epoch + 1, config.num_epoch + 1)),
        'train loss': all_train_loss,
        'valid loss': all_valid_loss
    })

    loss_curve_data.plot(x='epoch', y=['train loss', 'valid loss'])
    plt.savefig(os.path.join(config.save_dir, 'loss.png'))
    plt.show()
    plt.close()

    if config.validate:
        valid_acc_curve_data = pd.DataFrame({
            'epoch': list(range(start_epoch + 1, config.num_epoch + 1)),
            'valid accuracy': all_valid_acc
        })

        valid_acc_curve_data.plot(x='epoch', y='valid accuracy')
        plt.xlabel('epoch')
        plt.ylabel('validation accuracy')
        plt.savefig(os.path.join(config.save_dir, 'val_acc.png'))
        plt.show()

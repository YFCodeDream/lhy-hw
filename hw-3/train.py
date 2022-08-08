import logging
import os

import numpy as np
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

    lowest_valid_loss, early_stopping_epoch_cnt = np.inf, 0

    logging.info('start training')

    for epoch in range(start_epoch, config.num_epoch):
        model.train()
        train_loss = list()
        train_loader_processor = tqdm(train_loader, desc='processing batch data: ')
        for batch_index, (tfs_images, labels) in enumerate(train_loader_processor):
            tfs_images = tfs_images.to(device)
            logits = model(tfs_images)

            if isinstance(model, GoogLeNet):
                logits, aux_1_res, aux_2_res = logits
                loss_1 = loss_fn(logits, labels.to(device))
                loss_2 = loss_fn(aux_1_res, labels.to(device))
                loss_3 = loss_fn(aux_2_res, labels.to(device))
                loss = loss_1 + 0.3 * loss_2 + 0.3 * loss_3
            else:
                loss = loss_fn(logits, labels.to(device))

            train_loss.append(loss.item())
            train_loader_processor.set_description(f'processing batch data, index: {batch_index + 1}')
            train_loader_processor.set_postfix({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        logging.info(f'epoch [{epoch + 1} / {config.num_epoch}]: train loss: {np.mean(train_loss):.4f}')

        if config.validate:
            valid_loss = validater(valid_loader, model, loss_fn, device)
            logging.info('validation result: ')
            logging.info(f'epoch [{epoch + 1} / {config.num_epoch}]: valid loss: {valid_loss:.4f}')

            if valid_loss < lowest_valid_loss:
                lowest_valid_loss = valid_loss
                cur_state_dict = {
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict()
                }

                torch.save(cur_state_dict, os.path.join(config.save_dir, 'model.pt'))
                early_stopping_epoch_cnt = 0
            else:
                early_stopping_epoch_cnt += 1

            if early_stopping_epoch_cnt >= config.early_stopping_limit:
                logging.info('\nModel is not improving, so we halt the training session')
                return
        else:
            cur_state_dict = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict()
            }

            torch.save(cur_state_dict, os.path.join(config.save_dir, 'model.pt'))

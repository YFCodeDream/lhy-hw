import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from validate import validate


def trainer(train_loader, valid_loader, model, config, device):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(os.path.join(config.save_dir, 'trainer.log.txt'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    logging.info(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), config.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1 / config.lr_half_life))

    start_epoch = 0
    if config.restore:
        ckpt_path = os.path.join(config.save_dir, 'model.pt')
        ckpt_state_dict = torch.load(ckpt_path)
        start_epoch = ckpt_state_dict['epoch'] + 1
        model.load_state_dict(ckpt_state_dict['state_dict'])
        optimizer.load_state_dict(ckpt_state_dict['optimizer'])

    lowest_valid_loss, early_stopping_epoch = np.inf, 0

    logging.info('start training')

    for epoch in range(start_epoch, config.num_epoch):
        model.train()
        train_loss = list()
        for batch_idx, batch in enumerate(train_loader):
            progress = epoch + batch_idx / len(train_loader)

            covid_features, covid_labels = [x.to(device) for x in batch]

            covid_predictions = model(covid_features)
            loss = loss_fn(covid_predictions, covid_labels)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        logging.info(f'epoch [{epoch + 1} / {config.num_epoch}]: train loss: {np.mean(train_loss):.4f}')

        if config.validate:
            mean_valid_loss = validate(model, valid_loader, loss_fn, config, device)
            logging.info('validation result: ')
            logging.info(f'epoch [{epoch + 1} / {config.num_epoch}]: valid loss: {mean_valid_loss:.4f}')

            if config.early_stopping:
                if mean_valid_loss < lowest_valid_loss:
                    lowest_valid_loss = mean_valid_loss
                    cur_state_dict = {
                        'epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        'state_dict': model.state_dict()
                    }

                    torch.save(cur_state_dict, os.path.join(config.save_dir, 'model.pt'))
                    early_stopping_epoch = 0
                else:
                    early_stopping_epoch += 1

                if early_stopping_epoch >= config.early_stopping_limit:
                    logging.info('\nModel is not improving, so we halt the training session')
                    return
        else:
            cur_state_dict = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict()
            }

            torch.save(cur_state_dict, os.path.join(config.save_dir, 'model.pt'))

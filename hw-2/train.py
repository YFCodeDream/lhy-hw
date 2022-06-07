import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from validate import validater


def trainer(train_loader, valid_loader, model, args, device):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
    logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'trainer.log.txt'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    logging.info(model)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.5 ** (1 / args.lr_half_life))

    start_epoch = 0
    if args.restore:
        ckpt_path = os.path.join(args.save_dir, 'model.pt')
        ckpt_state_dict = torch.load(ckpt_path)
        start_epoch = ckpt_state_dict['epoch'] + 1
        model.load_state_dict(ckpt_state_dict['model_state_dict'])
        optimizer.load_state_dict(ckpt_state_dict['optimizer_state_dict'])

    lowest_valid_loss, early_stopping_epoch = np.inf, 0

    for epoch in range(start_epoch, args.num_epoch):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            phone_ids, batch_padded_sequences, sequences_len, labels = batch
            batch_padded_sequences = batch_padded_sequences.to(device)
            sequences_len = sequences_len.to(device)

            batch_seq_predictions = model(batch_padded_sequences, sequences_len)

            batch_seq_predictions_no_pad = list()
            batch_cur_seq_labels_no_pad = list()
            for i, seq_predictions in enumerate(batch_seq_predictions):
                cur_seq_len = sequences_len[i]
                cur_seq_labels = labels[i]

                seq_predictions_no_pad = seq_predictions[:cur_seq_len]
                cur_seq_labels_no_pad = cur_seq_labels[:cur_seq_len].long()

                batch_seq_predictions_no_pad.append(seq_predictions_no_pad)
                batch_cur_seq_labels_no_pad.append(cur_seq_labels_no_pad)

            cur_seq_loss = loss_fn(torch.cat(batch_seq_predictions_no_pad).to(device),
                                   torch.cat(batch_cur_seq_labels_no_pad).to(device))

            optimizer.zero_grad()
            cur_seq_loss.backward()
            optimizer.step()
            scheduler.step()

            logging.info(f'batch index: {batch_idx + 1} / {len(train_loader)}, train loss: {cur_seq_loss.item():4f}')

        if args.validate:
            mean_valid_loss, total_seq_valid_acc = validater(valid_loader, model, loss_fn, args, device)
            logging.info(f'validation result: epoch: {epoch + 1} / {args.num_epoch}, '
                         f'valid loss: {mean_valid_loss:4f}, valid accuracy: {total_seq_valid_acc:4f}')

            if args.early_stopping:
                if mean_valid_loss < lowest_valid_loss:
                    lowest_valid_loss = mean_valid_loss
                    ckpt_state_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }

                    torch.save(ckpt_state_dict, os.path.join(args.save_dir, 'model.pt'))
                    early_stopping_epoch = 0
                else:
                    early_stopping_epoch += 1

                if early_stopping_epoch >= args.early_stopping_limit:
                    logging.info('\nModel is not improving, so we halt the training session')
                    return
        else:
            cur_state_dict = {
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict()
            }

            torch.save(cur_state_dict, os.path.join(args.save_dir, 'model.pt'))

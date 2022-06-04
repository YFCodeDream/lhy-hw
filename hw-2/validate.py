import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm


def validater(valid_loader, model, loss_fn, args, device):
    model.eval()
    seq_valid_loss = list()

    for batch in tqdm(valid_loader, total=len(valid_loader), desc='validating...'):
        phone_ids, batch_padded_sequences, sequences_len, labels = batch
        batch_padded_sequences = batch_padded_sequences.to(device)
        sequences_len = sequences_len.to(device)

        with torch.no_grad():
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

            cur_seq_loss = loss_fn(torch.cat(batch_seq_predictions_no_pad), torch.cat(batch_cur_seq_labels_no_pad))

            seq_valid_loss.append(cur_seq_loss)

    return np.mean(seq_valid_loss)

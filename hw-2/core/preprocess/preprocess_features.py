import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split


def get_split_ids(split_path, random_seed):
    np.random.seed(random_seed)
    with open(split_path, 'r') as f:
        ids = [file_id.strip('\n') for file_id in f.readlines()]
    return ids


def generate_train_val_ids(trainval_split_path, val_ratio, random_seed):
    trainval_ids = get_split_ids(trainval_split_path, random_seed)

    train_ids, val_ids = train_test_split(trainval_ids, test_size=val_ratio, random_state=random_seed)

    return train_ids, val_ids


def generate_label_dict(label_path):
    with open(label_path, 'r') as f:
        label_lines = f.readlines()

    label_dict = dict()
    for line in label_lines:
        line_list = line.strip('\n').split(' ')
        label_dict.update({
            line_list[0]: [int(phone) for phone in line_list[1:]]
        })

    return label_dict


def generate_labels(file_ids, label_dict=None):
    labels = list()
    if label_dict is None:
        labels = [[0] for _ in file_ids]
    else:
        labels = [label_dict[file_id] for file_id in file_ids]
    return labels


def generate_padded_features(feature_path, file_ids):
    raw_features = list()
    raw_sequences_len = list()
    max_seq_len = 0
    for file_id in file_ids:
        feature = torch.load(os.path.join(feature_path, f'{file_id}.pt'))
        cur_seq_len = len(feature)
        raw_sequences_len.append(cur_seq_len)
        if cur_seq_len > max_seq_len:
            max_seq_len = cur_seq_len
        raw_features.append(feature.numpy())

    for i, feature in enumerate(raw_features):
        cur_seq_len, feature_dim = feature.shape
        if cur_seq_len < max_seq_len:
            raw_features[i] = np.concatenate([feature, np.zeros((max_seq_len - cur_seq_len, feature_dim))])

    return np.stack(raw_features), raw_sequences_len


def generate_pt_file(ids, padded_features, sequences_len, labels, output_pt_path):
    obj = {
        'ids': ids,
        'padded_sequences': padded_features,
        'sequences_len': sequences_len,
        'labels': labels
    }
    with open(output_pt_path, 'wb') as f:
        pickle.dump(obj, f)


def preprocess_main(args):
    if args.mode == 'train+val':
        train_ids, val_ids = generate_train_val_ids(args.split_path, args.valid_ratio, args.seed)
        label_dict = generate_label_dict(args.label_path)

        train_labels = generate_labels(train_ids, label_dict)
        val_labels = generate_labels(val_ids, label_dict)

        train_padded_features, train_sequences_len = generate_padded_features(args.feature_path, train_ids)
        val_padded_features, val_sequences_len = generate_padded_features(args.feature_path, val_ids)

        print(train_ids[0])
        print(train_padded_features[0])
        print(train_sequences_len[0])
        print(train_labels[0])

        generate_pt_file(train_ids,
                         train_padded_features,
                         train_sequences_len,
                         train_labels,
                         os.path.join(args.pt_dir, 'train.pt'))

        print(len(val_ids))
        print(val_padded_features.shape)
        print(len(val_sequences_len))
        print(len(val_labels))

        generate_pt_file(val_ids,
                         val_padded_features,
                         val_sequences_len,
                         val_labels,
                         os.path.join(args.pt_dir, 'val.pt'))
    elif args.mode == 'test':
        test_ids = get_split_ids(args.split_path, args.seed)
        test_labels = generate_labels(test_ids)
        test_padded_features, test_sequences_len = generate_padded_features(args.feature_path, test_ids)

        print(len(test_ids))
        print(test_padded_features.shape)
        print(len(test_sequences_len))
        print(len(test_labels))

        generate_pt_file(test_ids,
                         test_padded_features,
                         test_sequences_len,
                         test_labels,
                         os.path.join(args.pt_dir, 'test.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True)
    parser.add_argument('--split_path', required=True)
    parser.add_argument('--label_path')
    parser.add_argument('--valid_ratio', default=0.2)
    parser.add_argument('--seed', default=666)
    parser.add_argument('--feature_path', required=True)
    parser.add_argument('--pt_dir', default='../../dataset/generated/')
    args = parser.parse_args()
    preprocess_main(args)

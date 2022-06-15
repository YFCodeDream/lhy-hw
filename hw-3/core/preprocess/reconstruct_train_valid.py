import argparse
import json
import os
import random


def get_label_filename_dict(dataset_root_dir, raw_mode):
    raw_dataset_dir = os.path.join(dataset_root_dir, raw_mode)
    label_filename = dict()
    for image_filename in os.listdir(raw_dataset_dir):
        image_label, _ = str(image_filename).split("_")
        image_label = int(image_label)
        image_path = os.path.join(raw_dataset_dir, image_filename)
        if image_label not in label_filename:
            label_filename[image_label] = [str(image_path).replace('..\\..', '.')]
        else:
            label_filename[image_label].append(str(image_path).replace('..\\..', '.'))
    return label_filename


def reconstruct_train_valid_main(args):
    trainval_label_filename = get_label_filename_dict(args.dataset_root_dir, 'training')
    valid_label_filename = get_label_filename_dict(args.dataset_root_dir, 'validation')
    for image_label, valid_filenames in valid_label_filename.items():
        trainval_label_filename[image_label].extend(valid_filenames)

    new_train_label_filename = dict().fromkeys(range(max(trainval_label_filename.keys())), None)
    new_valid_label_filename = dict().fromkeys(range(max(trainval_label_filename.keys())), None)
    for image_label, filenames in trainval_label_filename.items():
        train_filenames = random.sample(filenames, int(len(filenames) * args.train_ratio))
        valid_filenames = list(set(filenames) - set(train_filenames))
        new_train_label_filename[image_label] = train_filenames
        new_valid_label_filename[image_label] = valid_filenames

    dataset_filename_json = {
        'train': new_train_label_filename,
        'valid': new_valid_label_filename
    }

    with open(args.dataset_filename_json, 'w') as f:
        json.dump(dataset_filename_json, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', default='..\\..\\dataset\\food11\\')
    parser.add_argument('--dataset_filename_json', default='..\\..\\dataset\\generated\\dataset_filename.json')
    parser.add_argument('--train_ratio', default=0.8)
    args = parser.parse_args()
    reconstruct_train_valid_main(args)

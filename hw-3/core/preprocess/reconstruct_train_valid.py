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
            label_filename[image_label] = [str(image_path)]
        else:
            label_filename[image_label].append(str(image_path))
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

    test_dataset_dir = os.path.join(args.dataset_root_dir, 'test')

    dataset_filename_json = {
        'train': new_train_label_filename,
        'valid': new_valid_label_filename,
        'test': {
            -1: [os.path.join(test_dataset_dir, filename) for filename in os.listdir(test_dataset_dir)]
        }
    }

    with open(args.dataset_filename_json, 'w', encoding='utf8') as f:
        json.dump(dataset_filename_json, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', default='D:\\课程\\05 李宏毅2021_2022机器学习\\Lhy HW Data\\2022\\HW3\\food11')
    parser.add_argument('--dataset_filename_json', default='..\\..\\dataset\\generated\\dataset_filename.json')
    parser.add_argument('--train_ratio', default=0.8)
    args = parser.parse_args()
    reconstruct_train_valid_main(args)

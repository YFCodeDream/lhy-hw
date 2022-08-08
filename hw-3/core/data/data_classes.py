import json
import math

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FoodDataset(Dataset):
    def __init__(self, dataset_tuples):
        self.dataset_tuples = dataset_tuples

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(22.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        filename, label = self.dataset_tuples[index]
        raw_image = Image.open(filename)
        tfs_image = self.transform(raw_image)
        return tfs_image, label

    def __len__(self):
        return len(self.dataset_tuples)


# noinspection PyTypeChecker
class FoodDataLoader(DataLoader):
    def __init__(self, **food_loader_kwargs):
        dataset_filename_json_path = str(food_loader_kwargs.pop('dataset_filename_json'))
        mode = str(food_loader_kwargs.pop('mode'))
        with open(dataset_filename_json_path, 'r') as f:
            dataset_filename_json = json.load(f)
        cur_dataset_dict = dataset_filename_json[mode]
        cur_dataset_tuples = transform_dataset_dict_to_tuples(cur_dataset_dict)

        self.dataset = FoodDataset(cur_dataset_tuples)

        if mode != 'test':
            food_loader_kwargs.update({'shuffle': True})

        super().__init__(self.dataset, **food_loader_kwargs)

    def __len__(self):
        if self.drop_last:
            return math.floor(len(self.dataset) / self.batch_size)
        return math.ceil(len(self.dataset) / self.batch_size)


def transform_dataset_dict_to_tuples(dataset_dict):
    dataset_tuple_list = list()
    for image_label, filenames in dataset_dict.items():
        for filename in filenames:
            dataset_tuple_list.append((filename, int(image_label)))
    return dataset_tuple_list

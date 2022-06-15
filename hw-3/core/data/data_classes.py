import json

from torch.utils.data import Dataset, DataLoader


class FoodDataset(Dataset):
    def __init__(self, dataset_tuples):
        self.dataset_tuples = dataset_tuples

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.dataset_tuples)


class FoodDataLoader(DataLoader):
    def __init__(self, **food_loader_kwargs):
        dataset_filename_json_path = str(food_loader_kwargs.pop('dataset_filename_json'))
        mode = str(food_loader_kwargs.pop('mode'))
        with open(dataset_filename_json_path, 'r') as f:
            dataset_filename_json = json.load(f)
        cur_dataset_dict = dataset_filename_json[mode]
        cur_dataset_tuples = transform_dataset_dict_to_tuples(cur_dataset_dict)

        self.dataset = FoodDataset(cur_dataset_tuples)

        super().__init__(self.dataset, **food_loader_kwargs)

    def __len__(self):
        pass


def transform_dataset_dict_to_tuples(dataset_dict):
    dataset_tuple_list = list()
    for image_label, filenames in dataset_dict.items():
        for filename in filenames:
            dataset_tuple_list.append((filename, image_label))
    return dataset_tuple_list

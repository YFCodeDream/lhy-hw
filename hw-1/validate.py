import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from core.data.data_classes import COVID19Dataloader
from core.model.model import COVID19Net


def validate(model, valid_loader, loss_fn, config, device):
    model.eval()
    valid_loss = list()
    for batch in tqdm(valid_loader, total=len(valid_loader), desc='validation'):
        covid_features, covid_labels = [x.to(device) for x in batch]
        with torch.no_grad():
            covid_predictions = model(covid_features)
            loss = loss_fn(covid_predictions, covid_labels)
        valid_loss.append(loss.item())

    mean_valid_loss = np.mean(valid_loss)
    return mean_valid_loss


def predict(test_loader, model, device):
    model.eval()
    predictions = list()
    for x in tqdm(test_loader, total=len(test_loader), desc='testing'):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            predictions.append(pred.detach().cpu())
    predictions = torch.cat(predictions, dim=0).numpy()
    return predictions


def save_test_predictions(test_predictions, test_predictions_filename):
    test_predictions_file_dict = {
        'id': list(range(len(test_predictions))),
        'predictions': test_predictions
    }
    pd.DataFrame(test_predictions_file_dict).to_excel(test_predictions_filename, index=False)


def model_test_main(args):
    test_data = pd.read_csv('./dataset/covid.test.csv').values

    test_loader_kwargs = {
        'covid_data': test_data,
        'drop_id': True,
        # 'select_features': None,
        'select_features': 'mic',
        # 'select_features': 'manual',
        'shuffle': False,
        'batch_size': args.batch_size,
        'mode': 'test'
    }

    if test_loader_kwargs.get('select_features') is not None:
        if not os.path.exists(os.path.join(args.save_dir, 'select_features_model.pkl')):
            raise ValueError('the model used to select features does\'t exist, '
                             'but loader kwargs: select_features is not None')
        with open(os.path.join(args.save_dir, 'select_features_model.pkl'), 'rb') as f:
            select_features_model = pickle.load(f)
            test_loader_kwargs.update({
                'select_features_model': select_features_model
            })

    test_loader = COVID19Dataloader(**test_loader_kwargs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = COVID19Net(test_loader.dataset.input_dim).to(device)

    ckpt_path = os.path.join(args.save_dir, 'model.pt')
    ckpt_state_dict = torch.load(ckpt_path)
    model.load_state_dict(ckpt_state_dict['state_dict'])

    test_predictions = predict(test_loader, model, device)

    print('saving test_predictions.xlsx')
    save_test_predictions(test_predictions, os.path.join(args.save_dir, 'test_predictions.xlsx'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='./ckpt/')
    parser.add_argument('--batch_size', default=256)
    args = parser.parse_args()
    model_test_main(args)

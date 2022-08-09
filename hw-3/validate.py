import numpy as np
import torch
from tqdm import tqdm

from core.model.GoogLeNet import GoogLeNet


# noinspection PyTypeChecker
def validater(valid_loader, model, loss_fn, device):
    model.eval()

    valid_loss = list()
    valid_num_correct = 0
    valid_num_total = 0

    for tfs_images, labels in tqdm(valid_loader, total=len(valid_loader), desc='validation'):
        with torch.no_grad():
            logits = model(tfs_images.to(device))

            if isinstance(model, GoogLeNet):
                logits, aux_1_res, aux_2_res = logits

            loss = loss_fn(logits, labels.to(device))

            prediction = torch.softmax(logits, dim=1).to('cpu')
            probs, classes = torch.max(prediction, dim=1)

            valid_loss.append(loss.item())

            valid_num_correct += torch.sum(classes == labels)
            valid_num_total += len(labels)

    return np.mean(valid_loss), valid_num_correct / valid_num_total

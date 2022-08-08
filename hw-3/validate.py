import numpy as np
import torch
from tqdm import tqdm

from core.model.GoogLeNet import GoogLeNet


def validater(valid_loader, model, loss_fn, device):
    model.eval()
    valid_loss = list()
    for tfs_images, labels in tqdm(valid_loader, total=len(valid_loader), desc='validation'):
        with torch.no_grad():
            logits = model(tfs_images.to(device))

            if isinstance(model, GoogLeNet):
                logits, aux_1_res, aux_2_res = logits

            loss = loss_fn(logits, labels.to(device))
            valid_loss.append(loss.item())
    return np.mean(valid_loss)

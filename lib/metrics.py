import numpy as np
import torch

def onehot_decode(label):
    return torch.argmax(label, dim=1)


def accuracy(predictions, targets):
    pred_decode = onehot_decode(predictions)
    true_decode = targets

    assert len(pred_decode) == len(true_decode)

    acc = torch.mean((pred_decode == true_decode).float())

    return float(acc)

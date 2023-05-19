import torch
import numpy as np
from .utils import print_log

# x: (N, num_channels)
# y: (N, 1)


def get_tensors(
    log=None,
):
    x = np.load("../data/data.npz")["data"].astype(np.float32)
    y = np.load("../data/label.npz")["data"].astype(np.int16)
    indices = np.load("../data/indices.npz")

    train_indices = indices["train"].astype(np.int16)
    val_indices = indices["val"].astype(np.int16)
    test_indices = indices["test"].astype(np.int16)

    print_log(f"x-{x.shape}\ty-{y.shape}", log=log)

    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)

    return x, y, train_indices, val_indices, test_indices

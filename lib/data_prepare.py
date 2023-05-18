import torch
import numpy as np
from .utils import print_log, StandardScaler

# x: (samples, num_grid_h, num_grid_w, num_channels)
# y: (samples, num_grid_h, num_grid_w, 1)


def get_dataloaders_from_tvt(
    n,
    p,
    batch_size=64,
    log=None,
):
    train_data = np.load(f"../data/n_{n}_p_{p}/train_{n}_{p}.npz")
    val_data = np.load(f"../data/n_{n}_p_{p}/val_{n}_{p}.npz")
    test_data = np.load(f"../data/n_{n}_p_{p}/test_{n}_{p}.npz")

    x_train, y_train = (
        train_data["x_train"].astype(np.float32),
        train_data["y_train"].astype(np.float32)[..., np.newaxis],
    )
    x_val, y_val = (
        val_data["x_val"].astype(np.float32),
        val_data["y_val"].astype(np.float32)[..., np.newaxis],
    )
    x_test, y_test = (
        test_data["x_test"].astype(np.float32),
        test_data["y_test"].astype(np.float32)[..., np.newaxis],
    )
    
    # scaler = StandardScaler(
    #     mean=x_train[..., 0].mean(), std=x_train[..., 0].std()
    # )

    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader

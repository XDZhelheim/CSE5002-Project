import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys

sys.path.append("..")
from lib.utils import (
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import accuracy
from lib.data_prepare import get_tensors
from models import model_select


@torch.no_grad()
def eval_model(model, x, y, val_indices):
    model.eval()

    out = model.forward(x)

    acc = accuracy(out[val_indices], y[val_indices])

    return acc


@torch.no_grad()
def predict(model, x):
    model.eval()

    out = model(x).cpu()
    return out


def train_one_epoch(
    model, x, y, train_indices, optimizer, scheduler, criterion, clip_grad, log=None
):
    model.train()

    out = model(x)

    loss = criterion(out[train_indices], y[train_indices])
    epoch_loss = loss.item()

    epoch_acc = accuracy(out[train_indices], y[train_indices])

    optimizer.zero_grad()
    loss.backward()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()

    scheduler.step()

    return epoch_loss, epoch_acc


def train(
    model,
    x,
    y,
    train_indices,
    val_indices,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    compile_model=False,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    if torch.__version__ >= "2.0.0" and compile_model:
        model = torch.compile(model)
    model = model.to(DEVICE)

    wait = 0
    max_val_acc = -np.inf

    train_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            x,
            y,
            train_indices,
            optimizer,
            scheduler,
            criterion,
            clip_grad,
            log=log,
        )
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        val_acc = eval_model(model, x, y, val_indices)
        val_acc_list.append(val_acc)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                "\tTrain Loss = %.5f" % train_loss,
                "Train acc = %.5f " % train_acc,
                "Val acc = %.5f " % val_acc,
                log=log,
            )

        if val_acc > max_val_acc:
            wait = 0
            max_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = model.state_dict()
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train acc = %.5f " % train_acc_list[best_epoch]
    out_str += "Val acc = %.5f " % val_acc_list[best_epoch]
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        plt.plot(range(0, epoch + 1), train_acc_list, "-", label="Train Acc")
        plt.plot(range(0, epoch + 1), val_acc_list, "-", label="Val Acc")
        plt.title("Epoch-Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, x, y, test_indices, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_pred = predict(model, x)
    end = time.time()

    test_acc = accuracy(y_pred[test_indices], y[test_indices])
    out_str = "Test acc = %.5f" % test_acc

    print_log(out_str, log=log)
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="gcn")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-c", "--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--cpus", type=int, default=1)
    args = parser.parse_args()

    seed_everything(args.seed)
    set_cpu_num(args.cpus)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = args.model.upper()
    model_class = model_select(model_name)
    model_name = model_class.__name__

    with open(f"../configs/{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # -------------------------------- load model -------------------------------- #

    # cfg.get(key, default_value=None): no need to write in the config if not used
    # cfg[key]: must be assigned in the config, else KeyError
    if cfg.get("pass_device"):
        cfg["model_args"]["device"] = DEVICE

    model = model_class(**cfg["model_args"])

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/{model_name}"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    x, y, train_indices, val_indices, test_indices = get_tensors(log=log)
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    print_log(log=log)

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/{model_name}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.get("milestones", []),
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["model_args"]["num_nodes"],
                cfg["model_args"]["input_dim"],
            ],
            verbose=0,
            device=DEVICE,
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        x,
        y,
        train_indices,
        val_indices,
        optimizer,
        scheduler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        compile_model=args.compile,
        verbose=1,
        log=log,
        save=save,
    )

    test_model(model, x, y, test_indices, log=log)

    log.close()

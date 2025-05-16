import os
import argparse
import json
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2015 import ASSIST2015
from data_loaders.algebra2005 import Algebra2005
from data_loaders.statics2011 import Statics2011
from data_loaders.my_classroom import MyClassroom

from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.gkt import PAM, MHA
from models.utils import collate_fn

import random
import numpy as np


class EarlyStopping:
    def __init__(self, patience=3, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0
        return self.early_stop


def main(model_name, dataset_name, pretrained_path=None):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name, dataset_name)
    os.makedirs(ckpt_path, exist_ok=True)

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]

    if pretrained_path:
        train_config = config["fine_tune_config"]
    else:
        train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]
    seq_len = train_config["seq_len"]

    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2015":
        dataset = ASSIST2015(seq_len)
    elif dataset_name == "Algebra2005":
        dataset = Algebra2005(seq_len)
    elif dataset_name == "Statics2011":
        dataset = Statics2011(seq_len)
    elif dataset_name == "MyClassroom":
        dataset = MyClassroom(seq_len)
    else:
        raise ValueError("Unknown dataset name")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    if model_name == "dkt":
        model = DKT(dataset.num_q, **model_config).to(device)
    elif model_name == "dkt+":
        model = DKTPlus(dataset.num_q, **model_config).to(device)
    elif model_name == "dkvmn":
        model = DKVMN(dataset.num_q, **model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(dataset.num_q, **model_config).to(device)
    elif model_name == "gkt":
        if model_config["method"] == "PAM":
            model = PAM(dataset.num_q, **model_config).to(device)
        elif model_config["method"] == "MHA":
            model = MHA(dataset.num_q, **model_config).to(device)
        else:
            raise ValueError("Unknown GKT method")
    else:
        raise ValueError("Unknown model name")

    if pretrained_path:
        print(f"🔁 Loading pretrained weights from {pretrained_path}")
        pretrained_weights = torch.load(pretrained_path)
        model_weights = model.state_dict()

        filtered_weights = {
            k: v for k, v in pretrained_weights.items()
            if k in model_weights and v.size() == model_weights[k].size()
        }

        model_weights.update(filtered_weights)
        model.load_state_dict(model_weights)
        print(f"✅ Loaded pretrained weights for {len(filtered_weights)} / {len(model_weights)} layers (skipped incompatible)")

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
        with open(os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb") as f:
            train_dataset.indices = pickle.load(f)
        with open(os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb") as f:
            test_dataset.indices = pickle.load(f)
    else:
        with open(os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb") as f:
            pickle.dump(train_dataset.indices, f)
        with open(os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb") as f:
            pickle.dump(test_dataset.indices, f)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True, collate_fn=collate_fn)

    if optimizer == "sgd":
        opt = SGD(filter(lambda p: p.requires_grad, model.parameters()), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    aucs, loss_means = model.train_model(train_loader, test_loader, num_epochs, opt, ckpt_path)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_filename = f"{model_name}_{dataset_name}_{timestamp}.pt"

    if pretrained_path:
        save_dir = "finaltrainedmodels"
        print("📌 Detected fine-tuning. Saving to finaltrainedmodels/")
    else:
        save_dir = "pretrainedmodels"
        print("📌 Detected pretraining. Saving to pretrainedmodels/")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_filename)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")

    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="dkt",
        help="Model to train: [dkt, dkt+, dkvmn, sakt, gkt]"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="ASSIST2009",
        help="Dataset to use: [ASSIST2009, ASSIST2015, Algebra2005, Statics2011, MyClassroom]"
    )
    parser.add_argument(
        "--pretrained_path", type=str, default=None,
        help="Path to pretrained model checkpoint (for fine-tuning)"
    )

    args = parser.parse_args()
    main(args.model_name, args.dataset_name, args.pretrained_path)

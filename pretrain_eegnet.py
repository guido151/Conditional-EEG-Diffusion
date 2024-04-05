# Authors: Guido Klein <guido.klein@ru.nl>
#
# License: BSD (3-clause)

import pandas as pd
import skorch
import torch as th
import torch.nn as nn
import numpy as np
from braindecode.models import EEGNetv4
from huggingface_hub import upload_folder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from skorch.net import NeuralNet

from data_utils.dataset import (
    EEGDataset,
    Lee2019Dataset,
)
from configs.Lee2019_ERP_CFG import *


def train_eegnet(dataset: EEGDataset, dataset_name: str, ch_names: list, n_times: int):
    """Train EEGNet to be used during FID and IS metrics compuation. Checkpoint is saved locally.

    Args:
        dataset (EEGDataset): Initialized dataset
        dataset_name (str): Name of the dataset. Used to create a new folder for each pretrained EEGNet.
        ch_names (list): Which channels to use. Note that this should be the same as during the training of the diffusion model.
        n_times (int): Lenght of the EEG data in total number of samples. Note that this should be the same as during the training of the diffusion model.
    """

    # split the dataset into train and test sets
    test_dataset, train_dataset = train_test_split(
        dataset,
        test_size=0.2,
        shuffle=True,
        stratify=dataset.y_df_train["label"],
    )

    label_idx = dataset.y_df_all.columns.to_list().index("label")
    X_train, y_train = np.array([i[0] for i in train_dataset]), np.array(
        [i[1][label_idx] for i in train_dataset]
    )
    X_test, y_test = np.array([i[0] for i in test_dataset]), np.array(
        [i[1][label_idx] for i in test_dataset]
    )

    # initialize model
    model = EEGNetv4(
        n_chans=len(ch_names),
        n_outputs=dataset.y_df_all["label"].nunique(),
        n_times=n_times,
        drop_prob=0.25,
    )
    print(dataset.rebalanced_weights)
    print(dataset.y_df_all["label"].value_counts())

    # initialize loss
    criterion = nn.CrossEntropyLoss(weight=th.tensor(dataset.rebalanced_weights))
    net = NeuralNet(
        model,
        criterion=criterion,
        max_epochs=1000,
        callbacks=[skorch.callbacks.Checkpoint(dirname=f"EEGNet/{dataset_name}")],
        device=RUN_CFG["device"],
    )
    net = net.initialize()

    # train the network
    net.fit(X_train, y_train)

    # save the parameters
    net.load_params(
        f"EEGNet/{dataset_name}/params.pt",
        f"EEGNet/{dataset_name}/optimizer.pt",
        f"EEGNet/{dataset_name}/criterion.pt",
        f"EEGNet/{dataset_name}/history.json",
    )
    y_pred = net.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


dataset = Lee2019Dataset(**DATASET_CFG)
n_times = dataset.X.shape[2]
train_eegnet(
    dataset, DATASET_CFG["dataset"], ch_names=DATASET_CFG["ch_names"], n_times=n_times
)

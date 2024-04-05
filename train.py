# Authors: Guido Klein <guido.klein@ru.nl>
#
# License: BSD (3-clause)

import datetime
import os

import lightning as l
import torch as th
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, random_split

from callbacks import EMACallback
from data_utils.dataset import Lee2019Dataset
from framework import SDEFramework

from configs.Lee2019_ERP_CFG import (
    DATASET_CFG,
    FRAMEWORK_CFG,
    MODEL_CFG,
    RUN_CFG,
    SAMPLING_CFG,
    SDE_CFG,
)


def main():
    """
    Function does the following:
    - Load the preprocessed dataset and initializes the conditional diffusion framework 
    - Dataset is split into a validation and training dataset
    - Callbacks are added to save the model and do the EMA of the weights
    - Train the conditional diffusion model
    
    Config for training can be changed in the configs/Leee2019_ERP_CFG.py
    """

    dataset = Lee2019Dataset(**DATASET_CFG)

    checkpoint_dir = os.path.join(RUN_CFG["project_name"], RUN_CFG["run_name"]).replace(
        "\\", "/"
    )

    # if conditional, configure the labels and create possible combinations
    if isinstance(DATASET_CFG["user_conditions"], list):
        MODEL_CFG["model__conditionals_combinations"] = dataset.condition_combinations

    # configure framework
    framework = SDEFramework(
        **RUN_CFG,
        **FRAMEWORK_CFG,
        **MODEL_CFG,
        **SDE_CFG,
        **SAMPLING_CFG,
        **DATASET_CFG,
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [1 - RUN_CFG["val_split"], RUN_CFG["val_split"]],
        th.Generator().manual_seed(42),
    )

    # create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{step}-{validation_loss:.3f}",
            every_n_train_steps=RUN_CFG["checkpoint_freq"],
            save_top_k=-1,
        ),
        EMACallback(),
    ]

    # create dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=RUN_CFG["batch_size"], shuffle=False, drop_last=False
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=RUN_CFG["batch_size"], shuffle=False, drop_last=False
    )

    trainer = l.Trainer(
        log_every_n_steps=1,
        callbacks=callbacks,
        gradient_clip_val=RUN_CFG["gradient_clip_val"],
        max_steps=RUN_CFG["steps"],
        strategy=DDPStrategy(timeout=datetime.timedelta(minutes=300)),
    )

    # train the model
    trainer.fit(framework, train_dataloader, val_dataloader)


main()

# Authors: Guido Klein <guido.klein@ru.nl>
# 
# License: BSD (3-clause)

from typing import Any, Optional

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT


class EMACallback(Callback):
    """Update the EMA model using the current trained model

    Args:
        Callback (lightning.Callback): lightning.Callback class
    """

    def on_train_batch_end(
        self,
        pl_module: LightningModule,
    ):
        """
        Updates the EMA model using the current trained model after each training batch

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The LightningModule.
            outputs (STEP_OUTPUT): The outputs of the training batch.
            batch (Any): The current batch.
            batch_idx (int): The index of the current batch.
        """
        pl_module.ema.step_ema(pl_module.ema_model, pl_module.model)

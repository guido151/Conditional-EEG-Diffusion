# Authors: Guido Klein <guido.klein@ru.nl>
#
# License: BSD (3-clause)

import copy

import lightning as l
import numpy as np
import torch as th
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, optim

import models.utils as mutils
from losses import get_loss_fn
from sampling import get_sampling_fn
from sde import SDE


class SDEFramework(l.LightningModule):
    """
    Framework that trains the neural networks, based on the specified SDE.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_arg = {k[7:]: v for k, v in kwargs.items() if "model__" in k}
        self.model = self.hparams.model(**self.model_arg)

        self.sde_arg = {k[5:]: v for k, v in kwargs.items() if "sde__" in k}
        self.sde = self.hparams.sde(**self.sde_arg)

        self.loss_fn = get_loss_fn(
            self.sde,
            self.hparams.reduce_mean,
            self.hparams.continuous,
            self.hparams.likelihood_weighting,
        )

        self.sampling_fn = get_sampling_fn(
            self.sde,
            self.hparams.sampler_name,
            self.hparams.predictor_name,
            self.hparams.corrector_name,
            self.hparams.noise_removal,
            self.hparams.snr,
            self.hparams.n_steps_each,
            self.hparams.probability_flow,
            self.hparams.continuous,
            self.hparams.device,
            self.hparams.sampling_eps,
        )

        self.ema = mutils.EMA(self.hparams.EMA_beta)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

    def training_step(self, batch, batch_idx):
        x, y = self._get_batch_data(batch)

        loss = self.loss_fn(self.model, x, y)
        self._log_training_loss(loss)
        return loss

    def _get_batch_data(self, batch):
        if isinstance(self.hparams.user_conditions, list):
            x, y = batch
        else:
            x = batch
            y = None
        return x, y

    def _log_training_loss(self, loss):
        self.log("batch loss", loss.item())
        self.log("epoch loss", loss.item(), on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = self._initialize_optimizer()
        scheduler = self._initialize_scheduler(optimizer)
        return [optimizer], [scheduler] if scheduler else []

    def _initialize_optimizer(self):
        if self.hparams.optimizer_type == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
                betas=self.hparams.lr_betas,
                eps=self.hparams.optim_eps,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_type == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.hparams.lr,
                betas=self.hparams.lr_betas,
                eps=self.hparams.optim_eps,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer_type == "radam":
            optimizer = optim.RAdam(
                self.model.parameters(),
                lr=self.hparams.lr,
                betas=self.hparams.lr_betas,
                eps=self.hparams.optim_eps,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise NotImplementedError(self.hparams.optimizer_type)
        return optimizer

    def _initialize_scheduler(self, optimizer):
        if self.hparams.lr_scheduler is None:
            return None
        elif self.hparams.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=1)
        elif self.hparams.lr_scheduler == "linear":
            return optim.lr_scheduler.LinearLR(optimizer)
        else:
            raise NotImplementedError(self.hparams.lr_scheduler)

    def validation_step(self, batch, batch_idx):
        x, y = self._get_batch_data(batch)
        loss = self.loss_fn(self.model, x, y)
        self.log("validation_loss", loss, sync_dist=True)

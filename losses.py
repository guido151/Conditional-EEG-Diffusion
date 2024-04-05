# source: https://github.com/yang-song/score_sde_pytorch/blob/main/losses.py

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications are limited to:
# 1. Adding y in multiple functions to allow for conditional sampling
# 2. Shape is changed in places where data should have a similar shape to EEG data

"""All functions related to loss computation and optimization.
"""

from typing import Optional

import torch as th

from models import utils as mutils
from sde import VESDE, VPSDE


def get_sde_loss_fn(
    sde, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5
):
    """Create a loss function for training with arbirary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
            ad-hoc interpolation to take continuous time steps.
        likelihood_weighting: If `True`, weight the mixture of score matching losses
            according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = (
        th.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * th.sum(*args, **kwargs)
    )

    def loss_fn(model, batch, y: Optional[th.Tensor] = None):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.
            y: Conditional label.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, continuous=continuous)
        t = th.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = th.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None] * z
        score = score_fn(perturbed_data, t, y)

        if not likelihood_weighting:
            losses = th.square(score * std[:, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(th.zeros_like(batch), t)[1] ** 2
            losses = th.square(score + z / std[:, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

        loss = th.mean(losses)
        return loss

    return loss_fn


def get_smld_loss_fn(vesde, reduce_mean=False):
    """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
    assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

    # Previous SMLD models assume descending sigmas
    smld_sigma_array = th.flip(vesde.discrete_sigmas, dims=(0,))
    reduce_op = (
        th.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * th.sum(*args, **kwargs)
    )

    def loss_fn(model, batch, y: Optional[th.Tensor] = None):
        labels = th.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        sigmas = smld_sigma_array.to(batch.device)[labels]
        noise = th.randn_like(batch) * sigmas[:, None, None]
        perturbed_data = noise + batch
        score = model(perturbed_data, labels, y)
        target = -noise / (sigmas**2)[:, None, None]
        losses = th.square(score - target)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas**2
        loss = th.mean(losses)
        return loss

    return loss_fn


def get_ddpm_loss_fn(vpsde, reduce_mean=True):
    """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
    assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

    reduce_op = (
        th.mean
        if reduce_mean
        else lambda *args, **kwargs: 0.5 * th.sum(*args, **kwargs)
    )

    def loss_fn(model, batch, y: th.Tensor = th.tensor([])):
        labels = th.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = th.randn_like(batch)
        perturbed_data = (
            sqrt_alphas_cumprod[labels, None, None] * batch
            + sqrt_1m_alphas_cumprod[labels, None, None] * noise
        )
        score = model(perturbed_data, labels, y)
        losses = th.square(score - noise)
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = th.mean(losses)
        return loss

    return loss_fn


def get_loss_fn(sde, reduce_mean: bool, continuous: bool, likelihood_weighting: bool):
    if continuous:
        loss_fn = get_sde_loss_fn(
            sde,
            reduce_mean=reduce_mean,
            continuous=continuous,
            likelihood_weighting=likelihood_weighting,
        )
    else:
        assert (
            not likelihood_weighting
        ), "Likelihood weighting is not supported for original SMLD/DDPM training."
        if isinstance(sde, VESDE):
            loss_fn = get_smld_loss_fn(sde, reduce_mean=reduce_mean)
        elif isinstance(sde, VPSDE):
            loss_fn = get_ddpm_loss_fn(sde, reduce_mean=reduce_mean)
        else:
            raise ValueError(
                f"Discrete training for {sde.__class__.__name__} is not recommended."
            )
    return loss_fn

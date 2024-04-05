# Authors: Guido Klein <guido.klein@ru.nl>
#
# License: BSD (3-clause)

"""All functions and modules related to model definition.
"""

import torch as th
from sde import subVPSDE, VESDE, VPSDE
import torch.nn as nn

# source: https://github.com/yang-song/score_sde_pyth/blob/main/models/utils.py
def get_score_fn(sde, model, continuous=False):
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

    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        model: A score model.
        train: `True` for training and `False` for evaluation.
        continuous: If `True`, the score-based model is expected to directly take continuous time steps.

    Returns:
        A score function.
    """

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):

        def score_fn(x, t, y: th.Tensor = th.tensor([])):
            # Scale neural network output by standard deviation and flip sign
            if continuous or isinstance(sde, subVPSDE):
                # For VP-trained models, t=0 corresponds to the lowest noise level
                # The maximum value of time embedding is assumed to 999 for
                # continuously-trained models.
                labels = t * 999
                score = model(x, labels, y)
                std = sde.marginal_prob(th.zeros_like(x), t)[1]
            else:
                # For VP-trained models, t=0 corresponds to the lowest noise level
                labels = t * (sde.N - 1)
                score = model(x, labels, y)
                std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

            score = -score / std[:, None, None]
            return score

    elif isinstance(sde, VESDE):

        def score_fn(x, t, y: th.Tensor = th.tensor([])):
            if continuous:
                labels = sde.marginal_prob(th.zeros_like(x), t)[1]
            else:
                # For VE-trained models, t=0 corresponds to the highest noise level
                labels = sde.T - t
                labels *= sde.N - 1
                labels = th.round(labels).long()
            score = model(x, labels, y)
            return score

    else:
        raise NotImplementedError(
            f"SDE class {sde.__class__.__name__} not yet supported."
        )

    return score_fn


def to_flattened_numpy(x):
    """Flatten a th tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a th tensor with the given `shape` from a flattened numpy array `x`."""
    return th.from_numpy(x.reshape(shape))


# Based on: https://wandb.ai/capecape/train_sd/reports/How-To-Train-a-Conditional-Diffusion-Model-From-Scratch--VmlldzoyNzIzNTQ1
class EMA:
    """Exponential moving average."""

    def __init__(self, beta: float):
        """Initialize Exponential Moving Average

        Args:
            beta (float): Percentage of the new EMA model that is the old EMA model
        """
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1.0 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_params(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# based on marginal in sde.py
def get_sde_forward_fn(sde):
    def forward_fn(batch, t=None, z=None, eps=1e-5):
        if t is None:
            t = th.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        else:
            t = t * (sde.T - eps) + eps
        if z is None:
            z = th.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None] * z
        return perturbed_data

    return forward_fn


def get_smld_forward_fn(vesde):
    # Previous SMLD models assume descending sigmas
    smld_sigma_array = th.flip(vesde.discrete_sigmas, dims=(0,))

    def forward_fn(batch, t=None, z=None):
        if t is None:
            t = th.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
        if z is None:
            z = th.randn_like(batch)
        sigmas = smld_sigma_array.to(batch.device)[t]
        noise = z * sigmas[:, None, None]
        perturbed_data = noise + batch
        return perturbed_data

    return forward_fn


def get_ddpm_forward_fn(vpsde):
    def forward_fn(batch, t=None, z=None):
        if t is None:
            t = th.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
        if z is None:
            z = th.randn_like(batch)
        sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
        sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
        noise = z
        perturbed_data = (
            sqrt_alphas_cumprod[t, None, None] * batch
            + sqrt_1m_alphas_cumprod[t, None, None] * noise
        )
        return perturbed_data

    return forward_fn


def get_forward_fn(sde, continuous: bool):
    if continuous:
        forward_fn = get_sde_forward_fn(sde)
    else:
        if isinstance(sde, VESDE):
            forward_fn = get_smld_forward_fn(sde)
        elif isinstance(sde, VPSDE):
            forward_fn = get_ddpm_forward_fn(sde)
        else:
            raise ValueError(f"There is no discrete forward process for {sde}")
    return forward_fn


# Based on: https://github.com/tszab/eegwave-dpm/tree/master
class DiffusionStepEmbedding(nn.Module):
    def __init__(
        self,
        max_timesteps: int,
        step_emb_in_dim: int,
        step_emb_hidden_dim: int,
        step_emb_out_dim: int,
    ):
        """Create a timestep embedding based on look-up table impelemented in Embedding layer

        Args:
            max_timesteps (int): Maximum number of timesteps used during the forward diffusion process.
            step_embd_in_dim (int): Size of the original timestep embedding.
            step_emb_hidden_dim (int): Size of the timestep embedding after one linear layer.
            step_emb_out_dim (int): Final size of the timestep embedding after second linear layer .
        """
        super(DiffusionStepEmbedding, self).__init__()
        self.step_emb_in_dim = step_emb_in_dim
        self.step_emb_hidden_dim = step_emb_hidden_dim
        self.step_emb_out_dim = step_emb_out_dim
        self.max_timesteps = max_timesteps

        self.half_dims = step_emb_in_dim // 2
        steps = th.arange(self.max_timesteps).unsqueeze(1)
        dims = th.arange(self.half_dims).unsqueeze(0)
        emb = 10.0 ** (dims * 4.0 / (self.half_dims - 1))
        table = steps * emb
        self.register_buffer("embedding", th.cat([th.sin(table), th.cos(table)], dim=1))

        self.step_embedding_linear = nn.Sequential(
            nn.Linear(step_emb_in_dim, step_emb_hidden_dim),
            nn.SiLU(),
            nn.Linear(step_emb_hidden_dim, step_emb_out_dim),
            nn.SiLU(),
        )

    def _interp_step(self, t: th.Tensor) -> th.Tensor:
        """Interpolate the timestep embedding based on float

        Args:
            step (th.Tensor): timestep of each sample. Shape: (batch_size,)

        Returns:
            th.Tensor: interpolated timestep embedding. Shape: (batch_size, step_emb_in_dim)
        """
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        low_val = self.embedding[low_idx].squeeze(1)
        high_val = self.embedding[high_idx].squeeze(1)

        t = low_val + (high_val - low_val) * (t - low_idx).unsqueeze(1)

        assert t.shape[1] == self.step_emb_in_dim
        return t

    def forward(self, t: th.Tensor) -> th.Tensor:
        """Create timestep embedding

        Args:
            t (th.Tensor): timestep of each sample. Shape: (batch_size,)

        Returns:
            th.Tensor: timestep embedding. Shape: (batch_size, step_emb_out_dim)
        """
        if t.dtype in [th.float]:
            t = self._interp_step(t)
        else:
            t = self.embedding[t].squeeze(1)
        t = self.step_embedding_linear(t)

        return t


def get_activations(model: nn.Module, x: th.Tensor, layer_idx: int):
    """Extract the activations of the layer at layer_idx

    Args:
        model (nn.Module): Pytorch model
        x (th.Tensor): EEG data of shape (batch_size, eeg_channels, length_eeg)
        layer_idx (int): index of the layer to extract the activations from

    Returns:
        th.Tensor: tensor with the activations of the layer at layer_idx
    """
    model.eval()
    activations = []

    def hook_fn(module, x, output):
        activation = output.detach()
        activations.append(activation)
        return activation

    hook = model[layer_idx].register_forward_hook(hook_fn)
    model.to(x.device)
    model.eval()
    with th.no_grad():
        model(x)
    hook.remove()
    return activations[0]

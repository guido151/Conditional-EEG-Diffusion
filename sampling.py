# source: https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py

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

# pylint: skip-file
# pytype: skip-file

# Modifications are limited to: 
# 1. Adding y in multiple functions to allow for conditional sampling
# 2. Shape is changed in places where data should have a similar shape to EEG data
# 3. Comments and indentations are changed in multiple places for clarity and consistency 

"""Various sampling methods."""

import abc
import functools
from typing import Optional

import numpy as np
import torch as th
from scipy import integrate

from models import utils as mutils
from models.utils import from_flattened_numpy, get_score_fn, to_flattened_numpy
from sde import VESDE, VPSDE, subVPSDE

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(
    sde,
    sampler_name: str,
    predictor_name: str,
    corrector_name: str,
    noise_removal: bool,
    snr: float,
    n_steps_each: int,
    probability_flow: bool,
    continuous: bool,
    device,
    eps: float,
):
    """Create a sampling function.

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == "ode":
        sampling_fn = get_ode_sampler(
            sde=sde,
            denoise=noise_removal,
            eps=eps,
            device=device,
        )
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == "pc":
        predictor = get_predictor(predictor_name.lower())
        corrector = get_corrector(corrector_name.lower())
        sampling_fn = get_pc_sampler(
            sde=sde,
            predictor=predictor,
            corrector=corrector,
            snr=snr,
            n_steps=n_steps_each,
            probability_flow=probability_flow,
            continuous=continuous,
            denoise=noise_removal,
            eps=eps,
            device=device,
        )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, y: Optional[th.Tensor]):
        """One update of the predictor.

        Args:
            x: A Pytorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            y: A Pytorch tensor of conditioning labels (if condioned model)

        Returns:
            x: A Pytorch tensor of the next state.
            x_mean: A Pytorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, y: Optional[th.Tensor]):
        """One update of the corrector.

        Args:
            x: A Pytorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.
            y: A Pytorch tensor of conditioning labels (if condioned model)

        Returns:
            x: A Pytorch tensor of the next state.
            x_mean: A Pytorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, y: Optional[th.Tensor]):
        dt = -1.0 / self.rsde.N
        z = th.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t, y)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, y: Optional[th.Tensor]):
        f, G = self.rsde.discretize(x, t, y)
        z = th.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None] * z
        return x, x_mean


@register_predictor(name="ancestral_sampling")
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        assert (
            not probability_flow
        ), "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t, y: Optional[th.Tensor]):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = th.where(
            timestep == 0,
            th.zeros_like(t),
            sde.discrete_sigmas.to(t.device)[timestep - 1],
        )
        score = self.score_fn(x, t, y)
        x_mean = x + score * (sigma**2 - adjacent_sigma**2)[:, None, None]
        std = th.sqrt((adjacent_sigma**2 * (sigma**2 - adjacent_sigma**2)) / (sigma**2))
        noise = th.randn_like(x)
        x = x_mean + std[:, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t, y: Optional[th.Tensor]):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t, y)
        x_mean = (x + beta[:, None, None] * score) / th.sqrt(1.0 - beta)[:, None, None]
        noise = th.randn_like(x)
        x = x_mean + th.sqrt(beta)[:, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t, y: Optional[th.Tensor]):
        if isinstance(self.sde, VESDE):
            return self.vesde_update_fn(x, t, y)
        elif isinstance(self.sde, VPSDE):
            return self.vpsde_update_fn(x, t, y)


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, y: Optional[th.Tensor]):
        return x, x


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, VPSDE)
            and not isinstance(sde, VESDE)
            and not isinstance(sde, subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t, y: Optional[th.Tensor]):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]  # subVPSDE does not have alphas?
        else:
            alpha = th.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t, y)
            noise = th.randn_like(x)
            grad_norm = th.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = th.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None] * grad
            x = x_mean + th.sqrt(step_size * 2)[:, None, None] * noise

        return x, x_mean


@register_corrector(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, VPSDE)
            and not isinstance(sde, VESDE)
            and not isinstance(sde, subVPSDE)
        ):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )

    def update_fn(self, x, t, y: Optional[th.Tensor]):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = th.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t, y)
            noise = th.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None] * grad
            x = x_mean + noise * th.sqrt(step_size * 2)[:, None, None]

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t, y: Optional[th.Tensor]):
        return x, x


def shared_predictor_update_fn(
    x, t, y, sde, model, predictor, probability_flow, continuous
):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, y)


def shared_corrector_update_fn(
    x, t, y, sde, model, corrector, continuous, snr, n_steps
):
    """A wrapper that configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t, y)


def get_pc_sampler(
    sde,
    predictor,
    corrector,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        sde: An `sde_lib.SDE` object representing the forward SDE.
        predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
        corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
        snr: A `float` number. The signal-to-noise ratio for configuring correctors.
        n_steps: An integer. The number of corrector steps per predictor update.
        probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
        continuous: `True` indicates that the score model was continuously trained.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        device: Pyth device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def pc_sampler(model, shape, z=None, y: Optional[th.Tensor] = None):
        """The PC sampler funciton.

        Args:
            model: A score model.
            shape: A sequence of integers. The expected shape of a single sample.
            z: The latent code.
            y: if present, generate samples from the conditional label

        Returns:
            Samples, number of function evaluations.
        """
        with th.no_grad():
            # Initial sample
            if z is None:
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z.to(device)
            y = y.to(device)
            timesteps = th.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = th.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, y, model=model)
                x, x_mean = predictor_update_fn(x, vec_t, y, model=model)

            return x_mean if denoise else x, sde.N * (n_steps + 1)

    return pc_sampler


def get_ode_sampler(
    sde,
    denoise=False,
    rtol=1e-5,
    atol=1e-5,
    method="RK45",
    eps=1e-3,
    device="cuda",
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: Pyth device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x, y: Optional[th.Tensor]):
        score_fn = get_score_fn(sde, model, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = th.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps, y)
        return x

    def drift_fn(model, x, t, y: Optional[th.Tensor]):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, y)[0]

    def ode_sampler(model, shape, z=None, y: Optional[th.Tensor] = None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            shape: A sequence of integers. The expected shape of a single sample.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with th.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z.to(device)
            y = y.to(device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(th.float32)
                vec_t = th.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t, y)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func,
                (sde.T, eps),
                to_flattened_numpy(x),
                rtol=rtol,
                atol=atol,
                method=method,
            )
            nfe = solution.nfev
            x = th.tensor(solution.y[:, -1]).reshape(shape).to(device).type(th.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x, y)

            return x, nfe

    return ode_sampler

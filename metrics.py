# Authors: Guido Klein <guido.klein@ru.nl>
# 
# License: BSD (3-clause)

from copy import deepcopy
from typing import Any, List, Optional, Sequence, Tuple, Union

import mne
import numpy as np
import torch as th
import torch.nn as nn
from braindecode.models import EEGNetv4
from ot.sliced import sliced_wasserstein_distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.pairwise import manhattan_distances
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from torchmetrics.image.fid import _compute_fid
from torchmetrics.image.inception import InceptionScore
from torch import Tensor
from torch.nn import Module
from torchmetrics.metric import Metric
import matplotlib.pyplot as plt


# Based on https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/21
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_fid_model(model: EEGNetv4) -> nn.Module:
    """
    Returns the model that is used to calculate FID score

    Args:
        model (EEGNetv4): The initialized EEGNetv4 model

    Returns:
        nn.Module: EEGNetv4 model without layers after last pooling layer
    """
    fid_model = deepcopy(model)
    for i in range(len(fid_model)):
        if i >= 14:
            fid_model[i] = Identity()

    fid_model.eval()
    for param in fid_model.parameters():
        param.requires_grad = False
    return fid_model


def get_is_model(model: EEGNetv4) -> nn.Module:
    """
    Returns the model that is used to calculate Inception score

    Args:
        model (EEGNetv4): The initialized EEGNetv4 model

    Returns:
        nn.Module: EEGNetv4 model in eval mode and gradients disabled
    """
    is_model = deepcopy(model)
    is_model.eval()
    for param in is_model.parameters():
        param.requires_grad = False
    return is_model


# https://github.com/Lightning-AI/thmetrics/blob/v1.3.0.post0/src/thmetrics/image/fid.py#L182-L436 with adaptated dummy input
class AdaptedFrechetInceptionDistance(Metric):
    r"""Calculate Fréchet inception distance (FID_) which is used to access the quality of generated images.

    .. math::
        FID = \|\mu - \mu_w\|^2 + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})

    where :math:`\mathcal{N}(\mu, \Sigma)` is the multivariate normal distribution estimated from Inception v3
    (`fid ref1`_) features calculated on real life images and :math:`\mathcal{N}(\mu_w, \Sigma_w)` is the
    multivariate normal distribution estimated from Inception v3 features calculated on generated (fake) images.
    The metric was originally proposed in `fid ref1`_.

    Using the default feature extraction (Inception v3 using the original weights from `fid ref2`_), the input is
    expected to be mini-batches of 3-channel RGB images of shape ``(3xHxW)``. If argument ``normalize``
    is ``True`` images are expected to be dtype ``float`` and have values in the ``[0,1]`` range, else if
    ``normalize`` is set to ``False`` images are expected to have dtype ``uint8`` and take values in the ``[0, 255]``
    range. All images will be resized to 299 x 299 which is the size of the original training data. The boolian
    flag ``real`` determines if the images should update the statistics of the real distribution or the
    fake distribution.

    This metric is known to be unstable in its calculatations, and we recommend for the best results using this metric
    that you calculate using `th.float64` (default is `th.float32`) which can be set using the `.set_dtype`
    method of the metric.

    .. note:: using this metrics requires you to have th 1.9 or higher installed

    .. note:: using this metric with the default feature extractor requires that ``th-fidelity``
        is installed. Either install as ``pip install thmetrics[image]`` or ``pip install th-fidelity``

    As input to ``forward`` and ``update`` the metric accepts the following input

    - ``x`` (:class:`~th.Tensor`): tensor with data to feed the feature extractor with
    - ``real`` (:class:`~bool`): bool indicating if ``imgs`` belong to the real or the fake distribution

    As output of `forward` and `compute` the metric returns the following output

    - ``fid`` (:class:`~th.Tensor`): float scalar tensor with mean FID value over samples

    Args:
        feature:
            nn.module
        reset_real_features: Whether to also reset the real features. Since in many cases the real dataset does not
            change, the features can be cached them to avoid recomputing them which is costly. Set this to ``False`` if
            your dataset does not change.
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.
        layer:

    Raises:
        ValueError:
            If th version is lower than 1.9
        ModuleNotFoundError:
            If ``feature`` is set to an ``int`` (default settings) and ``th-fidelity`` is not installed
        ValueError:
            If ``feature`` is set to an ``int`` not in [64, 192, 768, 2048]
        TypeError:
            If ``feature`` is not an ``str``, ``int`` or ``th.nn.Module``
        ValueError:
            If ``reset_real_features`` is not an ``bool``

    Example:
        >>> import th
        >>> _ = th.manual_seed(123)
        >>> from thmetrics.image.fid import FrechetInceptionDistance
        >>> fid = FrechetInceptionDistance(feature=64)
        >>> # generate two slightly overlapping image intensity distributions
        >>> imgs_dist1 = th.randint(0, 200, (100, 3, 299, 299), dtype=th.uint8)
        >>> imgs_dist2 = th.randint(100, 255, (100, 3, 299, 299), dtype=th.uint8)
        >>> fid.update(imgs_dist1, real=True)
        >>> fid.update(imgs_dist2, real=False)
        >>> fid.compute()
        tensor(12.7202)

    """

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    model: Module

    def __init__(
        self,
        feature: Module,
        dummy_input: Tensor,
        reset_real_features: bool = True,
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.model = feature
        dummy_out = self.model(dummy_input)
        num_features = dummy_out.reshape(dummy_out.shape[0], -1).shape[1]

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        mx_num_feats = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            th.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            th.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", th.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            th.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            th.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", th.tensor(0).long(), dist_reduce_fx="sum"
        )

    def update(self, x: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        features = self.model(x)
        features = features.reshape(features.shape[0], -1)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += x.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += x.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributed to compute FID"
            )
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(
            0
        )
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(
            0
        )

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        ).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[plt.Axes] = None,
    ):
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import th
            >>> from thmetrics.image.fid import FrechetInceptionDistance
            >>> imgs_dist1 = th.randint(0, 200, (100, 3, 299, 299), dtype=th.uint8)
            >>> imgs_dist2 = th.randint(100, 255, (100, 3, 299, 299), dtype=th.uint8)
            >>> metric = FrechetInceptionDistance(feature=64)
            >>> metric.update(imgs_dist1, real=True)
            >>> metric.update(imgs_dist2, real=False)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import th
            >>> from thmetrics.image.fid import FrechetInceptionDistance
            >>> imgs_dist1 = lambda: th.randint(0, 200, (100, 3, 299, 299), dtype=th.uint8)
            >>> imgs_dist2 = lambda: th.randint(100, 255, (100, 3, 299, 299), dtype=th.uint8)
            >>> metric = FrechetInceptionDistance(feature=64)
            >>> values = [ ]
            >>> for _ in range(3):
            ...     metric.update(imgs_dist1(), real=True)
            ...     metric.update(imgs_dist2(), real=False)
            ...     values.append(metric.compute())
            ...     metric.reset()
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)


class Metrics:
    def __init__(
        self,
        data: th.Tensor,
        feature: EEGNetv4,
        amplitude_latency_channel: str,
        fmin: float,
        fmax: float,
    ):
        self.data = data
        self.feature = feature
        self.amplitude_latency_channel = amplitude_latency_channel
        self.fmin = fmin
        self.fmax = fmax

        self.fid, self.inception_score = self.init_is_fid()

        self.session_specific_metrics = pd.DataFrame([])
        self.label_specific_metrics = pd.DataFrame([])
        self.full_dataset_metrics = pd.DataFrame([])

    def update_non_session_specific(
        self,
        real_epo,
        sampled_epo,
        sampling_combination,
    ):
        if self.fid is not None:
            self.fid.update(
                th.tensor(sampled_epo.get_data(units="uV"), dtype=th.float32), False
            )
        if self.inception_score is not None:
            self.inception_score.update(
                th.tensor(sampled_epo.get_data(units="uV"), dtype=th.float32)
            )
        self.label_specific_metrics = self.add_eeg_metrics(
            real_epo,
            sampled_epo,
            self.label_specific_metrics,
            self.amplitude_latency_channel,
            sampling_combination,
        )

    def update_session_specific(self, real_epo, sampled_epo, sampling_combination):
        self.session_specific_metrics = self.add_lda_score(
            real_epo,
            sampled_epo,
            self.session_specific_metrics,
            sampling_combination,
        )

    def log_dfs(self, step: int):
        self.session_specific_metrics["step"] = step
        self.label_specific_metrics["step"] = step

        if self.fid is not None:
            fid = self.fid.compute().numpy()

        if self.inception_score is not None:
            mean_is, std_is = self.inception_score.compute()
            mean_is = mean_is.numpy()
            std_is = std_is.numpy()

        self.full_dataset_metrics = pd.DataFrame(
            data={"FID": fid, "IS": mean_is, "SD-IS": std_is, "step": step}, index=[0]
        )

        self.full_dataset_metrics.to_csv("full_dataset_metrics.csv", index=False)
        self.session_specific_metrics.to_csv(
            "session_specific_metrics.csv", index=False
        )
        self.label_specific_metrics.to_csv("label_specific_metrics.csv", index=False)

    def init_is_fid(self):
        is_model = get_is_model(self.feature)
        fid_model = get_fid_model(self.feature)

        fid = AdaptedFrechetInceptionDistance(
            fid_model,
            dummy_input=th.rand_like(self.data[0:2]),
            reset_real_features=False,
        )
        fid.update(self.data, True)
        inception = InceptionScore(is_model)

        return fid, inception

    def reset(self):
        self.fid.reset()
        self.inception_score.reset()
        self.session_specific_metrics = pd.DataFrame([])
        self.label_specific_metrics = pd.DataFrame([])
        self.full_dataset_metrics = pd.DataFrame([])

    def compute_peak_amplitude_latency_distance(
        self, evoked_real: mne.Evoked, evoked_sampled: mne.Evoked, channel: str
    ) -> tuple:
        """Compute peak amplitude and latency distance between two evokeds for the channel with the highest peak

        Args:
            evoked_real (mne.Evoked): real averaged epochs
            evoked_sampled (mne.Evoked): sampled averaged epochs

        Returns:
            tuple: Peak amplitude in uV and latency distance in ms
        """
        evoked_real_data = evoked_real.pick(channel).get_data(units="uV")[0]
        evoked_sampled_data = evoked_sampled.pick(channel).get_data(units="uV")[0]

        real_peak_amplitude = np.max(evoked_real_data)
        sampled_peak_amplitude = np.max(evoked_sampled_data)

        real_peak_latency = np.argmax(np.abs(evoked_real_data))
        sampled_peak_latency = np.argmax(np.abs(evoked_sampled_data))

        peak_amplitude_distance = np.abs(real_peak_amplitude - sampled_peak_amplitude)
        peak_latency_distance = (
            np.abs(real_peak_latency - sampled_peak_latency) / evoked_real.info["sfreq"]
        )

        return peak_amplitude_distance, peak_latency_distance

    def calculate_eeg_metrics(
        self, real_epo: mne.EpochsArray, sampled_epo: mne.EpochsArray, channel: str
    ):
        """
        Calculate EEG metrics.

        Args:
            real_epo (mne.EpochsArray): Epochs data for real EEG.
            sampled_epo (mne.EpochsArray): Epochs data for sampled EEG.

        Returns:
            dict: EEG metrics.
        """
        metrics = {}

        real_epo_data = real_epo.copy().get_data(units="uV")
        sampled_epo_data = sampled_epo.copy().get_data(units="uV")

        std_real_data = real_epo.copy().standard_error().get_data(units="uV")
        std_sampled_data = sampled_epo.copy().standard_error().get_data(units="uV")

        metrics["PAD"], metrics["PLD"] = self.compute_peak_amplitude_latency_distance(
            real_epo.copy().average(), sampled_epo.copy().average(), channel
        )

        std_manhattan_distance = []

        metrics["SWD"] = np.mean(
            sliced_wasserstein_distance(
                real_epo_data.reshape(real_epo_data.shape[0], -1),
                sampled_epo_data.reshape(sampled_epo_data.shape[0], -1),
                n_projections=10000,
            )
        )

        # loop over channels
        for i in range(real_epo_data.shape[1]):
            std_manhattan_distance.append(
                manhattan_distances(
                    std_real_data[i].reshape(1, -1), std_sampled_data[i].reshape(1, -1)
                )
            )

        metrics["SD-MD"] = np.mean(std_manhattan_distance)

        return metrics

    def add_eeg_metrics(
        self,
        epo_real: Optional[mne.EpochsArray],
        epo_sampled: Optional[mne.EpochsArray],
        metrics_df: pd.DataFrame,
        channel: str = "Cz",
        sampling_combination: dict = None,
    ):
        """
        add EEG metrics.

        Args:
            epo_real (Optional[mne.EpochsArray]): Real EEG data.
            epo_sampled (Optional[mne.EpochsArray]): sampled EEG data.
            sampling_combination (dict, optional): Dictionary of conditionals. Defaults to None.
        """
        if epo_real is None or epo_sampled is None:
            return

        metrics = self.calculate_eeg_metrics(epo_real, epo_sampled, channel)

        return pd.concat(
            [metrics_df, pd.DataFrame([dict(sampling_combination | metrics)])]
        )

    def epochs_to_features(
        self,
        epo: mne.EpochsArray,
        twin: list[list[float]],
    ) -> np.ndarray:
        """
        Extracts features from MNE EpochsArray based on specified time windows.

        Args:
            epoch_data (mne.EpochsArray): The input MNE EpochsArray.
            time_windows (list[list[float]]): List of time windows to crop and extract features.

        Returns:
            np.ndarray: Extracted features as a NumPy array.
        """
        features = []
        n_epochs = len(epo)
        for tw in twin:
            features.append(
                epo.copy()
                .crop(*tw)
                .get_data(units="uV")
                .mean(axis=-1)  # mean accross time window
                .reshape(n_epochs, -1)
            )
        features = np.hstack(features)
        return features

    # https://stackoverflow.com/questions/58065055/floor-and-ceil-with-number-of-decimals
    def my_ceil(self, a, precision=0):
        return np.true_divide(np.ceil(a * 10**precision), 10**precision)

    def my_floor(self, a, precision=0):
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    def lda_score(self, real_epochs: mne.EpochsArray, sampled_epochs: mne.EpochsArray):
        """Calculate synthetic and max accuracy using Linear Discriminant Analysis.

        Args:
            real_epochs (mne.EpochsArray): Epochs data for real events.
            sampled_epochs (mne.EpochsArray): Epochs data for synthetic events.

        Returns:
            tuple: A tuple containing synthetic accuracy and max accuracy.
        """

        begin_interval = self.my_ceil(real_epochs.times[0], 1)
        end_interval = self.my_floor(real_epochs.times[-1], 1)

        time_intervals = np.arange(begin_interval, end_interval + 0.1, 0.1)
        time_intervals = np.vstack([time_intervals[:-1], time_intervals[1:]]).T

        X_sampled = self.epochs_to_features(sampled_epochs, time_intervals)
        y_sampled = sampled_epochs.events[:, 2]

        X_real = self.epochs_to_features(real_epochs, time_intervals)
        y_real = real_epochs.events[:, 2]

        cv = StratifiedKFold(n_splits=5, shuffle=False)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        synthetic_accuracy = []
        for train, _ in cv.split(X_sampled, y_sampled):
            lda.fit(X_sampled[train], y_sampled[train])
            synthetic_accuracy.append(
                balanced_accuracy_score(y_real, lda.predict(X_real))
            )
        synthetic_accuracy = np.mean(synthetic_accuracy)

        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        max_accuracy = cross_val_score(
            lda, X_real, y_real, cv=cv, scoring="balanced_accuracy"
        ).mean()

        lda_scores = {
            "Generated": synthetic_accuracy,
            "Real": max_accuracy,
        }

        return lda_scores

    def add_lda_score(
        self,
        epo_real: mne.EpochsArray,
        epo_sampled: mne.EpochsArray,
        lda_metrics_df: pd.DataFrame,
        conditionals_combination_dict: Optional[dict] = None,
    ):
        """add LDA scores.

        Args:
            epo_real (mne.EpochsArray): Real epochs data.
            epo_sampled (mne.EpochsArray): Synthetic epochs data.
            conditionals_combination_dict (dict, optional): Dictionary for conditionals combination.
                Defaults to None.
        """
        lda_scores = self.lda_score(epo_real, epo_sampled)

        return pd.concat(
            [
                lda_metrics_df,
                pd.DataFrame([dict(conditionals_combination_dict | lda_scores)]),
            ]
        )

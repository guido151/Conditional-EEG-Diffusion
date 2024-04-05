# Authors: Guido Klein <guido.klein@ru.nl>
#
# License: BSD (3-clause)

from typing import List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from pytorch_lightning import LightningModule
from metrics import Metrics
from .plot_util import Plotting
from sklearn.preprocessing import LabelEncoder


class ProcessConditionals:
    def __init__(
        self,
        split_condition: Optional[str] = None,
        conditionals_combinations: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.split_condition = split_condition
        self.conditionals_combinations = conditionals_combinations

    def process_conditions(self):
        """
        Process the conditionals.

        Returns:
            split_condition_combinations (pd.DataFrame or None): The split condition combinations.
            conditionals_combinations (pd.DataFrame or None): The conditionals combinations.
        """
        # Check if split_condition is a string
        if isinstance(self.split_condition, str):
            (
                split_condition_combinations,
                conditionals_combinations,
            ) = self._process_with_split_condition()
            print(split_condition_combinations)
            print(conditionals_combinations)
            return split_condition_combinations, conditionals_combinations
        else:
            return None, None

    def _process_with_split_condition(self):
        """
        Process conditionals when split_condition is a string.

        Returns:
            split_condition_combinations (pd.DataFrame): The split condition combinations.
            conditionals_combinations (pd.DataFrame or None): The conditionals combinations.
        """
        # Check if conditionals_combinations columns match the split_condition
        if all(self.conditionals_combinations.columns == self.split_condition):
            return self.conditionals_combinations, None
        else:
            return self._filter_conditionals_combinations()

    def _filter_conditionals_combinations(self):
        """
        Filter conditionals_combinations based on split_condition.

        Returns:
            split_condition_combinations (pd.DataFrame): The split condition combinations.
            conditionals_combinations (pd.DataFrame): The conditionals combinations.
        """

        # Filter conditionals_combinations based on split_condition
        split_condition_combinations = (
            self.conditionals_combinations[[self.split_condition]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Exclude split_condition column from conditionals_combinations
        conditionals_combinations = (
            self.conditionals_combinations.drop(self.split_condition, axis=1)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return split_condition_combinations, conditionals_combinations


class EpochHelperFunctions:
    def __init__(
        self,
        y_df: pd.DataFrame,
        mne_info: mne.Info,
        split_condition: Optional[str] = None,
        reverse_mapping: Optional[dict] = None,
    ):
        super().__init__()
        self.y_df = y_df
        self.mne_info = mne_info
        self.split_condition = split_condition
        self.reverse_mapping = reverse_mapping

    def create_epochs_array(
        self,
        dataset: th.Tensor,
        y_df: Optional[pd.DataFrame] = None,
    ) -> mne.EpochsArray:
        """Create mne.EpochsArray from dataset of shape (samples, channels, time)

        Args:
            dataset (th.Tensor): EEG data of shape (samples, channels, time)
            y_df (pd.DataFrame):

        Returns:
            mne.EpochsArray: EpochsArray of the dataset, with events based on the comparison condition
        """

        assert isinstance(
            dataset, (np.ndarray, th.Tensor)
        ), "dataset must be of type np.ndarray or th.Tensor"

        # Convert to numpy and convert the data to volts
        dataset = dataset.cpu().numpy() / 1e6
        n_samples = dataset.shape[0]

        # If there are no labels, return epochs without events
        if not isinstance(y_df, pd.DataFrame):
            return mne.EpochsArray(dataset, self.mne_info, verbose=False)

        # Create events (given that the data is already in epochs)
        events = np.column_stack(
            (
                np.arange(0, n_samples, 1),
                np.zeros(n_samples, dtype=int),
                y_df[self.split_condition],
            )
        )
        event_dict = self._create_event_dict(y_df)

        # TODO: tmin should be set through hparams if there is a tmin
        return mne.EpochsArray(
            dataset, self.mne_info, events=events, event_id=event_dict, verbose=False
        )

    def _create_event_dict(self, y_df: pd.DataFrame) -> dict:
        """Create event dictionary for the split condition

        Args:
            y_df (pd.DataFrame): Dataframe with the labels for each condition

        Returns:
            dict: The event dictionary that is used in the mne.EpochsArray
        """
        event_dict = {}

        unique_labels = (
            y_df[self.split_condition].drop_duplicates().reset_index(drop=True)
        )
        for unique_label in unique_labels:
            if isinstance(
                self.reverse_mapping[self.split_condition][unique_label], str
            ):
                event_dict[self.reverse_mapping[self.split_condition][unique_label]] = (
                    unique_label
                )
            else:
                event_name = f"{self.split_condition}_{self.reverse_mapping[self.split_condition][unique_label]}"
                event_dict[event_name] = unique_label

        return event_dict

    def generate_y(
        self,
        combination: Optional[dict],
        n_samples: int,
        device,
    ) -> Tuple[th.Tensor, pd.DataFrame]:
        """
        Create conditionals for the current combination

        Args:
            combination (dict): Combination of conditionals with each one label
            n_samples (int): Number of samples
            device: Device on which the model is running

        Returns:
            Tuple[th.Tensor, pd.DataFrame]: The conditionals and the labels DataFrame
        """
        if not isinstance(combination, dict):
            return None, None

        # Initialize an empty list for conditionals
        y = []

        # Initialize an empty DataFrame for labels
        y_df = pd.DataFrame()

        # Iterate through the combination items
        for key, value in combination.items():
            # Create a column in the DataFrame with the current key and value
            y_df[key] = [value] * n_samples

            # Create a conditional tensor and move it to the specified device
            y.append(
                th.full(size=(n_samples,), fill_value=value, dtype=th.long).to(device)
            )

        # Stack the conditional tensors to create the final tensor
        y = th.stack(y).T

        return y, y_df

    def encode_conditions(self, combination: dict) -> dict:
        """Encode the combination using the reverse mapping"""
        encoded_combination = {}
        for key in combination.keys():
            le = LabelEncoder()
            le.classes_ = self.reverse_mapping[key]
            encoded_combination[key] = le.transform([combination[key]])[0]
        return encoded_combination

    def add_epochs(
        self,
        epochs_array: mne.EpochsArray,
        name: str,
        epochs_dict: dict,
    ):
        if epochs_array is not None:
            epochs_dict[f"{name}"] = epochs_array
            return epochs_dict
        else:
            return epochs_dict

    def get_real_data(
        self, dataset: th.Tensor, combination: Optional[dict] = None
    ) -> Union[mne.EpochsArray, None]:
        """Select data that fits the current combination.

        Args:
            combination (dict): Current combination of conditionals.

        Returns:
            Union[mne.EpochsArray, None]: The selected epochs and labels,
            or None if there is no data that fits the combination.
        """

        # Clone the dataset to avoid modifying the original
        data = th.clone(dataset)

        # Create a copy of y_df to avoid modifying the original
        y_df = self.y_df.copy()

        # If no combination is provided, there is no need to filter the data
        if not isinstance(combination, dict):
            return self.create_epochs_array(data)

        combination = self.encode_conditions(combination)

        # Generate boolean masks for each key-value pair in the combination
        bool_mask = [y_df[key] == value for key, value in combination.items()]

        # Combine boolean masks using logical AND to get the final mask
        bool_mask = th.tensor([all(tup) for tup in zip(*bool_mask)])

        # Check if there is any data that fits the combination
        if th.any(bool_mask):
            # Apply the boolean mask to filter the data
            data = data[bool_mask, :, :]

            # Mark the selected data in the y_df DataFrame
            y_df["selected"] = bool_mask

            # Drop rows with unselected data and the 'selected' column
            y_df = y_df[y_df["selected"]].drop("selected", axis=1)

            # Create epochs array with the filtered data
            real_epochs = self.create_epochs_array(
                dataset=data,
                y_df=y_df,
            )

            return real_epochs
        else:
            # Return None if there is no data that fits the combination
            return None

    def split_epochs_by_events(self, epochs_dict: dict) -> dict:
        """
        Splits epochs arrays based on events and updates the names.

        Args:
            epochs_dict (dict): Dictionary of epochs arrays, keys are the corresponding names.

        Returns:
            dict: Dictionary in which the epochs are split by events.
        """
        # Collect all unique events across all epochs arrays
        all_events = set()
        for epochs_array in epochs_dict.values():
            if isinstance(epochs_array.event_id, dict):
                # Ignore the default event_id (str(1))
                if list(epochs_array.event_id.keys())[0] != str(1):
                    all_events.update(epochs_array.event_id.keys())

        # Sort the events for consistency
        sorted_events = sorted(all_events)

        # If there are no events, keep the original dict
        if not sorted_events:
            return epochs_dict

        new_epochs_dict = {}

        # Iterate over sorted events and original data to create a new dict
        for event in sorted_events:
            for name, epochs_array in epochs_dict.items():
                if (
                    isinstance(epochs_array.event_id, dict)
                    and event in epochs_array.event_id
                ):
                    new_epochs_dict[f"{name} {event}"] = epochs_array[event]

        return new_epochs_dict


class Sampling:
    def __init__(
        self,
        dataset: th.Tensor,
        sampling_fn,
        mne_info: mne.Info,
        metrics: Metrics,
        plotting: Plotting,
        conditionals_combinations: Optional[pd.DataFrame] = None,
        split_condition: Optional[str] = None,
        reverse_mapping: Optional[dict] = None,
        y_df: Optional[pd.DataFrame] = None,
    ):
        self.sampling_fn = sampling_fn
        self.dataset = dataset

        self.metrics = metrics
        self.plotting = plotting

        (
            self.split_condition_combinations,
            self.conditionals_combinations,
        ) = ProcessConditionals(
            split_condition=split_condition,
            conditionals_combinations=conditionals_combinations,
        ).process_conditions()
        self.split_condition = split_condition
        self.reverse_mapping = reverse_mapping
        self.y_df = y_df
        self.epoch_helper = EpochHelperFunctions(
            y_df, mne_info, split_condition, reverse_mapping
        )

    def sampling_logging(
        self,
        pl_module: LightningModule,
        sampling_shape: tuple,
        sampling_model: nn.Module,
        step: Optional[int] = None,
    ):
        """
        Perform sampling and logging based on specified conditions.

        args:
            pl_module (LightningModule): The PyTorch Lightning module.
            sampling_shape (tuple): The shape of the sampling.
            sampling_model (nn.Module): The sampling model.
        """

        if step is None:
            step = pl_module.global_step

        # Check if split condition is a string
        if isinstance(self.split_condition, str):
            # Check if conditionals_combinations is a DataFrame
            if isinstance(self.conditionals_combinations, pd.DataFrame):
                # Handle conditionals with split
                self._handle_conditionals_with_split(
                    pl_module,
                    sampling_shape,
                    sampling_model,
                )
            else:
                # Handle split conditional
                self._handle_split_conditional(
                    pl_module,
                    sampling_shape,
                    sampling_model,
                    None,
                )
        else:
            # Check if conditionals_combinations is a DataFrame
            if isinstance(self.conditionals_combinations, pd.DataFrame):
                # Handle conditionals without split
                self._handle_conditionals_without_split(
                    pl_module,
                    sampling_shape,
                    sampling_model,
                )
            else:
                # Handle unconditioned
                self._handle_unconditioned(
                    pl_module,
                    sampling_shape,
                    sampling_model,
                )

        self.metrics.log_dfs(step)
        self.metrics.reset()

    def _handle_unconditioned(
        self,
        pl_module: LightningModule,
        sampling_shape: tuple,
        sampling_model: nn.Module,
    ):
        """
        Generates the epochs arrays and names for the unconditioned case.

        Args:
            pl_module (LightningModule): The Lightning module.
            sampling_shape (tuple): The shape of the sampled epochs.
            sampling_model (nn.Module): The model to use for sampling.
            return_sampled_epochs_arrays (bool): Whether to return the sampled epochs arrays. Defaults to False.
        """
        epochs_dict = {}

        real_epochs = self.epoch_helper.get_real_data(self.dataset, None)
        epochs_dict = self.epoch_helper.add_epochs(real_epochs, "real", epochs_dict)

        # TODO: add if-statement
        if real_epochs is not None:
            sampling_shape = real_epochs.get_data().shape

        sampled_epochs = self.sampling_loop(
            pl_module=pl_module,
            sampling_shape=sampling_shape,
            sampling_model=sampling_model,
            combination=None,
        )

        self.metrics.update_non_session_specific(real_epochs, sampled_epochs, None)

        epochs_dict = self.epoch_helper.add_epochs(
            sampled_epochs, "generated", epochs_dict
        )
        self.plotting.get_plots(epochs_dict)

    def _handle_conditionals_without_split(
        self,
        pl_module: LightningModule,
        sampling_shape: tuple,
        sampling_model: nn.Module,
    ):
        """
        Generates the epochs arrays and names for the conditionals without split condition.

        Args:
            pl_module (LightningModule): The Lightning module.
            sampling_shape (tuple): The shape of the sampled epochs.
            sampling_model (nn.Module): The model to use for sampling.

        """

        for combination in self.conditionals_combinations.itertuples(index=False):
            sampling_dict = combination._asdict()

            epochs_dict = {}

            real_epochs = self.epoch_helper.get_real_data(self.dataset, sampling_dict)

            epochs_dict = self.epoch_helper.add_epochs(real_epochs, "real", epochs_dict)

            # TODO: add if-statement
            if real_epochs is not None:
                sampling_shape = real_epochs.get_data().shape

            sampled_epochs = self.sampling_loop(
                pl_module=pl_module,
                sampling_shape=sampling_shape,
                sampling_model=sampling_model,
                combination=sampling_dict,
            )

            epochs_dict = self.epoch_helper.add_epochs(
                sampled_epochs, "generated", epochs_dict
            )

            self.metrics.update_non_session_specific(
                real_epochs, sampled_epochs, sampling_dict
            )
            self.plotting.get_plots(epochs_dict)

    def _handle_conditionals_with_split(
        self,
        pl_module: LightningModule,
        sampling_shape: tuple,
        sampling_model: nn.Module,
    ):
        """
        Generates the epochs arrays and names for the conditionals without split condition.

        Args:
            pl_module (LightningModule): The Lightning module.
            sampling_shape (tuple): The shape of the sampled epochs.
            sampling_model (nn.Module): The model to use for sampling.

        """

        for conditionals_combination in self.conditionals_combinations.itertuples(
            index=False
        ):
            conditionals_combination_dict = conditionals_combination._asdict()

            self._handle_split_conditional(
                pl_module,
                sampling_shape,
                sampling_model,
                conditionals_combination_dict,
            )

    def _handle_split_conditional(
        self,
        pl_module: LightningModule,
        sampling_shape: tuple,
        sampling_model: nn.Module,
        conditionals_combination_dict: Optional[dict] = None,
    ):
        """Generate and handle split conditional epochs.

        Args:
            pl_module (LightningModule): The Lightning module.
            sampling_shape (tuple): The shape of the sampled epochs.
            sampling_model (nn.Module): The model to use for sampling.
            conditionals_combination_dict (Optional[dict], optional):
                The conditionals combination. Defaults to None.

        """
        sampled_epochs_arrays = []
        real_epochs_arrays = []

        for split_condition_combination in self.split_condition_combinations.itertuples(
            index=False
        ):
            sampling_dict = (
                {
                    **conditionals_combination_dict,
                    **split_condition_combination._asdict(),
                }
                if isinstance(conditionals_combination_dict, dict)
                else split_condition_combination._asdict()
            )

            real_epochs = self.epoch_helper.get_real_data(self.dataset, sampling_dict)

            # TODO: fix how shape is slected
            if real_epochs:
                sampling_shape = (
                    real_epochs.get_data().shape
                    if real_epochs.get_data().shape[0] != 0
                    else sampling_shape
                )
                real_epochs_arrays.append(real_epochs)

            sampled_epochs = self.sampling_loop(
                pl_module=pl_module,
                sampling_shape=sampling_shape,
                sampling_model=sampling_model,
                combination=sampling_dict,
            )
            sampled_epochs_arrays.append(sampled_epochs)

            self.metrics.update_non_session_specific(
                real_epochs, sampled_epochs, sampling_dict
            )

        real_epo = mne.concatenate_epochs(real_epochs_arrays)
        sampled_epo = mne.concatenate_epochs(sampled_epochs_arrays)

        self.metrics.update_session_specific(
            real_epo, sampled_epo, conditionals_combination_dict
        )

        epochs_dict = {}
        epochs_dict = self.epoch_helper.add_epochs(
            sampled_epo, "generated", epochs_dict
        )
        epochs_dict = self.epoch_helper.add_epochs(real_epo, "real", epochs_dict)
        epochs_dict = self.epoch_helper.split_epochs_by_events(epochs_dict)

        self.plotting.get_plots(epochs_dict)

    def sampling_loop(
        self,
        pl_module: LightningModule,
        sampling_shape: tuple,
        sampling_model: nn.Module,
        combination: Optional[dict] = None,
    ) -> list:
        """
        Sampling loop for generating epochs using a trained model.

        Args:
            pl_module (LightningModule): The trained LightningModule.
            sampling_shape (tuple): The shape of the sampled epochs.
            combination (dict, optional): A dictionary containing the combination of conditionals for sampling. Defaults to None.

        Returns:
            sampled_epochs (list): A list of sampled epochs.
        """
        # Ensure that the conditional labels during sampling are in the same order as the ones in the training data

        if isinstance(combination, dict):
            # Sort the combination dictionary based on conditionals in pl_module.hparams.conditionals
            sorted_combination = {
                condition: combination[condition]
                for condition in pl_module.hparams.conditionals
            }
            encoded_sorted_combination = self.epoch_helper.encode_conditions(
                sorted_combination
            )
            # Generate y and y_df
            y, df = self.epoch_helper.generate_y(
                encoded_sorted_combination, sampling_shape[0], pl_module.hparams.device
            )
        else:
            y = None
            df = None

        # Perform sampling
        sampled_data, _ = self.sampling_fn(sampling_model, sampling_shape, None, y)

        # Create sampled epochs
        sampled_epochs = self.epoch_helper.create_epochs_array(
            sampled_data,
            df,
        )

        return sampled_epochs


def compute_rebalanced_weights(df: pd.DataFrame) -> list:
    """
    Compute rebalanced weights based on the label distribution in the provided DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing a 'label' column.

    Returns:
        list: List of rebalanced weights.
    """
    # Calculate the counts of each label
    value_counts = df["label"].value_counts()

    # Calculate ratios of each label count to the total count
    ratios = 1 - (value_counts / len(df))

    # Find the minimum ratio and calculate a multiplier to rebalance the weights
    multiply = 1 / ratios.min()

    # Compute rebalanced weights and convert to a list
    rebalanced_weights = (ratios * multiply).tolist()

    return rebalanced_weights

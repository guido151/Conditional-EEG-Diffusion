# Authors: Guido Klein <guido.klein@ru.nl>
#
# License: BSD (3-clause)

import os
from glob import glob
from typing import Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
import torch as th
from moabb import paradigms
from moabb.datasets.base import BaseDataset
from moabb.datasets.utils import dataset_list
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from data_utils.data_utils import compute_rebalanced_weights


class EEGDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        user_conditions: list,
        sfreq: int,
        fmin: float,
        fmax: float,
        reject_treshold: float,
        cache: bool,
        path: str,
        use_presaved: bool,
        save: bool,
    ):
        """Initialize the class.

        Args:
            dataset (str): Name of the dataset.
            user_conditions (List): List of user-defined conditions.
            sfreq (int): Sampling frequency.
            fmin (float): Minimum frequency for bandpass filter.
            fmax (float): Maximum frequency for bandpass filter.
            reject_treshold (float): Maximum peak-to-peak for rejection.
            cache (bool): Whether to cache the data.
            path (str): Path to the data.
            use_presaved (bool): Whether to use pre-saved data.
        """
        self.dataset = dataset
        self.user_conditions = user_conditions
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax
        self.reject_treshold = reject_treshold
        self.cache = cache
        self.path = path
        self.use_presaved = use_presaved

        self.reverse_mapping = {}
        self.mne_info = {}
        self.condition_combinations = None
        self.rebalance_weights = []

        if use_presaved:
            epo, df = self.load_data()
        else:
            epo, df = self.get_data()
            if save:
                self.save_data(epo, df)
        (
            self.X,
            self.y_df,
        ) = self.apply_transforms(epo, df)

    def get_data(self):
        pass

    def load_data(self):
        epo = mne.read_epochs(self.path + f"/{self.dataset}-epo.fif")
        df = pd.read_csv(self.path + f"/{self.dataset}-df.csv")
        return epo, df

    def save_data(self, epo: mne.EpochsArray, df: pd.DataFrame):
        # useful if dataset cannot be loaded directly

        # epo_old = mne.read_epochs(self.path + f"/{self.dataset}-epo.fif")
        # df = pd.read_csv(self.path + f"/{self.dataset}-df.csv")

        # epo = mne.concatenate_epochs([epo_old, epo])
        # df = pd.concat([df_old, df])

        epo.save(self.path + f"/{self.dataset}-epo.fif", overwrite=True)
        df.to_csv(self.path + f"/{self.dataset}-df.csv", index=False)

    def apply_transforms(self, epo: mne.EpochsArray, df: pd.DataFrame) -> tuple:
        self.set_mne_info(epo)

        # makes it possible to preload dataset, but still select subjects, sessions, and runs
        if self.use_presaved:
            user_conditions_dict = self.create_user_conditions_dict()
            epo = epo.pick(self.ch_names)
            epo, df = self.apply_conditions(user_conditions_dict, epo, df, False)

        if self.reject_treshold > 0:
            epo, df = self.reject_epo(epo, df)

        if self.user_conditions:
            df = df[df.columns.intersection(self.user_conditions)]
            y_df = self.encode_conditions(df)
            self.condition_combinations = self.get_unique_combinations(df)
        else:
            y_df = None
            self.conditionals_combinations = None

        X = epo.get_data(units="uV")

        X = th.tensor(X, dtype=th.float32)

        return X, y_df

    def set_load_config(self):
        """Set and load configuration for the class.

        This function sets the configuration based on cache and path settings.
        If cache is True, it creates the specified path if it doesn't exist.
        It then sets the MNE_DATASETS_LEE2019_ERP_PATH configuration.

        Raises:
            AssertionError: If cache is True, path must be specified.

        """
        assert (
            self.cache and self.path
        ) or not self.cache, "If cache is True, path must be specified"

        if self.cache:
            self._create_path_if_not_exists()

        mne.set_config(f"MNE_DATASETS_{self.dataset.upper()}_PATH", self.path)

    def _create_path_if_not_exists(self):
        """Create the specified path if it doesn't exist."""
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def set_mne_info(self, epo: mne.EpochsArray):
        """Set MNE info and montage.

        Args:
            epo (mne.EpochsArray): The MNE EpochsArray object.
        """
        # Set MNE info
        self.mne_info = epo.info

        # Set montage
        montage = epo.get_montage()
        if montage is not None:
            self.mne_info.set_montage(montage)
        else:
            self.mne_info.set_montage("standard_1020")

    def reject_epo(
        self, epo: mne.EpochsArray, dataset_conditions: pd.DataFrame
    ) -> tuple:
        """Reject epochs.

        Args:
            epo (mne.EpochsArray): The MNE EpochsArray object containing the epochs.
            dataset_conditions (pd.Dataframe): dataframe with all the conditions

        Returns:
            mne.EpochsArray: The MNE EpochsArray object containing the epochs after rejection.
        """
        epo.metadata = dataset_conditions

        treshold = {"eeg": self.reject_treshold}
        epo.drop_bad(reject=treshold, verbose=False)
        epo.plot_drop_log()

        dataset_conditions = epo.metadata

        return epo, dataset_conditions

    def encode_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode the labels in each conditional and construct the reverse mapping"""
        le = LabelEncoder()

        encoded_df = pd.DataFrame([], columns=df.columns)
        for col in df.columns:
            encoded_df[col] = le.fit_transform(df[col])
            self.reverse_mapping[col] = le.classes_
        return encoded_df

    def standardize_meta(self, meta: pd.DataFrame) -> pd.DataFrame:
        """Standardize the meta data

        Args:
            meta (pd.DataFrame): Meta data

        Returns:
            pd.DataFrame: Standardized meta data
        """
        return meta

    def condition_indices(
        self,
        df: pd.DataFrame,
        conditions_df: pd.DataFrame,
        remove: bool,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Indices of the dataframe that match the conditions

        Args:
            df (pd.DataFrame): The dataframe
            conditions_df (pd.DataFrame): The conditions
            remove (bool): Whether to remove or keep the conditions.

        Returns:
            list[int]: The indices of the dataframe that match the conditions
        """
        # Merge dataframes based on conditions
        merged_df = pd.merge(
            df,
            conditions_df,
            indicator=True,
            how="left",
            on=list(conditions_df.columns),
        )

        if remove:
            # Return indices that do not match the conditionals
            return (
                merged_df[merged_df["_merge"] == "left_only"]
                .drop(columns=["_merge"])
                .index
            )

        else:
            # Return indices that match the conditionals
            return (
                merged_df[merged_df["_merge"] == "both"].drop(columns=["_merge"]).index
            )

    def preprocess_conditions(
        self,
        conditions: dict,
    ) -> pd.DataFrame:
        """
        Convert the conditions dictionary to a DataFrame

        Args:
            conditions (dict): The conditions.

        Returns:
            pd.DataFrame: Processed conditions.
        """
        # Convert the conditions dictionary to a DataFrame
        conditions_df = pd.DataFrame(
            {key: [value] for key, value in conditions.items()}
        )

        # Explode the DataFrame to handle lists in the columns
        for key in conditions_df.columns:
            conditions_df = conditions_df.explode(key).reset_index(drop=True)

        # TODO: update to newest version of moabb
        if "session" in conditions_df.columns:
            conditions_df["session"] = conditions_df["session"].apply(
                lambda x: f"session_{x}"
            )

        return conditions_df

    def apply_conditions(
        self,
        condition_df: dict,
        epo: mne.EpochsArray,
        df: pd.DataFrame,
        remove: bool,
    ):
        condition_df = self.preprocess_conditions(condition_df)
        idx = self.condition_indices(df, condition_df, remove)
        return epo[idx], df.iloc[idx]

    def create_user_conditions_dict():
        """Create dictionary that allows selecting relevant data

        Returns:
            dict: dictionary
        """
        pass

    def get_unique_combinations(self, df: pd.DataFrame):
        """Get unique combinations of a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with unique combinations.
        """
        # Drop duplicate rows and reset the index
        conditionals_combinations = df.drop_duplicates().reset_index(drop=True)
        return conditionals_combinations

    def __len__(self) -> int:
        """Number of samples in the dataset

        Returns:
            int: Number of samples in the dataset
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Union[Tuple[th.Tensor, th.Tensor], th.Tensor]:
        """
        Get the sample and label at the given index.

        Args:
            idx (int): The index at which to get the sample and label.

        Returns:
            Union[Tuple[th.Tensor, th.Tensor], th.Tensor]: The sample and label at the given index, or just the sample if label is not available.
        """
        sample = self.X[idx]
        if isinstance(self.y_df, pd.DataFrame):
            label = th.tensor(list(self.y_df.iloc[idx]))
            return sample, label
        else:
            return sample


class MOABBDataset(EEGDataset):
    def __init__(
        self,
        dataset: str,
        sfreq: int,
        fmin: float,
        fmax: float,
        subjects: list,
        ch_names: list,
        reject_treshold: float,
        cache: bool,
        path: str,
        use_presaved: bool,
        save: bool,
        user_conditions: list,
        **kwargs,
    ):
        self.dataset_params = kwargs

        self.subjects = subjects
        self.ch_names = ch_names

        super().__init__(
            dataset,
            user_conditions,
            sfreq,
            fmin,
            fmax,
            reject_treshold,
            cache,
            path,
            use_presaved,
            save,
        )

    def get_data(self):
        self.set_load_config()
        dataset_dict = {d.__name__: d for d in dataset_list}
        dataset = dataset_dict[self.dataset](**self.dataset_params)
        paradigm = getattr(
            paradigms,
            (
                "MotorImagery"
                if "imagery" in dataset.paradigm
                else f"{dataset.paradigm}".capitalize()
            ),
        )(fmin=self.fmin, fmax=self.fmax, resample=self.sfreq, channels=self.ch_names)

        epo, labels, meta = paradigm.get_data(
            dataset,
            self.subjects,
            return_epochs=True,
        )

        df = meta
        df["label"] = labels

        return epo, df

    def create_user_conditions_dict(
        self,
    ):
        pass


class Lee2019Dataset(MOABBDataset):
    def __init__(
        self,
        dataset: str,
        sfreq: int,
        fmin: float,
        fmax: float,
        subjects: list,
        ch_names: list,
        reject_treshold: float,
        cache: bool,
        path: str,
        use_presaved: bool,
        save: bool,
        user_conditions: list,
        sessions: list = [1, 2],
        train_run: bool = True,
        test_run: bool = True,
    ):
        kwargs = {
            "sessions": sessions,
            "train_run": train_run,
            "test_run": test_run,
        }

        self.train_run = train_run
        self.test_run = test_run
        self.sessions = sessions
        self.subjects = subjects

        super().__init__(
            dataset,
            sfreq,
            fmin,
            fmax,
            subjects,
            ch_names,
            reject_treshold,
            cache,
            path,
            use_presaved,
            save,
            user_conditions,
            **kwargs,
        )

    # TODO: update to new version of MOABB
    def create_user_conditions_dict(
        self,
    ):
        run = []
        if self.train_run:
            run.append("train")
        if self.test_run:
            run.append("test")
        return {
            "session": self.sessions,
            "run": run,
            "subject": self.subjects,
        }

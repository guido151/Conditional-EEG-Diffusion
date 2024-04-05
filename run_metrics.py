# Authors: Guido Klein <guido.klein@ru.nl>
# 
# License: BSD (3-clause)

from metrics import AdaptedFrechetInceptionDistance, get_fid_model, Metrics
from data_utils.data_utils import EpochHelperFunctions
from configs.Lee2019_ERP_CFG import DATASET_CFG
from data_utils.dataset import Lee2019Dataset
from braindecode.models import EEGNetv4
from skorch import NeuralNet
from huggingface_hub import hf_hub_download
import torch as th
import torch.nn as nn
import numpy as np

dataset = Lee2019Dataset(**DATASET_CFG)

EHF = EpochHelperFunctions(
    dataset.y_df, dataset.mne_info, "label", dataset.reverse_mapping
)
epochs_array = EHF.create_epochs_array(dataset.X, dataset.y_df)

path_params = hf_hub_download(
    repo_id="guido151/EEGNetv4",
    filename="EEGNetv4_Lee2019_ERP/params.pt",
)
path_optimizer = hf_hub_download(
    repo_id="guido151/EEGNetv4",
    filename="EEGNetv4_Lee2019_ERP/optimizer.pt",
)
path_history = hf_hub_download(
    repo_id="guido151/EEGNetv4",
    filename="EEGNetv4_Lee2019_ERP/history.json",
)
path_criterion = hf_hub_download(
    repo_id="guido151/EEGNetv4",
    filename="EEGNetv4_Lee2019_ERP/criterion.pt",
)

model = EEGNetv4(
    n_chans=19,
    n_outputs=2,
    n_times=128,
)

net = NeuralNet(
    model,
    criterion=nn.CrossEntropyLoss(weight=th.tensor([1, 1])),
)
net.initialize()
net.load_params(
    path_params,
    path_optimizer,
    path_criterion,
    path_history,
)

between_session_metrics = Metrics(
    dataset.X,
    net.module,
    "O1",
    DATASET_CFG["fmin"],
    DATASET_CFG["fmax"],
)


def create_two_random_halves(epochs_array):
    full_list = np.arange(len(epochs_array))
    half_1 = np.random.choice(full_list, size=int(len(epochs_array) / 2), replace=False)
    half_2 = [x for x in full_list if x not in half_1]
    epochs_array_1 = epochs_array.copy()[half_1]
    epochs_array_2 = epochs_array.copy()[half_2]
    return epochs_array_1, epochs_array_2


for subject in dataset.y_df["subject"].unique():
    for label in dataset.y_df["label"].unique():
        selected_epochs_array = epochs_array.copy()
        session1_indices = list(
            dataset.y_df[
                (dataset.y_df["subject"] == subject)
                & (dataset.y_df["label"] == label)
                & (dataset.y_df["session"] == 0)
            ].index
        )
        session2_indices = list(
            dataset.y_df[
                (dataset.y_df["subject"] == subject)
                & (dataset.y_df["label"] == label)
                & (dataset.y_df["session"] == 1)
            ].index
        )
        between_session_metrics.update_non_session_specific(
            selected_epochs_array.copy()[session1_indices],
            selected_epochs_array.copy()[session2_indices],
            {"subject": subject, "session": 1, "label": label},
        )

between_session_metrics.session_specific_metrics.to_csv(
    "between_session_session_specific_metrics.csv", index=False
)
between_session_metrics.label_specific_metrics.to_csv(
    "between_session_non_session_specific_metrics.csv", index=False
)
between_session_metrics.full_dataset_metrics.to_csv(
    "between_session_full_dataset_metrics.csv", index=False
)


# Compute FID baselines

fid_model = get_fid_model(model)
fid = AdaptedFrechetInceptionDistance(
    fid_model,
    th.tensor(epochs_array[0:2].copy().get_data(units="uV"), dtype=th.float32),
)
selected_epochs_array = epochs_array.copy()
session1_indices = list(
    dataset.y_df[(dataset.y_df["subject"].isin(np.arange(0, 26)))].index
)
session2_indices = list(
    dataset.y_df[dataset.y_df["subject"].isin(np.arange(26, 53))].index
)
ea1 = selected_epochs_array.copy()[session1_indices]
ea2 = selected_epochs_array.copy()[session2_indices]
fid.update(th.tensor(ea1.get_data(units="uV"), dtype=th.float32), True)
fid.update(th.tensor(ea2.get_data(units="uV"), dtype=th.float32), False)

print(f"FID between first half and second half of subjects: {fid.compute()}")
fid.reset()


n_bootstraps = 20
random_halves_fid = []
for i in range(n_bootstraps):
    selected_epochs_array = epochs_array.copy()
    ea1, ea2 = create_two_random_halves(selected_epochs_array)
    fid.update(th.tensor(ea1.get_data(units="uV"), dtype=th.float32), True)
    fid.update(th.tensor(ea2.get_data(units="uV"), dtype=th.float32), False)
    random_halves_fid.append(fid.compute())
    fid.reset()
print(
    f"Random halves FID mean: {np.mean(random_halves_fid)} std: {np.std(random_halves_fid)}"
)

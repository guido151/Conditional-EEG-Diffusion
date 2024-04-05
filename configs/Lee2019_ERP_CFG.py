import numpy as np
import torch as th

from models.EEGWave import EEGWave
from sde import VESDE, VPSDE, subVPSDE

RUN_CFG = dict(
    project_name="Article",  # name of the project
    run_name="Lee 2019 ERP",  # name of the run
    steps=950001,  # number of training steps
    batch_size=128,  # batch size
    val_split=0.1,  # percentage of the dataset to use for validation
    device=th.device("cuda") if th.cuda.is_available() else th.device("cpu"),
    gradient_clip_val=1.0,
    checkpoint_freq=50000,  # checkpoint frequency in number of steps (based on train steps)
)

DATASET_CFG = dict(
    dataset="Lee2019_ERP",
    sfreq=128,  # resampling frequency
    fmin=1.0,  # band pass filter minimum frequency
    fmax=40.0,  # band pass filter maximum frequency
    subjects=(
        np.arange(1, 17).tolist() + np.arange(18, 55).tolist()
    ),  # selected subjects
    ch_names=[
        "Fp1",
        "Fp2",
        "F7",
        "F8",
        "F3",
        "F4",
        "Fz",
        "T7",
        "T8",
        "C3",
        "C4",
        "Cz",
        "P7",
        "P8",
        "P3",
        "P4",
        "Pz",
        "O1",
        "O2",
    ],  # selected channels from ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'STI 014']
    reject_treshold=150e-6,  # peak-to-peak difference before rejection (if set to 0, no rejection is applie)
    cache=True,  # if True, cache the complete dataset
    path=r"datasets/",  # Folder of the dataset
    use_presaved=True,  # Load preprocessed epochs
    save=False,  # Save preprocessed epochs (redo necessary if you change anything above rejection treshold)
    user_conditions=[
        "label",
        "subject",
        "session",
    ],  # the variable(s) on which the model is conditioned select from ["session", "subject", "label", "run"], None means no conditional
    train_run=True,
    test_run=True,
    sessions=[1, 2],  # selected sessions
)

FRAMEWORK_CFG = dict(
    model=EEGWave,  # specify the model
    sde=VPSDE,
    optimizer_type="adam",
    lr=2e-4,  # learning rate
    lr_betas=(
        0.99,
        0.999,
    ),  # coefficients used for computing running averages of gradient and its square
    lr_scheduler=None,
    optim_eps=1e-8,
    weight_decay=0,
    reduce_mean=True,  # If true, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous=True,  # If true, the model is defined to take continuous time steps. Otherwise the model is defined to take discrete time steps.
    likelihood_weighting=False,  # if true, use different type of likelihood weighting in the loss (https://arxiv.org/abs/2101.09258)
)

SDE_CFG = dict(
    sde__beta_min=0.1,  # amount of noise to add at timestep 0
    sde__beta_max=20,  # amount of noise to add at final timestep
    sde__N=1000,  # number of timesteps
)


# settings for the model
MODEL_CFG = dict(
    model__eeg_channels=len(DATASET_CFG["ch_names"]),
    model__inp_channels=1,
    model__out_channels=len(DATASET_CFG["ch_names"]),
    model__res_layers=40,  # number of residual blocks
    model__res_channels=128,  # number of channels in each residual block
    model__skip_channels=128,
    model__step_emb_in_dim=128,
    model__step_emb_hidden_dim=512,
    model__step_emb_out_dim=512,
    model__dilation_cycle=7,  # maximum size of the dilation
    model__res_kernel_size=3,  # kernel size of the residual block
    model__max_timesteps=SDE_CFG["sde__N"],
    model__embedding_type="nn.embedding_summed" # Cannot be changed
)

SAMPLING_CFG = dict(
    snapshot_sampling=False,  # if true, sample from the model during training
    calculate_metrics=False,  # calculating FID + IS score with model that has lowest validation loss
    sampling_freq=50000,  # number of epochs between sample generation
    sampling_shape=(
        500,
        len(DATASET_CFG["ch_names"]),
        DATASET_CFG["sfreq"],
    ),  # fallback shape of the generated samples during training if there is no real data to base the shape on
    split_condition="label",  # condition for comparison, make sure it is in DATASET_CFG["conditionals"]
    plot_channels=[
        "O1",
        "Pz",
        "Cz",
    ],  # channels that are plotted in plots that do not have all channels (to reduce clutter)
    sampler_name="pc",
    predictor_name="euler_maruyama",
    corrector_name="langevin",
    noise_removal=True,  # if true, do not add noise during last timestep of sampling
    snr=0.16,  # signal-to-noise ratio for configuring correctors.
    n_steps_each=1,  # number of corrector steps per predictor update.
    probability_flow=False,  # If true, solve the reverse-time probability flow ODE when running the predictor.
    sampling_eps=1e-5,  # minumum timestep for sampling, resolves numerical issues (default: VESDE: 1e-5; (sub)VPSDE: 1e-3 during training and 1e-5 during sampling)
    EMA=True,  # if true, use Exponential Moving Average during sampling
    EMA_beta=0.9999,  # percentage of the new EMA model that is the old EMA model (default VE: 0.999 and default VP: 0.9999)
)

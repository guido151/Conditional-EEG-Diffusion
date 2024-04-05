# Authors: Guido Klein <guido.klein@ru.nl>
#
# License: BSD (3-clause)
#
# Based on: https://github.com/tszab/eegwave-dpm/tree/master

from typing import Optional, Tuple

import pandas as pd
import torch as th
import torch.nn as nn
from .utils import DiffusionStepEmbedding


class EEGWaveResidualLayer(nn.Module):
    def __init__(
        self,
        res_kernel_size: int,
        res_channels: int,
        skip_channels: int,
        step_emb_out_dim: int,
        dilation: int,
        norm_value: th.Tensor = th.sqrt(th.tensor(2.0)),
        embedding_type: Optional[str] = None,
        conditionals_combinations: Optional[pd.DataFrame] = None,
    ):
        """Residual layer used in EEGWave

        Args:
            res_kernel_size (int): Size of the residual kernel
            res_channels (int): Number of channels in the residual layer
            skip_channels (int): Number of channels in the skip-connection
            step_emb_out_dim (int): Final dimensions of the timestep embedding
            dilation (int): Dilation size for the bi-directional dilation convolution
            norm_value (float): Normalizes the residual connection
            conditionals_combinations (Optional[pd.DataFrame], optional): Conditional combinations. Defaults to None.
        """
        super(EEGWaveResidualLayer, self).__init__()

        self.skip_channels = skip_channels
        self.res_channels = res_channels
        self.register_buffer("norm_value", norm_value.clone().detach())
        self.step_emb_out_dim = step_emb_out_dim
        self.conditionals_combinations = conditionals_combinations

        if isinstance(self.conditionals_combinations, pd.DataFrame):
            (
                self.y_embedding_layers,
                self.step_embedding_layer,
            ) = self._create_res_embedding_layers()
        else:
            self.step_embedding_layer = nn.Linear(step_emb_out_dim, res_channels)

        self.bidilconv = nn.Conv2d(
            res_channels,
            2 * res_channels,
            (1, res_kernel_size),
            padding="same",
            dilation=(1, dilation),
        )
        self.pointwise_conv = nn.Conv2d(
            res_channels, skip_channels + res_channels, (1, 1)
        )

    def _create_res_embedding_layers(self) -> Tuple[nn.ModuleList, nn.Linear]:
        """
        Create embedding layers for conditional combinations.

        Returns:
            Tuple[nn.ModuleList, nn.Linear]: Tuple of embedding layers.
        """
        # Check if conditionals_combinations is a DataFrame
        if isinstance(self.conditionals_combinations, pd.DataFrame):

            step_embedding_layer = nn.Linear(self.step_emb_out_dim, self.res_channels)

            # Create linear layers for each condition
            y_embedding_layers = [
                nn.Linear(self.step_emb_out_dim, self.res_channels)
                for _ in self.conditionals_combinations.columns
            ]

            # Convert y_embedding_layers to nn.ModuleList
            y_embedding_layers = nn.ModuleList(y_embedding_layers)

        # Return the tuple of embedding layers
        return y_embedding_layers, step_embedding_layer

    def forward(self, x: th.Tensor, t: th.Tensor, y: Optional[list] = None):
        """Residual layer forward pass

        Args:
            x (th.Tensor): x to residual layer. Shape (batch_size, eeg_channels, 1, length_eeg)
            t (th.Tensor): timestep embedding. Shape (batch_size, step_emb_out_dim)

        Returns:
            Tuple[th.Tensor, th.Tensor]: skip connection and residual connection. Shape (batch_size, skip_channels + res_channels, length_eeg)
        """
        assert (
            x.ndim == 4
        ), f"x should have 4 dimensions (batch_size, eeg_channels, 1, length_eeg), but got {x.ndim}"
        assert (
            t.ndim == 2
        ), f"t should have 2 dimensions (batch_size, step_emb_out_dim), but got {t.ndim}"
        assert x.shape[2] == 1, f"x should have 1 dimension (1), but got {x.shape[2]}"
        assert (
            t.shape[0] == x.shape[0]
        ), f"timestep embedding should have the same batch size as the x, but got {t.shape[0]} and {x.shape[0]}"
        assert (
            t.shape[1] == self.step_emb_out_dim
        ), f"timestep embedding should have the same dimension as output of timestep embedding, but got {t.shape[1]} and {self.step_emb_out_dim}"

        # x: (batch_size, res_channels, 1, length_eeg)

        # t: (batch_size, step_emb_out_dim)

        t = self.step_embedding_layer(t)
        # t: (batch_size, res_channels)

        # add label embeddings to the timestep embedding
        if isinstance(y, list):
            for i in range(len(y)):
                # y[i]: (batch_size, step_emb_out_dim)
                output = self.y_embedding_layers[i](y[i])
                # output: (batch_size, res_channels)
                t += output

        t = t.reshape(*t.shape, 1, 1)
        # t: (batch_size, res_channels, 1, 1)

        xt = x + t
        # xt: (batch_size, res_channels, 1, length_eeg)

        xt = self.bidilconv(xt)
        # xt: (batch_size, 2*res_channels, 1, length_eeg)

        gate, filt = th.chunk(xt, 2, dim=1)
        # gate: (batch_size, res_channels, 1, length_eeg)
        # filt: (batch_size, res_channels, 1, length_eeg)

        xt = th.sigmoid(gate) * th.tanh(filt)
        # xt: (batch_size, res_channels, 1, length_eeg)

        xt = self.pointwise_conv(xt)
        # xt: (batch_size, 2*res_channels, 1, length_eeg)

        skip, res = th.split(xt, [self.skip_channels, self.res_channels], dim=1)
        # skip: (batch_size, skip_channels, 1, length_eeg)
        # res: (batch_size, res_channels, 1, length_eeg)

        res = (x + res) / self.norm_value
        # res: (batch_size, res_channels, 1, length_eeg)

        assert (
            res.shape == x.shape
        ), f"residual connection and x of residual layer should have the same shape, but got {res.shape} and {x.shape}"
        assert (
            skip.shape == x.shape
        ), f"skip connection and x of residual layer should have the same shape, but got {skip.shape} and {x.shape}"
        return skip, res


class EEGWave(nn.Module):
    def __init__(
        self,
        eeg_channels: int = 64,
        inp_channels: int = 128,
        out_channels: int = 64,
        res_layers: int = 40,
        res_channels: int = 128,
        skip_channels: int = 128,
        step_emb_in_dim: int = 128,
        step_emb_hidden_dim: int = 512,
        step_emb_out_dim: int = 512,
        dilation_cycle: int = 7,
        res_kernel_size: int = 3,
        max_timesteps: int = 1000,
        embedding_type: Optional[str] = None,
        conditionals_combinations: Optional[pd.DataFrame] = None,
    ):
        """Network that learns to predict the noise

        Args:
            eeg_channels (int, optional): Number of EEG data channels. Defaults to 64.
            inp_channels (int, optional): Number of x channels
            res_layers (int, optional): Number of residual layers. Defaults to 40.
            res_channels (int, optional): Number of channels in the residual layers. Defaults to 128.
            skip_channels (int, optional): Number of channels in the skip connections. Defaults to 128.
            step_emb_in_dim (int, optional): Size of the original timestep embedding. Defaults to 128.
            step_emb_hidden_dim (int, optional): Size of the timestep embedding after one linear layer. Defaults to 512.
            step_emb_out_dim (int, optional): Final size of the timestep embedding after second linear layer. Defaults to 512.
            dilation_cycle (int, optional): Maximum size of the dilation in the residual layers 2^(i%dilation_cycle), with i being current layer. Defaults to 7.
            res_kernel_size (int, optional): Size of the kernel in the residual layers. Defaults to 3.
            max_timesteps (int, optional): Maximum number of timesteps used during the forward diffusion process. Defaults to 1000.
            conditionals_combinations (Optional[pd.DataFrame], optional): List of conditioning labels. Defaults to None.
        """
        super(EEGWave, self).__init__()
        self.out_norm_value = th.sqrt(th.tensor(res_layers))
        self.res_channels = res_channels
        self.max_timesteps = max_timesteps
        self.eeg_channels = eeg_channels
        self.step_emb_in_dim = step_emb_in_dim
        self.step_emb_hidden_dim = step_emb_hidden_dim
        self.step_emb_out_dim = step_emb_out_dim
        self.conditionals_combinations = conditionals_combinations


        self.x_conv = nn.Sequential(
            nn.Conv2d(inp_channels, res_channels, (eeg_channels, 1)),
            nn.ReLU(),
        )
        self.step_embedding = DiffusionStepEmbedding(
            max_timesteps, step_emb_in_dim, step_emb_hidden_dim, step_emb_out_dim
        )

        if isinstance(self.conditionals_combinations, pd.DataFrame):
            self.embedding_layers = self._create_embedding_layers()

        self.res_layers_list = nn.ModuleList(
            [
                EEGWaveResidualLayer(
                    res_kernel_size=res_kernel_size,
                    res_channels=res_channels,
                    skip_channels=skip_channels,
                    step_emb_out_dim=step_emb_out_dim,
                    dilation=2 ** (i % dilation_cycle),
                    conditionals_combinations=self.conditionals_combinations,
                )
                for i in range(res_layers)
            ]
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(skip_channels, out_channels, (1, 1)),
        )

    def _create_embedding_layers(self):
        """
        Create embedding layers based on the specified embedding type.

        Returns:
        embedding_layers (nn.ModuleList): List of embedding layers.
        """
        embedding_layers = []

        for condition in self.conditionals_combinations.columns:
            embedding_layers.append(self._create_nn_embedding_layer(condition))

        return nn.ModuleList(embedding_layers)

    def _create_nn_embedding_layer(self, condition):
        """
        Create an embedding layer using nn.embedding for the given condition.

        Args:
        condition (str): The column name for the condition.

        Returns:
        nn.Sequential: Sequential neural network module.
        """
        return nn.Sequential(
            nn.Embedding(
                self.conditionals_combinations[condition].nunique(),
                self.step_emb_in_dim,
            ),
            nn.Linear(self.step_emb_in_dim, self.step_emb_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.step_emb_hidden_dim, self.step_emb_out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        x: th.Tensor,
        t: th.Tensor,
        y: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """compute forward pass

        Args:
            x (th.Tensor): x data. Shape: (batch_size, eeg_channels, length_eeg)
            t (th.Tensor): timestep of each sample. Shape: (batch_size,)
            y (Optional[th.Tensor], optional): y data. Shape: (batch_size, conditionals).

        Returns:
            th.Tensor: score with same shap as x. Shape: (batch_size, out_channels, length_eeg)
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)

        assert (
            x.ndim == 3
        ), f"x should have 3 dimensions (batch_size, eeg_channels, eeg_length), but got {x.ndim}"
        assert (
            x.shape[1] == self.eeg_channels
        ), f"x and eeg_channels should have the same number of eeg channels, but got {x.shape[1]} and {self.eeg_channels}"
        assert t.ndim == 1, f"t should have 1 dimension (batch_size), but got {t.ndim}"
        assert (
            t.shape[0] == x.shape[0]
        ), f"t and x should have the same batch size, but got {t.shape[0]} and {x.shape[0]}"
        assert t.dtype in [
            th.long,
            th.float,
        ], "t must be a long or float tensor, encoding the timestep)"

        # x: (batch_size, eeg_channels, eeg_length)

        x = x.unsqueeze(1)
        # x: (batch_size, 1, eeg_channels, eeg_length)

        x = self.x_conv(x)
        # x: (batch_size, res_channels, 1, eeg_length)

        # t: (batch_size, )
        t = self.step_embedding(t)
        # t: (batch_size, step_emb_out_dim)

        # add label embedding to the timestep embedding if labels are provided
        if isinstance(y, th.Tensor):
            y_embedding = []
            for i, condition in enumerate(self.conditionals_combinations.columns):
                # y_shape[i]: (batch_size, )
                y_embedding.append(self.embedding_layers[i](y[:, i]))
                # y_embedding[i]: (batch_size, step_emb_out_dim)
            y = y_embedding

        # run residual layers
        skip, res = self.res_layers_list[0](x, t, y)
        if len(self.res_layers_list) > 1:
            for res_layer in self.res_layers_list[1:]:
                skip_add, res = res_layer(res, t, y)
                skip = skip + skip_add

        # normalization
        skip = skip / self.out_norm_value

        # skip: (batch_size, skip_channels, 1, length_eeg)
        out_data = self.output_conv(skip)
        # out_data: (batch_size, out_channels, 1, length_eeg)

        out_data = out_data.squeeze(2)
        # out_data: (batch_size, out_channels, length_eeg)

        return out_data

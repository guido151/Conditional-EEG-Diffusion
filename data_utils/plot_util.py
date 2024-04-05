# Authors: Guido Klein <guido.klein@ru.nl>
#
# License: BSD (3-clause)

import math
from typing import List, Optional, Union
import itertools
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
import pandas as pd
import torch as th
from mne.viz.utils import plot_sensors


class Plotting:
    def __init__(
        self,
        channels: list,
    ):
        self.channels = channels

    def evoked_plot(self, epochs_dict: dict) -> plt.figure:
        epoch_df = pd.DataFrame([])
        for name, epochs_array in epochs_dict.items():
            current_df = epochs_array.copy().pick(self.channels).to_data_frame()
            current_df["Type"] = name.split(" ")[0]
            epoch_df = pd.concat([epoch_df, current_df], ignore_index=True)
        id_vars = ["time", "condition", "epoch", "Type"]
        melted_df = pd.melt(epoch_df, id_vars=id_vars, var_name="channel")
        melted_df = melted_df.rename(
            columns={
                "time": "Time (s)",
                "value": "\u03BCV",
                "condition": "Class",
            }
        )
        melted_df["Class"] = melted_df["Class"].map(
            {"Target": "target", "NonTarget": "non-target"}
        )
        melted_df["Type"] = melted_df["Type"] + ": " + melted_df["Class"]
        print(melted_df["Type"].unique())

        sns.set_context("paper", font_scale=3)

        linewidth = 7

        g = sns.relplot(
            melted_df,
            x="Time (s)",
            y="\u03BCV",
            col="channel",
            hue="Type",
            hue_order=[
                "real: target",
                "real: non-target",
                "generated: target",
                "generated: non-target",
            ],
            col_order=["Cz", "Pz", "O1"],
            style="Type",
            style_order=[
                "real: target",
                "real: non-target",
                "generated: target",
                "generated: non-target",
            ],
            dashes={
                "real: target": (1, 0),
                "real: non-target": (1, 0),
                "generated: target": (1.5, 1),
                "generated: non-target": (1.5, 1),
            },
            palette={
                "real: target": "#f77189",
                "real: non-target": "#bb9832",
                "generated: target": "#f79c71",
                "generated: non-target": "#9abb32",
            },
            kind="line",
            col_wrap=1,
            errorbar="sd",
            legend="full",
            height=3,
            aspect=3,
            linewidth=linewidth,
            err_kws={"alpha": 0.15},
        )

        g.figure.set_size_inches(11.69, 13.69)

        for ax in g.axes.flatten():
            label = ax.title.get_text().split("=")[-1].strip()
            ax.set_ylabel(f"Aplitude at {label} (\u03BCV)")
            ax.set_title("")
        plt.tight_layout()

        for legobj in g.legend.legendHandles:
            legobj.set_linewidth(linewidth)

        sns.move_legend(g, "center", bbox_to_anchor=(1.18, 0.85))

        isax = g.axes.flat[-1].inset_axes(bounds=(0.8, 1.3, 1, 1))

        easycap_montage = mne.channels.make_standard_montage("easycap-M1")

        epochs = list(epochs_dict.values())[0].copy()
        epochs.set_montage(easycap_montage)
        plot_sensors(
            epochs.pick(self.channels).info,
            show_names=False,
            axes=isax,
        )

        g.savefig(
            "figures/subject52_session1_timecourse.pdf",
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=None,
            transparent=False,
            bbox_inches="tight",
            pad_inches=0,
            metadata=None,
        )

        plt.close(g.fig)

        return g.fig

    def SCM_plot(self, epochs_dict: dict) -> plt.figure:
        """Create a plot of the covariance matrices

        Args:
            epochs_dict (dict): A dictionary of epochs arrays, with keys corresponding to names of the epochs.

        Returns:
            plt.figure: A plot of the covariance matrices
        """

        def create_SCM(epochs_dict: dict) -> pd.DataFrame:
            """Create a dataframe with the covariance matrices of the averaged epochs

            Args:
                epochs_dict (dict): A dictionary of epochs arrays, with keys corresponding to names of the epochs.

            Returns:
                pd.DataFrame: A dataframe with the covariance matrices of the averaged epochs
            """
            labels = next(iter(epochs_dict.values())).ch_names

            matrices = np.array(
                [
                    np.cov(
                        epochs_array.average().get_data(units="uV").T, rowvar=False
                    ).flatten()
                    for epochs_array in epochs_dict.values()
                ]
            ).flatten()

            combination = list(itertools.product(epochs_dict.keys(), labels, labels))

            SCM_df = pd.DataFrame(
                combination, columns=["name", "channels1", "channels2"]
            )

            SCM_df["covariance"] = matrices

            SCM_df["data type"] = SCM_df["name"].apply(lambda x: x.split(" ")[-2])
            SCM_df["label"] = SCM_df["name"].apply(lambda x: x.split(" ")[-1])
            SCM_df.drop(columns=["name"], inplace=True)

            return SCM_df

        def create_SCM_plot(SCM: pd.DataFrame) -> plt.figure:
            """Create a plot of the covariance matrices

            Args:
                SCM (pd.DataFrame): A dataframe with the covariance matrices

            Returns:
                plt.figure: A plot of the covariance matrices
            """
            with sns.plotting_context():
                g = sns.FacetGrid(
                    SCM,
                    col="data type",
                    row="label",
                    col_wrap=None,
                    aspect=1,
                    sharex=True,
                    sharey=True,
                    height=6,
                )

            cbar_positions = []
            for i, row_name in enumerate(g.row_names):
                subplot_height = g.axes[i][0].get_position().height
                cbar_y = g.axes[i][0].get_position().y0
                cbar_positions.append([0.92, cbar_y, 0.02, subplot_height])

            cbar_axes = [g.fig.add_axes(pos) for pos in cbar_positions]

            for i, row_name in enumerate(g.row_names):
                for j, col_name in enumerate(g.col_names):
                    current_ax = g.facet_axis(i, j)
                    data_pv = SCM[
                        (SCM["data type"] == col_name) & (SCM["label"] == row_name)
                    ].pivot(index="channels1", columns="channels2", values="covariance")
                    sns.heatmap(
                        data_pv, ax=current_ax, cmap="rocket", cbar_ax=cbar_axes[i]
                    )
                    current_ax.set_xlabel(col_name)
                    current_ax.set_ylabel(row_name)

            g.set_titles(template="")
            g.fig.subplots_adjust(right=0.9)
            return g.fig

        SCM_df = create_SCM(epochs_dict)
        figure = create_SCM_plot(SCM_df)

        plt.savefig(
            "figures/SCM.pdf",
            dpi=None,
            facecolor="w",
            edgecolor="w",
            orientation="portrait",
            format=None,
            transparent=False,
            bbox_inches="tight",
            pad_inches=0,
            metadata=None,
        )

        plt.close(figure)
        return figure

    def get_plots(
        self,
        epochs_dict: dict,
    ):
        """
        Generate a list of plots based on the given epochs arrays.

        Args:
            epochs_dict (dict): A dictionary of epochs arrays, with keys corresponding to names of the epochs.

        Returns:
            list: A list of plot figures generated from the epochs arrays.
        """
        self.SCM_plot(epochs_dict)
        self.evoked_plot(epochs_dict)

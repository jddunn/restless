import os

import numpy as np

from pandas import DataFrame

import matplotlib.pyplot as plt
import seaborn as sbn
from datetime import datetime

# Path to save visualization output images
DEFAULT_SCREENSHOTS_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "screenshots", "model_results"
    )
)


class StatsVisUtils:
    def __init__(self):
        return

    def visualize_correlation_diagonal_matrix(
        self,
        corr,
        plot_title: str = None,
        save_image: bool = False,
        output_fp: str = None,
        show: bool = True,
    ) -> None:
        """
        Graphs df correlation with diagonal matrix. Optionally
        saves output to image.
        https://seaborn.pydata.org/examples/many_pairwise_correlations.html
        """
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=np.float32))
        # mask = np.zeros_like(corr, dtype=np.bool)
        # mask[np.triu_indices_from(mask)] = True
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a custom diverging colormap
        cmap = sbn.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        hmap = sbn.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=0.3,
            center=0,
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            # cmap="RdBu_r",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )
        if plot_title:
            plt.title = plot_title
        if show:
            plt.show()
        if save_image:
            if not output_fp:
                now = datetime.now()
                output_fp = os.path.abspath(
                    os.path.join(DEFAULT_SCREENSHOTS_PATH, str(now))
                )
            plt.savefig(output_fp + ".png", dpi=300)
        return

    def visualize_correlation(
        self,
        corr,
        plot_title: str = None,
        save_image: bool = False,
        output_fp: str = None,
        show: bool = True,
    ) -> None:
        """
        Graphs df correlation with heatmap. Optionally saves
        output to image.
        """
        hmap = sbn.heatmap(
            corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap="RdBu_r",
            annot=True,
            linewidth=0.5,
        )
        if plot_title:
            plt.title = plot_title
        if show:
            plt.show()
        if save_image:
            if not output_fp:
                now = datetime.now()
                output_fp = os.path.abspath(
                    os.path.join(DEFAULT_SCREENSHOTS_PATH, str(now))
                )
            plt.savefig(output_fp + ".png", dpi=300)
        return

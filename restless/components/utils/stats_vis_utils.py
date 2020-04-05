import os

from pandas import DataFrame

import matplotlib as pt
from seaborn import heatmap

# Path to save visualization output images
DEFAULT_SCREENSHOTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "screenshots")
)


class StatsVisUtils:
    def __init__(self):
        return

    def visualize_correlation(
        self,
        corr,
        save_img: bool = False,
        output_fp: str = DEFAULT_SCREENSHOTS_PATH,
        show: bool = True,
    ) -> None:
        """
        Graphs df correlation with heatmap. Optionally saves
        output to image.
        """
        heatmap(
            pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap="RdBu_r",
            annot=True,
            linewidth=0.5,
        )
        if show:
            plt.show()
        plt.savefig("output" + key + ".png", dpi=300)
        return

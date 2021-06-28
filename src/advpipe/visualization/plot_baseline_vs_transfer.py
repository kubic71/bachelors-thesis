from __future__ import annotations
import pandas as pd
import numpy as np
import seaborn as sns
from os import path
import matplotlib.pyplot as plt
from PIL import Image

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Sequence
    from typing_extensions import Literal


def _load_dataframe(results_dir: str) -> pd.DataFrame:
    return pd.read_csv(results_dir + "/blackbox_query_results.txt", delimiter=" ", names=["img_fn", "label", "loss"])

# plot loss-distributions for baseline original images vs. adversarial examples constructed from them
def plot_baseline_vs_transfer(baseline_results_dir: str, transfer_results_dir: str,
                   plot_fn: Optional[str],
                   title: str, clip_to_same_size=True) -> None:
    if plot_fn is None:
        plot_fn = transfer_results_dir + "/baseline_vs_transfer.png"

    dataframes: pd.Dataframe = []
    baseline_df = _load_dataframe(baseline_results_dir)
    baseline_df['type'] = "original images"
    transfer_df = _load_dataframe(transfer_results_dir)
    transfer_df['type'] = "transferred adversarial examples"

    if clip_to_same_size:
        s = min(len(baseline_df), len(transfer_df))
        baseline_df = baseline_df[:s]
        transfer_df = transfer_df[:s]

    dist_plot = sns.displot(data=pd.concat([baseline_df, transfer_df]), x="loss", hue="type", kind="hist")
    dist_plot.set(xlabel="Loss", ylabel="number of images", title=f"{title} (n_images={s})")
    dist_plot.savefig(plot_fn)
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_results_dir", required=True, type=str)
    parser.add_argument("--transfer_results_dir", required=False, type=str)
    parser.add_argument("-o", "--output_plot_filename", required=False, type=str, default=None)
    parser.add_argument("--title", required=False, type=str, default="Performance of target model on original images vs. transferred adversarial examples")
    args = parser.parse_args()

    plot_baseline_vs_transfer(args.baseline_results_dir, args.transfer_results_dir, args.output_plot_filename, args.title)
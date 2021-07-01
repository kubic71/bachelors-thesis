from __future__ import annotations
import pandas as pd
import numpy as np
import seaborn as sns
from os import path
import pathlib
import matplotlib.pyplot as plt
from PIL import Image

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Sequence
    from typing_extensions import Literal


def _get_success_rate(df: pd.DataFrame) -> float:
    return len(df[df['loss'] < 0]) / len(df)


def _load_dataframe(results_dir: str) -> pd.DataFrame:
    return pd.read_csv(results_dir + "/blackbox_query_results.txt", delimiter="\t", names=["img_fn", "label", "loss"])


# plot loss-distributions for baseline original images vs. adversarial examples constructed from them
def plot_baseline_vs_transfer(baseline_results_dir: str,
                              transfer_results_dir: str,
                              plot_fn: Optional[str],
                              title: str,
                              clip_to_same_size: bool = False,
                              show_plot: bool = False) -> None:
    print(f"Generating plot: {title}")

    if plot_fn is None:
        plot_fn = transfer_results_dir + "/baseline_vs_transfer.png"

    pathlib.Path(path.dirname(plot_fn)).mkdir(parents=True, exist_ok=True)
    dataframes: pd.Dataframe = []
    baseline_df = _load_dataframe(baseline_results_dir)
    baseline_df['type'] = "original images"
    transfer_df = _load_dataframe(transfer_results_dir)
    transfer_df['type'] = "transferred adversarial examples"

    if clip_to_same_size:
        s = min(len(baseline_df), len(transfer_df))
        baseline_df = baseline_df[:s]
        transfer_df = transfer_df[:s]

    sns.set_theme(style="darkgrid")
    # dist_plot = sns.displot(data=pd.concat([baseline_df, transfer_df]), x="loss", hue="type", kind="hist", stat="density")
    f, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[2, 1]))
    f.suptitle(f"{title}\n(#baselines_images={len(baseline_df)}, #transfer_images={len(transfer_df)})")

    colors = sns.color_palette()
    dist_plot = sns.histplot(data=baseline_df, x="loss", stat="density", ax=axs[0], color=colors[0])
    dist_plot = sns.histplot(data=transfer_df, x="loss", stat="density", ax=axs[0], color=colors[1])
    axs[0].set(xlabel="Loss", ylabel="density", title="Target loss values distribution")
    axs[0].legend(labels=["Original images", "Transferred adv examples"])

    barplot = sns.barplot(
        x=["baseline foolrate", "transfer attack foolrate"],
        y=[_get_success_rate(baseline_df), _get_success_rate(transfer_df)],
        ax=axs[1],
    )
    barplot.set_ylim(0, 1)

    f.tight_layout()

    print(f"Saving exported plot to: {plot_fn}\n")
    f.savefig(plot_fn)

    if show_plot:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_results_dir", required=True, type=str)
    parser.add_argument("--transfer_results_dir", required=False, type=str)
    parser.add_argument("-o", "--plot_fn", required=False, type=str, default=None)
    parser.add_argument("--title",
                        required=False,
                        type=str,
                        default="Performance of target model on original images vs. transferred adversarial examples")
    parser.add_argument("--show_plot", action="store_true", default=False)

    args = parser.parse_args()

    plot_baseline_vs_transfer(args.baseline_results_dir, args.transfer_results_dir, args.plot_fn,
                              title=args.title, show_plot = args.show_plot)

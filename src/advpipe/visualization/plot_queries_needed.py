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



def plot_n_queries(queries_log_file_path: Sequence[str],
                   plot_fn: Optional[str],
                   title: str = "Number of queries needed to craft successful adversarial example",
                   type: Literal["histogram", "quantiles"] = "histogram",
                   query_limit: int = 200,
                   x_adv_limit: int = 50000
                   ) -> None:
    if plot_fn is None:
        if len(queries_log_file_path) != 1:
            raise Exception("Plot filename is None, but multiple log_files were given")
        plot_fn = path.dirname(queries_log_file_path[0]) + "/n_queries_plot.png"

    dataframes: pd.Dataframe = []
    for queries_fn in queries_log_file_path:
        df = pd.read_csv(queries_fn, delimiter=" ", names=["img_fn", "label", "n_queries"])

        # take the parent directory as the experiment name
        exp_name = path.basename(path.dirname(queries_fn))
        df['exp_name'] = exp_name

        dataframes.append(df[:x_adv_limit])

    data = pd.concat(dataframes)

    # replace the 'inf' of the unsuccessful adversarial examples by the query_limit of the attack
    # otherwise seaborn refuses to plot the cumulative distribution graph
    data.loc[data['n_queries'] == float('inf'), 'n_queries'] = query_limit

    dist_plot = sns.displot(data=data, x="n_queries", hue="exp_name", kind="ecdf")
    dist_plot.set(xlabel="Number of queries", ylabel="Percent of successful adversarial examples")
    dist_plot.savefig(plot_fn)
    plt.show()

    # queries = df["n_queries"]
    # quantiles = queries.quantile(np.array(range(100)) / 100)
    # plot_fn = os.path.dirname(queries_log_file_path) + "/n_queries.png"
    # title = os.path.basename(queries_log_file_path) if title is None else title
    # plt.plot(list(quantiles), quantiles.index)

    # plt.title(title)
    # plt.xlabel()
    # plt.ylabel()

    # plt.savefig(plot_fn)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--queries_log_files", nargs="+", required=True, type=str)
    parser.add_argument("-o", "--output_plot_filename", required=False, type=str, default=None)
    parser.add_argument("--title", required=False, type=str, default="Number of queries needed to craft successful adversarial example")
    parser.add_argument("--query_limit", required=False, type=int, default=200)
    parser.add_argument("--x_adv_limit", required=False, type=int, default=50000)
    args = parser.parse_args()


    if args.output_plot_filename is None and len(args.queries_log_files) == 1:
        args.output_plot_filename = path.dirname(args.queries_log_files[0]) + "/n_queries_plot.png"

    plot_n_queries(args.queries_log_files, args.output_plot_filename, args.title, query_limit=args.query_limit, x_adv_limit=args.x_adv_limit)
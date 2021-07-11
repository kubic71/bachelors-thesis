from __future__ import annotations

import pandas as pd
import yaml
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Dict, Optional, Sequence, Iterator


def get_config(res_dir: str) -> Optional[Dict]:
    for filename in os.listdir(res_dir):
        if filename.endswith(".yaml"):
            return yaml.load(filename) # type: ignore
    return None


def get_all_csvs(res_dir: str) -> Iterator[str]:
    for sub_dir, _, _ in os.walk(res_dir):
        for filename in os.listdir(sub_dir):
            if filename.endswith(".csv"):
                # print("getallcsv")
                # sub_dir = sub_dir[len(res_dir) + 1:]
                # print(res_dir, sub_dir, filename)
                yield os.path.join(sub_dir, filename)


def gather_results(
        results_dir: str,
        experiment_filter: Callable[[Dict], bool] = lambda config: True,
        dataframe_transform: Callable[[pd.DataFrame, Dict], pd.DataFrame] = lambda data, config: data) -> pd.Dataframe:
    """Recursively scans through results directories and constructs one pandas dataframe containing all experiments
        args:
            - results_dir: str
                - path to the root results directory

            - experiment_filter: Callable[[Dict], bool]
                 - filters experiments based on their configs parsed as Dict

            - dataframe_transform: Callable[[pd.DataFrame, Dict], pd.DataFrame]
                - postprocessing for each experiment dataframe. It can also use Dict experiment config
    """


    data_fs = []
    for res_dir, _, _ in os.walk(results_dir):
        config = get_config(res_dir)
        if config is None or not experiment_filter(config):
            continue
    
        print("Experiment dir: ", res_dir)
        # we are in experiment directory

        exp_pds = []

        for csv_file in get_all_csvs(res_dir):
            exp_pds.append(pd.read_csv(csv_file, delimiter="\t"))
        
        exp_pd = dataframe_transform(pd.concat(exp_pds), config)

        # print(exp_pd)

        data_fs.append(exp_pd)

    return pd.concat(data_fs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str)

    args = parser.parse_args()

    gather_results(args.dir)

from __future__ import annotations

import pandas as pd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Callable, Dict


# def list_all_results()


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
    raise NotImplementedError
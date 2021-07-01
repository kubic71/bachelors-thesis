from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import TargetBlackBoxConfig
    from typing import Tuple, Iterator
    import numpy as np


class TargetBlackBox(ABC):
    """
    Generic BlackBox target model
    Abstraction for cloud MLaaS classifiers as well as for local pretrained CNNs
    """

    last_query_result: BlackboxLabels

    def __init__(self, blackbox_config: TargetBlackBoxConfig):
        self.blackbox_config = blackbox_config

    @abstractmethod
    def loss(self, pertubed_image: np.ndarray) -> float:
        """ Override this in Cloud/Local blackbox child class
        # In case of Cloud blackbox, query the API and pass it through the language-model translation mapping layer
        # In case of Local blackbox, pass split the ImageNet categories using imagenet_utils.py
        """
        ...


    # returns BlackBox labels and saves the results to last_query_result variable
    @abstractmethod
    def classify(self, pertubed_image: np.ndarray) -> BlackboxLabels:
        ...


class BlackboxLabels(ABC):

    @abstractmethod
    def get_top_organism(self) -> Tuple[str, float]:
        ...

    @abstractmethod
    def get_top_object(self) -> Tuple[str, float]:
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[str, float]]:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

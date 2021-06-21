from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import TargetBlackBoxConfig
    import numpy as np


class TargetBlackBox(ABC):
    """
    Generic BlackBox target model
    Abstraction for cloud MLaaS classifiers as well as for local pretrained CNNs
    """
    def __init__(self, blackbox_config: TargetBlackBoxConfig):
        self.blackbox_config = blackbox_config

    @abstractmethod
    def loss(self, pertubed_image: np.ndarray) -> float:
        """ Override this in Cloud/Local blackbox child class
        # In case of Cloud blackbox, query the API and pass it through the language-model translation mapping layer
        # In case of Local blackbox, pass split the ImageNet categories using imagenet_utils.py
        """
        ...
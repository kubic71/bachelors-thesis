from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import TargetModelConfig
    from typing import Tuple, Iterator, Sequence
    import numpy as np


class TargetModel(torch.nn.Module):
    """
    Generic target model
    Abstraction for cloud MLaaS classifiers as well as for local pretrained CNNs
    They are all torch.nn.Modules, such that they can be passed to 3-rd party attack libraries without any modifications

    LocalModels are differentiable, CloudModels are not.

    """

    last_query_result: BlackboxLabels
    model_config: TargetModelConfig

    def __init__(self, model_config: TargetModelConfig):
        super(TargetModel, self).__init__() # type: ignore

        self.model_config = model_config 


    def loss(self, pertubed_images: torch.Tensor) -> torch.Tensor:
        """ Override this in Cloud/Local blackbox child class
        # In case of Cloud blackbox, query the API and pass it through the language-model translation mapping layer
        # In case of Local model, pass split the ImageNet categories using imagenet_utils.py
        """
        raise NotImplementedError

    # returns BlackBox labels and saves the results to last_query_result variable
    def classify(self, pertubed_images: torch.Tensor) -> BlackboxLabels:
        raise NotImplementedError


class BlackboxLabels(ABC):

    @abstractmethod
    def get_top_organism_labels(self) -> Sequence[str]:
        ...
    

    @abstractmethod
    def get_top_organism_logits(self) -> torch.Tensor:
        ...

    @abstractmethod
    def get_top_object_labels(self) -> Sequence[str]:
        ...
    
    @abstractmethod
    def get_top_object_logits(self) -> torch.Tensor:
        ...
    
    def __iter__(self) -> Iterator[Tuple[str, float]]:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

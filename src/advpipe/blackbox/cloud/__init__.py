from __future__ import annotations
import torch
from advpipe.blackbox import TargetModel, BlackboxLabels
from advpipe.blackbox.loss import margin_loss
from advpipe.language_model.label_classification import OrganismLabelClassifier
from abc import ABC, abstractmethod
from advpipe.log import logger, CloudDataLogger

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Tuple, Iterator
    from advpipe.config_datamodel import CloudBlackBoxConfig
    import numpy as np


# TODO: CloudLabels should contain entire batch, as LocalLabels do
class CloudLabels(BlackboxLabels):
    _organism_label_classifier = OrganismLabelClassifier()

    all_labels: Sequence[Tuple[str, float]]
    correct: Sequence[Tuple[str, float]]
    adversarial: Sequence[Tuple[str, float]]

    def __init__(self, labels_and_scores: Sequence[Tuple[str, float]]):
        self.all_labels = labels_and_scores

        self.correct = []
        self.adversarial = []
        for label, score in labels_and_scores:
            if self._organism_label_classifier.is_organism_label(label):
                self.correct.append((label, score))
            else:
                self.adversarial.append((label, score))

        # sort by score
        self.correct.sort(key=lambda x: x[1], reverse=True)
        self.adversarial.sort(key=lambda x: x[1], reverse=True)

        for labels in [self.correct, self.adversarial]:
            if labels == []:
                labels.append(("no-label", 0))

    def __iter__(self) -> Iterator[Tuple[str, float]]:
        return iter(self.all_labels)

    def get_top_organism(self) -> Tuple[str, float]:
        return self.correct[0]

    def get_top_object(self) -> Tuple[str, float]:
        return self.adversarial[0]

    def __str__(self) -> str:
        """Convert labels and scores tuple-list to pretty-printable string"""
        return "\n".join(list(map(lambda l_s: l_s[0] + ": " + str(l_s[1]), self.all_labels)))


# TODO: make this into torch.nn.Module
class CloudBlackBox(TargetModel):
    """Pretrained ImageNet model"""
    cloud_data_logger: CloudDataLogger
    last_query_result: CloudLabels

    def __init__(self, blackbox_config: CloudBlackBoxConfig):
        super().__init__(blackbox_config)
        self.cloud_data_logger = blackbox_config.cloud_data_logger

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def classify(self, images: torch.Tensor) -> Sequence[CloudLabels]:
        raise NotImplementedError

    def loss(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

        # cloud_labels = self.classify(images)
        # return [label.get_top_organism()[1] - label.get_top_object()[1] + self.blackbox_config.loss.margin for label in cloud_labels]


from .gvision_blackbox import GVisionBlackBox

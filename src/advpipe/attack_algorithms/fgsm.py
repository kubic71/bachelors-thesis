from __future__ import annotations
from advpipe.attack_algorithms import BlackBoxTransferAlgorithm
from advpipe.blackbox.local import LocalModel, PytorchOrganismClassifier
import numpy as np
from advpipe.log import logger
from advpipe import utils
import torch
import foolbox as fb
import eagerpy as ep
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.blackbox.local import LocalModel
    from typing import Generator, Sequence


class FgsmTransferAlgorithm(BlackBoxTransferAlgorithm):
    epsilon: float

    def __init__(self, surrogate: LocalModel, epsilon: float):
        super().__init__(surrogate)

        org_classifier = PytorchOrganismClassifier(surrogate)
        org_classifier.eval()
        self.fmodel = fb.PyTorchModel(org_classifier, bounds=(0, 1))
        self.attack = fb.attacks.FGSM()
        self.epsilon = epsilon

    def run(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, advs, success = self.attack(self.fmodel, images, labels, epsilons=[self.epsilon])

        return advs[0]

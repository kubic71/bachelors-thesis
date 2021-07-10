from __future__ import annotations
from advpipe.blackbox.local import LocalModel
import foolbox as fb
from advpipe.attack_algorithms import BlackBoxTransferAlgorithm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from advpipe.blackbox.local import LocalModel
    import torch

class FgsmTransferAlgorithm(BlackBoxTransferAlgorithm):
    epsilon: float

    def __init__(self, surrogate: LocalModel, epsilon: float):
        super().__init__(surrogate)

        self.fmodel = fb.PyTorchModel(surrogate, bounds=(0, 1))
        self.attack = fb.attacks.FGSM()
        self.epsilon = epsilon

    def run(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        _, advs, success = self.attack(self.fmodel, images, labels, epsilons=[self.epsilon])

        return advs[0]

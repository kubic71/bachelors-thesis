from __future__ import annotations
from advpipe.attack_algorithms import BlackBoxTransferAlgorithm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.blackbox.local import LocalModel
    import torch


class PassthroughTransferAttackAlgorithm(BlackBoxTransferAlgorithm):
    def __init__(self, surrogate: LocalModel):
        super().__init__(surrogate)

    def run(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return images
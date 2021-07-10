from __future__ import annotations
from advpipe.attack_algorithms import BlackBoxTransferAlgorithm
import numpy as np
from advpipe.log import logger
from advpipe import utils
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from typing import Generator
    from advpipe.blackbox.local import LocalModel


class PassthroughTransferAttackAlgorithm(BlackBoxTransferAlgorithm):
    def __init__(self, surrogate: LocalModel):
        super().__init__(surrogate)

    def run(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return images
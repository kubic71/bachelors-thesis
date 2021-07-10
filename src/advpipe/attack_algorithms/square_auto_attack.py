from __future__ import annotations
from advpipe.attack_algorithms import BlackBoxTransferAlgorithm
from advpipe.blackbox.local import LocalModel
import numpy as np
from advpipe.log import logger
from advpipe import utils
import torch
from autoattack.square import SquareAttack
from advpipe.config_datamodel.attack_algorithm_config import SquareAutoAttackConfig
import eagerpy as ep

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.blackbox.local import LocalModel
    from typing import Generator, Sequence


class SquareAutoAttack(BlackBoxTransferAlgorithm):

    def __init__(self, surrogate: LocalModel, config: SquareAutoAttackConfig):
        super().__init__(surrogate)

        self.config = config

        def predict(x: torch.Tensor) -> torch.Tensor:
            with torch.no_grad(): # type: ignore
                return surrogate(x) # type: ignore

        self.attack = SquareAttack(predict, norm=self.config.metric, n_restarts = self.config.n_restarts, n_queries=self.config.n_iters, eps=self.config.epsilon, loss="margin", device="cuda")

    def run(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.attack.perturb(images, labels) # type: ignore
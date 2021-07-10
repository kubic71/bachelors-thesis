from __future__ import annotations
from abc import ABC, abstractmethod
from advpipe.log import logger
import numpy as np
import eagerpy as ep

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.blackbox.local import LocalModel
    from advpipe.types import TensorTypeVar
    from typing import Generator
    import torch


class Norm:
    _supported_norms = {"l0": ep.norms.l0, "l1": ep.norms.l1, "l2": ep.norms.l2, "linf": ep.norms.linf}

    def __init__(self, norm_order: str):
        assert norm_order in self._supported_norms
        self._name = norm_order
        self._norm = self._supported_norms[norm_order]

    def __call__(self, imgs: TensorTypeVar) -> TensorTypeVar:
        return self._norm(ep.astensor(imgs), axis=(1,2,3)).raw # type: ignore

    @property
    def name(self) -> str:
        return self._name


class BlackBoxIterativeAlgorithm():
    loss_fn: LossCallCounter
    images: torch.Tensor

    def __init__(self, images: torch.Tensor, loss_fn: LossCallCounter):
        self.images = images
        self.loss_fn = loss_fn
    
    def run(self) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError


class BlackBoxTransferAlgorithm():
    surrogate: LocalModel

    def __init__(self, surrogate: LocalModel):
        self.surrogate = surrogate

    def run(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


from .fgsm import FgsmTransferAlgorithm
from .rays import RaySAttackAlgorithm
from .passthrough_attack import PassthroughTransferAttackAlgorithm
from .square_attack import SquareL2AttackAlgorithm
from __future__ import annotations
from abc import ABC, abstractmethod
from advpipe.log import logger
import numpy as np
import eagerpy as ep

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from typing import Generator


class Norm:
    _supported_norms = {"l0": ep.norms.l0, "l1": ep.norms.l1, "l2": ep.norms.l2, "linf": ep.norms.linf}

    def __init__(self, norm_order: str):
        assert norm_order in self._supported_norms
        self._name = norm_order
        self._norm = self._supported_norms[norm_order]

    def __call__(self, img: ep.types.NativeTensor) -> float:
        return float(self._norm(ep.astensor(img)).numpy())

    @property
    def name(self) -> str:
        return self._name


class BlackBoxAlgorithm(ABC):
    image: np.ndarray
    pertubation: np.ndarray

    def __init__(self, image: np.ndarray):
        # Image should have [H, W, C] shape with values in [0, 1.0] range
        self.image = image
        self.pertubation = np.zeros_like(self.image)

    # Algoritm should yield to the attack manager the current pertubation
    # transfer attacks yield only once and then StopIteration
    @abstractmethod
    def run(self) -> Generator[np.ndarray, None, None]:
        ...


class BlackBoxIterativeAlgorithm(BlackBoxAlgorithm):
    loss_fn: LossCallCounter

    def __init__(self, image: np.ndarray, loss_fn: LossCallCounter):
        super().__init__(image)
        self.loss_fn = loss_fn


class BlackBoxTransferAlgorithm(BlackBoxAlgorithm):
    def __init__(self, image: np.ndarray):
        super().__init__(image)


from .rays import RaySAttackAlgorithm
from .passthrough_attack import PassthroughTransferAttackAlgorithm
from .square_attack import SquareL2AttackAlgorithm

from __future__ import annotations
from advpipe.log import logger
import numpy as np
import eagerpy as ep

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from typing import Iterator


class Norm:
    _supported_norms = {"l0": ep.norms.l0, "l1": ep.norms.l1, "l2": ep.norms.l2, "linf": ep.norms.linf}

    def __init__(self, norm_order: str):
        assert norm_order in self._supported_norms
        self._norm = self._supported_norms[norm_order]

    def __call__(self, img: ep.types.NativeTensor) -> float:
        return float(self._norm(ep.astensor(img)).numpy())


class BlackBoxAlgorithm:
    image: np.ndarray
    loss_fn: LossCallCounter
    pertubation: np.ndarray

    def __init__(self, image: np.ndarray, loss_fn: LossCallCounter):
        self.image = image
        self.loss_fn = loss_fn
        self.pertubation = np.zeros_like(self.image)

    def run(self) -> Iterator[np.ndarray]:
        for i in range(100):
            logger.info(f"i: {i}, blackbox-loss: {self.loss_fn(self.image + self.pertubation)}")

            # Algoritm should yield to the attack manager the current pertubation
            yield self.pertubation


class BlackBoxIterativeAlgorithm(BlackBoxAlgorithm):
    def __init__(self, image: np.ndarray, loss_fn: LossCallCounter):
        super().__init__(image, loss_fn)

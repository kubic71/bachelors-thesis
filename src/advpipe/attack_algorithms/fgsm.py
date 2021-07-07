from __future__ import annotations
from advpipe.attack_algorithms import BlackBoxTransferAlgorithm
from advpipe.blackbox.local import LocalModel
import numpy as np
from advpipe.log import logger
from advpipe import utils
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.blackbox.local import WhiteBoxSurrogate
    from typing import Generator, Sequence


class FgsmTransferAlgorithm(BlackBoxTransferAlgorithm):
    epsilon: float

    def __init__(self, image: np.ndarray, surrogate: WhiteBoxSurrogate, epsilon: float):
        super().__init__(image, surrogate)

        self.epsilon = epsilon

    def run(self) -> Generator[np.ndarray, None, None]:
        grad = self.surrogate.grad(self.image)
        np_img = np.clip(self.image - self.epsilon * np.sign(grad), 0, 1)

        yield np_img
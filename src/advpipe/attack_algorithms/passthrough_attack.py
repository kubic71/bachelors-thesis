from __future__ import annotations
from advpipe.attack_algorithms import BlackBoxTransferAlgorithm
import numpy as np
from advpipe.log import logger
from advpipe import utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from typing import Generator
    from advpipe.blackbox.local import WhiteBoxSurrogate


class PassthroughTransferAttackAlgorithm(BlackBoxTransferAlgorithm):
    def __init__(self, image: np.ndarray, surrogate: WhiteBoxSurrogate):
        super().__init__(image, surrogate)

    def run(self) -> Generator[np.ndarray, None, None]:
        yield self.image
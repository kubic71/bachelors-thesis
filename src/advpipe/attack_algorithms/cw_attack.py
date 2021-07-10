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


# class FgsmTransferAlgorithm(BlackBoxTransferAlgorithm):
#     epsilon: float
#     n_iters: int
# 
#     def __init__(self, image: np.ndarray, surrogate: WhiteBoxSurrogate, epsilon: float, n_iters: int = 10):
#         super().__init__(image, surrogate)
# 
#         self.epsilon = epsilon
#         self.n_iters = n_iters
# 
# 
#     def run(self) -> Generator[np.ndarray, None, None]:
#         current = self.image
#         for i in range(self.n_iters):
#             grad = self.surrogate.grad(current)
#             current = current - self.epsilon*np.sign(grad)
#             current = utils.clip_linf(self.image, current, self.epsilon)
# 
#         yield current
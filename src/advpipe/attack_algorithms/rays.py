from __future__ import annotations
from advpipe.attack_algorithms import BlackBoxIterativeAlgorithm
import numpy as np
import torch
from advpipe.log import logger
import eagerpy as ep

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.attack_algorithms import Norm
    from typing import Generator, Optional, Iterator, AsyncGenerator


class RaySAttackAlgorithm(BlackBoxIterativeAlgorithm):
    def __init__(self, image: np.ndarray, loss_fn: LossCallCounter, epsilon: float, norm: Norm, early_stopping: bool):
        super().__init__(image, loss_fn)
        self.pertubation = np.zeros_like(image)

        self.epsilon: float = epsilon
        self.norm: Norm = norm
        self.early_stopping: bool = early_stopping

        self.queries: int = 0
        self.sgn_t: torch.Tensor = torch.sign(torch.ones(self.image.shape))
        self.d_t: float = np.inf
        self.x_final: torch.Tensor = torch.from_numpy(self.image)
        self.lin_search_rad: int = 10
        self.pre_set = {1, -1}

    def get_xadv(self, x: torch.Tensor, v: torch.Tensor, d: float, lb: float = 0., rb: float = 1.) -> torch.Tensor:
        out = x + d * v
        return torch.clamp(out, lb, rb)

    def run(self, seed: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        x: torch.Tensor = self.x_final
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        dist = torch.tensor(np.inf)
        block_level = 0
        block_ind = 0

        # Number of iterations is set to a large number, because attack termination is handled upstream
        for i in range(1000000):

            block_num = 2**block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[:, start:end] *= -1.
            attempt = attempt.view(shape)

            yield from self.binary_search(x, attempt)

            block_ind += 1
            if block_ind == 2**block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.tensor(self.norm(self.x_final - x))

            if i % 1 == 0:
                logger.debug(f"Rays iteration: {i}, Queries: {self.queries} d_t {self.d_t:.8f} dist {dist:.8f}")

            if self.early_stopping and (dist <= self.epsilon):
                logger.info(f"Rays early stopping. dist: {dist} <= {self.epsilon}")
                break

    def is_adversarial(self, img: torch.Tensor) -> Generator[np.ndarray, None, bool]:
        loss_val = self.loss_fn(img.detach().cpu().numpy())
        # yield evey time loss_fn is evaluated, so that the attack manager can retake the execution control
        yield self.x_final.detach().cpu().numpy()

        return loss_val <= 0

    def search_succ(self, x: torch.Tensor) -> Generator[np.ndarray, None, bool]:
        self.queries += 1
        ret = (yield from self.is_adversarial(x))
        return ret

    def lin_search(self, x: torch.Tensor, sgn: torch.Tensor) -> Generator[np.ndarray, None, float]:
        d_end: float = np.inf
        for d in range(1, self.lin_search_rad + 1):
            succ = yield from self.search_succ(self.get_xadv(x, sgn, d))
            if succ:
                d_end = d
                break
        return d_end

    def binary_search(self, x: torch.Tensor, sgn: torch.Tensor, tol: float = 1e-3) -> Generator[np.ndarray, None, bool]:
        sgn_unit: torch.Tensor = sgn / torch.norm(sgn)    # type: ignore
        sgn_norm: torch.Tensor = torch.norm(sgn)    # type: ignore

        d_start = 0
        if np.inf > self.d_t:    # already have current result
            succ = yield from self.search_succ(self.get_xadv(x, sgn_unit, self.d_t))
            if not succ:
                return False
            d_end = self.d_t
        else:    # init run, try to find boundary distance
            d = yield from self.lin_search(x, sgn)
            if d < np.inf:
                d_end = d * sgn_norm    # type: ignore
            else:
                return False

        while (d_end - d_start) > tol:
            d_mid = (d_start + d_end) / 2.0
            succ = yield from self.search_succ(self.get_xadv(x, sgn_unit, d_mid))
            if succ:
                d_end = d_mid
            else:
                d_start = d_mid    # type: ignore
        if d_end < self.d_t:
            self.d_t = d_end
            self.x_final = self.get_xadv(x, sgn_unit, d_end)
            self.sgn_t = sgn
            return True
        else:
            return False

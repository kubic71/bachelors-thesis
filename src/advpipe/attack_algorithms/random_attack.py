from advpipe.attack_algorithms import BlackBoxIterativeAlgorithm
import numpy as np
from advpipe.log import logger
from advpipe import utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from typing import Generator


class RandomAttackAlgorithm(BlackBoxIterativeAlgorithm):
    def __init__(self, image: np.ndarray, loss_fn: LossCallCounter, epsilon: float = 0.3, max_iters: int = 1000):
        super().__init__(image, loss_fn)
        self.pertubation = np.zeros_like(image)
        self.best_loss = self.loss_fn(image)
        self.epsilon = epsilon
        self.max_iters = max_iters

    def run(self) -> Generator:
        for i in range(self.max_iters):
            self.new_pertubation = self.pertubation + np.random.normal(size=self.image.shape) * self.epsilon / 3
            pertubed_img = utils.clip_linf(self.image + self.new_pertubation, self.epsilon)

            loss_val = self.loss_fn(pertubed_img)
            if loss_val < self.best_loss:
                self.pertubation = np.clip(self.new_pertubation, -self.epsilon, self.epsilon)

            yield self.image + self.pertubation
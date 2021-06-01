from advpipe.attack_algorithms import BlackBoxIterativeAlgorithm
import numpy as np
from advpipe.log import logger
from advpipe import utils

class RandomAttackAlgorithm(BlackBoxIterativeAlgorithm):
    def __init__(self, image, loss_fn, epsilon=0.3, max_iters=1000):
        super().__init__(image, loss_fn)
        self.pertubation = np.zeros_like(image)
        self.best_loss = self.loss_fn(image)
        self.epsilon = epsilon
        self.max_iters = max_iters
    
    def run(self):
        for i in range(self.max_iters):
            self.new_pertubation = self.pertubation + np.random.normal(size=self.image.shape)*self.epsilon/3
            pertubed_img = utils.clip_linf(self.image + self.new_pertubation, self.epsilon)

            loss_val = self.loss_fn(pertubed_img)
            if loss_val < self.best_loss:
                self.pertubation = np.clip(self.new_pertubation, -self.epsilon, self.epsilon)


            inf_dist = utils.l_inf(self.image, self.image + self.pertubation)
            logger.info(f"l_inf dist: {inf_dist}")
            yield self.image + self.pertubation
from advpipe.log import logger


def zero_pertubation():
    # TODO
    return 0


class BlackBoxAlgorithm:
    image = None
    pertubation = None

    def __init__(self, image, loss_fn):
        self.image = image
        self.loss_fn = loss_fn
        self.pertubation = zero_pertubation()

    def run(self):
        for i in range(100):
            logger.info(f"i: {i}, blackbox-loss: {self.loss_fn(self.image)}")

            # Algoritm should yield to the attack manager the current pertubation
            yield self.pertubation


class BlackBoxIterativeAlgorithm(BlackBoxAlgorithm):
    def __init__(self, image, loss_fn):
        super().__init__(image, loss_fn)

    @staticmethod
    def createBlackBoxAttackInstance(algorithm_name, image, loss_fn):
        return blackbox_iterative_algorithms[algorithm_name](image, loss_fn)


from .random_attack import RandomAttackAlgorithm
from .rays import RaySAttackAlgorithm

blackbox_iterative_algorithms = {
    "random": RandomAttackAlgorithm,
    "rays": RaySAttackAlgorithm
}

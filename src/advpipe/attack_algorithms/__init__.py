
def zero_pertubation():
    # TODO
    return 0

class BlackBoxAlgorithm:
    image = None
    pertubation = None

    def __init__(self, image):
        self.image = image
        self.pertubation = zero_pertubation()


class BlackBoxIterativeAlgorithm(BlackBoxAlgorithm):
    iter_counter = 0

    def __init__(self, image):
        super().__init__(image)


    @staticmethod
    def createBlackBoxAttackInstance(algorithm_name, image):
        return blackbox_iterative_algorithms[algorithm_name](image)

    def _pertube(self, loss):
        # Override this in Child attack class
        # default is No-pertubation
        return self.pertubation

    def step(self, loss):
        self.pertubation = self._pertube(loss)
        self.iter_counter += 1
        return self.pertubation


from .rays import RaysAttackAlgorithm
blackbox_iterative_algorithms = {"rays": RaysAttackAlgorithm}
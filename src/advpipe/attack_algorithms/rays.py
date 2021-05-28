from advpipe.attack_algorithms import BlackBoxIterativeAlgorithm

class RaysAttackAlgorithm(BlackBoxIterativeAlgorithm):
    def __init__(self, image):
        super().__init__(image)
    
    # Override the _pertube method
    def _pertube(self, loss):
        # TODO
        # Image is left unchanged for now just for testing purposes

        # New adversarial image pertubation is returned
        return self.pertubation

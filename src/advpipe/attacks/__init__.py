from advpipe.config_datamodel import AdvPipeConfig
from advpipe.log import logger
import numpy as np


class MaxFunctionCallsExceededException(Exception):
    pass


class LossCallCounter:
    def __init__(self, loss_fn, max_calls):
        self.loss_fn = loss_fn
        self.last_loss_val = np.inf
        self.last_img = None
        self.max_calls = max_calls
        self.i = 0

    def __call__(self, pertubed_image):
        if self.i >= self.max_calls:
            msg = f"Max number of function calls exceeded (max_calls={self.max_calls})"
            logger.info(f"LossCallCounter: {msg}")
            raise MaxFunctionCallsExceededException(msg)

        self.i += 1
        self.last_loss_val = self.loss_fn(pertubed_image)
        self.last_img = pertubed_image
        return self.last_loss_val


from advpipe.blackbox import TargetBlackBox
class Attack:
    target_blackbox: TargetBlackBox

    def __init__(self, attack_regime_config):
        self.config = attack_regime_config

        # Initialize the connection to the target blackbox
        self.target_blackbox = self.config.target_blackbox_config.getBlackBoxInstance()

    @staticmethod
    def from_config(config):
        """Factory method creating AdvPipe attack experiment objects"""
        return attack_list[config.attack_regime.name](config.attack_regime)

    def run(self):
        # Override this in corresponding AdvPipe attack class
        pass


from .simple_iterative_attack import SimpleIterativeAttack

attack_list = {"simple-iterative-attack": SimpleIterativeAttack}

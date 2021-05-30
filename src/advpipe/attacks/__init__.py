from advpipe.blackbox import TargetBlackBox
from advpipe.config_datamodel import AdvPipeConfig


class Attack:
    target_blackbox : TargetBlackBox

    def __init__(self, attack_regime_config ):
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

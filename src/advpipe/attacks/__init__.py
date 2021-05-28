from advpipe.blackbox import TargetBlackBox


class Attack:
    target_blackbox : TargetBlackBox

    def __init__(self, config):
        self.config = config

        # Initialize the connection to the target blackbox
        self.target_blackbox = TargetBlackBox(config.target_blackbox)


    @staticmethod
    def from_config(config):
        """Factory method creating AdvPipe attack experiment objects"""
        return attack_list[config.name](config)

    def run(self):
        # Override this in corresponding AdvPipe attack class
        pass
        

from .simple_iterative_attack import SimpleIterativeAttack
attack_list = {"simple-iterative-attack": SimpleIterativeAttack}

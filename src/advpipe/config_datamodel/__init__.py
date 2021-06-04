from dataclasses import dataclass
from abc import ABC, abstractmethod


class DatasetConfig:
    def __init__(self, dataset_config_yaml):
        self.data_dir = dataset_config_yaml.data_dir




class AdvPipeConfig:
    def __init__(self, yaml_config):
        # TODO - parse and check attack_regime data config
        self.attack_regime = yaml_config.attack_regime
        self.attack_regime.dataset_config = DatasetConfig(self.attack_regime.dataset_config)
        self.attack_regime.target_blackbox_config = TargetBlackBoxConfig.loadFromYamlConfig(self.attack_regime.target_blackbox_config)



class TargetBlackBoxConfig:

    def __init__(self, blackbox_name, loss_config, blackbox_class_ref):
        self.name = blackbox_name
        self.blackbox_class_ref = blackbox_class_ref
        self.loss = loss_config


    @staticmethod
    def loadFromYamlConfig(target_blackbox_config):
        """Factory method for TargetBlackBoxConfig

        returns: either LocalBlackBoxConfig or CloudBlackBoxConfig depending on blackbox_type YAML field
        """
        assert target_blackbox_config.blackbox_type in TARGET_BLACKBOX_TYPES

        if target_blackbox_config.blackbox_type == "local":
            return LocalBlackBoxConfig(target_blackbox_config)

        elif target_blackbox_config.blackbox_type == "cloud":
            return CloudBlackBoxConfig(target_blackbox_config)

    def getBlackBoxInstance(self):
        return self.blackbox_class_ref(blackbox_config = self)

from advpipe.blackbox.local import LOCAL_BLACKBOXES
class LocalBlackBoxConfig(TargetBlackBoxConfig):
    blackbox_type = "local"
    def __init__(self, local_blackbox_config):
        class_ref = LOCAL_BLACKBOXES[local_blackbox_config.name]
        super().__init__(blackbox_name = local_blackbox_config.name, loss_config = local_blackbox_config.loss, blackbox_class_ref = class_ref)


from advpipe.blackbox.cloud import CLOUD_BLACKBOXES
@dataclass
class CloudBlackBoxConfig(TargetBlackBoxConfig):
    blackbox_type = "cloud"

    def __init__(self, cloud_blackbox_config):
        class_ref = CLOUD_BLACKBOXES[cloud_blackbox_config.name]
        super().__init__(blackbox_name = cloud_blackbox_config.name, loss_config = cloud_blackbox_config.loss, blackbox_class_ref = class_ref)


from advpipe.blackbox import TargetBlackBox
from advpipe.blackbox import TARGET_BLACKBOX_TYPES
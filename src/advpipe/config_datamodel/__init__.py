from dataclasses import dataclass

class DatasetConfig:
    def __init__(self, dataset_config_yaml):
        self.data_dir = dataset_config_yaml.data_dir



from advpipe.blackbox import TargetBlackBox
from advpipe.blackbox.local import LOCAL_BLACKBOXES, LocalBlackBox

class AdvPipeConfig:
    def __init__(self, yaml_config):
        # TODO - parse and check attack_regime data config
        self.attack_regime = yaml_config.attack_regime
        self.attack_regime.dataset_config = DatasetConfig(self.attack_regime.dataset_config)
        self.attack_regime.target_blackbox_config = TargetBlackBoxConfig.loadFromConfig(self.attack_regime.target_blackbox_config)

@dataclass
class TargetBlackBoxConfig:
    blackbox_type: str
    class_ref: TargetBlackBox

    @staticmethod
    def loadFromConfig(target_blackbox_config):
        assert target_blackbox_config.blackbox_type in TARGET_BLACKBOX_TYPES

        if target_blackbox_config.blackbox_type == "local":
            return LocalBlackBoxConfig(target_blackbox_config)

        elif target_blackbox_config.blackbox_type == "cloud":
            return CloudBlackBoxConfig(target_blackbox_config)

    def getBlackBoxInstance(self) -> TargetBlackBox:
        return self.class_ref(self)

@dataclass
class LocalBlackBoxConfig(TargetBlackBoxConfig):
    blackbox_type = "local"
    name: str
    class_ref: LocalBlackBox

    def __init__(self, local_blackbox_config):
        assert local_blackbox_config.name in LOCAL_BLACKBOXES

        self.name = local_blackbox_config.name
        self.class_ref = LOCAL_BLACKBOXES[self.name]


@dataclass
class CloudBlackBoxConfig(TargetBlackBoxConfig):
    blackbox_type = "cloud"
    name: str

    def __init__(self, cloud_blackbox_config):
        raise NotImplementedError


from advpipe.blackbox import TARGET_BLACKBOX_TYPES
        



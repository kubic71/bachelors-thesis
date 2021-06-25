from __future__ import annotations
from advpipe.config_datamodel import DatasetConfig
from advpipe.log import CloudDataLogger
from datetime import datetime
import os
from advpipe import utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from munch import Munch
    from .attack_algorithm_config import AttackAlgorithmConfig


class AttackRegimeConfig:
    name: str
    skip_already_adversarial: bool = True
    dataset_config: DatasetConfig
    target_blackbox_config: TargetBlackBoxConfig
    attack_algorithm_config: AttackAlgorithmConfig

    def __init__(self, attack_regime_config: Munch):
        self.name = attack_regime_config.name
        self.skip_already_adversarial = utils.get_config_attr(attack_regime_config, "skip_already_adversarial",
                                                              self.skip_already_adversarial)
        self.dataset_config = DatasetConfig.loadFromYamlConfig(attack_regime_config.dataset_config)

        self.target_blackbox_config = TargetBlackBoxConfig.loadFromYamlConfig(
            attack_regime_config.target_blackbox_config)

        # leaky-abstraction, because attack algorithm needs to access epsilon, norm and other constraints,
        # which are global to the AttackRegime whole AdvPipe Attack
        self.attack_algorithm_config = AttackAlgorithmConfig.loadFromYamlConfig(attack_regime_config,
                                                                                attack_regime_config.attack_algorithm)

        # log destination includes information about the whole attack regime
        # default is saving to /tmp/cloud_data
        if isinstance(self.target_blackbox_config, CloudBlackBoxConfig):
            # Create cloud data logger
            bbox_name = self.target_blackbox_config.name
            dataset_name = self.dataset_config.name
            attack_name = self.name
            algo_name = self.attack_algorithm_config.name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            cloud_log_path = os.path.join(utils.get_abs_module_path(), "collected_cloud_data", bbox_name, dataset_name,
                                          attack_name, algo_name, timestamp)

            self.target_blackbox_config.cloud_data_logger = CloudDataLogger(cloud_log_path)

    @staticmethod
    def loadFromYamlConfig(attack_regime_config: Munch) -> AttackRegimeConfig:
        return ATTACK_REGIME_CONFIGS[attack_regime_config.name](attack_regime_config)


class IterativeRegimeConfig(AttackRegimeConfig):
    name: str = "iterative-regime"

    def __init__(self, attack_regime_config: Munch):
        super().__init__(attack_regime_config)
        self.max_iter = attack_regime_config.max_iter


class TransferRegimeConfig(AttackRegimeConfig):
    name: str = "transfer-regime"
    show_images: bool = True

    def __init__(self, attack_regime_config: Munch):
        super().__init__(attack_regime_config)
        self.show_images = utils.get_config_attr(attack_regime_config, "show_images", self.show_images)


ATTACK_REGIME_CONFIGS = {
    IterativeRegimeConfig.name: IterativeRegimeConfig,
    TransferRegimeConfig.name: TransferRegimeConfig
}

from .blackbox_config import TargetBlackBoxConfig, CloudBlackBoxConfig
from .attack_algorithm_config import AttackAlgorithmConfig
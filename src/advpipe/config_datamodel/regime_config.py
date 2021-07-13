from __future__ import annotations
from advpipe.config_datamodel import DatasetConfig
from advpipe.log import CloudDataLogger, logger
from datetime import datetime
import os
from advpipe import utils

from advpipe.attack_algorithms import Norm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from munch import Munch
    from typing import Optional, Sequence
    from .attack_algorithm_config import AttackAlgorithmConfig


class AttackRegimeConfig:
    name: str
    skip_already_adversarial: bool = True
    show_images: bool = True
    save_only_successful_examples: bool = False
    dont_save_images: bool = False
    dataset_config: DatasetConfig
    target_model_config: TargetModelConfig
    results_dir: str
    config_filename: str
    attack_algorithm_config: AttackAlgorithmConfig
    batch_size: int = 16

    norm: Norm
    epsilon: float




    _unparsed_config: Munch

    def __init__(self, attack_regime_config: Munch):
        self._unparsed_config = attack_regime_config

        self.config_filename = os.path.basename(attack_regime_config.config_filename)

        self.name = attack_regime_config.name
        self.skip_already_adversarial = utils.get_config_attr(attack_regime_config, "skip_already_adversarial",
                                                              self.skip_already_adversarial)
        self.show_images = utils.get_config_attr(attack_regime_config, "show_images", self.show_images)
        self.dataset_config = DatasetConfig.loadFromYamlConfig(attack_regime_config.dataset_config)

        try:
            target_config = getattr(attack_regime_config, "target_model_config")
            self.target_model_config = TargetModelConfig.loadFromYamlConfig(target_config)
        except AttributeError:
            # Multiple targets regime, assigning None
            # Works only with local models currently
            logger.debug("Mutliple-targets regime")
            self.target_model_config = None # type: ignore

        default_exp_name = self.config_filename.split(".")[0]
        self.results_dir = utils.convert_to_absolute_path(
            utils.get_config_attr(attack_regime_config, "results_dir", "results/" + default_exp_name))

        self.save_only_successful_images = utils.get_config_attr(attack_regime_config, "save_only_successful_examples",
                                                                 AttackRegimeConfig.save_only_successful_examples)
        self.dont_save_images = utils.get_config_attr(attack_regime_config, "dont_save_images",
                                                                 AttackRegimeConfig.dont_save_images)


        self.epsilon = attack_regime_config.epsilon
        self.norm = Norm(attack_regime_config.norm)
        self.batch_size = utils.get_config_attr(attack_regime_config, "batch_size", TransferRegimeConfig.batch_size)


        # log destination includes information about the whole attack regime
        # default is saving to /tmp/cloud_data
        if isinstance(self.target_model_config, CloudBlackBoxConfig):
            # Create cloud data logger
            bbox_name = self.target_model_config.name
            dataset_name = self.dataset_config.name
            attack_name = self.name
            algo_name = self.attack_algorithm_config.name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            cloud_log_path = os.path.join(utils.get_abs_module_path(), "collected_cloud_data", bbox_name, dataset_name,
                                          attack_name, algo_name, timestamp)
            self.target_model_config.cloud_data_logger = CloudDataLogger(cloud_log_path)

    @staticmethod
    def loadFromYamlConfig(attack_regime_config: Munch) -> AttackRegimeConfig:
        attack_regime_config.attack_algorithm.epsilon = attack_regime_config.epsilon
        attack_regime_config.attack_algorithm.norm = attack_regime_config.norm


        return  ATTACK_REGIME_CONFIGS[attack_regime_config.name](attack_regime_config)


class IterativeRegimeConfig(AttackRegimeConfig):
    name: str = "iterative-regime"
    attack_algorithm_config: IterativeAttackAlgorithmConfig

    def __init__(self, attack_regime_config: Munch):
        self.attack_algorithm_config = IterativeAttackAlgorithmConfig.loadFromYamlConfig(attack_regime_config.attack_algorithm)

        super().__init__(attack_regime_config)
        self.max_iter = attack_regime_config.max_iter


class TransferRegimeConfig(AttackRegimeConfig):

    name: str = "transfer-regime"
    surrogate_config: TargetModelConfig
    attack_algorithm_config: TransferAttackAlgorithmConfig

    def __init__(self, attack_regime_config: Munch):
        self.attack_algorithm_config = TransferAttackAlgorithmConfig.loadFromYamlConfig(attack_regime_config.attack_algorithm)

        super().__init__(attack_regime_config)

        # passthrough attack doesn't need surrogate
        if utils.get_config_attr(attack_regime_config, "surrogate", None):
            self.surrogate_config = TargetModelConfig.loadFromYamlConfig(attack_regime_config.surrogate)
        else:
            self.surrogate_config = DummySurrogateConfig()


class TransferRegimeMultipleTargetsConfig(TransferRegimeConfig):
    # Works only with local models currently

    name: str = "transfer-regime-multiple-targets"
    multiple_target_configs: Sequence[LocalModelConfig]

    def __init__(self, attack_regime_config: Munch):
        super().__init__(attack_regime_config)

        self.multiple_target_configs = []
        for target_name in attack_regime_config.multiple_target_configs.names:
            # substitute target name
            attack_regime_config.multiple_target_configs.name = target_name

            # and parse it in a standard way
            self.multiple_target_configs.append(LocalModelConfig(attack_regime_config.multiple_target_configs))

        
        

ATTACK_REGIME_CONFIGS = {
    IterativeRegimeConfig.name: IterativeRegimeConfig,
    TransferRegimeConfig.name: TransferRegimeConfig,
    TransferRegimeMultipleTargetsConfig.name: TransferRegimeMultipleTargetsConfig
}

from .blackbox_config import TargetModelConfig, CloudBlackBoxConfig, LocalModelConfig, DummySurrogateConfig
from .attack_algorithm_config import AttackAlgorithmConfig, IterativeAttackAlgorithmConfig, TransferAttackAlgorithmConfig
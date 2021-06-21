from __future__ import annotations
from abc import ABC, abstractmethod
import os
from datetime import datetime
from advpipe.attack_algorithms import Norm
from advpipe import utils
from advpipe.log import CloudDataLogger
import numpy as np
from advpipe.blackbox.local import LOCAL_BLACKBOXES
from advpipe.blackbox.cloud import CLOUD_BLACKBOXES
from advpipe.data_loader import DataLoader, DAmageNetDatasetLoader    # noqa: 402
from typing import TypeVar
from typing_extensions import Literal

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.attack_algorithms import BlackBoxAlgorithm
    from advpipe.blackbox import TargetBlackBox
    from munch import Munch
    from typing import Type, Dict


class DatasetConfig:
    def __init__(self, dataset_config_yaml: Munch):
        data_path = dataset_config_yaml.data_dir

        # If path is relative
        if not (data_path.startswith("/") or data_path.startswith("~")):
            self.full_path = os.path.join(utils.get_abs_module_path(), data_path)
        else:
            self.full_path = data_path

    @property
    def name(self) -> str:
        return os.path.basename(self.full_path)

    def getDatasetInstance(self) -> DataLoader:
        if self.name == "DAmageNet":
            return DAmageNetDatasetLoader(self)
        else:
            return DataLoader(self)


class AdvPipeConfig:
    def __init__(self, advpipe_config: Munch):
        self.attack_regime_config = AttackRegimeConfig(advpipe_config.attack_regime)

    def getAttackInstance(self) -> Attack:
        return ATTACK_LIST[self.attack_regime_config.name](self.attack_regime_config)


class AttackRegimeConfig:
    name: str
    max_iter: int
    dataset_config: DatasetConfig
    target_blackbox_config: TargetBlackBoxConfig
    attack_algorithm_config: AttackAlgorithmConfig

    def __init__(self, attack_regime_config: Munch):
        self.name = attack_regime_config.name
        self.max_iter = attack_regime_config.max_iter
        self.dataset_config = DatasetConfig(attack_regime_config.dataset_config)

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


class AttackAlgorithmConfig(ABC):
    name: str
    norm: Norm
    epsilon: float

    def __init__(self, attack_regime_config: Munch):
        self.attack_regime_config = attack_regime_config
        self.norm = Norm(attack_regime_config.norm)
        self.epsilon = attack_regime_config.epsilon

    @staticmethod
    def loadFromYamlConfig(attack_regime_config: Munch, algorithm_config: Munch) -> AttackAlgorithmConfig:
        if algorithm_config.name not in ATTACK_ALGORITHM_CONFIGS:
            raise ValueError(f"Invalid attack algorithm: {algorithm_config.name}")

        return ATTACK_ALGORITHM_CONFIGS[algorithm_config.name](attack_regime_config, algorithm_config)

    @abstractmethod
    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> BlackBoxAlgorithm:
        """Factory method creating AdvPipe attack experiment objects"""


class RaySAlgorithmConfig(AttackAlgorithmConfig):
    name: str = "rays"

    def __init__(self, attack_regime_config: Munch, rays_config: Munch):
        super().__init__(attack_regime_config)
        self.early_stopping = rays_config.early_stopping

    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> BlackBoxAlgorithm:
        return RaySAttackAlgorithm(image,
                                   loss_fn,
                                   epsilon=self.epsilon,
                                   norm=self.norm,
                                   early_stopping=self.early_stopping)


ATTACK_ALGORITHM_CONFIGS = {"rays": RaySAlgorithmConfig}

BLACKBOX_TYPE = Literal["cloud", "local"]


class TargetBlackBoxConfig:
    name: str
    blackbox_class_ref: Type[TargetBlackBox]
    loss: Munch
    blackbox_type: BLACKBOX_TYPE

    def __init__(self, target_blackbox_config: Munch):
        self.name = target_blackbox_config.name
        self.loss = target_blackbox_config.loss

    @staticmethod
    def loadFromYamlConfig(target_blackbox_config: Munch) -> TargetBlackBoxConfig:
        """Factory method for TargetBlackBoxConfig

        returns: either LocalBlackBoxConfig or CloudBlackBoxConfig depending on blackbox_type YAML field
        """
        b_type = target_blackbox_config.blackbox_type
        if b_type == "cloud":
            return CloudBlackBoxConfig(target_blackbox_config)
        elif b_type == "local":
            return LocalBlackBoxConfig(target_blackbox_config)
        else:
            raise ValueError(f"Invalid blackbox type: {b_type}")

    def getBlackBoxInstance(self) -> TargetBlackBox:
        return self.blackbox_class_ref(blackbox_config=self)


class LocalBlackBoxConfig(TargetBlackBoxConfig):
    blackbox_type: BLACKBOX_TYPE = "local"

    def __init__(self, local_blackbox_config: Munch):
        self.blackbox_class_ref = LOCAL_BLACKBOXES[local_blackbox_config.name]
        super().__init__(local_blackbox_config)


class CloudBlackBoxConfig(TargetBlackBoxConfig):
    blackbox_type: BLACKBOX_TYPE = "cloud"
    cloud_data_logger: CloudDataLogger

    def __init__(self, cloud_blackbox_config: Munch):
        self.blackbox_class_ref = CLOUD_BLACKBOXES[cloud_blackbox_config.name]
        super().__init__(cloud_blackbox_config)

        # Default cloud data logger
        self.cloud_data_logger = CloudDataLogger("/tmp/cloud_data")


from advpipe.attack_algorithms.rays import RaySAttackAlgorithm    # noqa: 402
from advpipe.attacks import Attack    # noqa: 402

from advpipe.attacks.simple_iterative_attack import SimpleIterativeAttack    # noqa: 402

ATTACK_LIST = {"simple-iterative-attack": SimpleIterativeAttack}

from __future__ import annotations
from abc import ABC, abstractmethod
import os
from advpipe import utils
from advpipe.log import CloudDataLogger
import numpy as np
from advpipe.data_loader import DataLoader, ImageNetValidationDataloader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from munch import Munch
    from advpipe.attacks import AttackRegime
    from typing import Optional


class DatasetConfig:
    dataset_type: str = "generic"
    size_limit: Optional[int] = None

    def __init__(self, dataset_config_yaml: Munch):
        data_path = dataset_config_yaml.data_dir
        self.size_limit = utils.get_config_attr(dataset_config_yaml, "size_limit", DatasetConfig.size_limit)

        # If path is relative
        if not (data_path.startswith("/") or data_path.startswith("~")):
            self.full_path = os.path.join(utils.get_abs_module_path(), data_path)
        else:
            self.full_path = data_path

    @staticmethod
    def loadFromYamlConfig(dataset_config_yaml: Munch) -> DatasetConfig:
        dataset_type = utils.get_config_attr(dataset_config_yaml, "dataset_type", DatasetConfig.dataset_type)

        if dataset_type == ImageNetValidationDatasetConfig.dataset_type:
            return ImageNetValidationDatasetConfig(dataset_config_yaml)
        else:
            return DatasetConfig(dataset_config_yaml)

    @property
    def name(self) -> str:
        return os.path.basename(self.full_path)

    def getDatasetInstance(self) -> DataLoader:
        return DataLoader(self)


class ImageNetValidationDatasetConfig(DatasetConfig):
    dataset_type: str = "imagenet-validation"
    load_only_organisms: bool = True

    def __init__(self, dataset_config_yaml: Munch):
        super().__init__(dataset_config_yaml)
        self.load_only_organisms = utils.get_config_attr(dataset_config_yaml, "load_only_organisms",
                                                         ImageNetValidationDatasetConfig.load_only_organisms)

    def getDatasetInstance(self) -> ImageNetValidationDataloader:
        return ImageNetValidationDataloader(self)


class AdvPipeConfig:
    def __init__(self, advpipe_config_fn: str):
        advpipe_config = utils.load_yaml(advpipe_config_fn)
        advpipe_config.attack_regime.config_filename = advpipe_config_fn
        self.attack_regime_config = AttackRegimeConfig.loadFromYamlConfig(advpipe_config.attack_regime)

    def getAttackInstance(self) -> AttackRegime:
        return ATTACK_REGIMES[self.attack_regime_config.name](self.attack_regime_config)


# add regime config to namespace
from .regime_config import AttackRegimeConfig, IterativeRegimeConfig, TransferRegimeConfig

#  add attack algorithm config to namespace
from .attack_algorithm_config import AttackAlgorithmConfig, RaySAlgorithmConfig

#  add blackbox config to namespace
from .blackbox_config import TargetModelConfig, LocalModelConfig, CloudBlackBoxConfig

from advpipe.attack_regimes import SimpleIterativeRegime, SimpleTransferRegime

ATTACK_REGIMES = {IterativeRegimeConfig.name: SimpleIterativeRegime, TransferRegimeConfig.name: SimpleTransferRegime}

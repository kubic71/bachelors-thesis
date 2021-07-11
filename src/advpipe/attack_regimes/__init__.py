from __future__ import annotations
from abc import ABC, abstractmethod
from advpipe import utils
from advpipe.log import logger
import numpy as np
import shutil
from os import path
import munch
import yaml
import imageio

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import AttackRegimeConfig
    from advpipe.blackbox import TargetModel
    from advpipe.data_loader import DataLoader
    import numpy as np


class AttackRegime(ABC):
    target_model: TargetModel
    regime_config: AttackRegimeConfig
    dataloader: DataLoader

    def __init__(self, attack_regime_config: AttackRegimeConfig):
        self.regime_config = attack_regime_config
        self.dataloader = self.regime_config.dataset_config.getDatasetInstance()

        self.init_target()
        self.create_results_dir()
        self.copy_config_to_results_dir()

    def init_target(self) -> None:
        # Initialize the connection to the target model
        self.target_model = self.regime_config.target_model_config.getModelInstance()

    @abstractmethod
    def run(self) -> None:
        ...

    def save_adv_img(self, x_adv: np.ndarray, img_fn: str, dest_dir: str) -> None:
        """Saves successful adversarial image to results directory"""
        utils.mkdir_p(dest_dir)
        if self.regime_config.dont_save_images:
            return
        img_fn = img_fn.split(".")[0] + ".png"
        imageio.imwrite(dest_dir + "/" + img_fn, x_adv)

    def create_results_dir(self) -> None:
        if path.exists(self.regime_config.results_dir):
            logger.info(f"Results directory {self.regime_config.results_dir} already exists!")
            ## TODO: DEBUG
            # choice = input("Do you want to overwrite the results? (y/n):")
            choice = 'y'
            if choice != 'y':
                raise KeyboardInterrupt

            # delete the old results dir
            shutil.rmtree(self.regime_config.results_dir)

        utils.mkdir_p(self.regime_config.results_dir)

    def copy_config_to_results_dir(self) -> None:
        unmunched = munch.unmunchify(self.regime_config._unparsed_config)
        with open(self.regime_config.results_dir + "/" + self.regime_config.config_filename, "w") as f:
            yaml.dump(unmunched, stream=f)


# attack regimes are exported to advpipe.attack_regimes module namespace
# they are imported from this namespace by AdvPipeConfig, when it creates the regime instance (def getAttackInstance(self) -> AttackRegime)
from .simple_iterative_regime import SimpleIterativeRegime
from .simple_transfer_regime import SimpleTransferRegime
from .transfer_regime_multiple_targets import TransferRegimeMultipleTargets
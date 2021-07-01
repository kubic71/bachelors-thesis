from __future__ import annotations
from abc import ABC, abstractmethod
from advpipe import utils
import numpy as np
import shutil
from os import path
import munch
import yaml
import imageio


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import AttackRegimeConfig
    from advpipe.blackbox import TargetBlackBox
    from advpipe.data_loader import DataLoader
    import numpy as np


class AttackRegime(ABC):
    target_blackbox: TargetBlackBox
    regime_config: AttackRegimeConfig
    dataloader: DataLoader

    def __init__(self, attack_regime_config: AttackRegimeConfig):
        self.regime_config = attack_regime_config
        self.dataloader = self.regime_config.dataset_config.getDatasetInstance()

        # Initialize the connection to the target blackbox
        self.target_blackbox = self.regime_config.target_blackbox_config.getBlackBoxInstance()

        self.create_results_dir()
        self.create_adv_img_dir()
        self.copy_config_to_results_dir()


    @abstractmethod
    def run(self) -> None:
        ...

    def create_adv_img_dir(self) -> None:
        self.adv_img_dir = self.regime_config.results_dir + "/adv_examples"
        utils.mkdir_p(self.adv_img_dir)

    def save_adv_img(self, x_adv: np.ndarray, img_fn: str) -> None:
        """Saves successful adversarial image to results directory"""
        img_fn = img_fn.split(".")[0] + ".png"
        imageio.imwrite(self.adv_img_dir + "/" + img_fn, x_adv)


    def create_results_dir(self) -> None:
        if path.exists(self.regime_config.results_dir):
            print(f"Results directory {self.regime_config.results_dir} already exists!")
            choice = input("Do you want to overwrite the results? (y/n):")
            if choice != 'y':
                raise KeyboardInterrupt

            # delete the old results dir
            shutil.rmtree(self.regime_config.results_dir)

        utils.mkdir_p(self.regime_config.results_dir)


    def copy_config_to_results_dir(self) -> None:
        unmunched = munch.unmunchify(self.regime_config._unparsed_config)
        with open(self.regime_config.results_dir + "/" + self.regime_config.config_filename, "w") as f:
            yaml.dump(unmunched, stream=f)


from .simple_iterative_regime import SimpleIterativeRegime
from .simple_transfer_regime import SimpleTransferRegime
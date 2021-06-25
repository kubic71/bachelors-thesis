from __future__ import annotations
from advpipe.attack_regimes import AttackRegime
from advpipe.utils import MaxFunctionCallsExceededException, LossCallCounter
from advpipe.log import logger
from advpipe import utils
import numpy as np
from os import path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import TransferRegimeConfig


# TODO
class SimpleTransferRegime(AttackRegime):
    regime_config: TransferRegimeConfig

    def __init__(self, attack_regime_config: TransferRegimeConfig):
        # Initialize the black-box
        super().__init__(attack_regime_config)

    def run(self) -> None:
        super().run()

        logger.info(f"running Simple transfer regime for {len(self.dataloader)} images")

        n_successful = 0
        total = 0
        for img_path, np_img, label in self.dataloader:
            logger.info(f"Running transfer attack algorithm for {img_path}")
            _, img_fn = path.split(img_path)    # get image file name

            if self.regime_config.skip_already_adversarial:
                initial_loss_val = self.target_blackbox.loss(np_img)
                if initial_loss_val < 0:
                    logger.info(f"Skipping already adversarial image - initial loss: {initial_loss_val}")
                    continue
            
            total += 1

            # Set max_loss_count to 10, such that MaxCall exception isn't raised
            blackbox_loss = LossCallCounter(self.target_blackbox.loss, 10)

            self.transfer_algorithm = self.regime_config.attack_algorithm_config.getAttackAlgorithmInstance(
                np_img.copy(), blackbox_loss)

            running_attack = self.transfer_algorithm.run()
            pertubation = next(running_attack)

            # check that attack algorithm is used in transfer mode
            transfer_mode = False
            try:
                _ = next(running_attack)
            except StopIteration:
                transfer_mode = True

            assert transfer_mode, f"Attack algorithm {self.regime_config.attack_algorithm_config.name} isn't used in transfer mode, because it returned more than one pertubation"


            
            x_adv = np_img + pertubation
            loss = blackbox_loss(x_adv)
            img_name = path.basename(img_fn)
            norm = self.regime_config.attack_algorithm_config.norm
            dist = norm(pertubation)

            if loss < 0:
                n_successful += 1

            logger.debug(f"Img: {img_name}\t{norm.name} pertubation norm:{dist}\tloss: {loss}\tsuccess_rate: {n_successful}/{total} = {(n_successful/total*100):.2f}%")

            if self.regime_config.show_images:
                utils.show_img(x_adv)

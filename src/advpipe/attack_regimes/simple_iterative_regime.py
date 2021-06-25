from __future__ import annotations
from advpipe.attack_regimes import AttackRegime
from advpipe.utils import MaxFunctionCallsExceededException, LossCallCounter
from advpipe.log import logger
from advpipe import utils
import numpy as np
from os import path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import IterativeRegimeConfig


class SimpleIterativeRegime(AttackRegime):
    regime_config: IterativeRegimeConfig

    def __init__(self, attack_regime_config: IterativeRegimeConfig):
        # Initialize the black-box
        super().__init__(attack_regime_config)

    def run(self) -> None:
        super().run()

        logger.info(f"running Simple iterative attack for {len(self.dataloader)} images")

        for img_path, np_img, label in self.dataloader:
            logger.info(f"Running Simple iterative attack for {img_path}")
            _, img_fn = path.split(img_path)    # get image file name

            blackbox_loss = LossCallCounter(self.target_blackbox.loss, self.regime_config.max_iter)

            self.iterative_algorithm = self.regime_config.attack_algorithm_config.getAttackAlgorithmInstance(
                np_img.copy(), blackbox_loss)

            running_attack = self.iterative_algorithm.run()

            best_loss = np.inf
            best_img = np_img
            best_dist = np.inf

            i = 1
            while True:
                try:
                    _ = next(running_attack)
                    dist = np.max(np.abs(blackbox_loss.last_img - np_img))
                    logger.debug(
                        f"Manager: Img: {img_fn}\t\tIter: {i}\tdist:{dist}\tloss: {blackbox_loss.last_loss_val}")

                    if blackbox_loss.last_loss_val == 0 and dist < best_dist:
                        best_loss = 0
                        best_dist = dist
                        best_img = blackbox_loss.last_img
                        # utils.show_img(best_img)
                        logger.info(
                            f"New best img: {img_fn}\t\tQuery: {i}\tdist:{dist}\tloss: {blackbox_loss.last_loss_val}")

                    i += 1
                except (MaxFunctionCallsExceededException, StopIteration):
                    break

            dist = np.max(np.abs(np_img - best_img))
            logger.info(f"Img {img_fn} attack result: dist:{dist} best_loss: {best_loss}")
            utils.show_img(best_img)

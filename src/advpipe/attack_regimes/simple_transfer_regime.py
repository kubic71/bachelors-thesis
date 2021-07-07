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
    from advpipe.blackbox.local import WhiteBoxSurrogate
    from typing import Optional


# TODO
class SimpleTransferRegime(AttackRegime):
    regime_config: TransferRegimeConfig
    surrogate: Optional[WhiteBoxSurrogate]
    results_file: str

    def __init__(self, attack_regime_config: TransferRegimeConfig):
        # Initialize the black-box
        super().__init__(attack_regime_config)
        self.regime_config = attack_regime_config

        self.create_results_file()

        if self.regime_config.surrogate_config is None:
            self.surrogate = None
        else:
            self.surrogate = self.regime_config.surrogate_config.getSurrogateInstance()

    def run(self) -> None:
        super().run()

        logger.info(f"running Simple transfer regime with dataset {self.dataloader.name}")

        n_successful = 0
        total = 0
        for img_path, np_img, label, human_readable_label in self.dataloader:
            logger.debug(f"Running transfer attack algorithm for {img_path}")
            _, img_fn = path.split(img_path)    # get image file name

            if self.regime_config.skip_already_adversarial:
                initial_loss_val = self.target_blackbox.loss(np_img)
                if initial_loss_val < 0:
                    logger.debug(f"Skipping already adversarial image - initial loss: {initial_loss_val}")
                    continue

            total += 1

            self.transfer_algorithm = self.regime_config.attack_algorithm_config.getAttackAlgorithmInstance(
                np_img.copy(), self.surrogate)    # type: ignore

            running_attack = self.transfer_algorithm.run()
            x_adv = next(running_attack)

            # check that attack algorithm is used in transfer mode
            transfer_mode = False
            try:
                _ = next(running_attack)
            except StopIteration:
                transfer_mode = True

            assert transfer_mode, f"Attack algorithm {self.regime_config.attack_algorithm_config.name} isn't used in transfer mode, because it yielded more than once"

            success = False
            pertubation = x_adv - np_img

            loss = self.target_blackbox.loss(x_adv)
            labels = self.target_blackbox.last_query_result
            norm = self.regime_config.attack_algorithm_config.norm
            dist = norm(pertubation)

            self.write_result_to_file(img_fn, human_readable_label, loss,
                                      labels.get_top_organism()[0],
                                      labels.get_top_object()[0], dist)

            if loss < 0:
                n_successful += 1
                success = True

            if (not self.regime_config.save_only_successful_images) or success:
                self.save_adv_img(x_adv, img_fn)

            logger.debug(
                f"Img: {img_fn}\t{norm.name} pertubation norm:{dist}\tloss: {loss}\tsuccess_rate: {n_successful}/{total} = {(n_successful/total*100):.2f}%"
            )

            if self.regime_config.show_images:
                utils.show_img(x_adv)

        self.write_summary(n_successful, total)

    def create_results_file(self) -> None:
        self.results_file = self.regime_config.results_dir + "/transfer_attack_results.csv"
        # write header
        with open(self.results_file, "w") as f:
            f.write("img_fn\thuman_readable_label\tloss_val\ttop_organism_label\ttop_object_label\tdist\n")

    def write_result_to_file(self, img_fn: str, human_readable_label: Optional[str], loss_val: float,
                             top_organism_label: str, top_object_label: str, dist: float) -> None:

        with open(self.results_file, "a") as f:
            label_str = "label-NA" if human_readable_label is None else human_readable_label
            f.write(f"{img_fn}\t{label_str}\t{loss_val:.5f}\t{top_organism_label}\t{top_object_label}\t{dist:.5f}\n")

    def write_summary(self, n_successful: int, n_total: int) -> None:
        summary = f"successfully transferred / total = {n_successful}/{n_total} = {(n_successful/n_total*100):.2f}%"
        logger.info(summary)

        with open(self.regime_config.results_dir + "/summary.txt", "w") as f:
            f.write(summary + "\n")

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
    from typing import Optional


class SimpleIterativeRegime(AttackRegime):
    regime_config: IterativeRegimeConfig

    def __init__(self, attack_regime_config: IterativeRegimeConfig):
        # Initialize the black-box
        super().__init__(attack_regime_config)

        self.create_results_file()

    def run(self) -> None:
        super().run()

        logger.info(f"running Simple iterative attack for {len(self.dataloader)} images")

        n_successful = 0
        total = 0
        for img_path, np_img, label, human_readable_label in self.dataloader:
            logger.info(f"Running Simple iterative attack for {img_path}")
            _, img_fn = path.split(img_path)    # get image file name

            blackbox_loss = LossCallCounter(self.target_blackbox.loss, self.regime_config.max_iter)

            self.iterative_algorithm = self.regime_config.attack_algorithm_config.getAttackAlgorithmInstance(
                np_img.copy(), blackbox_loss)

            running_attack = self.iterative_algorithm.run()

            best_loss = np.inf
            best_img = np_img
            norm = self.regime_config.attack_algorithm_config.norm
            n_queries_needed = None
            success = False
            i = 1

            while True:
                try:
                    _ = next(running_attack)

                    dist = norm(blackbox_loss.last_img - np_img)
                    logger.debug(
                        f"Manager: Img: {img_fn}\t\tIter: {i}\tdist:{dist}\tloss: {blackbox_loss.last_loss_val}")

                    # x_adv satisfies l_p norm constarint and has best loss val so far
                    if blackbox_loss.last_loss_val < best_loss and dist <= self.regime_config.attack_algorithm_config.epsilon:
                        best_loss = blackbox_loss.last_loss_val
                        best_img = blackbox_loss.last_img
                        # utils.show_img(best_img)
                        logger.info(
                            f"New best img: {img_fn}\t\tQuery: {i}\tdist:{dist}\tloss: {blackbox_loss.last_loss_val}")

                        if best_loss < 0:
                            n_successful += 1
                            n_queries_needed = i
                            success = True
                            break

                    i += 1
                except (MaxFunctionCallsExceededException, StopIteration):
                    break

            total += 1

            loss_val = self.target_blackbox.loss(best_img)
            labels = self.target_blackbox.last_query_result

            logger.info(
                f"Img {img_fn} attack result: dist:{dist} best_loss: {loss_val} success_rate: {n_successful}/{total} = {(n_successful/total*100):.2f}%"
            )
            self.write_result_to_file(img_fn, human_readable_label, loss_val, n_queries_needed, labels.get_top_organism()[0], labels.get_top_object()[0], dist)

            if (not self.regime_config.save_only_successful_images) or success:
                self.save_adv_img(best_img, img_fn)

            if self.regime_config.show_images:
                utils.show_img(best_img, method="pyplot")



    def write_result_to_file(self, img_fn: str, human_readable_label: Optional[str], loss_val: float,
                             n_queries_needed: Optional[int], top_organism_label: str, top_object_label: str, dist: float) -> None:

        results_file = self.regime_config.results_dir + "/iterative_attack_results.csv"
        with open(results_file, "a") as f:
            label_str = "label-NA" if human_readable_label is None else human_readable_label
            n_queries_str = "inf" if n_queries_needed is None else str(n_queries_needed)
            f.write(f"{img_fn}\t{label_str}\t{loss_val:.5f}\t{n_queries_str}\t{top_organism_label}\t{top_object_label}\t{dist:.5f}\n")

    def create_results_file(self) -> None:
        self.results_file = self.regime_config.results_dir + "/iterative_attack_results.csv"
        # write header
        with open(self.results_file, "w") as f:
            f.write("img_fn\thuman_readable_label\tloss_val\tn_queries\ttop_organism_label\ttop_object_label\tdist\n")

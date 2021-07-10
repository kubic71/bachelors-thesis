from __future__ import annotations
from advpipe.attack_regimes import AttackRegime
from advpipe.data_loader import DataLoader
from advpipe.utils import MaxFunctionCallsExceededException, LossCallCounter
from advpipe.log import logger
from advpipe import utils
import numpy as np
from os import path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import TransferRegimeConfig
    from advpipe.blackbox.local import LocalModel
    from typing import Optional


class SimpleTransferRegime(AttackRegime):
    regime_config: TransferRegimeConfig
    surrogate: Optional[LocalModel]
    results_file: str

    def __init__(self, attack_regime_config: TransferRegimeConfig):
        # Initialize the black-box
        super().__init__(attack_regime_config)
        self.regime_config = attack_regime_config

        self.create_results_file()

        if self.regime_config.surrogate_config is None:
            self.surrogate = None
        else:
            self.surrogate = self.regime_config.surrogate_config.getBlackBoxInstance()

        self.transfer_algorithm = self.regime_config.attack_algorithm_config.getAttackAlgorithmInstance(self.surrogate) # type: ignore   

    def run(self) -> None:

        logger.info(f"running Simple transfer regime with dataset {self.dataloader.name}")

        n_successful = 0
        total = 0
        for img_paths, imgs, labels, human_readable_labels in DataLoader.create_batches(self.dataloader, self.regime_config.batch_size):
            img_fns = list(map(path.basename, img_paths))    # get image file name

            # if self.regime_config.skip_already_adversarial:
                # initial_loss_val = self.target_blackbox.loss(np_img)
                # if initial_loss_val < 0:
                    # logger.debug(f"Skipping already adversarial image - initial loss: {initial_loss_val}")
                    # continue

            total += len(img_paths)

            x_advs = self.transfer_algorithm.run(imgs, labels)

            pertubations = x_advs - imgs
            losses = self.target_blackbox.loss(x_advs)
            local_labels = self.target_blackbox.last_query_result
            norm = self.regime_config.norm
            dists = norm(pertubations)
        
            # logger.debug(f"Labels: {local_labels}")
            # logger.debug(f"Distance: {dist}")

            n_successful += int((losses < 0).sum())

            for img_fn, human_label, target_loss, top_org_label, top_obj_label, dist, x_adv in zip(img_fns, human_readable_labels, losses, local_labels.get_top_organism_labels(), local_labels.get_top_object_labels(), dists, x_advs.detach().cpu().numpy()):

                self.write_result_to_file(str(img_fn), human_label, target_loss, top_org_label, top_obj_label, dist)

                if (not self.regime_config.save_only_successful_images) or target_loss < 0:
                    self.save_adv_img(x_adv.transpose(1, 2, 0), img_fn)

                logger.debug(
                    f"Img: {human_label} {img_fn}\t{norm.name} pertubation norm:{dist}\tloss: {target_loss}\tsuccess_rate: {n_successful}/{total} = {(n_successful/total*100):.2f}%" )


            if self.regime_config.show_images:
                utils.show_img(x_adv[0])

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

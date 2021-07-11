from __future__ import annotations
from advpipe.attack_regimes import AttackRegime
from advpipe.data_loader import DataLoader
from advpipe.utils import MaxFunctionCallsExceededException, LossCallCounter
from advpipe.log import logger
from advpipe import utils
import numpy as np
from advpipe.blackbox import BlackboxLabels
from dataclasses import dataclass
from os import path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import TransferRegimeConfig
    from advpipe.blackbox.local import LocalModel, TargetModel
    import torch
    from typing import Optional, Sequence, Iterator, Union


# Corresponds to one line in the transfer_results.csv
@dataclass
class TransferResult:
    surrogate_name: str
    target_name: str
    img_fn: str
    human_label: str
    target_loss: float
    top_org_label: str
    top_obj_label: str
    dist: float
    x_adv: torch.Tensor


@dataclass
class BatchTransferResult:
    surrogate_name: str
    target_name: str
    img_fns: Sequence[str]
    human_labels: Sequence[str]
    model_labels: BlackboxLabels
    target_losses: torch.Tensor
    dists: torch.Tensor
    x_advs: torch.Tensor

    @property
    def n_successful(self) -> int:
        return int((self.target_losses < 0).sum())

    def items(self) -> Iterator[TransferResult]:
        for img_fn, human_label, target_loss, top_org_label, top_obj_label, dist, x_adv in zip(
                self.img_fns, self.human_labels, self.target_losses, self.model_labels.get_top_organism_labels(),
                self.model_labels.get_top_object_labels(), self.dists,
                self.x_advs.detach().cpu()):
            yield TransferResult(self.surrogate_name, self.target_name, img_fn, human_label, target_loss, top_org_label,
                                 top_obj_label, dist, x_adv)

RESULTS_FILENAME: str = "transfer_attack_results.csv"

class SimpleTransferRegime(AttackRegime):
    regime_config: TransferRegimeConfig
    surrogate: TargetModel


    def __init__(self, attack_regime_config: TransferRegimeConfig):
        # Initialize the black-box
        super().__init__(attack_regime_config)
        self.regime_config = attack_regime_config

        self.create_results_file()
        self.surrogate = self.regime_config.surrogate_config.getModelInstance()

        # can be dummy model in passthrough mode, which isn't a LocalModel, so ignore the type
        self.transfer_algorithm = self.regime_config.attack_algorithm_config.getAttackAlgorithmInstance(self.surrogate) # type: ignore

    def run(self) -> None:

        logger.info(f"running Simple transfer regime with dataset {self.dataloader.name}")

        self.n_successful = 0
        self.total = 0
        for img_paths, imgs, labels, human_readable_labels in DataLoader.create_batches(
                self.dataloader, self.regime_config.batch_size):
            # get image file name
            img_fns: Sequence[str] = list(map(path.basename, img_paths))    # type: ignore

            # if self.regime_config.skip_already_adversarial:
            # initial_loss_val = self.target_model.loss(np_img)
            # if initial_loss_val < 0:
            # logger.debug(f"Skipping already adversarial image - initial loss: {initial_loss_val}")
            # continue

            self.total += len(img_paths)

            x_advs = self.transfer_algorithm.run(imgs, labels)
            results = self.evaluate_batch_on_target(self.target_model, x_advs, imgs, img_fns, human_readable_labels)

            self.n_successful += results.n_successful
            self.save_results(results, self.regime_config.results_dir + "/" + RESULTS_FILENAME, self.n_successful, self.total)

            if self.regime_config.show_images:
                utils.show_img(next(results.items()).x_adv)

        self.write_summary(self.n_successful, self.total, self.regime_config.results_dir)

    def evaluate_batch_on_target(self, target_model: TargetModel, x_advs: torch.Tensor, original_imgs: torch.Tensor, img_fns: Sequence[str],
                                 human_labels: Sequence[str]) -> BatchTransferResult:
        pertubations = x_advs - original_imgs
        losses = target_model.loss(x_advs)
        model_labels = target_model.last_query_result
        norm = self.regime_config.norm
        dists = norm(pertubations)

        return BatchTransferResult(self.surrogate.model_config.name, target_model.model_config.name, img_fns,
                                   human_labels, model_labels, losses, dists, x_advs)

    def create_results_file(self) -> None:
        # write header
        with open(self.regime_config.results_dir + "/" + RESULTS_FILENAME, "w") as f:
            f.write("img_fn\thuman_readable_label\ttarget_loss\ttop_organism_label\ttop_object_label\tdist\tsurrogate\ttarget\n")

    def save_results(self, results: BatchTransferResult, results_file: str, n_successful: int, total: int) -> None:
        res_dir = path.dirname(results_file)


        with open(results_file, "a") as f:
            for r in results.items():
                label_str = "label-NA" if r.human_label is None else r.human_label
                f.write(
                    f"{r.img_fn}\t{label_str}\t{r.target_loss:.5f}\t{r.top_org_label}\t{r.top_obj_label}\t{r.dist:.5f}\t{r.surrogate_name}\t{r.target_name}\n"
                )

                if (not self.regime_config.save_only_successful_images) or r.target_loss < 0:
                    self.save_adv_img(r.x_adv.detach().cpu().numpy().transpose(1, 2, 0), r.img_fn, res_dir + "/adv_examples") 

                logger.debug(
                    f"Img: {r.human_label} {r.img_fn}\t{self.regime_config.norm.name} pertubation norm:{r.dist}\tloss: {r.target_loss}\tsuccess_rate: {n_successful}/{total} = {(n_successful/total*100):.2f}%"
                )

    def write_summary(self, n_successful: int, n_total: int, destination_dir: str) -> None:
        summary = f"successfully transferred / total = {n_successful}/{n_total} = {(n_successful/n_total*100):.2f}%"
        logger.info(summary)

        with open(destination_dir + "/summary.txt", "w") as f:
            f.write(summary + "\n")

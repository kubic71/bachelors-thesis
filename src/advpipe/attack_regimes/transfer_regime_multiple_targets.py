from __future__ import annotations
from advpipe.attack_regimes import AttackRegime
from advpipe.data_loader import DataLoader
from advpipe.utils import MaxFunctionCallsExceededException, LossCallCounter
from advpipe.log import logger
from advpipe.attack_regimes.simple_transfer_regime import SimpleTransferRegime, BatchTransferResult, TransferResult, RESULTS_FILENAME, CSV_HEADER
from advpipe import utils
import numpy as np
from collections import defaultdict
from advpipe.blackbox import BlackboxLabels
from dataclasses import dataclass
from os import path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import TransferRegimeMultipleTargetsConfig
    from advpipe.config_datamodel.blackbox_config import LocalModelConfig
    from advpipe.blackbox.local import LocalModel, TargetModel
    import torch
    from typing import Optional, Sequence, Iterator, Union, Dict


class TransferRegimeMultipleTargets(SimpleTransferRegime):
    regime_config: TransferRegimeMultipleTargetsConfig
    surrogate: TargetModel
    
    # create separate directory for each target
    # targets must have unique names!
    target_result_dirs: Dict[str, str]
    target_models: Dict[str, TargetModel]

    n_successful: Dict[str, int]  # type: ignore

    def __init__(self, attack_regime_config: TransferRegimeMultipleTargetsConfig):
        # Initialize the black-box
        super().__init__(attack_regime_config)
            
        self.n_successful = defaultdict(int)

    # @override
    def create_results_file(self) -> None:
        self.target_result_dirs = {}

        for target_config in self.regime_config.multiple_target_configs:
            results_dir = path.join(self.regime_config.results_dir, "target_" + target_config.name)
            utils.mkdir_p(results_dir)
            self.target_result_dirs[target_config.name] = results_dir

            results_file = path.join(results_dir, RESULTS_FILENAME)
            # write header
            with open(results_file, "w") as f:
                f.write(CSV_HEADER)

    # @override from TransferRegime
    def init_target(self) -> None:
        self.target_models = {}
        # Initialize the connection to the target model
        for target_config in self.regime_config.multiple_target_configs:
            self.target_models[target_config.name] = target_config.getModelInstance()

    # @override
    def run(self) -> None:

        logger.info(f"running transfer-regime-multiple-targets with dataset {self.dataloader.name}")
        self.total = 0
        for img_paths, imgs, labels, human_readable_labels in DataLoader.create_batches(
                self.dataloader, self.regime_config.batch_size):
            # get image file name
            img_fns: Sequence[str] = list(map(path.basename, img_paths))    # type: ignore

            if self.regime_config.skip_already_adversarial:
                raise NotImplementedError

            self.total += len(img_paths)
            x_advs = self.transfer_algorithm.run(imgs, labels)

            for target_name, target_model in self.target_models.items():
                results = self.evaluate_batch_on_target(target_model, x_advs, imgs, img_fns, human_readable_labels)
                self.n_successful[target_name] += results.n_successful
                self.save_results(results, self.target_result_dirs[target_name], self.n_successful[target_name], self.total)

            if self.regime_config.show_images:
                raise NotImplementedError
        
        for target_config in self.regime_config.multiple_target_configs:
            target_name = target_config.name
            self.write_summary(self.n_successful[target_name], self.total, self.target_result_dirs[target_name], self.surrogate.model_config.name, target_name)
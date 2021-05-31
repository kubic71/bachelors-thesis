from advpipe.attacks import Attack, MaxFunctionCallsExceededException, LossCallCounter
from advpipe.log import logger
from typing import Union
from advpipe.attack_algorithms import BlackBoxIterativeAlgorithm
from advpipe.data_loader import DataLoader
from advpipe import utils
import numpy as np
from os import path


class SimpleIterativeAttack(Attack):
    iterative_algorithm : BlackBoxIterativeAlgorithm
    dataloader : DataLoader

    def __init__(self, config):
        # Initialize the black-box 
        super().__init__(config)
        self.dataloader = DataLoader(config.dataset_config)

    
    def run(self):
        super().run()

        logger.info(f"running Simple iterative attack for {len(self.dataloader)} images")
        blackbox_loss = LossCallCounter(self.target_blackbox.loss, self.config.max_iter)

        for img_path, np_img in self.dataloader:
            logger.info(f"Running Simple iterative attack for {img_path}")
            _, img_fn = path.split(img_path)  # get image file name

            self.iterative_algorithm = BlackBoxIterativeAlgorithm.createBlackBoxAttackInstance(self.config.algorithm_name, np_img, blackbox_loss)

            running_attack = self.iterative_algorithm.run()

            i = 1
            while True:
                try:
                    pertubation = next(running_attack)
                    logger.info(f"Manager: Img: {img_fn}\t\tIter: {i}\tloss: {blackbox_loss.last_loss_val}")
                    i += 1
                except (MaxFunctionCallsExceededException, StopIteration):
                    break
            
            adv_img = np_img + pertubation
            utils.show_img(adv_img)

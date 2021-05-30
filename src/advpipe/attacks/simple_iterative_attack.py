from advpipe.attacks import Attack
from advpipe.log import logger
from typing import Union
from advpipe.attack_algorithms import BlackBoxIterativeAlgorithm
from advpipe.data_loader import DataLoader
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

        for img_path, np_img in self.dataloader:
            logger.info(f"Running Simple iterative attack for {img_path}")
            _, img_fn = path.split(img_path)

            self.iterative_algorithm = BlackBoxIterativeAlgorithm.createBlackBoxAttackInstance(self.config.algorithm_name, np_img)
            loss = self.target_blackbox.loss(np_img)

            logger.debug(f"np_img shape after first forward pass: {np_img.shape}")

            for i in range(self.config.max_iter):
                pertubed_img = np_img + self.iterative_algorithm.pertubation

                logger.info(f"Img: {img_fn}\t\tIter: {i}\tloss: {loss}")
                self.iterative_algorithm.step(loss)

                # new loss
                loss = self.target_blackbox.loss(pertubed_img)




            
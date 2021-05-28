import os
import numpy as np
from PIL import Image
from advpipe.log import logger
from advpipe import utils
import functools

class DataLoader:
    """Data iterator that returns all images from dataset directory as numpy tensors"""

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

        
        # convert data_dir to absolute path
        data_dir = dataset_config.data_dir
        
        # Windows users may have to tweak this
        if not (data_dir.startswith("/") or data_dir.startswith("~")):
            data_dir = os.path.join(utils.get_abs_module_path(), data_dir)


        self.dataset_list = list(
            map(functools.partial(os.path.join, data_dir), os.listdir(data_dir)))

        self.index = 0


    def __iter__(self):
        return self

    def __next__(self):
        try:
            img_path = self.dataset_list[self.index]
            np_img = self._load_image_to_numpy(img_path)
        except IndexError:
            raise StopIteration
        self.index += 1
        return (img_path, np_img)

    def __len__(self):
        return len(self.dataset_list)

    def _load_image_to_numpy(self, img_path):
        logger.debug(f"Dataloader - loading {img_path}")
        image = Image.open(img_path)
        # convert image to numpy array
        np_img = np.asarray(image)
        logger.debug(f"img dtype: {type(np_img)}")
        # summarize shape
        logger.debug(f"img shape: {np_img.shape}")
        return np_img

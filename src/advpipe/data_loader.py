from __future__ import annotations
import os
import skimage.transform
from advpipe import utils
from advpipe.imagenet_utils import is_organism, get_human_readable_label, get_imagenet_validation_label
from advpipe.log import logger
import torch
import functools
import numpy as np

# Typing and stuff
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import DatasetConfig, ImageNetValidationDatasetConfig
    from typing import Dict, Tuple, Sequence, Optional, Iterator
    from numpy import np


class DataLoader:
    """Data iterator that returns all images from dataset directory as numpy tensors"""

    dataset_config: DatasetConfig
    dataset_list: Sequence[str]

    @property
    def name(self) -> str:
        return self.dataset_config.name

    def __init__(self, dataset_config: DatasetConfig):
        self.dataset_config = dataset_config

        self.dataset_list = list(
            map(
                functools.partial(os.path.join, self.dataset_config.full_path),
                list(filter(utils.is_img_filename, os.listdir(self.dataset_config.full_path))),
            ))

        self.dataset_list.sort()

        self.index = 0


    @staticmethod
    def create_batches(loader: DataLoader, bs: int = 16) -> Iterator[Tuple[Sequence[str], torch.Tensor, torch.Tensor, Sequence[str]]]: 
        for batch in utils.batches(loader, bs):
            paths = [p for p, _, _, _ in batch]
            imgs = torch.stack([img for _, img, _, _ in batch])
            labels = torch.tensor([label for _, _, label, _ in batch], dtype=torch.long)
            human_readable_labels = [human_label for _, _, _, human_label in batch]
            yield (paths, imgs.to("cuda"), labels.to("cuda"), human_readable_labels)

    def __iter__(self) -> DataLoader:
        return self

    def __next__(self) -> Tuple[str, torch.Tensor, Optional[int], Optional[str]]:
        """Return (img_path, numpy_img, label)
        label == 0 -> object
        label == 1 -> organism

        if gold label isn't known, return None
        """

        try:
            if self.dataset_config.size_limit is not None and self.index >= self.dataset_config.size_limit:
                raise StopIteration
            img_path = self.dataset_list[self.index]
            torch_img = utils.load_image_to_torch(img_path)

        except IndexError:
            raise StopIteration
        self.index += 1
        return (img_path, torch_img, None, None)


    def __len__(self) -> int:
        return len(self.dataset_list)


class ImageNetValidationDataloader(DataLoader):
    dataset_config: ImageNetValidationDatasetConfig

    # Some images can be filtered out, so index isn't enough to keep track of total returned images
    image_counter: int = 0

    def __init__(self, dataset_config: ImageNetValidationDatasetConfig):
        super().__init__(dataset_config)

    def __next__(self) -> Tuple[str, np.ndarray, Optional[int], Optional[str]]:
        try:
            while True:
                if self.dataset_config.size_limit is not None and self.image_counter >= self.dataset_config.size_limit:
                    raise StopIteration
                img_path = self.dataset_list[self.index]
                img_fn = os.path.basename(img_path)

                torch_img = utils.load_image_to_torch(img_path)

                imagenet_id = get_imagenet_validation_label(img_fn)
                organism_label = int(is_organism(imagenet_id))
                human_readable = get_human_readable_label(imagenet_id)
                self.index += 1

                # organism_label == 0 is an object label, == 1 is an organism
                if self.dataset_config.load_only_organisms and organism_label == 0:
                    continue
                else:
                    logger.debug(
                        f"ImageNetValidationDataset loader: {img_fn}, label: {human_readable}  is_organism: {organism_label == 1}"
                    )
                    self.image_counter += 1
                    return (img_path, torch_img, organism_label, human_readable)
        except IndexError:
            raise StopIteration

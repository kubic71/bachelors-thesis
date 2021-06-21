from __future__ import annotations
import os
import skimage.transform
from advpipe import utils
from advpipe.imagenet_utils import is_organism
import functools

# Typing and stuff
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import DatasetConfig    # noqa: 402
    from typing import Dict, Tuple, Sequence
    from numpy import np


class DataLoader:
    """Data iterator that returns all images from dataset directory as numpy tensors"""

    resize: Tuple[int, int] | None
    dataset_config: DatasetConfig
    dataset_list: Sequence[str]

    def __init__(self, dataset_config: DatasetConfig, resize: Tuple[int, int] | None = (224, 224)):
        self.dataset_config = dataset_config
        self.resize = resize

        self.dataset_list = list(
            map(
                functools.partial(os.path.join, self.dataset_config.full_path),
                list(filter(utils.is_img_filename, os.listdir(self.dataset_config.full_path))),
            ))

        self.index = 0

    def __iter__(self) -> DataLoader:
        return self

    def __next__(self) -> Tuple[str, np.ndarray, int | None]:
        """Return (img_path, numpy_img, label)
        label == 0 -> object
        label == 1 -> organism

        if gold label isn't known, return None
        """

        try:
            img_path = self.dataset_list[self.index]
            np_img = self._transform(utils.load_image_to_numpy(img_path))
        except IndexError:
            raise StopIteration
        self.index += 1
        return (img_path, np_img, None)

    def _transform(self, np_img: np.ndarray) -> np.ndarray:
        if self.resize:
            np_img = skimage.transform.resize(np_img, output_shape=self.resize)
        return np_img

    def __len__(self) -> int:
        return len(self.dataset_list)


class DAmageNetDatasetLoader(DataLoader):
    labels: Dict[str, int]

    def __init__(self, dataset_config: DatasetConfig):
        super().__init__(dataset_config, resize=None)

        self.labels = {}
        with open(os.path.join(dataset_config.full_path, "val_damagenet.txt"), "r") as f:
            for line in f.read().strip().split("\n"):
                img_fn, imagenet_label = line.split()
                self.labels[img_fn] = int(is_organism(int(imagenet_label)))

    def __next__(self) -> Tuple[str, np.ndarray, int | None]:
        try:
            img_path = self.dataset_list[self.index]
            np_img = utils.load_image_to_numpy(img_path)
            label = self.labels[os.path.basename(img_path)]
        except IndexError:
            raise StopIteration
        self.index += 1

        return (img_path, np_img, label)

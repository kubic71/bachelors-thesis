from __future__ import annotations
import logging
import logging.config
from os import path
import imageio
from datetime import datetime
import pathlib

# TODO: cyclic dependency
# from advpipe import utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.blackbox.cloud import CloudLabels
    from logging import Logger
    import numpy as np
    from typing import Any, Text
    from typing_extensions import Literal


class CloudDataLogger:
    def __init__(self, img_log_path: str):
        self.img_log_path = img_log_path
        pathlib.Path(img_log_path).mkdir(parents=True, exist_ok=True)

    def save_img(self, np_img: np.ndarray, img_desc: str = "") -> str:
        if img_desc != "":
            img_desc = "_" + img_desc
        # save img
        timestamp = datetime.now().strftime("%H:%M:%S")
        img_filename = f"{timestamp}{img_desc}.png"

        img_full_path = path.join(self.img_log_path, img_filename)
        imageio.imwrite(img_full_path, np_img)

        # utils.show_img(np_img)

        return img_full_path

    def save_cloud_labels(self, classification_labels: CloudLabels, img_filename: str) -> None:
        """
        args:
            classification_labels: list of tuples (label, score)
            img_filename as returned by save_img method
        """

        # remove .png file extension
        labels_filename = img_filename[:-4] + ".txt"

        with open(labels_filename, "w") as f:
            for label, score in classification_labels:
                f.write(f"{label};{score}\n")


LOG_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "error": logging.ERROR}


class CustomLogger:
    """Wrapper around standard python logger"""
    def __init__(self, logger: Logger):
        self.logger = logger

    def set_level(self, level: Literal["debug", "info", "error"]) -> None:
        self.logger.setLevel(LOG_LEVELS[level])

    def __getattr__(self, attrname: Text) -> Any:
        return getattr(self.logger, attrname)


log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_file_path)

# create logger
logger = CustomLogger(logging.getLogger('advpipe'))

from advpipe import utils    # noqa: 402

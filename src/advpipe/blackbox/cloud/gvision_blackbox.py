from __future__ import annotations
import io
import os
import numpy as np
from google.cloud import vision
from advpipe import utils
from advpipe.blackbox.cloud import CloudBlackBox, CloudLabels

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import CloudBlackBoxConfig

API_KEY_FILENAME = os.path.join(utils.get_abs_module_path(), "keys/gvision_api_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY_FILENAME


class GVisionBlackBox(CloudBlackBox):
    name: str = "gvision"

    # For safety, max number of requests is by default set to conservative number
    def __init__(self, blackbox_config: CloudBlackBoxConfig):
        super().__init__(blackbox_config)

    def loss(self, pertubed_image: np.ndarray) -> float:
        labels_and_scores = self._gvision_classify(pertubed_image)
        return self._loss(labels_and_scores)

    def _gvision_classify(self, img: np.ndarray) -> CloudLabels:
        """Return the labels and scores by calling the cloud API"""
        img_filename = self.cloud_data_logger.save_img(img, "")

        # new vision.ImageAnnotatorClient instance must be created for each classifiied image
        client = vision.ImageAnnotatorClient()

        # Loads the image into memory
        with io.open(img_filename, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Performs label detection on the image file
        response = client.label_detection(image=image, max_results=100)

        result = CloudLabels([(annotation.description, annotation.score) for annotation in response.label_annotations])
        self.cloud_data_logger.save_cloud_labels(result, img_filename)

        return result

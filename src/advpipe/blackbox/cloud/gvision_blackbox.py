import io
import os
# Imports the Google Cloud client library
from google.cloud import vision
import numpy as np 
from PIL import Image
from advpipe import utils
from advpipe.log import logger
import random

API_KEY_FILENAME = os.path.join(utils.get_abs_module_path(), "keys/gvision_api_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = API_KEY_FILENAME




from advpipe.blackbox.cloud import CloudBlackBox 


class GVisionBlackBox(CloudBlackBox):
    """Google Vision API wrapper for RayS_Single.py"""


    # For safety, max number of requests is by default set to conservative number
    def __init__(self, blackbox_config):
        super().__init__(blackbox_config)


    def loss(self, pertubed_image):
        labels_and_scores = self._gvision_classify(pertubed_image)
        return self._loss(labels_and_scores)


    def _gvision_classify(self, img):
        """Return the labels and scores by calling the cloud API"""
        img_filename = logger.save_img(img, self.blackbox_config.name, "")

        # new vision.ImageAnnotatorClient instance must be created for each classifiied image
        client = vision.ImageAnnotatorClient()

        # Loads the image into memory
        with io.open(img_filename, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Performs label detection on the image file
        response = client.label_detection(image=image, max_results=100) # pylint: disable=no-member
        
        result = [(annotation.description, annotation.score) for annotation in response.label_annotations]
        return result

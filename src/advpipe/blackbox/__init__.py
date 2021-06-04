from advpipe.config_datamodel import TargetBlackBoxConfig
from abc import ABC, abstractmethod

class TargetBlackBox(ABC):
    """
    Generic BlackBox target model
    Abstraction for cloud MLaaS classifiers as well as for local pretrained CNNs
    """

    def __init__(self, blackbox_config : TargetBlackBoxConfig):
        self.blackbox_config = blackbox_config

    @abstractmethod
    def loss(self, pertubed_image):
        # Override this in Cloud/Local blackbox child class
        # In case of Cloud blackbox, query the API and pass it through the language-model translation mapping layer
        # In case of Local blackbox, pass split the ImageNet categories using imagenet_utils.py
        pass

        



from .local import LocalBlackBox
from .cloud import CloudBlackBox

TARGET_BLACKBOX_TYPES = {"local":LocalBlackBox,
    "cloud":CloudBlackBox 
}




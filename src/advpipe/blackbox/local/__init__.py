from advpipe.blackbox import TargetBlackBox
import numpy as np

from torchvision import models
import torch



class LocalModel():
    """
    Wrapper around different types of local models
    This wrapping is neccessary, because tensorflow nad pytorch models differ for example in the order of channels for images

    """
    def __init__(self):
        pass

    def __call__(self, np_img):
        """Take [W, H, C] numpy image in any resolution and return probabilities over ImageNet categories"""
        # Override in Pytorch/Tensorflow/Other Model class
        pass


class LocalBlackBox(TargetBlackBox):
    """Pretrained ImageNet model"""
    local_model : LocalModel

    def __init__(self, blackbox_config):
        super().__init__(blackbox_config)

    def loss(self, pertubed_image):
        probs = self.local_model([pertubed_image], )

    @staticmethod
    def getPytorchResnet18(blackbox_config):
        return PytorchResnet18(blackbox_config)

    
class PytorchModel(LocalModel):
    """Wrapper around pytorch ImageNet models"""

    def __init__(self, pytorch_model):
        super().__init__()
        self.pytorch_model = pytorch_model

    def __call__(self, np_img):
        np_img = self._preprocess(np_img)
        return np.array(self.pytorch_model([np_img])[0])

    def _preprocess(self, np_img):
        # TODO
        return np_img


class PytorchResnet18(LocalBlackBox):
    local_model = PytorchModel(models.resnet18(pretrained=True))
    def __init__(self, blackbox_config):
        super().__init__(blackbox_config)

class PytorchResnet50(LocalBlackBox):
    local_model = PytorchModel(models.resnet50(pretrained=True))
    def __init__(self, blackbox_config):
        super().__init__(blackbox_config)
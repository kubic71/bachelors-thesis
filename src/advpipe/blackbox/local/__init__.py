import numpy as np
from .imagenet_utils import get_organism_indeces, get_object_indeces, get_label
from torchvision import models
from advpipe.log import logger
import torch


#-----------Local Model----------
# - wraps different types of Local white-box models
# - actual white-box model instances are singletons to save memory

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

class PytorchModel(LocalModel):
    """Wrapper around pytorch ImageNet models"""

    def __init__(self, pytorch_model : torch.nn.Module, cuda=True):
        self.model = pytorch_model
        self.cuda = cuda
        if cuda:
            self.model.cuda()
        self.model.eval()

    def __call__(self, np_img):
        img_tensor = self._preprocess(np_img)
        return self.model(img_tensor).cpu().detach().numpy()[0]

    def _preprocess(self, np_img):
        np_img = np_img.transpose(2, 0, 1)
        tensor = torch.from_numpy(np_img).float().unsqueeze(0) # pylint: disable=no-member
        if self.cuda:
            tensor = tensor.cuda()
        return tensor



class PytorchResnet18(PytorchModel):
    # keep only one instance of Resnet-18 in memory
    _resnet18 = models.resnet18(pretrained=True)
    def __init__(self):
        super().__init__(self._resnet18)


class PytorchResnet50(PytorchModel):
    # keep only one instance of Resnet-50 in memory
    _resnet50 = models.resnet50(pretrained=True)
    def __init__(self):
        super().__init__(self._resnet50)




#------------Local BlackBox------------
# - BlackBox wrapping around LocalModel

from advpipe.blackbox import TargetBlackBox
class LocalBlackBox(TargetBlackBox):
    """Pretrained ImageNet model"""

    def __init__(self, blackbox_config, local_model : LocalModel):
        super().__init__(blackbox_config)
        self.local_model = local_model

    def loss(self, pertubed_image):
        probs = self.local_model(pertubed_image)

        organims_probs = probs.copy()
        object_probs = probs.copy()
        organims_probs[get_object_indeces()] = -np.inf
        object_probs[get_organism_indeces()] = -np.inf

        logger.debug(f"top-1 organism label: {get_label(np.argmax(organims_probs))}, prob: {np.max(organims_probs)}")
        logger.debug(f"top-1 object label: {get_label(np.argmax(object_probs))}, prob: {np.max(object_probs)}")

        return np.max(probs[get_organism_indeces()]) - np.max(probs[get_object_indeces()])




class PytorchBlackBoxResnet18(LocalBlackBox):
    def __init__(self, blackbox_config):
        super().__init__(blackbox_config, PytorchResnet18())


class PytorchBlackBoxResnet50(LocalBlackBox):
    def __init__(self, blackbox_config):
        super().__init__(blackbox_config, PytorchResnet50())


LOCAL_BLACKBOXES = {
    "resnet18": PytorchBlackBoxResnet18,
    "resnet50": PytorchBlackBoxResnet50
}
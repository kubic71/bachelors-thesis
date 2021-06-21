from __future__ import annotations
from abc import abstractmethod

import numpy as np
import torch
from advpipe.blackbox.loss import margin_loss
from advpipe.log import logger
from torchvision import models
import eagerpy as ep

from advpipe.imagenet_utils import get_label, get_object_indeces, get_organism_indeces
from advpipe.blackbox import TargetBlackBox

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import LocalBlackBoxConfig


# -----------Local Model----------
# - wraps different types of Local white-box models
# - actual white-box model instances are singletons to save memory
class LocalModel():
    """
    Wrapper around different types of local models
    This wrapping is neccessary, because tensorflow and pytorch models differ
    for example in the order of channels for images
    """

    @abstractmethod
    def __call__(self, np_img: np.ndarray) -> ep.types.NativeTensor:
        """Take [W, H, C] numpy image in any resolution and return probabilities over ImageNet categories"""
        # Override in Pytorch/Tensorflow/Other Model class


class PytorchModel(LocalModel):
    model: torch.nn.Module
    cuda: bool
    """Wrapper around pytorch ImageNet models"""
    def __init__(self, pytorch_model: torch.nn.Module, cuda: bool =True):
        self.model = pytorch_model
        self.cuda = cuda
        if cuda:
            self.model.cuda()
        self.model.eval()

    def __call__(self, np_img: np.ndarray) -> ep.types.NativeTensor:
        img_tensor = self._preprocess(np_img)
        return ep.astensor(self.model(img_tensor)[0])

    def _preprocess(self, np_img: np.ndarray) -> torch.Tensor:
        np_img = np_img.transpose(2, 0, 1)
        tensor = torch.from_numpy(np_img).float().unsqueeze(0)
        if self.cuda:
            tensor = tensor.cuda()
        return tensor


class PytorchResnet18(PytorchModel):
    # keep only one instance of Resnet-18 in memory
    _resnet18: torch.nn.Module = models.resnet18(pretrained=True)

    def __init__(self) -> None:
        super().__init__(PytorchResnet18._resnet18)


class PytorchResnet50(PytorchModel):
    # keep only one instance of Resnet-50 in memory
    _resnet50: torch.nn.Module = models.resnet50(pretrained=True)

    def __init__(self) -> None:
        super().__init__(PytorchResnet50._resnet50)


# ------------Local BlackBox------------
# - BlackBox wrapping around LocalModel
class LocalBlackBox(TargetBlackBox):
    """Pretrained ImageNet model"""
    local_model: LocalModel

    def __init__(self, blackbox_config: LocalBlackBoxConfig, local_model: LocalModel):
        super().__init__(blackbox_config)
        self.local_model = local_model

    def loss(self, pertubed_image: np.ndarray) -> float:
        probs = self.local_model(pertubed_image)
        probs = probs.numpy()


        organims_probs = probs.copy()
        object_probs = probs.copy()
        organims_probs[get_object_indeces()] = -np.inf
        object_probs[get_organism_indeces()] = -np.inf

        logger.debug(f"top-1 organism label: {get_label(np.argmax(organims_probs))}, prob: {np.max(organims_probs)}")
        logger.debug(f"top-1 object label: {get_label(np.argmax(object_probs))}, prob: {np.max(object_probs)}")

        if self.blackbox_config.loss.name == "margin_loss":
            loss_val = np.max(probs[get_organism_indeces()]) - np.max(probs[get_object_indeces()])
            margin_loss_val = margin_loss(loss_val, self.blackbox_config.loss.margin)
            return margin_loss_val
        else:
            raise NotImplementedError(f"Loss {self.blackbox_config.loss.name} is not implemented")


class PytorchBlackBoxResnet18(LocalBlackBox):
    def __init__(self, blackbox_config: LocalBlackBoxConfig):
        super().__init__(blackbox_config, PytorchResnet18())


class PytorchBlackBoxResnet50(LocalBlackBox):
    def __init__(self, blackbox_config: LocalBlackBoxConfig):
        super().__init__(blackbox_config, PytorchResnet50())


LOCAL_BLACKBOXES = {"resnet18": PytorchBlackBoxResnet18, "resnet50": PytorchBlackBoxResnet50}

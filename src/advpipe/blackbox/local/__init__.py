from __future__ import annotations
from abc import abstractmethod

import numpy as np
import torch
import functools
from efficientnet_pytorch import EfficientNet
from advpipe.blackbox.loss import margin_loss
from advpipe.log import logger
import torchvision
from torchvision import transforms
from advpipe import utils
from PIL import Image
import eagerpy as ep

from advpipe.imagenet_utils import get_human_readable_label, get_object_indeces, get_organism_indeces
from advpipe.blackbox import TargetBlackBox, BlackboxLabels

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import LocalBlackBoxConfig
    from typing import Tuple, Iterator, Dict, Any


class LocalLabels(BlackboxLabels):
    logits: np.ndarray

    organism_logits: np.ndarray
    object_logits: np.ndarray

    def __init__(self, logits: np.ndarray):
        self.logits = logits

        self.organism_logits = logits.copy()
        self.object_logits = logits.copy()
        self.organism_logits[get_object_indeces()] = -np.inf
        self.object_logits[get_organism_indeces()] = -np.inf

        self.top_org_id = np.argmax(self.organism_logits)
        self.top_obj_id = np.argmax(self.object_logits)

    def get_top_organism(self) -> Tuple[str, float]:
        return get_human_readable_label(self.top_org_id), self.organism_logits[self.top_org_id]

    def get_top_object(self) -> Tuple[str, float]:
        return get_human_readable_label(self.top_obj_id), self.object_logits[self.top_obj_id]

    def __iter__(self) -> Iterator[Tuple[str, float]]:
        for i, logit in self.logits:
            yield get_human_readable_label(i), logit

    def __str__(self) -> str:
        return "\n".join(list(map(lambda l_s: l_s[0] + ": " + str(l_s[1]), iter(self))))


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
        """Take [H, W, C] numpy image and return probabilities over ImageNet categories"""
        # Override in Pytorch/Tensorflow/Other Model classes

    @classmethod
    def getLocalModel(cls, model_name: str, resize_and_crop: bool) -> LocalModel:
        if model_name in PytorchModel.models:
            return PytorchModel(model_name, resize_and_center_crop=resize_and_crop)
        else:
            raise ValueError(f"Invalid local model name: {model_name}")


class PytorchModel(LocalModel):

    models: Dict[str, Any] = {
        "resnet18":
        functools.partial(torchvision.models.resnet18, pretrained=True),
        "resnet34":
        functools.partial(torchvision.models.resnet18, pretrained=True),
        "resnet50":
        functools.partial(torchvision.models.resnet50, pretrained=True),
        "resnet101":
        functools.partial(torchvision.models.resnet101, pretrained=True),
        "efficientnet-b0":
        functools.partial(EfficientNet.from_pretrained, model_name="efficientnet-b0", advprop=False),
        "efficientnet-b0-advtrain":
        functools.partial(EfficientNet.from_pretrained, model_name="efficientnet-b0", advprop=True),
        "efficientnet-b1":
        functools.partial(EfficientNet.from_pretrained, model_name="efficientnet-b1", advprop=False),
        "efficientnet-b1-advtrain":
        functools.partial(EfficientNet.from_pretrained, model_name="efficientnet-b1", advprop=True),
    }

    model: torch.nn.Module
    cuda: bool
    resize_and_center_crop: bool

    resize_and_center_crop_transform: transforms.Compose = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224),
         transforms.ToTensor()])
    normalize_transform: transforms.Normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    # transform: torchvision.
    """Wrapper around pytorch ImageNet models"""
    def __init__(self, pytorch_model_name: str, cuda: bool = True, resize_and_center_crop: bool = True):

        self.model = PytorchModel.models[pytorch_model_name]()
        self.cuda = cuda
        self.resize_and_center_crop = resize_and_center_crop

        # adversarially-trained models require slightly different normalization
        # taken from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/imagenet/main.py
        if pytorch_model_name.endswith("advtrain"):
            print("Adversarially-trained model")
            self.normalize_transform = transforms.Lambda(lambda img: img * 2.0 - 1.0)

        if cuda:
            self.model.cuda()
        self.model.eval()

    def __call__(self, np_img: np.ndarray) -> ep.types.NativeTensor:
        with torch.no_grad():    # type: ignore
            img_tensor = self._preprocess(np_img.copy())
            return ep.astensor(self.model(img_tensor)[0])

    def _normalize(self, torch_img: torch.Tensor) -> torch.Tensor:
        return self.normalize_transform(torch_img)    # type: ignore

    def _preprocess(self, np_img: np.ndarray) -> torch.Tensor:
        if self.resize_and_center_crop:
            pil_img = utils.convert_to_pillow(np_img)
            torch_img = PytorchModel.resize_and_center_crop_transform(pil_img)
        else:
            torch_img = torch.from_numpy(np_img.copy().transpose(2, 0, 1)).float()

        torch_img = self._normalize(torch_img).unsqueeze(0)

        if self.cuda:
            torch_img = torch_img.cuda()

        return torch_img    # type: ignore


# ------------Local BlackBox------------
# - BlackBox wrapping around LocalModel
class LocalBlackBox(TargetBlackBox):
    """Pretrained ImageNet model"""
    local_model: LocalModel
    blackbox_config: LocalBlackBoxConfig
    last_query_result: LocalLabels

    def __init__(self, blackbox_config: LocalBlackBoxConfig):
        super().__init__(blackbox_config)
        self.local_model = LocalModel.getLocalModel(blackbox_config.name, blackbox_config.resize_and_center_crop)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def loss(self, pertubed_image: np.ndarray) -> float:
        labels = self.classify(pertubed_image)
        top_org = labels.get_top_organism()
        top_obj = labels.get_top_object()
        logger.debug(f"top-1 organism label: {top_org[0]}, prob: {top_org[1]}")
        logger.debug(f"top-1 object label: {top_obj[0]}, prob: {top_obj[1]}")

        if self.blackbox_config.loss.name == "margin_loss":
            loss_val = top_org[1] - top_obj[1]
            margin_loss_val = margin_loss(loss_val, self.blackbox_config.loss.margin)
            return margin_loss_val
        else:
            raise NotImplementedError(f"Loss {self.blackbox_config.loss.name} is not implemented")

    def classify(self, pertubed_image: np.ndarray) -> LocalLabels:
        probs = self.local_model(self._preprocess(pertubed_image))
        self.last_query_result = LocalLabels(probs.numpy())
        return self.last_query_result
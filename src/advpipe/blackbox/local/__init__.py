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
import kornia as K
import functools

from advpipe.imagenet_utils import get_human_readable_label, get_object_indeces, get_organism_indeces
from advpipe.blackbox import TargetBlackBox, BlackboxLabels

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import LocalBlackBoxConfig
    from typing import Tuple, Iterator, Dict, Any, Callable, Text


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
# - TODO: local models aren't quite blackboxes, probably should be in different package
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
        ...

    @abstractmethod
    def grad(self, np_img: np.ndarray, loss: str) -> np.ndarray:
        """Take [H, W, C] numpy image and return gradient of organism-loss with respect to the input image"""
        # Override in Pytorch/Tensorflow/Other Model classes
        ...

    @classmethod
    def getLocalModel(cls, model_name: str, resize_and_crop: bool = False) -> LocalModel:
        if model_name in PytorchModel.models:
            return PytorchModel(model_name, resize_and_center_crop=resize_and_crop)
        else:
            raise ValueError(f"Invalid local model name: {model_name}")


class WhiteBoxSurrogate:
    loss: str
    local_model: LocalModel

    def __init__(self, local_model: LocalModel, loss: str):
        self.local_model = local_model
        self.loss = loss

    def grad(self, np_img: np.ndarray) -> np.ndarray:
        return self.local_model.grad(np_img, self.loss)

    def __getattr__(self, attrname: Text) -> Any:
        return getattr(self.local_model, attrname)

def get_pytorch_model_map() -> Dict[str, Any]:
    torch_models = {}

    # standard resnet18, 34, 50 and 101
    for resnet_variant in [18, 34, 50, 101]:
        resnet_name = f"resnet{resnet_variant}"
        torch_models[resnet_name] = functools.partial(getattr(torchvision.models, resnet_name), pretrained=True)

    # efficientnet with/without adversarial training

    for efnet_variant in [0, 1, 2, 3, 4, 5, 6, 7]:
        efnet_name = f"efficientnet-b{efnet_variant}"

        torch_models[efnet_name] = functools.partial(EfficientNet.from_pretrained, model_name=efnet_name, advprop=False)
        torch_models[f"{efnet_name}-advtrain"] = functools.partial(EfficientNet.from_pretrained, model_name=efnet_name, advprop=True)
    
    return torch_models


class PytorchModel(LocalModel):
    models: Dict[str, Any] = get_pytorch_model_map()
    model: torch.nn.Module
    cuda: bool
    resize_and_center_crop: bool
    adv_train: bool

    gold_object_dist: torch.Tensor

    # resize_and_center_crop_transform: transforms.Compose = transforms.Compose(
    # [transforms.Resize(256), transforms.CenterCrop(224),
    #  transforms.ToTensor()])

    def _resize_and_center_crop_transform(self, torch_img: torch.Tensor) -> torch.Tensor:
        return K.center_crop(K.resize(torch_img, size=256), size=(224, 224))

    # transform: torchvision.
    """Wrapper around pytorch ImageNet models"""

    def __init__(self, pytorch_model_name: str, cuda: bool = True, resize_and_center_crop: bool = True):

        self.model = PytorchModel.models[pytorch_model_name]()
        self.cuda = cuda
        self.resize_and_center_crop = resize_and_center_crop
        self.adv_train = pytorch_model_name.endswith("advtrain")

        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        # Debug
        if self.adv_train:
            print(f"{pytorch_model_name} is adversarially-trained model. Using different preprocessing normalization")

        self.organism_indeces = torch.tensor(get_organism_indeces(), dtype=torch.long)
        self.object_indeces = torch.tensor(get_object_indeces(), dtype=torch.long)

        self.gold_object_dist = torch.zeros(1000)
        self.gold_object_dist[self.object_indeces] = 1
        self.gold_object_dist /= self.gold_object_dist.sum()

        if cuda:
            self.model.cuda()
            self.gold_object_dist = self.gold_object_dist.cuda()
            self.organism_indeces = self.organism_indeces.cuda()
            self.object_indeces = self.object_indeces.cuda()

        # switch-off dropout, set batch-norm to eval regime
        self.model.eval()

    def normalize(self, torch_img: torch.Tensor) -> torch.Tensor:
        # adversarially-trained models require slightly different normalization
        # taken from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/imagenet/main.py
        if self.adv_train:
            return torch_img * 2.0 - 1.0
        else:
            return K.normalize(torch_img,
                               mean=torch.tensor([0.485, 0.456, 0.406]),
                               std=torch.tensor([0.229, 0.224, 0.225]))

    def __call__(self, np_img: np.ndarray) -> ep.types.NativeTensor:
        with torch.no_grad():    # type: ignore
            torch_img = self._preprocess(self._convert_to_tensor(np_img))
            return ep.astensor(self.model(torch_img)[0])

    def _convert_to_tensor(self, np_img: np.ndarray) -> torch.Tensor:
        # torch does wierd in-place things with tensors loaded from numpy
        np_img = np_img.copy()

        if np_img.shape[0] != 3:
            # convert np_img to [C, H, W] format
            np_img = np_img.transpose(2, 0, 1)

        torch_img = torch.from_numpy(np_img).float().unsqueeze(0)
        if self.cuda:
            return torch_img.cuda()
        else:
            return torch_img

    def grad(self, np_img: np.ndarray, loss: str = "cross-entropy") -> np.ndarray:
        """compute gradient of the organism-score with respect to the input numpy image
            loss:
                - 'cross-entropy' - KL-divergence with the organism gold tensor
                - 'cw' - max(organism_logit) - max(object_logit)
        """
        input_img = self._convert_to_tensor(np_img)
        input_img.requires_grad = True

        output = self.model(self._preprocess(input_img))

        if loss == "cross-entropy":
            kl_div = torch.nn.KLDivLoss()

            # KL-divergence to uniform distribution over object labels
            loss_val = kl_div(self.log_softmax(output), self.gold_object_dist.unsqueeze(0))

        elif loss == "cw":
            loss_val = output[0, self.organism_indeces].max() - output[0, self.object_indeces].max()
        else:
            raise ValueError(f"LocalModel.grad: invalid loss - {loss}")

        self.model.zero_grad()
        loss_val.backward()

        np_grad: np.ndarray = input_img.grad.squeeze().detach().cpu().numpy()
        if np_img.shape[0] == 3: # channels-first
            return np_grad
        else:
            # convert-back to channels-last
            return np_grad.transpose(1, 2, 0)


    def _preprocess(self, torch_img: torch.Tensor) -> torch.Tensor:
        if self.resize_and_center_crop:
            torch_img = self._resize_and_center_crop_transform(torch_img)

        return self.normalize(torch_img)


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
from __future__ import annotations
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
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
    from advpipe.config_datamodel import LocalModelConfig
    from typing import Tuple, Iterator, Dict, Any, Callable, Text, Sequence, List
    from typing_extensions import Literal




class LocalLabels(BlackboxLabels):
    logits: np.ndarray
    organism_logits: np.ndarray
    object_logits: np.ndarray

    def __init__(self, logits: torch.Tensor):
        assert len(logits.shape) == 2
        self.logits = logits.clone().detach().cpu().numpy()

        # We need to make copy with the same shape to get the the index to the full logit array
        self.organism_logits = self.logits.copy()
        self.object_logits = self.logits.copy()

        self.organism_logits[:, get_object_indeces()] = -np.inf
        self.object_logits[:, get_organism_indeces()] = -np.inf

        self.top_org_ids = np.argmax(self.organism_logits, axis=1)
        self.top_obj_ids = np.argmax(self.object_logits, axis=1)

    def get_top_organism_labels(self) -> Sequence[str]:
        return [get_human_readable_label(org_id) for org_id in self.top_org_ids]
    
    def get_top_organism_logits(self) -> torch.Tensor:
        return torch.tensor(self.organism_logits.max(axis=1))

    def get_top_object_labels(self) -> Sequence[str]:
        return [get_human_readable_label(obj_id) for obj_id in self.top_obj_ids]
    
    def get_top_object_logits(self) -> torch.Tensor:
        return torch.tensor(self.object_logits.max(axis=1))

    def __repr__(self) -> str:
        res = []
        for org_label, org_logit, obj_label, obj_logit in zip(self.get_top_organism_labels(), self.get_top_organism_logits(), self.get_top_object_labels(), self.get_top_object_logits()):
            res.append(f"top organism: {org_label}, {org_logit},\ttop object: {obj_label}, {obj_logit}")
        return "\n".join(res)


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
        torch_models[f"{efnet_name}-advtrain"] = functools.partial(EfficientNet.from_pretrained,
                                                                   model_name=efnet_name,
                                                                   advprop=True)

    return torch_models


PYTORCH_MODELS: Dict[str, Any] = get_pytorch_model_map()


def get_pytorch_model(model_name: str) -> torch.nn.Module:
    return PYTORCH_MODELS[model_name]() # type: ignore


def normalize(torch_img: torch.Tensor, adv_train: bool = False) -> torch.Tensor:
    # adversarially-trained models require slightly different normalization
    # taken from: https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/imagenet/main.py
    if adv_train:
        return torch_img * 2.0 - 1.0
    else:
        return K.normalize(torch_img, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))


def resize_and_center_crop_transform(torch_img: torch.Tensor) -> torch.Tensor:
    return K.center_crop(K.resize(torch_img, size=256), size=(224, 224))


organism_indeces_tensor: torch.Tensor = torch.tensor(get_organism_indeces(), dtype=torch.long)
object_indeces_tensor: torch.Tensor = torch.tensor(get_object_indeces(), dtype=torch.long)


def imagenet_logits_to_organism_probs(logits: torch.Tensor) -> torch.Tensor:
    org_indeces = organism_indeces_tensor.to(logits.device)
    obj_indeces = object_indeces_tensor.to(logits.device)
    # sum up the probabilities corresponding to organisms and objects
    imagenet_probs = F.softmax(logits, dim=1)
    organism_probs = imagenet_probs[:, org_indeces].sum(dim=1)
    object_probs = imagenet_probs[:, obj_indeces].sum(dim=1)
    return torch.stack([object_probs, organism_probs], dim=1)


def imagenet_logits_to_simulated_organism_logits(logits: torch.Tensor) -> torch.Tensor:
    organism_probs = imagenet_logits_to_organism_probs(logits)
    return torch.log(organism_probs)


def imagenet_logits_to_max_logits(logits: torch.Tensor) -> torch.Tensor:
    org_indeces = organism_indeces_tensor.to(logits.device)
    obj_indeces = object_indeces_tensor.to(logits.device)
    # sum up the probabilities corresponding to organisms and objects
    organism_logits = logits[:, org_indeces]
    object_logits = logits[:, obj_indeces]
    return torch.stack([object_logits.max(dim=1).values, organism_logits.max(dim=1).values], dim=1)


class PytorchOrganismClassifier(torch.nn.Module):
    """ wraps standard ImageNet pytorch model and maps its outputs to {object, organism}
        args:
            model: PytorchModel
            output: Literal['logits', 'probs', 'max_logits'] = 'logits'
                - output == 'probs':
                    - the output of model is passed through softmax and probabilies of all object classes are summed up, as well as the probabilities of all organisms

                - output == 'logits':
                    - simulated logits are computed from summed-up probabilities by taking their logarithm.
                    - This is sometimes necessary, because for example foolbox expects the target model to output logits.
                    - This is useful when dealing with adversarial algorithms using cross-entropy loss (FGSM, PGD)
                    - Note that softmax is shift-invariant and therefore it's inverse is not unique.
                
                - output == 'max_logits':
                    - outputs [max_object_logit, max_organism_logit]
                    - useful when attacking algorithm is using CW-loss

                - in each case, the output tensor has shape (batch_size, 2)
    """
    def __init__(self, model: torch.nn.Module, output: Literal["logits", "probs", "max_logits"] = "logits"):
        super(PytorchOrganismClassifier, self).__init__()    # type: ignore

        self.model = model
        self.output = output

        if output == "logits":
            self.output_mapping = imagenet_logits_to_simulated_organism_logits
        elif output == "probs":
            self.output_mapping = imagenet_logits_to_organism_probs
        elif output == "max_logits":
            self.output_mapping = imagenet_logits_to_max_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.output_mapping(self.model(x))

        # DEBUG
        # print(f"PytorchOrganismClassifier output (output type - {self.output}):")
        # print(result)

        return result


class PytorchModel(torch.nn.Module):
    """Wrapper around various pytorch ImageNet models"""

    model: torch.nn.Module
    adv_train: bool
    device: str

    def __init__(self,
                 pytorch_model_name: str,
                 device: str = "cuda",
                 preprocess_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = []):
        super(PytorchModel, self).__init__()    # type: ignore

        self.model = get_pytorch_model(pytorch_model_name).to(device)

        # switch-off dropout, set batch-norm to eval regime
        self.model.eval()

        self.adv_train = pytorch_model_name.endswith("advtrain")
        self.device = device

        self.preprocess_transforms = preprocess_transforms + [functools.partial(normalize, adv_train=self.adv_train)
                                                              ]    # every model needs normalization
        self.preprocess = torchvision.transforms.Compose(self.preprocess_transforms)

        # Debug
        if self.adv_train:
            logger.debug(
                f"{pytorch_model_name} is adversarially-trained model. Using different preprocessing normalization")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = self.preprocess(x.to(self.device))
        return self.model(inp)    # type: ignore

    # def _convert_to_tensor(self, np_img: np.ndarray) -> torch.Tensor:
    #     # torch does wierd in-place things with tensors loaded from numpy
    #     assert len(np_img.shape) == 4
    #     np_img = np_img.copy()

    #     if np_img.shape[1] != 3:
    #         # convert np_img to [C, H, W] format
    #         np_img = np_img.transpose(0, 3, 1, 2)

    #     return torch.from_numpy(np_img).float().to(self.device)

    # def grad(self, np_img: np.ndarray, loss: str = "cross-entropy") -> np.ndarray:
    #     """compute gradient of the organism-score with respect to the input numpy image
    #         loss:
    #             - 'cross-entropy' - KL-divergence with the organism gold tensor
    #             - 'cw' - max(organism_logit) - max(object_logit)
    #     """
    #     input_img = self._convert_to_tensor(np_img)
    #     input_img.requires_grad = True

    #     output = self.model(self._preprocess(input_img))

    #     if loss == "cross-entropy":
    #         kl_div = torch.nn.KLDivLoss()

    #         # KL-divergence to uniform distribution over object labels
    #         loss_val = kl_div(self.log_softmax_layer(output), self.gold_object_dist.unsqueeze(0))

    #     elif loss == "cw":
    #         loss_val = output[0, self.organism_indeces].max() - output[0, self.object_indeces].max()
    #     else:
    #         raise ValueError(f"LocalModel.grad: invalid loss - {loss}")

    #     self.model.zero_grad()
    #     loss_val.backward()

    #     np_grad: np.ndarray = input_img.grad.squeeze().detach().cpu().numpy()
    #     if np_img.shape[0] == 3:    # channels-first
    #         return np_grad
    #     else:
    #         # convert-back to channels-last
    #         return np_grad.transpose(1, 2, 0)


# ------------Local BlackBox------------
# - BlackBox wrapping around LocalModel
class LocalModel(TargetBlackBox):
    """Pretrained ImageNet model"""
    local_model: PytorchModel
    blackbox_config: LocalModelConfig
    last_query_result: LocalLabels

    def __init__(self, blackbox_config: LocalModelConfig):
        super(LocalModel, self).__init__(blackbox_config)
        self.local_model = PytorchModel(
            blackbox_config.name,
            preprocess_transforms=[resize_and_center_crop_transform] if blackbox_config.resize_and_center_crop else [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.local_model(x)
        self.last_query_result = LocalLabels(logits)
        return logits  # type: ignore

    def loss(self, pertubed_image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): # type: ignore
            logits = self.local_model(pertubed_image)
            self.last_query_result = LocalLabels(logits)
            
            return (self.last_query_result.get_top_organism_logits() - self.last_query_result.get_top_object_logits()) + self.blackbox_config.loss.margin # type: ignore
from __future__ import annotations
from advpipe.blackbox.cloud import GVisionBlackBox
from advpipe.log import CloudDataLogger
from typing_extensions import Literal
from advpipe import utils
from abc import ABC, abstractmethod
from munch import Munch
import kornia as K

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.blackbox.local import LocalModel, DummyModel
    from advpipe.blackbox.cloud import CloudBlackBox
    from advpipe.blackbox import TargetModel
    from typing import Type, Dict, Optional, Tuple, Text, Callable, Sequence, List
    import torch

MODEL_TYPE = Literal["cloud", "local", "ensemble", "dummy"]


class TargetModelConfig(ABC):
    name: str
    loss: Munch
    model_type: MODEL_TYPE

    def __init__(self, target_model_config: Munch):
        self.name = target_model_config.name

        default_loss = Munch.fromDict({"name": "margin_loss", "margin": 0})
        self.loss = utils.get_config_attr(target_model_config, "loss", default_loss)



    @staticmethod
    def loadFromYamlConfig(target_model_config: Munch) -> TargetModelConfig:
        """Factory method for TargetModelConfig

        returns: either LocalModelConfig or CloudBlackBoxConfig depending on model_type YAML field
        """
        m_type = target_model_config.model_type
        if m_type == CloudBlackBoxConfig.model_type:
            return CloudBlackBoxConfig(target_model_config)
        elif m_type == LocalModelConfig.model_type:
            return LocalModelConfig(target_model_config)
        elif m_type == EnsembleConfig.model_type:
            return EnsembleConfig(target_model_config)
        else:
            raise ValueError(f"Invalid model type: {m_type}")

    @abstractmethod
    def getModelInstance(self) -> TargetModel:
        ...


def get_image_augmentation(config: Munch) -> Tuple[str, Callable[[torch.Tensor], torch.Tensor]]:
    if config.name == "box-blur":
        return (f"blur-{utils.zfill_align_float(config.kernel_size)}", K.augmentation.RandomBoxBlur(kernel_size=(config.kernel_size, config.kernel_size), border_type="reflect", normalized=True, return_transform=False, same_on_batch=True, p=1))
    elif config.name == "gaussian-noise":
        return (f"noise-{utils.zfill_align_float(config.std)}", K.augmentation.RandomGaussianNoise(mean=0.0, std=config.std/255, return_transform=False, same_on_batch=True, p=1))
    elif config.name == "elastic":
        return (f"elastic-{utils.zfill_align_float(config.alpha)}" , K.augmentation.RandomElasticTransform(kernel_size=(63, 63), sigma=(32.0, 32.0), alpha=(config.alpha, config.alpha), align_corners=False, mode='bilinear', return_transform=False, same_on_batch=True, p=1))
    else:
        raise NotImplementedError


class EnsembleConfig(TargetModelConfig):
    model_type: MODEL_TYPE = "ensemble"
    model_configs: Sequence[TargetModelConfig]
    name: str 


    def __init__(self, ensemble_config: Munch):
        self.model_configs = []

        names = []
        for model_config in ensemble_config.model_configs:
            self.model_configs.append(TargetModelConfig.loadFromYamlConfig(model_config))
            names.append(model_config.name)
        self.name = ",".join(names)
        self.loss = ensemble_config.loss


        self.augmentations, aug_desc = load_augmentations(ensemble_config)
        if aug_desc != "":
            self.name += "." +  aug_desc

        self.output_mapping = ensemble_config.output_mapping


    def getModelInstance(self) -> EnsembleModel:
        model = EnsembleModel(self)
        model.cuda()
        model.eval()
        return model

def load_augmentations(model_conf: Munch) -> Tuple[List, str]:
    augmentations = [] # type: ignore
    augmentations_desc = []
    for aug_conf in utils.get_config_attr(model_conf, "augmentations", []):
        desc, func = get_image_augmentation(aug_conf)
        augmentations_desc.append(desc)
        augmentations.append(func.cuda()) # type: ignore

    return augmentations, ",".join(augmentations_desc)


class LocalModelConfig(TargetModelConfig):
    model_type: MODEL_TYPE = "local"

    # taken from: https://jhui.github.io/2018/02/09/PyTorch-Data-loading-preprocess_torchvision/
    # standard ImageNet Resize(256) and CenterCrop(224)
    resize_and_center_crop: bool = False
    output_mapping: Literal["logits", "probs", "organism_margin"]

    augmentations: List[Callable[[torch.Tensor], torch.Tensor]]

    def __init__(self, local_model_config: Munch):
        super().__init__(local_model_config)
        self.resize_and_center_crop = utils.get_config_attr(local_model_config, "resize_and_center_crop",
                                                            self.resize_and_center_crop)

        self.augmentations, aug_desc = load_augmentations(local_model_config)
        if aug_desc != "":
            self.name += "." +  aug_desc

        self.output_mapping = utils.get_config_attr(local_model_config, "output_mapping", "organism_margin")

    def getModelInstance(self) -> LocalModel:
        model = LocalModel(self)
        model.cuda()
        model.eval()
        return model


# AUGMENTATIONS = {"box-blur": RandomBoxBlur(kernel_size=(3, 3), border_type='reflect', normalized=True, return_transform=False, same_on_batch=True, p=1) }
# class AugmentationConfig():
    # def __init__(self, augment_conf: Munch):
        # pass



class DummySurrogateConfig(TargetModelConfig):
    model_type: MODEL_TYPE = "dummy"
    name: str = "dummy_model_name"

    def __init__(self) -> None:
        pass

    def getModelInstance(self) -> DummyModel:
        return DummyModel(self)


class CloudBlackBoxConfig(TargetModelConfig):
    model_type: MODEL_TYPE = "cloud"
    cloud_data_logger: CloudDataLogger

    def __init__(self, cloud_blackbox_config: Munch):
        self.blackbox_class_ref = CLOUD_BLACKBOXES[cloud_blackbox_config.name]
        super().__init__(cloud_blackbox_config)

        # Default cloud data logger
        self.cloud_data_logger = CloudDataLogger("/tmp/cloud_data")

    def getModelInstance(self) -> CloudBlackBox:
        return CLOUD_BLACKBOXES[self.name](self)


from advpipe.blackbox.local import LocalModel, EnsembleModel, DummyModel

CLOUD_BLACKBOXES = {GVisionBlackBox.name: GVisionBlackBox}
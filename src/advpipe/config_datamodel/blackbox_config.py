from __future__ import annotations
from advpipe.blackbox.cloud import GVisionBlackBox
from advpipe.log import CloudDataLogger
from typing_extensions import Literal
from advpipe import utils
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.blackbox.local import LocalModel, DummyModel
    from advpipe.blackbox.cloud import CloudBlackBox
    from advpipe.blackbox import TargetModel
    from munch import Munch
    from typing import Type, Dict, Optional, Tuple, Text

MODEL_TYPE = Literal["cloud", "local", "dummy"]


class TargetModelConfig(ABC):
    name: str
    loss: Munch
    model_type: MODEL_TYPE

    def __init__(self, target_model_config: Munch):
        self.name = target_model_config.name
        self.loss = target_model_config.loss

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
        else:
            raise ValueError(f"Invalid model type: {m_type}")

    @abstractmethod
    def getModelInstance(self) -> TargetModel:
        ...


class LocalModelConfig(TargetModelConfig):
    model_type: MODEL_TYPE = "local"

    # taken from: https://jhui.github.io/2018/02/09/PyTorch-Data-loading-preprocess_torchvision/
    # standard ImageNet Resize(256) and CenterCrop(224)
    resize_and_center_crop: bool = False
    output_mapping: Literal["logits", "probs", "organism_margin"]

    def __init__(self, local_model_config: Munch):
        super().__init__(local_model_config)
        self.resize_and_center_crop = utils.get_config_attr(local_model_config, "resize_and_center_crop",
                                                            self.resize_and_center_crop)

        self.output_mapping = local_model_config.output_mapping

    def getModelInstance(self) -> LocalModel:
        model = LocalModel(self)
        model.cuda()
        model.eval()
        return model


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


from advpipe.blackbox.local import LocalModel, DummyModel

CLOUD_BLACKBOXES = {GVisionBlackBox.name: GVisionBlackBox}
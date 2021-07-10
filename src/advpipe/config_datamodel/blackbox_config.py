from __future__ import annotations
from advpipe.blackbox.cloud import GVisionBlackBox
from advpipe.log import CloudDataLogger
from typing_extensions import Literal
from advpipe import utils
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.blackbox.local import LocalModel
    from advpipe.blackbox.cloud import CloudBlackBox
    from advpipe.blackbox import TargetBlackBox
    from munch import Munch
    from typing import Type, Dict, Optional, Tuple, Text

BLACKBOX_TYPE = Literal["cloud", "local"]


class TargetBlackBoxConfig(ABC):
    name: str
    loss: Munch
    blackbox_type: BLACKBOX_TYPE

    def __init__(self, target_blackbox_config: Munch):
        self.name = target_blackbox_config.name
        self.loss = target_blackbox_config.loss

    @staticmethod
    def loadFromYamlConfig(target_blackbox_config: Munch) -> TargetBlackBoxConfig:
        """Factory method for TargetBlackBoxConfig

        returns: either LocalBlackBoxConfig or CloudBlackBoxConfig depending on blackbox_type YAML field
        """
        b_type = target_blackbox_config.blackbox_type
        if b_type == "cloud":
            return CloudBlackBoxConfig(target_blackbox_config)
        elif b_type == "local":
            return LocalModelConfig(target_blackbox_config)
        else:
            raise ValueError(f"Invalid blackbox type: {b_type}")

    @abstractmethod
    def getBlackBoxInstance(self) -> TargetBlackBox:
        ...


class LocalModelConfig(TargetBlackBoxConfig):
    blackbox_type: BLACKBOX_TYPE = "local"

    # taken from: https://jhui.github.io/2018/02/09/PyTorch-Data-loading-preprocess_torchvision/
    # standard ImageNet Resize(256) and CenterCrop(224)
    resize_and_center_crop: bool = False

    def __init__(self, local_blackbox_config: Munch):
        super().__init__(local_blackbox_config)
        self.resize_and_center_crop = utils.get_config_attr(local_blackbox_config, "resize_and_center_crop",
                                                            self.resize_and_center_crop)

    def getBlackBoxInstance(self) -> LocalModel:
        return LocalModel(self)



class CloudBlackBoxConfig(TargetBlackBoxConfig):
    blackbox_type: BLACKBOX_TYPE = "cloud"
    cloud_data_logger: CloudDataLogger

    def __init__(self, cloud_blackbox_config: Munch):
        self.blackbox_class_ref = CLOUD_BLACKBOXES[cloud_blackbox_config.name]
        super().__init__(cloud_blackbox_config)

        # Default cloud data logger
        self.cloud_data_logger = CloudDataLogger("/tmp/cloud_data")

    def getBlackBoxInstance(self) -> CloudBlackBox:
        return CLOUD_BLACKBOXES[self.name](self)


from advpipe.blackbox.local import LocalModel

CLOUD_BLACKBOXES = {GVisionBlackBox.name: GVisionBlackBox}
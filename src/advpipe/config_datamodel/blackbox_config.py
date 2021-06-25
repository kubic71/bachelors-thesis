from __future__ import annotations
from advpipe.blackbox.local import PytorchBlackBoxResnet18, PytorchBlackBoxResnet50
from advpipe.blackbox.cloud import GVisionBlackBox
from advpipe.log import CloudDataLogger
from typing_extensions import Literal
from advpipe import utils
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.attack_algorithms import BlackBoxAlgorithm
    from advpipe.blackbox.local import LocalBlackBox
    from advpipe.blackbox.cloud import CloudBlackBox
    from advpipe.blackbox import TargetBlackBox
    from munch import Munch
    from typing import Type, Dict, Optional, Tuple

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
            return LocalBlackBoxConfig(target_blackbox_config)
        else:
            raise ValueError(f"Invalid blackbox type: {b_type}")

    @abstractmethod
    def getBlackBoxInstance(self) -> TargetBlackBox:
        ...


class LocalBlackBoxConfig(TargetBlackBoxConfig):
    blackbox_type: BLACKBOX_TYPE = "local"

    # taken from: https://jhui.github.io/2018/02/09/PyTorch-Data-loading-preprocess_torchvision/
    # standard ImageNet Resize(256) and CenterCrop(224)
    resize_and_center_crop: bool = True

    # standard ImageNet normalization for the CNNs
    normalize: Optional[Tuple[Tuple[float, float, float], Tuple[float, float,
                                                       float]]] = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __init__(self, local_blackbox_config: Munch):
        super().__init__(local_blackbox_config)
        self.resize_and_center_crop = utils.get_config_attr(local_blackbox_config, "resize_and_center_crop",
                                                            self.resize_and_center_crop)

        if not utils.get_config_attr(local_blackbox_config, "normalize", True):
            self.normalize = None

    def getBlackBoxInstance(self) -> LocalBlackBox:
        return LOCAL_BLACKBOXES[self.name](self)


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


LOCAL_BLACKBOXES = {
    PytorchBlackBoxResnet18.name: PytorchBlackBoxResnet18,
    PytorchBlackBoxResnet50.name: PytorchBlackBoxResnet50
}
CLOUD_BLACKBOXES = {GVisionBlackBox.name: GVisionBlackBox}
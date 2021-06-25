from __future__ import annotations
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import AttackRegimeConfig
    from advpipe.blackbox import TargetBlackBox
    from advpipe.data_loader import DataLoader


class AttackRegime(ABC):
    target_blackbox: TargetBlackBox
    regime_config: AttackRegimeConfig
    dataloader: DataLoader

    def __init__(self, attack_regime_config: AttackRegimeConfig):
        self.regime_config = attack_regime_config
        self.dataloader = self.regime_config.dataset_config.getDatasetInstance()

        # Initialize the connection to the target blackbox
        self.target_blackbox = self.regime_config.target_blackbox_config.getBlackBoxInstance()

    @abstractmethod
    def run(self) -> None:
        ...


from .simple_iterative_regime import SimpleIterativeRegime
from .simple_transfer_regime import SimpleTransferRegime
from __future__ import annotations
from advpipe.data_loader import DataLoader
from abc import ABC, abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.config_datamodel import AttackRegimeConfig
    from advpipe.blackbox import TargetBlackBox


class Attack(ABC):
    target_blackbox: TargetBlackBox
    regime_config: AttackRegimeConfig

    def __init__(self, attack_regime_config: AttackRegimeConfig):
        self.regime_config = attack_regime_config
        self.dataloader = DataLoader(self.regime_config.dataset_config)

        # Initialize the connection to the target blackbox
        self.target_blackbox = self.regime_config.target_blackbox_config.getBlackBoxInstance()

    @abstractmethod
    def run(self) -> None:
        ...

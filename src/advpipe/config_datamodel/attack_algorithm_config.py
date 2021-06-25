from __future__ import annotations
from abc import ABC, abstractmethod
from advpipe.attack_algorithms import Norm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.attack_algorithms import BlackBoxAlgorithm
    from advpipe.blackbox import TargetBlackBox
    from munch import Munch
    import numpy as np


class AttackAlgorithmConfig(ABC):
    name: str
    norm: Norm
    epsilon: float

    def __init__(self, attack_regime_config: Munch):
        self.attack_regime_config = attack_regime_config
        self.norm = Norm(attack_regime_config.norm)
        self.epsilon = attack_regime_config.epsilon

    @staticmethod
    def loadFromYamlConfig(attack_regime_config: Munch, algorithm_config: Munch) -> AttackAlgorithmConfig:
        if algorithm_config.name not in ATTACK_ALGORITHM_CONFIGS:
            raise ValueError(f"Invalid attack algorithm: {algorithm_config.name}")

        return ATTACK_ALGORITHM_CONFIGS[algorithm_config.name](attack_regime_config, algorithm_config)

    @abstractmethod
    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> BlackBoxAlgorithm:
        """Factory method creating AdvPipe attack experiment objects"""


class PassthroughTransferAlgorithmConfig(AttackAlgorithmConfig):
    name: str = "passthrough"

    def __init__(self, attack_regime_config: Munch, attack_config: Munch):
        super().__init__(attack_regime_config)

    def getAttackAlgorithmInstance(self, image: np.ndarray, _: LossCallCounter) -> PassthroughTransferAttackAlgorithm:
        return PassthroughTransferAttackAlgorithm(image)


class RaySAlgorithmConfig(AttackAlgorithmConfig):
    name: str = "rays"

    def __init__(self, attack_regime_config: Munch, rays_config: Munch):
        super().__init__(attack_regime_config)
        self.early_stopping = rays_config.early_stopping

    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> RaySAttackAlgorithm:
        return RaySAttackAlgorithm(image,
                                   loss_fn,
                                   epsilon=self.epsilon,
                                   norm=self.norm,
                                   early_stopping=self.early_stopping)


from advpipe.attack_algorithms import RaySAttackAlgorithm    # noqa: 402
from advpipe.attack_algorithms import PassthroughTransferAttackAlgorithm    # noqa: 402

ATTACK_ALGORITHM_CONFIGS = {
    RaySAlgorithmConfig.name: RaySAlgorithmConfig,
    PassthroughTransferAlgorithmConfig.name: PassthroughTransferAlgorithmConfig
}

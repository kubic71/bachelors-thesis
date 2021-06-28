from __future__ import annotations
from abc import ABC, abstractmethod
from advpipe.attack_algorithms import Norm
from advpipe import utils

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

        return ATTACK_ALGORITHM_CONFIGS[algorithm_config.name](attack_regime_config, algorithm_config)    # type: ignore

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


class SquareAttackAlgorithmConfig(AttackAlgorithmConfig):
    name: str = "square_attack"
    n_iters: int = 1000
    p_init: float = 0.2

    def __init__(self, attack_regime_config: Munch, square_attack_config: Munch):
        super().__init__(attack_regime_config)
        self.n_iters = utils.get_config_attr(square_attack_config, "n_iters", SquareAttackAlgorithmConfig.n_iters)
        self.p_init = utils.get_config_attr(square_attack_config, "p_init", SquareAttackAlgorithmConfig.p_init)

    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> SquareL2AttackAlgorithm:
        return SquareL2AttackAlgorithm(image, loss_fn, epsilon=self.epsilon, n_iters=self.n_iters, p_init=self.p_init)


from advpipe.attack_algorithms import RaySAttackAlgorithm, SquareL2AttackAlgorithm
from advpipe.attack_algorithms import PassthroughTransferAttackAlgorithm

ATTACK_ALGORITHM_CONFIGS = {
    RaySAlgorithmConfig.name: RaySAlgorithmConfig,
    PassthroughTransferAlgorithmConfig.name: PassthroughTransferAlgorithmConfig,
    SquareAttackAlgorithmConfig.name: SquareAttackAlgorithmConfig
}

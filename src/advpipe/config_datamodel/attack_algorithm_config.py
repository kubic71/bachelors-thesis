from __future__ import annotations
from abc import ABC, abstractmethod
from advpipe.attack_algorithms import Norm
from advpipe import utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from advpipe.utils import LossCallCounter
    from advpipe.attack_algorithms import BlackBoxAlgorithm, BlackBoxIterativeAlgorithm, BlackBoxTransferAlgorithm
    from advpipe.blackbox import TargetBlackBox
    from advpipe.blackbox.local import WhiteBoxSurrogate
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

        

class IterativeAttackAlgorithmConfig(AttackAlgorithmConfig):

    def __init__(self, attack_regime_config: Munch):
        super().__init__(attack_regime_config)


    @staticmethod
    def loadFromYamlConfig(attack_regime_config: Munch, algorithm_config: Munch) -> IterativeAttackAlgorithmConfig:
        if algorithm_config.name not in ITERATIVE_ATTACK_ALGORITHM_CONFIGS:
            raise ValueError(f"Invalid attack algorithm: {algorithm_config.name}")

        return ITERATIVE_ATTACK_ALGORITHM_CONFIGS[algorithm_config.name](attack_regime_config, algorithm_config)

    @abstractmethod
    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> BlackBoxIterativeAlgorithm:
        ...


class RaySAlgorithmConfig(IterativeAttackAlgorithmConfig):
    name: str = "rays"
    early_stopping: bool

    def __init__(self, attack_regime_config: Munch, rays_config: Munch):
        super().__init__(attack_regime_config)
        self.early_stopping = rays_config.early_stopping

    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> RaySAttackAlgorithm:
        return RaySAttackAlgorithm(image,
                                   loss_fn,
                                   epsilon=self.epsilon,
                                   norm=self.norm,
                                   early_stopping=self.early_stopping)


class SquareAttackAlgorithmConfig(IterativeAttackAlgorithmConfig):
    name: str = "square_attack"
    n_iters: int = 1000
    p_init: float = 0.2

    def __init__(self, attack_regime_config: Munch, square_attack_config: Munch):
        super().__init__(attack_regime_config)
        self.n_iters = utils.get_config_attr(square_attack_config, "n_iters", SquareAttackAlgorithmConfig.n_iters)
        self.p_init = utils.get_config_attr(square_attack_config, "p_init", SquareAttackAlgorithmConfig.p_init)

    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> SquareL2AttackAlgorithm:
        return SquareL2AttackAlgorithm(image, loss_fn, epsilon=self.epsilon, n_iters=self.n_iters, p_init=self.p_init)




class TransferAttackAlgorithmConfig(AttackAlgorithmConfig):
    def __init__(self, attack_regime_config: Munch):
        super().__init__(attack_regime_config)

    @staticmethod
    def loadFromYamlConfig(attack_regime_config: Munch, algorithm_config: Munch) -> TransferAttackAlgorithmConfig:
        if algorithm_config.name not in TRANSFER_ATTACK_ALGORITHM_CONFIGS:
            raise ValueError(f"Invalid attack algorithm: {algorithm_config.name}")

        return TRANSFER_ATTACK_ALGORITHM_CONFIGS[algorithm_config.name](attack_regime_config, algorithm_config)

    @abstractmethod
    def getAttackAlgorithmInstance(self, image: np.ndarray, surrogate: WhiteBoxSurrogate) -> BlackBoxTransferAlgorithm:
        ...

class PassthroughTransferAlgorithmConfig(TransferAttackAlgorithmConfig):
    name: str = "passthrough"

    def __init__(self, attack_regime_config: Munch, attack_config: Munch):
        super().__init__(attack_regime_config)

    def getAttackAlgorithmInstance(self, image: np.ndarray, _: WhiteBoxSurrogate) -> PassthroughTransferAttackAlgorithm:
        return PassthroughTransferAttackAlgorithm(image, _)


class FgsmTransferAlgorithmConfig(TransferAttackAlgorithmConfig):
    name: str = "fgsm"

    def __init__(self, attack_regime_config: Munch, attack_config: Munch):
        super().__init__(attack_regime_config)

    def getAttackAlgorithmInstance(self, image: np.ndarray, surrogate: WhiteBoxSurrogate) -> FgsmTransferAlgorithm:
        return FgsmTransferAlgorithm(image, surrogate, self.epsilon)


from advpipe.attack_algorithms import RaySAttackAlgorithm, SquareL2AttackAlgorithm, PassthroughTransferAttackAlgorithm, FgsmTransferAlgorithm

ITERATIVE_ATTACK_ALGORITHM_CONFIGS = {
    RaySAlgorithmConfig.name: RaySAlgorithmConfig,
    SquareAttackAlgorithmConfig.name: SquareAttackAlgorithmConfig,
}


TRANSFER_ATTACK_ALGORITHM_CONFIGS = {
    FgsmTransferAlgorithmConfig.name: FgsmTransferAlgorithmConfig,
    PassthroughTransferAlgorithmConfig.name: PassthroughTransferAlgorithmConfig,
}

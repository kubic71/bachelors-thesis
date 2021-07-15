from __future__ import annotations
from abc import ABC, abstractmethod
from advpipe.attack_algorithms import Norm
from advpipe import utils

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing_extensions import Literal
    from typing import Optional
    from advpipe.utils import LossCallCounter
    import torch
    from advpipe.attack_algorithms import BlackBoxIterativeAlgorithm, BlackBoxTransferAlgorithm
    from advpipe.blackbox import TargetModel
    from advpipe.blackbox.local import LocalModel
    from munch import Munch
    import numpy as np


class AttackAlgorithmConfig(ABC):
    name: str
    norm: Norm
    epsilon: float

    def __init__(self, algorithm_config: Munch):
        self.norm = Norm(algorithm_config.norm)
        self.epsilon = algorithm_config.epsilon

        

class IterativeAttackAlgorithmConfig(AttackAlgorithmConfig):

    def __init__(self, algorithm_config: Munch):
        super().__init__(algorithm_config)


    @staticmethod
    def loadFromYamlConfig(algorithm_config: Munch) -> IterativeAttackAlgorithmConfig:
        if algorithm_config.name not in ITERATIVE_ATTACK_ALGORITHM_CONFIGS:
            raise ValueError(f"Invalid attack algorithm: {algorithm_config.name}")

        return ITERATIVE_ATTACK_ALGORITHM_CONFIGS[algorithm_config.name](algorithm_config)

    @abstractmethod
    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> BlackBoxIterativeAlgorithm:
        ...


class RaySAlgorithmConfig(IterativeAttackAlgorithmConfig):
    name: str = "rays"
    early_stopping: bool

    def __init__(self, rays_config: Munch):
        super().__init__(rays_config)
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
    p_init: float = 0.8

    def __init__(self, square_attack_config: Munch):
        super().__init__(square_attack_config)
        self.n_iters = utils.get_config_attr(square_attack_config, "n_iters", SquareAttackAlgorithmConfig.n_iters)
        self.p_init = utils.get_config_attr(square_attack_config, "p_init", SquareAttackAlgorithmConfig.p_init)

    def getAttackAlgorithmInstance(self, image: np.ndarray, loss_fn: LossCallCounter) -> SquareL2AttackAlgorithm:
        return SquareL2AttackAlgorithm(image, loss_fn, epsilon=self.epsilon, n_iters=self.n_iters, p_init=self.p_init)




class TransferAttackAlgorithmConfig(AttackAlgorithmConfig):
    def __init__(self, algorithm_config: Munch):
        super().__init__(algorithm_config)

    @staticmethod
    def loadFromYamlConfig(algorithm_config: Munch) -> TransferAttackAlgorithmConfig:
        if algorithm_config.name not in TRANSFER_ATTACK_ALGORITHM_CONFIGS:
            raise ValueError(f"Invalid attack algorithm: {algorithm_config.name}")

        return TRANSFER_ATTACK_ALGORITHM_CONFIGS[algorithm_config.name](algorithm_config) # type: ignore

    @abstractmethod
    def getAttackAlgorithmInstance(self, surrogate: LocalModel) -> BlackBoxTransferAlgorithm:
        ...

class PassthroughTransferAlgorithmConfig(TransferAttackAlgorithmConfig):
    name: str = "passthrough"

    def __init__(self, attack_config: Munch):
        super().__init__(attack_config)

    def getAttackAlgorithmInstance(self, surrogate: LocalModel) -> PassthroughTransferAttackAlgorithm:
        return PassthroughTransferAttackAlgorithm(surrogate)


class FgsmTransferAlgorithmConfig(TransferAttackAlgorithmConfig):
    name: str = "fgsm"

    def __init__(self, attack_config: Munch):
        super().__init__(attack_config)

    def getAttackAlgorithmInstance(self, surrogate: LocalModel) -> FgsmTransferAlgorithm:
        return FgsmTransferAlgorithm(surrogate, self.epsilon)


class AdamPGDConfig(TransferAttackAlgorithmConfig):
    name: str = "adam-pgd"
    n_iters: int
    gradient_samples: int
    lr: float

    def __init__(self, attack_config: Munch):
        super().__init__(attack_config)
        self.n_iters = attack_config.n_iters
        self.gradient_samples = attack_config.gradient_samples
        self.lr = attack_config.lr

    def getAttackAlgorithmInstance(self, surrogate: LocalModel) -> FgsmTransferAlgorithm:
        return AdamPGD(surrogate, self.epsilon, self.n_iters, self.lr, gradient_samples=self.gradient_samples) # type: ignore


class SquareAutoAttackConfig(TransferAttackAlgorithmConfig):
    name: str = "square-autoattack"
    metric: Literal["L1", "L2", "Linf"]
    n_iters: int = 1000
    n_restarts: int = 1

    def __init__(self, attack_config: Munch):
        super().__init__(attack_config)

        # Square attack has slightly different norm names
        self.metric = {"l1": "L1", "l2": "L2", "linf": "Linf"}[self.norm.name] # type: ignore
        self.n_iters = attack_config.n_iters
        self.n_restarts = attack_config.n_restarts

    def getAttackAlgorithmInstance(self, surrogate: LocalModel) -> SquareAutoAttack:
        return SquareAutoAttack(surrogate, self)


class APGDAlgorithmConfig(TransferAttackAlgorithmConfig):
    name: str = "apgd"
    metric: Literal["L2", "Linf"]
    n_iters: int = 100
    eot_iter: int = 1
    n_restarts: int = 1
    early_stop_at: Optional[int] = None
    # surrogate_output

    def __init__(self, attack_config: Munch):
        super().__init__(attack_config)
        self.metric = {"l2": "L2", "linf": "Linf"}[self.norm.name] # type: ignore
        self.n_iters = attack_config.n_iters
        self.early_stop_at = utils.get_config_attr(attack_config, "early_stop_at", APGDAlgorithmConfig.early_stop_at)
        self.eot_iter = utils.get_config_attr(attack_config, "eot_iter", APGDAlgorithmConfig.eot_iter)

    def getAttackAlgorithmInstance(self, surrogate: LocalModel) -> APGDAutoAttack:
        return APGDAutoAttack(surrogate, self)

from advpipe.attack_algorithms.rays import RaySAttackAlgorithm
from advpipe.attack_algorithms.square_attack import SquareL2AttackAlgorithm
from advpipe.attack_algorithms.passthrough_attack import PassthroughTransferAttackAlgorithm
from advpipe.attack_algorithms.fgsm import FgsmTransferAlgorithm
from advpipe.attack_algorithms.adam_pgd import AdamPGD
from advpipe.attack_algorithms.square_auto_attack import SquareAutoAttack
from advpipe.attack_algorithms.apgd_auto_attack import APGDAutoAttack

ITERATIVE_ATTACK_ALGORITHM_CONFIGS = {
    RaySAlgorithmConfig.name: RaySAlgorithmConfig,
    SquareAttackAlgorithmConfig.name: SquareAttackAlgorithmConfig,
}


TRANSFER_ATTACK_ALGORITHM_CONFIGS = {
    FgsmTransferAlgorithmConfig.name: FgsmTransferAlgorithmConfig,
    PassthroughTransferAlgorithmConfig.name: PassthroughTransferAlgorithmConfig,
    SquareAutoAttackConfig.name: SquareAutoAttackConfig,
    APGDAlgorithmConfig.name: APGDAlgorithmConfig,
    AdamPGDConfig.name: AdamPGDConfig
}

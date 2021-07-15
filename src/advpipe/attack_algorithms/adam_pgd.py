from __future__ import annotations
from advpipe.attack_algorithms import BlackBoxTransferAlgorithm
from advpipe.log import logger
import torch.nn.functional as F
import numpy as np
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from advpipe.blackbox.local import LocalModel
    from typing import Callable

class AdamPGD(BlackBoxTransferAlgorithm):
    epsilon: float
    n_iters: int
    # lr_schedule: Callable[[float], float]
    lr: float
    gradient_samples: int

    def exp_lr_schedule(self, t:float, start:float, end:float, warmup_len = 0.0) -> float:
        if t < warmup_len:
            lr = t / 0.3 * start
        else:
            lr = np.exp(np.log(start) + t*(np.log(end) - np.log(start)))

        logger.debug(f"Learnig rate: {lr}")
        return lr
    
    def norm_factor(self) -> float:
        s = 0
        for t in range(self.n_iters):
            s += (1 - self.b1**(t+1)) / np.sqrt(1 - self.b2**(t+1))
        return s

    


    
    # def __init__(self, surrogate: LocalModel, epsilon: float, n_iters: int, lr_schedule: Callable[[float], float] = lambda x: 0.1):
    def __init__(self, surrogate: LocalModel, epsilon: float, n_iters: int, lr: float = 0.001, gradient_samples = 1, b1 = 0.9, b2 = 0.99):
        super().__init__(surrogate)
        self.epsilon = epsilon 
        self.n_iters = n_iters
        self.gradient_samples = gradient_samples
        # self.lr_schedule = lr_schedule # type: ignore
        self.lr = lr      

        self.b1 = b1
        self.b2 = b2
        self.u1 = 1.5
        self.u2 = 1.9
        self.s = self.norm_factor()
        self.stability = 10e-8

        self.lamb = 1.3

        # self.l2_reg = 100

    def margin_loss(self, logits: torch.Tensor, onehot_labels: torch.Tensor) -> torch.Tensor:
        return torch.max(logits*onehot_labels, dim=1).values - torch.max(logits*(1-onehot_labels), dim=1).values # type: ignore



    def run(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            m = torch.zeros_like(images)
            v = torch.zeros_like(images)


            gold = F.one_hot(labels, num_classes=2)
            x_adv = images.clone()
            x = images.clone()

            for t in range(1, self.n_iters + 1):
                assert x_adv.requires_grad == False

                grad = torch.zeros_like(x_adv).cuda()

                for j in range(self.gradient_samples):
                    x_adv_1 = x_adv.clone()
                    x_adv_1.requires_grad = True
                    with torch.enable_grad():
                        logits = self.surrogate(x_adv_1)
                        print(logits)
                        loss_indiv = self.margin_loss(logits, gold) # type: ignore
                        print(loss_indiv)
                        # print("l2_reg:", torch.exp(((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3)).sqrt() / 100)*self.l2_reg)
                        # loss_indiv += torch.exp(((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3)).sqrt() / 100)*self.l2_reg
                        loss = loss_indiv.sum()
                        # L2 regularization

                        # loss.backward()

                    oneg = torch.autograd.grad(loss, [x_adv_1])[0].detach()
                    grad += torch.nan_to_num(oneg) # 1 backward pass (eot_iter = 1)

                grad /= self.gradient_samples

                logger.debug(f"Grad l2 norm: {(grad**2).sum(dim=(1,2,3)).sqrt()}")

                assert not torch.isnan(grad.max())

                m = m + self.u1 * grad
                v = v + self.u2*(grad**2)
                a_t = self.epsilon / (self.s * 255)  * (1 - self.b1**t)/np.sqrt(1 - self.b2**t)

                assert not torch.isnan(m.max())
                assert not torch.isnan(v.max())

                # print(self.lr * m_hat/(v_hat + self.stability))

                # x_adv -= self.exp_lr_schedule(t / self.n_iters, self.lr, self.lr/10) * m_hat/(v_hat.sqrt() + self.stability)
                x_adv -= a_t * torch.tanh(self.lamb * m / (v.sqrt() + self.stability))

                assert not torch.isnan(x_adv.max())

                x_adv_det = x_adv.detach()
                x_adv = torch.clamp(x + (x_adv_det - x) / (((x_adv_det - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                            self.epsilon * torch.ones(x.shape).to("cuda").detach(), ((x_adv_det - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)



                logger.debug(f"Iteration: {t}, L2 dist: {((x_adv - x) ** 2).sum(dim=(1, 2, 3)).sqrt()}")
                                        #    (x_adv_det - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                assert not torch.isnan(x_adv.max())

            return x_adv.clone().detach()

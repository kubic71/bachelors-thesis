from advpipe.attack_algorithms import BlackBoxIterativeAlgorithm
import numpy as np
import torch
from advpipe.log import logger
from advpipe import utils


class RaySAttackAlgorithm(BlackBoxIterativeAlgorithm):
    def __init__(self,
                 image,
                 loss_fn,
                 epsilon=0.05,
                 order=np.inf,
                 early_stopping=True):
        super().__init__(image, loss_fn)
        self.pertubation = np.zeros_like(image)

        self.epsilon = epsilon
        self.order = order
        self.early_stopping = early_stopping

        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.lin_search_rad = 10
        self.pre_set = {1, -1}


    def get_xadv(self, x, v, d, lb=0., rb=1.):
        out = x + d * v
        return torch.clamp(out, lb, rb) # pylint: disable=no-member

    def run(self, seed=None):

        x = torch.from_numpy(self.image) # pylint: disable=no-member
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        self.queries = 0
        self.d_t = np.inf
        self.sgn_t = torch.sign(torch.ones(shape)) # pylint: disable=no-member
        self.x_final = x
        

        dist = torch.tensor(np.inf) # pylint: disable=not-callable
        block_level = 0
        block_ind = 0

        # Number of iterations is set to a large number, because attack termination is handled upstream
        for i in range(1000000): 

            block_num = 2**block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) *
                                                     block_size)

            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[:, start:end] *= -1.
            attempt = attempt.view(shape)

            yield from self.binary_search(x, attempt)

            block_ind += 1
            if block_ind == 2**block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm(self.x_final - x, self.order)

            if i % 1 == 0:
                logger.debug(f"Rays iteration: {i}, Queries: {self.queries} d_t {self.d_t:.8f} dist {dist:.8f}")

            if self.early_stopping and (dist <= self.epsilon):
                logger.info(f"Rays early stopping. dist: {dist} <= {self.epsilon}")
                break

        print("Iter %3d d_t %.6f dist %.6f queries %d" %
              (i + 1, self.d_t, dist, self.queries))
        # return self.x_final, self.queries, dist, (dist <= self.epsilon).float()

    def is_adversarial(self, img):
        loss_val = self.loss_fn(img.detach().cpu().numpy())
        # yield evey time loss_fn is evaluated, so that the attack manager can retake the execution control
        yield self.x_final.detach().cpu().numpy()

        return loss_val <= 0

    def search_succ(self, x):
        self.queries += 1
        ret = yield from self.is_adversarial(x)
        return ret

    def lin_search(self, x, sgn):
        d_end = np.inf
        for d in range(1, self.lin_search_rad + 1):
            succ = yield from self.search_succ(self.get_xadv(x, sgn, d))
            if succ:
                d_end = d
                break
        return d_end

    def binary_search(self, x, sgn, tol=1e-3):
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)

        d_start = 0
        if np.inf > self.d_t:  # already have current result
            succ = yield from self.search_succ(self.get_xadv(x, sgn_unit, self.d_t))
            if not succ:
                return False
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = yield from self.lin_search(x, sgn)
            if d < np.inf:
                d_end = d * sgn_norm
            else:
                return False

        while (d_end - d_start) > tol:
            d_mid = (d_start + d_end) / 2.0
            succ = yield from self.search_succ(self.get_xadv(x, sgn_unit, d_mid))
            if succ:
                d_end = d_mid
            else:
                d_start = d_mid
        if d_end < self.d_t:
            self.d_t = d_end
            self.x_final = self.get_xadv(x, sgn_unit, d_end)
            self.sgn_t = sgn
            return True
        else:
            return False

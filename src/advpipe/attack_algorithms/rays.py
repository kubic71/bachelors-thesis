from advpipe.attack_algorithms import BlackBoxIterativeAlgorithm
import numpy as np
import torch
from advpipe.log import logger
from advpipe import utils


# TODO

class RaySAttackAlgorithm(BlackBoxIterativeAlgorithm):
    def __init__(self, image, loss_fn, epsilon=0.3, order=np.inf, max_iters=1000, early_stopping=True):
        super().__init__(image, loss_fn)
        self.pertubation = np.zeros_like(image)
        self.best_loss = self.loss_fn(image)
        self.epsilon = epsilon
        self.order = order
        self.max_iters = max_iters
        self.early_stopping = early_stopping

        self.min_boundary = np.clip(image - self.epsilon*np.ones_like(image), 0, 1)
        self.max_boundary = np.clip(image + self.epsilon*np.ones_like(image), 0, 1)
    
        self.sgn_t = None
        self.d_t = None
        self.x_final = None
        self.lin_search_rad = 10
        self.pre_set = {1, -1}

    def run(self):
        for i in range(self.max_iters):
            self.new_pertubation = self.pertubation + np.random.normal(size=self.image.shape)*self.epsilon/3
            pertubed_img = np.clip(self.image + self.new_pertubation, self.min_boundary, self.max_boundary)

            loss_val = self.loss_fn(pertubed_img)
            if loss_val < self.best_loss:
                self.pertubation = np.clip(self.new_pertubation, -self.epsilon, self.epsilon)

            inf_dist = utils.l_inf(self.image, self.image + self.pertubation)
            logger.info(f"l_inf dist: {inf_dist}")
            yield self.pertubation


class RayS(object):

    def get_xadv(self, x, v, d, lb=0., rb=1.):
        out = x + d * v
        return torch.clamp(out, lb, rb)

    def attack_hard_label(self, x, y, target_label=None, query_limit=10000, seed=None):
        """ Attack the original image and return adversarial example.
            model: (pytorch model)
            (x, y): original image
        """
        x = x.cuda()
        shape = list(x.shape)
        dim = np.prod(shape[1:])
        if seed is not None:
            np.random.seed(seed)

        self.queries = 0
        self.d_t = np.inf
        self.sgn_t = torch.sign(torch.ones(shape)).cuda()
        self.x_final = self.get_xadv(x, self.sgn_t, self.d_t)
        dist = torch.tensor(np.inf)
        block_level = 0
        block_ind = 0

        for i in range(query_limit):

            block_num = 2 ** block_level
            block_size = int(np.ceil(dim / block_num))
            start, end = block_ind * block_size, min(dim, (block_ind + 1) * block_size)

            attempt = self.sgn_t.clone().view(shape[0], dim)
            attempt[:, start:end] *= -1.
            attempt = attempt.view(shape)

            self.binary_search(x, y, target_label, attempt)

            block_ind += 1
            if block_ind == 2 ** block_level or end == dim:
                block_level += 1
                block_ind = 0

            dist = torch.norm(self.x_final - x, self.order)
            if self.early_stopping and (dist <= self.epsilon):
                break

            if self.queries >= query_limit:
                print('out of queries')
                break

            if i % 10 == 0:
                print("Iter %3d d_t %.8f dist %.8f queries %d" % (i + 1, self.d_t, dist, self.queries))

        print("Iter %3d d_t %.6f dist %.6f queries %d" % (i + 1, self.d_t, dist, self.queries))
        return self.x_final, self.queries, dist, (dist <= self.epsilon).float()

    def search_succ(self, x, y, target):
        self.queries += 1
        if target:
            return self.model.predict_label(x) == target
        else:
            return self.model.predict_label(x) != y

    def lin_search(self, x, y, target, sgn):
        d_end = np.inf
        for d in range(1, self.lin_search_rad + 1):
            if self.search_succ(self.get_xadv(x, sgn, d), y, target):
                d_end = d
                break
        return d_end

    def binary_search(self, x, y, target, sgn, tol=1e-3):
        sgn_unit = sgn / torch.norm(sgn)
        sgn_norm = torch.norm(sgn)

        d_start = 0
        if np.inf > self.d_t:  # already have current result
            if not self.search_succ(self.get_xadv(x, sgn_unit, self.d_t), y, target):
                return False
            d_end = self.d_t
        else:  # init run, try to find boundary distance
            d = self.lin_search(x, y, target, sgn)
            if d < np.inf:
                d_end = d * sgn_norm
            else:
                return False

        while (d_end - d_start) > tol:
            d_mid = (d_start + d_end) / 2.0
            if self.search_succ(self.get_xadv(x, sgn_unit, d_mid), y, target):
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

    def __call__(self, data, label, target=None, seed=None, query_limit=10000):
        return self.attack_hard_label(data, label, target_label=target, seed=seed, query_limit=query_limit)

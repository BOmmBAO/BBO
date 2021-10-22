'''
Numpy implementation of REPES optimizer and upper-level policy
'''

import torch.autograd.numpy as np
import copy
from torch.autograd import grad
import scipy as sc
from scipy import optimize
from scipy import stats

class UpperPolicy:
    '''
    Controller for RL agents
    '''

    def __init__(self, dm_act, cov0):
        self.dm_act = dm_act

        self.mu = np.random.randn(self.dm_act)
        self.cov = cov0 * np.eye(self.dm_act)

    def action(self, n):
        aux = sc.stats.multivariate_normal(mean=self.mu, cov=self.cov).rvs(n)
        return aux.reshape((n, self.dm_act))

    def loglik(self, pi, x):
        mu, cov = pi.mu, pi.cov
        c = mu.shape[0] * np.log(2.0 * np.pi)

        ans = - 0.5 * (np.einsum('nk,kh,nh->n', mu - x, np.linalg.inv(cov), mu - x) +
                       np.log(np.linalg.det(cov)) + c)
        return ans

    def kli(self, pi):
        diff = self.mu - pi.mu

        kl = 0.5 * (np.trace(np.linalg.inv(self.cov) @ pi.cov) + diff.T @ np.linalg.inv(self.cov) @ diff
                    - self.dm_act + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov)))
        return kl

    def klm(self, pi):
        diff = pi.mu - self.mu

        kl = 0.5 * (np.trace(np.linalg.inv(pi.cov) @ self.cov) + diff.T @ np.linalg.inv(pi.cov) @ diff
                    - self.dm_act + np.log(np.linalg.det(pi.cov) / np.linalg.det(self.cov)))
        return kl

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(self.cov * 2.0 * np.pi * np.exp(1.0)))

    def wml(self, x, w, eta=np.array([0.0])):
        pol = copy.deepcopy(self)

        pol.mu = (np.sum(w[:, np.newaxis] * x, axis=0) + eta * self.mu) / (np.sum(w, axis=0) + eta)

        diff = x - pol.mu
        tmp = np.einsum('nk,n,nh->nkh', diff, w, diff)
        pol.cov = (np.sum(tmp, axis=0) + eta * self.cov +
                   eta * np.outer(pol.mu - self.mu, pol.mu - self.mu)) / (np.sum(w, axis=0) + eta)

        return pol

    def dual(self, eta, x, w, eps):
        pol = self.wml(x, w, eta)
        return np.sum(w * self.loglik(pol, x)) + eta * (eps - self.klm(pol))

    def wmap(self, x, w, eps=np.array([0.1])):
        res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                   method='SLSQP',
                                   jac=grad(self.dual),
                                   args=(x, w, eps),
                                   bounds=((1e-8, 1e8),))
        eta = res['x']
        pol = self.wml(x, w, eta)

        return pol


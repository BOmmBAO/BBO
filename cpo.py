import autograd.numpy as np

import scipy as sc
from scipy import optimize
from scipy import special

from sklearn.preprocessing import PolynomialFeatures

import copy


EXP_MAX = 700.0
EXP_MIN = -700.0


class cREPS:

    def __init__(self, func,
                 n_episodes, kl_bound,
                 vdgr, pdgr,
                 vreg, preg, **kwargs):

        self.func = func
        self.dm_act = self.func.dm_act
        self.dm_cntxt = self.func.dm_cntxt

        self.n_episodes = n_episodes
        self.kl_bound = kl_bound

        self.vreg = vreg
        self.preg = preg

        self.vdgr = vdgr
        self.pdgr = pdgr

        if 'cov0' in kwargs:
            cov0 = kwargs.get('cov0', False)
            self.ctl = Policy(self.dm_act, self.dm_cntxt,
                              self.pdgr, cov0)
        else:
            self.ctl = Policy(self.dm_act, self.dm_cntxt,
                              self.pdgr, 100.0)

        self.nb_pfeat = self.ctl.nb_feat

        self.vfunc = Vfunction(self.dm_cntxt, self.vdgr)
        self.nb_vfeat = self.vfunc.nb_feat

        self.eta = np.array([1.0])

        self.data = None
        self.vfeatures = None
        self.w = None

    def sample(self, n_episodes):
        data = {'c': self.func.context(n_episodes)}
        data['x'] = self.ctl.action(data['c'])
        data['r'] = self.func.eval(data['c'], data['x'])
        return data

    def weights(self, eta, omega, r, phi):
        adv = r - np.dot(phi, omega)
        delta = adv - np.max(adv)
        w = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        return w, delta, np.max(adv)

    def dual(self, var, eps, r, phi):
        eta, omega = var[0], var[1:]
        w, delta, max_adv = self.weights(eta, omega, r, phi)
        g = eta * eps + max_adv + np.dot(np.mean(phi, axis=0), omega) +\
            eta * np.log(np.mean(w, axis=0))
        g = g + self.vreg * np.sum(omega ** 2)
        return g

    def grad(self, var, eps, r, phi):
        eta, omega = var[0], var[1:]
        w, delta, max_adv = self.weights(eta, omega, r, phi)

        deta = eps + np.log(np.mean(w, axis=0)) - \
               np.sum(w * delta, axis=0) / (eta * np.sum(w, axis=0))

        domega = np.mean(phi, axis=0) - \
                 np.sum(w[:, np.newaxis] * phi, axis=0) / np.sum(w, axis=0)
        domega = domega + self.vreg * 2 * omega

        return np.hstack((deta, domega))

    def kl_samples(self, w):
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def run(self, nb_iter=100, verbose=False):
        _trace = {'rwrd': [],
                  'kls': [], 'kli': [], 'klm': [],
                  'ent': []}

        for it in range(nb_iter):
            self.data = self.sample(self.n_episodes)
            rwrd = np.mean(self.data['r'])

            self.vfeatures = self.vfunc.features(self.data['c'])

            res = sc.optimize.minimize(self.dual,
                                       np.hstack((1.0, 1e-8 * np.random.randn(self.nb_vfeat))),
                                       method='L-BFGS-B',
                                       jac=self.grad,
                                       # jac=grad(self.dual),
                                       args=(
                                           self.kl_bound,
                                           self.data['r'],
                                           self.vfeatures),
                                       bounds=((1e-8, 1e8), ) + ((-np.inf, np.inf), ) * self.nb_vfeat)

            self.eta, self.vfunc.omega = res.x[0], res.x[1:]
            self.w, _, _ = self.weights(self.eta, self.vfunc.omega,
                                        self.data['r'], self.vfeatures)

            # pol = self.ctl.wml(self.data['c'], self.data['x'], self.w, self.preg)
            pol = self.ctl.wmap(self.data['c'], self.data['x'], self.w, eps=self.kl_bound)

            kls = self.kl_samples(self.w)
            kli = self.ctl.kli(pol, self.data['c'])
            klm = self.ctl.klm(pol, self.data['c'])

            self.ctl = pol
            ent = self.ctl.entropy()

            _trace['rwrd'].append(rwrd)
            _trace['kls'].append(kls)
            _trace['kli'].append(kli)
            _trace['klm'].append(klm)
            _trace['ent'].append(ent)

            if verbose:
                print('it=', it,
                      f'rwrd={rwrd:{5}.{4}}',
                      f'kls={kls:{5}.{4}}',
                      f'kli={kli:{5}.{4}}',
                      f'klm={klm:{5}.{4}}',
                      f'ent={ent:{5}.{4}}')

            if ent < -3e2:
                break

        return _trace

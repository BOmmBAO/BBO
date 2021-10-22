import numpy as np
from copy import copy
from scipy.optimize import minimize
import mushroom_rl.algorithms
from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from sklearn.feature_selection import mutual_info_regression


class REPS_MI(BlackBoxOptimization):
    """
    Episodic Relative Entropy Policy Search algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, eps, m, features=None):
        """
        Constructor.

        Args:
            eps (float): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.

        """
        self.m = m
        self.eps = eps

        self._add_save_attr(eps='primitive')

        super().__init__(mdp_info, distribution, policy, features)#inherit from BBO

    def _update(self, Jep, theta):

        #variable of reduced dimention m befor update

        m_mi = self.MI_features_selection(Jep, theta)
        theta_mi = theta[m_mi, :]



        eta_start = np.ones(1)

        res = minimize(REPS_MI._dual_function, eta_start,
                       jac=REPS_MI._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self.eps, Jep, theta_mi))

        eta_opt = res.x.item()

        Jep -= np.max(Jep)

        d = np.exp(Jep / eta_opt)
        Z = (np.sum(d)**2-np.sum(d**2))/np.sum(d)
        mu_mi = d@theta[:, m_mi] / np.sum(d)
        delta2 = (theta[:, m_mi] - mu_mi) ** 2
        std_mi = np.sqrt(d @ delta2 / Z)

        # get old mu, std
        mu = self.distribution._mu
        std = self.distribution._std

        # update using new mu, std
        mu_new = copy(mu)
        std_new = copy(std)
        mu_new[m_mi] = mu_mi
        std_new[m_mi] = std_mi


        KL_full = REPS_MI._KL_M_Projection(mu, mu_new, np.diag(std), np.diag(std_new))
        KL_reduced = REPS_MI._KL_M_Projection(mu[m_mi], mu_new[m_mi], np.diag(std[m_mi]), np.diag(std_new[m_mi]))
        print('Equal?', round(KL_full, 6) == round(KL_reduced, 6), '| KL_full', KL_full, '| KL_reduced', KL_reduced)

        rho = np.concatenate((mu_new, std_new))
        self.distribution.set_parameters(rho)

        self.distribution.mle(theta, d)

    @staticmethod
    def _dual_function(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J
        sum1 = np.mean(np.exp(r / eta))

        return eta * eps + eta * np.log(sum1) + max_J

    @staticmethod
    def _dual_function_diff(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J

        sum1 = np.mean(np.exp(r / eta))
        sum2 = np.mean(np.exp(r / eta) * r)

        gradient = eps + np.log(sum1) - sum2 / (eta * sum1)

        return np.array([gradient])


    def MI_features_selection(self, y, x):
        mi = mutual_info_regression(x, y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
        #n_mi = np.argsort(np.sum(mi, axis = 1))[-n:][::-1]
        n_mi = mi.argsort()[-self.m:][::-1]
        return n_mi

    @staticmethod
    def _KL_M_Projection(mu_pre, mu_post, sigma_pre, sigma_post):
        diff = mu_post - mu_pre
        inverse = np.linalg.inv(sigma_post)
        _, slog_post = np.linalg.slogdet(sigma_post)
        _, slog_pre = np.linalg.slogdet(sigma_pre)
        kl = 0.5 * (np.trace(inverse @ sigma_pre) + diff.T @ inverse @ diff + slog_post - slog_pre - len(mu_pre))
        return kl
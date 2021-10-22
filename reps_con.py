import numpy as np

from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization

import autograd.numpy as np
from autograd import grad

import traceback

class REPS_con(BlackBoxOptimization):
    """
    Episodic Relative Entropy Policy Search algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, eps, kappa, features=None):
        """
        Constructor.

        Args:
            eps (float): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
            kappa (float): entropy constraint for the policy update. H(pi_t) - kappa <= H(pi_t+1)

        """
        self.eps = eps
        self.kappa = kappa

        self._add_save_attr(eps='primitive')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):

        n = len(self.distribution._mu)
        dist_params = self.distribution.get_parameters()
        mu_t = dist_params[:n]
        chol_sig_empty = np.zeros((n,n))
        chol_sig_empty[np.tril_indices(n)] = dist_params[n:]
        chol_sig_t = chol_sig_empty.dot(chol_sig_empty.T)

        # REPS
        eta_start = np.ones(1)
        res = minimize(REPS_con._dual_function, eta_start,
                       jac=REPS_con._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self.eps, Jep, theta),
                       method=None)
        eta_opt = res.x.item()

        Jep -= np.max(Jep)
        d = np.exp(Jep / eta_opt)

        # optimize for Langrangian multipliers
        eta_omg_start = np.ones(2)
        res = minimize(REPS_con._lagrangian_eta_omg, eta_omg_start,
                       jac=grad(REPS_con._lagrangian_eta_omg),
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(d, theta, mu_t, chol_sig_t, n, self.eps, self.kappa),
                       method=None)
        eta_opt, omg_opt  = res.x[0], res.x[1]

        mu_t1, sig_t1 = REPS_con.optimal_mu_t1_sig_t1(d, theta, mu_t, chol_sig_t, n, self.eps, self.kappa, eta_opt, omg_opt)
        
        try:
            dist_params = np.concatenate((mu_t1.flatten(), np.linalg.cholesky(sig_t1)[np.tril_indices(n)].flatten()))
        except np.linalg.LinAlgError:
            traceback.print_exc()
            print('error in setting dist_params - sig_t1 not positive definite')
            print('sig_t1', sig_t1)
            print('eta_opt', eta_opt)
            print('omg_opt', omg_opt)
            exit(42)

        self.distribution.set_parameters(dist_params)


    @staticmethod
    def optimal_mu_t1_sig_t1(*args):
        
        W, theta, mu_t, sig_t, n, eps, eta, omg, kappa = args
        W_sum = np.sum(W)

        mu_t1 = (np.sum(W@theta) + eta * mu_t) / (W_sum + eta)

        sig_wa = (theta - mu_t1).T @ np.diag(W) @ (theta - mu_t1)
        sig_t1 = (sig_wa + eta * sig_t + eta * (mu_t1 - mu_t) @ (mu_t1 - mu_t).T) / (W_sum + eta - omg) 

        return mu_t1, sig_t1

    @staticmethod
    def _lagrangian_eta_omg(lag_array, *args):
        W, theta, mu_t, sig_t, n, eps, kappa = args
        eta, omg = lag_array[0], lag_array[1]
        W_sum = np.sum(W)
        mu_t1 = (np.sum(W@theta) + eta * mu_t) / (W_sum + eta)
        sig_wa = (theta - mu_t1).T @ np.diag(W) @ (theta - mu_t1)
        sig_t1 = (sig_wa + eta * sig_t + eta * (mu_t1 - mu_t) @ (mu_t1 - mu_t).T) / (W_sum + eta - omg)

        if len(sig_t1.shape) == 1:
            sig_t1 = sig_t1[:,np.newaxis]
        
        try:
            sig_t1_inv = np.linalg.inv(sig_t1)
        except np.linalg.LinAlgError:
            traceback.print_exc()
            print('error in lagrangian - cant inverse')
            print(sig_t1)
            exit(42)

        sum1 = - 0.5 * np.sum([w_i * np.trace(sig_t1_inv @ (theta_i - mu_t1)[:,np.newaxis] @ (theta_i - mu_t1)[:,np.newaxis].T) for w_i, theta_i in zip(W,theta)])

        sum2 = - 0.5 * eta * np.trace(sig_t1_inv @ sig_t1) - 0.5 * eta * np.trace(sig_t1_inv @ (mu_t1 - mu_t)[:,np.newaxis] @ (mu_t1 - mu_t)[:,np.newaxis].T)

        # log determinants
        (sign_sig_t, logdet_sig_t) = np.linalg.slogdet(sig_t)
        (sign_sig_t1, logdet_sig_t1) = np.linalg.slogdet(sig_t1)

        c = n * np.log(2*np.pi)
        H_t = 0.5 * logdet_sig_t + c/2 + n/2
        
        beta = H_t - kappa

        sum3 = 0.5 * (omg - eta - W_sum) * logdet_sig_t1 + 0.5 * eta * (n + logdet_sig_t + 2 * eps)
        sum4 = 0.5 * omg * (c + n - 2 * beta) - 0.5 * W_sum * c
        
        return (sum1 + sum2 + sum3 + sum4)

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

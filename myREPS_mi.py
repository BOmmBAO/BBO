from sklearn.feature_selection import mutual_info_regression
import autograd.numpy as np
from autograd import grad


from scipy.optimize import minimize
import mushroom_rl.algorithms
from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization


class myREPS_mi(BlackBoxOptimization):
    """
    Episodic Relative Entropy Policy Search algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """

    def __init__(self, mdp_info, distribution, policy, eps, k, features=None):
        """
        Constructor.

        Args:
            eps (float): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.

        """
        self.k = k
        self.eps = eps

        self._add_save_attr(eps='primitive')
        #self.distribution = distribution

        super().__init__(mdp_info,distribution,  policy, features)  # inherit from BBO

    def _update(self, Jep, theta):
        max1, max2 = self.MI_features_selection(Jep, theta)
        eta_start = np.ones(1)

        res = minimize(myREPS_mi._dual_function, eta_start,
                       jac=myREPS_mi._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self.eps, Jep, theta))

        eta_opt = res.x.item()

        Jep -= np.max(Jep)
        d = np.exp(Jep / eta_opt)
        #self.distribution.mle(theta, d)

        ##
        eta_in = np.ones(2)
        res = minimize(myREPS_mi._lag_function_constrained, eta_in,
                       method='SLSQP',
                       jac=grad(myREPS_mi._lag_function_constrained),
                       args=(d, theta, self.distribution,self.eps,self.k),
                       bounds= ((0.0,np.inf), (0.0, np.inf)))
        eta_opt, omeg_opt = res.x[0], res.x[1]

        mu_pre, std_pre = self.distribution._mu, self.distribution._std
        sigma_pre = std_pre**2

        mu_change = (d @ theta + eta_opt * mu_pre) / (np.sum(d) + eta_opt)
        mu_post = mu_pre
        mu_post[max1] = mu_change[max1]
        mu_post[max2] = mu_change[max2]
        diff = theta - mu_post
        tmp = np.einsum('nk,n,nh->kh', diff, d, diff)
        sigma_post = (tmp + eta_opt * sigma_pre + eta_opt * np.outer(mu_post - mu_pre, mu_post - mu_pre)) / (np.sum(d) + eta_opt-omeg_opt)
        std_post = np.sqrt(sigma_post)

        self.distribution._mu=mu_post
        self.distribution._std = std_post
        kl = myREPS_mi._KL_M_Projection(mu_pre,sigma_pre,mu_post,sigma_post)
        entropydiff = myREPS_mi._entropy(sigma_pre)-myREPS_mi._entropy(sigma_post)




    @staticmethod
    def _KL_M_Projection(mu_pre, sigma_pre, mu_post, sigma_post):
        diff = mu_post - mu_pre
        inverse = np.linalg.inv(sigma_post)
        _, slog_post = np.linalg.slogdet(sigma_post)
        _, slog_pre = np.linalg.slogdet(sigma_pre)
        kl = 0.5 * (np.trace(inverse@ sigma_pre) + diff.T @ inverse @ diff + slog_post-slog_pre-mu_pre.shape[0])
        return kl
    @staticmethod
    def _entropy(sigma):
        c = sigma.shape[0]*np.log(2*np.pi)
        _, slog = np.linalg.slogdet(sigma)
        return 0.5 * slog+c/2+sigma.shape[0]/2

    @staticmethod
    def _lag_function_constrained(eta_array, *args):
        # constrained reps
        eta, omeg = eta_array
        d, theta, distribution, eps, k = args
        mu_pre, std_pre = distribution._mu, distribution._std
        sigma_pre = std_pre**2
        mu_post = (d @ theta + eta * mu_pre) / (np.sum(d) + eta)
        diff = theta - mu_post
        tmp = diff.T@np.diag(d)@diff
        sigma_post = (tmp + eta * sigma_pre + eta * np.outer(mu_post - mu_pre, mu_post - mu_pre)) / (np.sum(d) + eta-omeg)
        new_sum = myREPS_mi.logLikelihood(theta, mu_post, sigma_post, d) + eta * (eps - myREPS_mi._KL_M_Projection(mu_pre, sigma_pre, mu_post, sigma_post))+omeg*(myREPS_mi._entropy(sigma_post)-myREPS_mi._entropy(sigma_pre)+k)
        return new_sum

    @staticmethod
    def logLikelihood(theta, mu, sigma, d):
        c = mu.shape[0] * np.log(2 * np.pi)
        _, slog = np.linalg.slogdet(sigma)
        lik = - 0.5* d @ (np.einsum('nk,kh,nh->n', mu - theta, np.linalg.inv(sigma), mu - theta) + slog + c)
        return lik


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
        print('nocodi is', mi, )
        mi_sum = np.sum(mi, axis=1).tolist()
        max1 = mi_sum.index(max(mi_sum))
        del mi_sum[max1]
        max2 = mi_sum.index(max(mi_sum))

        return max1, max2



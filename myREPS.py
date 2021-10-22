import autograd.numpy as np
from autograd import grad


from scipy.optimize import minimize
import mushroom_rl.algorithms
from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization


class myREPS(BlackBoxOptimization):
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

        super().__init__(mdp_info, distribution, policy, features)  # inherit from BBO

    def _update(self, Jep, theta):
        eta_start = np.ones(1)

        res = minimize(myREPS._dual_function, eta_start,
                       jac=myREPS._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self.eps, Jep, theta))

        eta_opt = res.x.item()

        Jep -= np.max(Jep)
        d = np.exp(Jep / eta_opt)
        #self.distribution.mle(theta, d)

        ##
        eta_in = np.ones(2)
        res = minimize(myREPS._lag_function_constrained, eta_in,
                       method='SLSQP',
                       jac=grad(myREPS._lag_function_constrained),
                       args=(d, theta, self.distribution,self.eps,self.k),
                       bounds= ((0.0,np.inf), (0.0, np.inf)))
        print('the result is:',res.x, 'success is',res.success)
        eta_opt, omeg_opt = res.x[0], res.x[1]

        mu_pre, cholsigma_pre = self.distribution._mu, self.distribution._chol_sigma
        sigma_pre = cholsigma_pre @ cholsigma_pre.T
        mu_post = (d @ theta + eta_opt * mu_pre) / (np.sum(d) + eta_opt)
        diff = theta - mu_post
        tmp = np.einsum('nk,n,nh->kh', diff, d, diff)
        sigma_post = (tmp + eta_opt * sigma_pre + eta_opt * np.outer(mu_post - mu_pre, mu_post - mu_pre)) / (np.sum(d) + eta_opt-omeg_opt)

        self.distribution._mu=mu_post
        self.distribution._chol_sigma = np.linalg.cholesky(sigma_post)
        kl = myREPS._KL_M_Projection(mu_pre,sigma_pre,mu_post,sigma_post)
        entropydiff = myREPS._entropy(sigma_pre)-myREPS._entropy(sigma_post)
        print('KL is :',kl)
        print('entropy difference is:',entropydiff)




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
        mu_pre, choysigma_pre = distribution._mu, distribution._chol_sigma
        sigma_pre = choysigma_pre@choysigma_pre.T
        mu_post = (d @ theta + eta * mu_pre) / (np.sum(d) + eta)
        diff = theta - mu_post
        tmp = diff.T@np.diag(d)@diff
        sigma_post = (tmp + eta * sigma_pre + eta * np.outer(mu_post - mu_pre, mu_post - mu_pre)) / (np.sum(d) + eta-omeg)
        new_sum = myREPS.logLikelihood(theta, mu_post, sigma_post, d) + eta * (eps - myREPS._KL_M_Projection(mu_pre, sigma_pre, mu_post, sigma_post))+omeg*(myREPS._entropy(sigma_post)-myREPS._entropy(sigma_pre)+k)
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

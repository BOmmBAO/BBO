
import numpy as np
import torch
import torch.distributions

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class ProjectionModule:
    def __init__(self, mu_q, sigma_q=None, kl_ubound=0.01, entropy_lbound=None, lr = 0.0001, max_steps = 2000, device='cpu'):
        self.device = torch.device(device)
        self.mu_q = torch.tensor(mu_q, dtype=torch.float64, requires_grad=False, device=self.device)

        self.sigma_q = torch.tensor(sigma_q, dtype=torch.float64, device=self.device)
        self.sigma_q_inv = torch.inverse(self.sigma_q)

        self.epsilon = torch.tensor(kl_ubound, dtype=torch.float64, requires_grad=False, device=self.device)

        if entropy_lbound is not None:
            self.beta = torch.tensor(entropy_lbound, dtype=torch.float64, requires_grad=False, device=self.device)
            self.use_entropy_proj = True
        else:
            self.use_entropy_proj = False

        self.d = self.mu_q.shape[0]
        self.lr = lr
        self.max_steps = max_steps

        self.mu = torch.tensor(mu_q, dtype=torch.float64, requires_grad=True, device=self.device)
        self.L = tril2vec(torch.cholesky(self.sigma_q)).requires_grad_(True)

    def forward(self, Jep, theta, old_log_pdf):
        L = vec2tril(self.L, self.d)
        sigma = torch.mm(L, L.t())
        mu = self.mu
        if self.use_entropy_proj:
            sigma = self.entropy_projection(sigma)

        mu, sigma = self.kl_projection(mu, sigma)

        pi_proj = torch.distributions.MultivariateNormal(mu, sigma)
        imp_samp = torch.exp(pi_proj.log_prob(theta) - old_log_pdf)

        # log_pdf = pi_proj.log_prob(theta)
        # return {'loss':-torch.mean(log_pdf * Jep), 'sigma':sigma, 'mu':mu}
        return {'loss':-torch.mean(imp_samp * Jep), 'sigma':sigma, 'mu':mu}

    def optimize(self, theta, Jep):
        optimizer = torch.optim.Adam([self.mu, self.L], lr=0.01)
        # Jep = (Jep - torch.mean(Jep)) / (torch.std(Jep, unbiased=True) + torch.tensor(1e-8))
        # Jep = (Jep - torch.mean(Jep))
        old_dist = torch.distributions.MultivariateNormal(self.mu_q, self.sigma_q)
        old_log_pdf = old_dist.log_prob(theta)
        for i in range(self.max_steps):
            optimizer.zero_grad()
            result = self.forward(Jep, theta, old_log_pdf)
            loss = result['loss']
            loss.backward()

            optimizer.step()

            # kl_step = torch.distributions.kl_divergence(torch.distributions.MultivariateNormal(self.mu, self.sigma),
            #                                             torch.distributions.MultivariateNormal(self.mu_q, self.sigma_q))

            kl = torch.distributions.kl_divergence(torch.distributions.MultivariateNormal(result['mu'], result['sigma']),
                                              torch.distributions.MultivariateNormal(self.mu_q,self.sigma_q))
            entropy = torch.distributions.MultivariateNormal(result['mu'], result['sigma']).entropy()

            print("Iteration: {}, Projected KL: {:.5f}, Entropy: {:.5f}, Loss: {:.5f}".format(i, kl.item(), entropy.item(), loss.item()))

            # if torch.abs(kl_cur - kl_step)< 1e-8 or i == self.max_steps -1:
            if i == self.max_steps - 1:
                self.sigma = result['sigma']
                self.mu = result['mu']
                break
        return self.mu, self.sigma

    def kl_projection(self, mu, sigma):
        r = self.r(sigma)
        m = self.m(mu)
        e = self.e(sigma)

        eta_g = torch.tensor(1.)
        eta_m = torch.tensor(1.)

        init_kl = m + r + e
        if m + r + e > self.epsilon + 1e-6:
                eta_g = self.epsilon/ init_kl

        # L_proj = eta_g * self.L + (1. - eta_g) * self.L_q
        sigma = eta_g * sigma + (1. - eta_g) * self.sigma_q

        r = self.r(sigma)
        e = self.e(sigma)

        if m + r + e > self.epsilon + 1e-6:
            eta_m = torch.sqrt((self.epsilon - r - e) / m)
        mu = eta_m * self.mu + (1. - eta_m) * self.mu_q

        return mu, sigma

    def entropy_projection(self, sigma):
        # print(sigma)
        L = torch.cholesky(sigma)
        h = self.h(L)
        if h < -1e-8:
            L = L * torch.exp(-h/self.d)
        return torch.mm(L, L.t())

    def r(self, sigma):
        return 0.5*(torch.trace(torch.mm(self.sigma_q_inv , sigma)) - self.d)

    def e(self, sigma):
        return 0.5 * (torch.log(torch.det(self.sigma_q)) - torch.log(torch.det(sigma)))

    def m(self, mu):
        diff = mu-self.mu_q
        return 0.5*torch.dot(torch.mv(self.sigma_q_inv, diff),diff)

    def h(self, L):
        return 0.5 * self.d * (torch.log(torch.tensor(2. * np.pi, device=self.device)) + 1.) + torch.sum(torch.log(torch.diag(L))) - self.beta

def vec2tril(vec, dim):
    assert int(dim*(dim+1)/2) == vec.shape[0]
    tril = torch.zeros((dim, dim), dtype=torch.float64)
    tril_indices = torch.tril_indices(dim, dim)
    tril[tril_indices[0], tril_indices[1]]=vec
    return tril

def tril2vec(tril_src):
    return tril_src[torch.tril(torch.ones_like(tril_src)) == 1]

if __name__ == "__main__":

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    n_samples = 200
    q_dist = torc

    h.distributions.MultivariateNormal(torch.tensor([1., 1.],device=device), torch.tensor([[2.5, 1.5],[1.5,4.5]], device=device))
    kl_bound = 10
    entropy_lbound = 4

    sample_dist = torch.distributions.MultivariateNormal(torch.tensor([0.,0.], device=device),
                                                         torch.tensor([[5.,0.],[0.,5.]], device=device))

    pps = ProjectionModule([0.,0.], sigma_q=[[5.,0.],[0.,5.]],kl_ubound=kl_bound, max_steps=2000, entropy_lbound = entropy_lbound, device=device)
    # torch.manual_seed(5)
    theta = sample_dist.sample(torch.Size((n_samples,)))
    Jep = torch.exp(q_dist.log_prob(theta))
    r = pps.optimize(theta, Jep)
    print(r[0])
    print(r[1])
    # print(pps.forward(Jep, theta)['loss'].item())
    print(torch.distributions.kl_divergence(torch.distributions.MultivariateNormal(r[0], r[1]), torch.distributions.MultivariateNormal(pps.mu_q, pps.sigma_q)))
    print(torch.distributions.MultivariateNormal(r[0], r[1]).entropy())

    plt.figure()
    ax = plt.gca()
    plt.scatter(theta.cpu().numpy()[:,0], theta.cpu().numpy()[:,1], marker='.', alpha=0.2)

    # Learned distribution
    cov = r[1].detach().numpy()
    mu = r[0].detach().numpy()
    w, v = np.linalg.eigh(cov)
    a = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    width, height = 2 * np.sqrt(w)
    ell = Ellipse(mu, width, height, a, edgecolor='r', fc='None')
    ax.add_patch(ell)

    # Plot true distribution
    mu2 = np.array([1., 1.])
    cov2 = np.array([[2.5, 1.5],[1.5,4.5]])
    w2, v2 = np.linalg.eigh(cov2)
    a2 = np.degrees(np.arctan2(v2[1, 0], v2[0, 0]))
    width2, height2 = 2 * np.sqrt(w2)
    ell2 = Ellipse(mu2, width2, height2, a2, edgecolor='b', fc='None')
    ax.add_patch(ell2)

    # Start Distribution
    mu3 = np.array([0., 0.])
    cov3 = np.array([[2., 0.], [0., 2.]])
    w3, v3 = np.linalg.eigh(cov3)
    a3 = np.degrees(np.arctan2(v3[1, 0], v3[0, 0]))
    width3, height3 = 2 * np.sqrt(w3)
    ell2 = Ellipse(mu3, width3, height3, a3, edgecolor='k', fc='None')
    ax.add_patch(ell2)

    plt.show()
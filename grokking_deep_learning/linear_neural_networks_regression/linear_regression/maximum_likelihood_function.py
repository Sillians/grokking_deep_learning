import numpy as np
from normal_distribution import normal

# Likelihood function
# data and variance
x = np.array([2.5, 1.8, 2.5, 2.3, 1.7])
sigma = 1.0

# define likelihood function (product of densities)
def likelihood(x, mu, sigma):
    densities = normal(x, mu, sigma)
    return np.prod(densities)

# search range for μ
mu_values = np.linspace(-5, 5, 2000)

# compute the likelihood for each candidate μ
likelihoods = [likelihood(x, mu, sigma) for mu in mu_values]

# find the μ that maximizes the likelihood
mle_mu = mu_values[np.argmax(likelihoods)]
print("MLE for mu:", mle_mu)
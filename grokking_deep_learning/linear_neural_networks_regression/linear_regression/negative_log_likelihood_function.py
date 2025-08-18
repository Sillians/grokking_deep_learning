import numpy as np


# This implementation is tailored for regression tasks.
# This simply computes the negative log-likelihood for given inputs.
def negative_log_likelihood(y_true, y_pred, variance):
    """
    Compute the negative log-likelihood for a Gaussian distribution.

    Parameters:
    y_true (np.ndarray): True values (observations).
    y_pred (np.ndarray): Predicted values (mean of the Gaussian distribution).
    variance (float): Variance of the Gaussian distribution.

    Returns:
    float: Negative log-likelihood value.
    """
    n = len(y_true)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * variance) - np.sum((y_true - y_pred)**2) / (2 * variance)
    return -log_likelihood






# This implementation is focused on estimating the mean(mu) using MLE
# This includes a search grid for possible (mu) values and computes the MLE for mu by minimizing the negative likelihood estimation (MLE)
data = np.array([2.3, 1.9, 2.7, 2.5, 1.8])

# known variance
sigma = 1.0

# negative log-likelihood
def negative_log_likelihood(mu, data, sigma):
    n = len(data)
    squared_term = np.sum((data - mu)**2) / (2 * sigma**2)
    constant_term = (n / 2) * np.log(2 * np.pi * sigma**2)
    return squared_term + constant_term

# search grid for possible mu values
mu_values = np.linspace(-5, 5, 2000)

# compute negative log-likelihoods
nll_values = [negative_log_likelihood(mu, data, sigma) for mu in mu_values]

# minimizer of negative log-likelihood
mle_mu = mu_values[np.argmin(nll_values)]
print("MLE for mu:", mle_mu)




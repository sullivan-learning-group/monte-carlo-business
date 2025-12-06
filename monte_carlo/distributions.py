"""
Common probability distributions for business decision modeling
"""

import numpy as np
from scipy import stats
from typing import Optional


def normal_distribution(mean: float, std: float, size: int,
                       random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a normal (Gaussian) distribution.

    Useful for modeling variables with symmetric uncertainty around a mean value.

    Parameters
    ----------
    mean : float
        Mean (center) of the distribution
    std : float
        Standard deviation (spread)
    size : int
        Number of samples to generate
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of samples
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.normal(mean, std, size)


def lognormal_distribution(mean: float, std: float, size: int,
                          random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a lognormal distribution.

    Useful for modeling variables that cannot be negative (e.g., prices, revenues)
    and have right-skewed distributions.

    Parameters
    ----------
    mean : float
        Mean of the underlying normal distribution
    std : float
        Standard deviation of the underlying normal distribution
    size : int
        Number of samples to generate
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of samples (all positive)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.lognormal(mean, std, size)


def uniform_distribution(low: float, high: float, size: int,
                        random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a uniform distribution.

    Useful when all values in a range are equally likely.

    Parameters
    ----------
    low : float
        Lower bound (inclusive)
    high : float
        Upper bound (exclusive)
    size : int
        Number of samples to generate
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of samples
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.uniform(low, high, size)


def triangular_distribution(low: float, mode: float, high: float, size: int,
                           random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a triangular distribution.

    Useful for modeling expert estimates with min, most likely, and max values.

    Parameters
    ----------
    low : float
        Minimum value
    mode : float
        Most likely value (peak of distribution)
    high : float
        Maximum value
    size : int
        Number of samples to generate
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of samples
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.triangular(low, mode, high, size)


def beta_distribution(alpha: float, beta: float, low: float, high: float,
                     size: int, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a beta distribution scaled to [low, high].

    Useful for modeling percentages, probabilities, or bounded variables.

    Parameters
    ----------
    alpha : float
        First shape parameter (alpha > 0)
    beta : float
        Second shape parameter (beta > 0)
    low : float
        Lower bound
    high : float
        Upper bound
    size : int
        Number of samples to generate
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of samples between low and high
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    samples = np.random.beta(alpha, beta, size)
    return low + (high - low) * samples


def pert_distribution(low: float, mode: float, high: float, size: int,
                     lambda_param: float = 4.0,
                     random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a PERT (Program Evaluation and Review Technique) distribution.

    Similar to triangular but with smoother shape. Very popular in project management
    and business modeling.

    Parameters
    ----------
    low : float
        Minimum value (optimistic estimate)
    mode : float
        Most likely value
    high : float
        Maximum value (pessimistic estimate)
    size : int
        Number of samples to generate
    lambda_param : float, default=4.0
        Shape parameter (typically 4, higher values make it more peaked)
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of samples
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate PERT mean and standard deviation
    mean = (low + lambda_param * mode + high) / (lambda_param + 2)

    # Calculate alpha and beta for equivalent beta distribution
    if high == low:
        return np.full(size, mode)

    alpha = ((mean - low) * (2 * mode - low - high)) / ((mode - mean) * (high - low))
    beta = alpha * (high - mean) / (mean - low)

    # Generate beta distribution and scale
    samples = np.random.beta(alpha, beta, size)
    return low + samples * (high - low)


def gamma_distribution(shape: float, scale: float, size: int,
                      random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a gamma distribution.

    Useful for modeling waiting times, lifetimes, and other positive continuous variables.

    Parameters
    ----------
    shape : float
        Shape parameter (k)
    scale : float
        Scale parameter (theta)
    size : int
        Number of samples to generate
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of samples (all positive)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    return np.random.gamma(shape, scale, size)


def discrete_distribution(values: list, probabilities: list, size: int,
                         random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate samples from a discrete probability distribution.

    Useful for modeling scenarios with specific discrete outcomes.

    Parameters
    ----------
    values : list
        Possible values
    probabilities : list
        Probability of each value (must sum to 1)
    size : int
        Number of samples to generate
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Array of samples

    Examples
    --------
    >>> # Market scenarios: bear (30%), neutral (50%), bull (20%)
    >>> values = [-0.20, 0.05, 0.30]  # returns
    >>> probs = [0.30, 0.50, 0.20]
    >>> samples = discrete_distribution(values, probs, 1000)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    probabilities = np.array(probabilities)
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("Probabilities must sum to 1")

    return np.random.choice(values, size=size, p=probabilities)


def correlated_normal(mean1: float, std1: float, mean2: float, std2: float,
                     correlation: float, size: int,
                     random_seed: Optional[int] = None) -> tuple:
    """
    Generate correlated samples from two normal distributions.

    Useful for modeling related business variables (e.g., sales and marketing spend).

    Parameters
    ----------
    mean1 : float
        Mean of first variable
    std1 : float
        Standard deviation of first variable
    mean2 : float
        Mean of second variable
    std2 : float
        Standard deviation of second variable
    correlation : float
        Correlation coefficient between -1 and 1
    size : int
        Number of samples to generate
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple of np.ndarray
        (samples1, samples2) - two correlated arrays

    Examples
    --------
    >>> # Revenue and costs are often correlated
    >>> revenue, costs = correlated_normal(100000, 10000, 60000, 5000, 0.7, 1000)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if not -1 <= correlation <= 1:
        raise ValueError("Correlation must be between -1 and 1")

    # Generate correlated standard normal variables
    cov_matrix = [[1, correlation], [correlation, 1]]
    samples = np.random.multivariate_normal([0, 0], cov_matrix, size)

    # Scale and shift to desired distributions
    samples1 = mean1 + std1 * samples[:, 0]
    samples2 = mean2 + std2 * samples[:, 1]

    return samples1, samples2

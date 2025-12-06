"""
Core Monte Carlo Simulator for Business Decision Making
"""

import numpy as np
from scipy import stats
from typing import Callable, Dict, List, Optional, Union
import warnings


class MonteCarloSimulator:
    """
    Monte Carlo Simulator for business decision making and risk analysis.

    This class provides methods to run Monte Carlo simulations with various
    probability distributions to model uncertainty in business decisions.
    """

    def __init__(self, n_simulations: int = 10000, random_seed: Optional[int] = None):
        """
        Initialize the Monte Carlo Simulator.

        Parameters
        ----------
        n_simulations : int, default=10000
            Number of simulation iterations to run
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def simulate(self, model: Callable, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Run Monte Carlo simulation with a given model and input distributions.

        Parameters
        ----------
        model : callable
            Function that takes keyword arguments and returns a scalar result.
            Each keyword should correspond to a key in the inputs dict.
        inputs : dict
            Dictionary where keys are parameter names and values are numpy arrays
            of sampled values (should all have length n_simulations)

        Returns
        -------
        np.ndarray
            Array of simulation results (length n_simulations)

        Examples
        --------
        >>> sim = MonteCarloSimulator(n_simulations=1000)
        >>> revenue = np.random.normal(100000, 10000, 1000)
        >>> costs = np.random.normal(60000, 5000, 1000)
        >>> inputs = {'revenue': revenue, 'costs': costs}
        >>> results = sim.simulate(lambda revenue, costs: revenue - costs, inputs)
        """
        # Validate inputs
        lengths = [len(v) for v in inputs.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All input arrays must have the same length")
        if lengths[0] != self.n_simulations:
            warnings.warn(f"Input length ({lengths[0]}) differs from n_simulations ({self.n_simulations})")

        # Run simulation
        results = np.zeros(lengths[0])
        for i in range(lengths[0]):
            kwargs = {key: inputs[key][i] for key in inputs}
            results[i] = model(**kwargs)

        return results

    def analyze_results(self, results: np.ndarray, confidence_levels: List[float] = [0.90, 0.95, 0.99]) -> Dict:
        """
        Analyze simulation results and compute statistics.

        Parameters
        ----------
        results : np.ndarray
            Array of simulation results
        confidence_levels : list of float, default=[0.90, 0.95, 0.99]
            Confidence levels for computing confidence intervals

        Returns
        -------
        dict
            Dictionary containing statistical analysis:
            - mean: average outcome
            - median: median outcome
            - std: standard deviation
            - min: minimum outcome
            - max: maximum outcome
            - percentiles: key percentile values
            - confidence_intervals: confidence intervals at specified levels
            - probability_positive: probability of positive outcome
        """
        analysis = {
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'min': np.min(results),
            'max': np.max(results),
            'percentiles': {
                'p5': np.percentile(results, 5),
                'p10': np.percentile(results, 10),
                'p25': np.percentile(results, 25),
                'p75': np.percentile(results, 75),
                'p90': np.percentile(results, 90),
                'p95': np.percentile(results, 95),
            },
            'confidence_intervals': {},
            'probability_positive': np.mean(results > 0),
            'probability_negative': np.mean(results < 0),
        }

        # Calculate confidence intervals
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower = np.percentile(results, 100 * alpha / 2)
            upper = np.percentile(results, 100 * (1 - alpha / 2))
            analysis['confidence_intervals'][f'{int(conf_level*100)}%'] = (lower, upper)

        return analysis

    def probability_threshold(self, results: np.ndarray, threshold: float,
                            direction: str = 'greater') -> float:
        """
        Calculate probability that results exceed or fall below a threshold.

        Parameters
        ----------
        results : np.ndarray
            Array of simulation results
        threshold : float
            Threshold value to compare against
        direction : str, default='greater'
            Either 'greater' or 'less' to specify comparison direction

        Returns
        -------
        float
            Probability (between 0 and 1)
        """
        if direction == 'greater':
            return np.mean(results > threshold)
        elif direction == 'less':
            return np.mean(results < threshold)
        else:
            raise ValueError("direction must be 'greater' or 'less'")

    def value_at_risk(self, results: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) at a given confidence level.

        VaR represents the maximum expected loss at a given confidence level.
        For example, 95% VaR means there's only a 5% chance of losing more than this amount.

        Parameters
        ----------
        results : np.ndarray
            Array of simulation results (e.g., profits/losses)
        confidence_level : float, default=0.95
            Confidence level (between 0 and 1)

        Returns
        -------
        float
            Value at Risk
        """
        return np.percentile(results, (1 - confidence_level) * 100)

    def expected_shortfall(self, results: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR) at a given confidence level.

        Expected Shortfall is the expected loss given that the loss exceeds VaR.

        Parameters
        ----------
        results : np.ndarray
            Array of simulation results
        confidence_level : float, default=0.95
            Confidence level (between 0 and 1)

        Returns
        -------
        float
            Expected Shortfall
        """
        var = self.value_at_risk(results, confidence_level)
        return np.mean(results[results <= var])

    def sensitivity_analysis(self, model: Callable, base_inputs: Dict[str, np.ndarray],
                           variable: str, variation_range: tuple = (0.8, 1.2),
                           n_points: int = 10) -> Dict:
        """
        Perform sensitivity analysis on a specific input variable.

        Parameters
        ----------
        model : callable
            Model function
        base_inputs : dict
            Base input distributions
        variable : str
            Name of variable to analyze
        variation_range : tuple, default=(0.8, 1.2)
            Range to vary the variable (as multipliers)
        n_points : int, default=10
            Number of points to sample in the range

        Returns
        -------
        dict
            Dictionary with 'multipliers' and 'mean_results' arrays
        """
        multipliers = np.linspace(variation_range[0], variation_range[1], n_points)
        mean_results = []

        for mult in multipliers:
            inputs = base_inputs.copy()
            inputs[variable] = base_inputs[variable] * mult
            results = self.simulate(model, inputs)
            mean_results.append(np.mean(results))

        return {
            'multipliers': multipliers,
            'mean_results': np.array(mean_results),
            'variable': variable
        }

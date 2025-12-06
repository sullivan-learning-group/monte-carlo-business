"""
Pytest configuration and shared fixtures for Monte Carlo simulation tests
"""

import pytest
import numpy as np
from monte_carlo import MonteCarloSimulator


@pytest.fixture
def small_sim():
    """Fixture providing a simulator with small number of simulations for fast tests"""
    return MonteCarloSimulator(n_simulations=100, random_seed=42)


@pytest.fixture
def standard_sim():
    """Fixture providing a simulator with standard number of simulations"""
    return MonteCarloSimulator(n_simulations=1000, random_seed=42)


@pytest.fixture
def large_sim():
    """Fixture providing a simulator with large number of simulations"""
    return MonteCarloSimulator(n_simulations=10000, random_seed=42)


@pytest.fixture
def sample_results():
    """Fixture providing sample simulation results for testing analysis methods"""
    np.random.seed(42)
    # Generate sample results: mix of positive and negative values
    return np.random.normal(100, 50, 1000)


@pytest.fixture
def simple_model():
    """Fixture providing a simple profit model for testing"""
    def profit_model(revenue, costs):
        return revenue - costs
    return profit_model


@pytest.fixture
def sample_inputs():
    """Fixture providing sample inputs for simulation"""
    np.random.seed(42)
    return {
        'revenue': np.random.normal(1000, 100, 1000),
        'costs': np.random.normal(600, 50, 1000)
    }

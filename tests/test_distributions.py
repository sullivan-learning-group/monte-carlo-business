"""
Unit tests for probability distribution functions
"""

import pytest
import numpy as np
from monte_carlo.distributions import (
    normal_distribution,
    lognormal_distribution,
    uniform_distribution,
    triangular_distribution,
    beta_distribution,
    pert_distribution,
    gamma_distribution,
    discrete_distribution,
    correlated_normal
)


class TestNormalDistribution:
    """Tests for normal_distribution function"""

    def test_output_size(self):
        """Test that output has correct size"""
        samples = normal_distribution(100, 10, size=1000, random_seed=42)
        assert len(samples) == 1000

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        samples1 = normal_distribution(100, 10, size=1000, random_seed=42)
        samples2 = normal_distribution(100, 10, size=1000, random_seed=42)
        np.testing.assert_array_equal(samples1, samples2)

    def test_statistical_properties(self):
        """Test that mean and std are approximately correct"""
        samples = normal_distribution(100, 10, size=10000, random_seed=42)
        assert np.abs(np.mean(samples) - 100) < 1  # Mean within 1 of expected
        assert np.abs(np.std(samples) - 10) < 0.5  # Std within 0.5 of expected

    def test_different_seeds(self):
        """Test that different seeds produce different results"""
        samples1 = normal_distribution(100, 10, size=1000, random_seed=42)
        samples2 = normal_distribution(100, 10, size=1000, random_seed=43)
        assert not np.array_equal(samples1, samples2)


class TestLognormalDistribution:
    """Tests for lognormal_distribution function"""

    def test_output_size(self):
        """Test that output has correct size"""
        samples = lognormal_distribution(4, 0.5, size=1000, random_seed=42)
        assert len(samples) == 1000

    def test_all_positive(self):
        """Test that all values are positive"""
        samples = lognormal_distribution(4, 0.5, size=1000, random_seed=42)
        assert np.all(samples > 0)

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        samples1 = lognormal_distribution(4, 0.5, size=1000, random_seed=42)
        samples2 = lognormal_distribution(4, 0.5, size=1000, random_seed=42)
        np.testing.assert_array_equal(samples1, samples2)

    def test_right_skewed(self):
        """Test that distribution is right-skewed (mean > median)"""
        samples = lognormal_distribution(4, 0.5, size=10000, random_seed=42)
        assert np.mean(samples) > np.median(samples)


class TestUniformDistribution:
    """Tests for uniform_distribution function"""

    def test_output_size(self):
        """Test that output has correct size"""
        samples = uniform_distribution(10, 20, size=1000, random_seed=42)
        assert len(samples) == 1000

    def test_bounds(self):
        """Test that all values are within bounds"""
        samples = uniform_distribution(10, 20, size=1000, random_seed=42)
        assert np.all(samples >= 10)
        assert np.all(samples < 20)

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        samples1 = uniform_distribution(10, 20, size=1000, random_seed=42)
        samples2 = uniform_distribution(10, 20, size=1000, random_seed=42)
        np.testing.assert_array_equal(samples1, samples2)

    def test_mean_approximately_center(self):
        """Test that mean is approximately at center of range"""
        samples = uniform_distribution(10, 20, size=10000, random_seed=42)
        expected_mean = (10 + 20) / 2
        assert np.abs(np.mean(samples) - expected_mean) < 0.2


class TestTriangularDistribution:
    """Tests for triangular_distribution function"""

    def test_output_size(self):
        """Test that output has correct size"""
        samples = triangular_distribution(10, 15, 20, size=1000, random_seed=42)
        assert len(samples) == 1000

    def test_bounds(self):
        """Test that all values are within bounds"""
        samples = triangular_distribution(10, 15, 20, size=1000, random_seed=42)
        assert np.all(samples >= 10)
        assert np.all(samples <= 20)

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        samples1 = triangular_distribution(10, 15, 20, size=1000, random_seed=42)
        samples2 = triangular_distribution(10, 15, 20, size=1000, random_seed=42)
        np.testing.assert_array_equal(samples1, samples2)

    def test_mode_influences_mean(self):
        """Test that mode closer to low results in lower mean"""
        samples_low_mode = triangular_distribution(10, 11, 20, size=10000, random_seed=42)
        samples_high_mode = triangular_distribution(10, 19, 20, size=10000, random_seed=43)
        assert np.mean(samples_low_mode) < np.mean(samples_high_mode)


class TestBetaDistribution:
    """Tests for beta_distribution function"""

    def test_output_size(self):
        """Test that output has correct size"""
        samples = beta_distribution(2, 5, 0, 100, size=1000, random_seed=42)
        assert len(samples) == 1000

    def test_bounds(self):
        """Test that all values are within specified bounds"""
        samples = beta_distribution(2, 5, 10, 20, size=1000, random_seed=42)
        assert np.all(samples >= 10)
        assert np.all(samples <= 20)

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        samples1 = beta_distribution(2, 5, 0, 100, size=1000, random_seed=42)
        samples2 = beta_distribution(2, 5, 0, 100, size=1000, random_seed=42)
        np.testing.assert_array_equal(samples1, samples2)

    def test_shape_parameters(self):
        """Test that alpha > beta results in right-skewed distribution"""
        samples = beta_distribution(5, 2, 0, 100, size=10000, random_seed=42)
        # With alpha > beta, mean should be > 50 (center)
        assert np.mean(samples) > 50


class TestPertDistribution:
    """Tests for pert_distribution function"""

    def test_output_size(self):
        """Test that output has correct size"""
        samples = pert_distribution(10, 15, 20, size=1000, random_seed=42)
        assert len(samples) == 1000

    def test_bounds(self):
        """Test that all values are within bounds"""
        samples = pert_distribution(10, 15, 20, size=1000, random_seed=42)
        assert np.all(samples >= 10)
        assert np.all(samples <= 20)

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        samples1 = pert_distribution(10, 15, 20, size=1000, random_seed=42)
        samples2 = pert_distribution(10, 15, 20, size=1000, random_seed=42)
        np.testing.assert_array_equal(samples1, samples2)

    def test_edge_case_equal_values(self):
        """Test edge case where low == high"""
        samples = pert_distribution(15, 15, 15, size=1000, random_seed=42)
        assert np.all(samples == 15)

    def test_lambda_parameter_effect(self):
        """Test that different lambda values affect the distribution"""
        samples_lambda_2 = pert_distribution(10, 15, 20, size=10000, lambda_param=2.0, random_seed=42)
        samples_lambda_6 = pert_distribution(10, 15, 20, size=10000, lambda_param=6.0, random_seed=43)
        # Higher lambda should result in lower variance (more peaked)
        assert np.std(samples_lambda_6) < np.std(samples_lambda_2)


class TestGammaDistribution:
    """Tests for gamma_distribution function"""

    def test_output_size(self):
        """Test that output has correct size"""
        samples = gamma_distribution(2, 2, size=1000, random_seed=42)
        assert len(samples) == 1000

    def test_all_positive(self):
        """Test that all values are positive"""
        samples = gamma_distribution(2, 2, size=1000, random_seed=42)
        assert np.all(samples > 0)

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        samples1 = gamma_distribution(2, 2, size=1000, random_seed=42)
        samples2 = gamma_distribution(2, 2, size=1000, random_seed=42)
        np.testing.assert_array_equal(samples1, samples2)

    def test_mean_approximately_correct(self):
        """Test that mean is approximately shape * scale"""
        shape, scale = 3, 4
        samples = gamma_distribution(shape, scale, size=10000, random_seed=42)
        expected_mean = shape * scale
        assert np.abs(np.mean(samples) - expected_mean) < 0.5


class TestDiscreteDistribution:
    """Tests for discrete_distribution function"""

    def test_output_size(self):
        """Test that output has correct size"""
        samples = discrete_distribution([1, 2, 3], [0.2, 0.5, 0.3], size=1000, random_seed=42)
        assert len(samples) == 1000

    def test_only_specified_values(self):
        """Test that output only contains specified values"""
        values = [10, 20, 30]
        samples = discrete_distribution(values, [0.3, 0.4, 0.3], size=1000, random_seed=42)
        assert np.all(np.isin(samples, values))

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        samples1 = discrete_distribution([1, 2, 3], [0.2, 0.5, 0.3], size=1000, random_seed=42)
        samples2 = discrete_distribution([1, 2, 3], [0.2, 0.5, 0.3], size=1000, random_seed=42)
        np.testing.assert_array_equal(samples1, samples2)

    def test_probability_distribution(self):
        """Test that values appear with approximately correct frequencies"""
        values = [1, 2, 3]
        probs = [0.2, 0.5, 0.3]
        samples = discrete_distribution(values, probs, size=10000, random_seed=42)

        for value, expected_prob in zip(values, probs):
            actual_prob = np.mean(samples == value)
            assert np.abs(actual_prob - expected_prob) < 0.02  # Within 2%

    def test_invalid_probabilities_sum(self):
        """Test that error is raised if probabilities don't sum to 1"""
        with pytest.raises(ValueError, match="Probabilities must sum to 1"):
            discrete_distribution([1, 2, 3], [0.2, 0.3, 0.3], size=100, random_seed=42)

    def test_single_value(self):
        """Test with single value (probability = 1)"""
        samples = discrete_distribution([42], [1.0], size=1000, random_seed=42)
        assert np.all(samples == 42)


class TestCorrelatedNormal:
    """Tests for correlated_normal function"""

    def test_output_size(self):
        """Test that output has correct size"""
        samples1, samples2 = correlated_normal(100, 10, 50, 5, 0.7, size=1000, random_seed=42)
        assert len(samples1) == 1000
        assert len(samples2) == 1000

    def test_reproducibility(self):
        """Test that same seed produces same results"""
        s1_a, s2_a = correlated_normal(100, 10, 50, 5, 0.7, size=1000, random_seed=42)
        s1_b, s2_b = correlated_normal(100, 10, 50, 5, 0.7, size=1000, random_seed=42)
        np.testing.assert_array_equal(s1_a, s1_b)
        np.testing.assert_array_equal(s2_a, s2_b)

    def test_positive_correlation(self):
        """Test that positive correlation is achieved"""
        samples1, samples2 = correlated_normal(100, 10, 50, 5, 0.8, size=10000, random_seed=42)
        correlation = np.corrcoef(samples1, samples2)[0, 1]
        assert correlation > 0.75  # Should be close to 0.8

    def test_negative_correlation(self):
        """Test that negative correlation is achieved"""
        samples1, samples2 = correlated_normal(100, 10, 50, 5, -0.8, size=10000, random_seed=42)
        correlation = np.corrcoef(samples1, samples2)[0, 1]
        assert correlation < -0.75  # Should be close to -0.8

    def test_zero_correlation(self):
        """Test that zero correlation results in independent variables"""
        samples1, samples2 = correlated_normal(100, 10, 50, 5, 0.0, size=10000, random_seed=42)
        correlation = np.corrcoef(samples1, samples2)[0, 1]
        assert np.abs(correlation) < 0.1  # Should be close to 0

    def test_invalid_correlation_high(self):
        """Test that error is raised for correlation > 1"""
        with pytest.raises(ValueError, match="Correlation must be between -1 and 1"):
            correlated_normal(100, 10, 50, 5, 1.5, size=100, random_seed=42)

    def test_invalid_correlation_low(self):
        """Test that error is raised for correlation < -1"""
        with pytest.raises(ValueError, match="Correlation must be between -1 and 1"):
            correlated_normal(100, 10, 50, 5, -1.5, size=100, random_seed=42)

    def test_statistical_properties(self):
        """Test that means and stds are approximately correct"""
        mean1, std1, mean2, std2 = 100, 10, 50, 5
        samples1, samples2 = correlated_normal(mean1, std1, mean2, std2, 0.5, size=10000, random_seed=42)

        assert np.abs(np.mean(samples1) - mean1) < 1
        assert np.abs(np.std(samples1) - std1) < 0.5
        assert np.abs(np.mean(samples2) - mean2) < 0.5
        assert np.abs(np.std(samples2) - std2) < 0.3


class TestDistributionCommonProperties:
    """Test common properties across all distributions"""

    @pytest.mark.parametrize("dist_func,args", [
        (normal_distribution, (100, 10, 1000)),
        (lognormal_distribution, (4, 0.5, 1000)),
        (uniform_distribution, (10, 20, 1000)),
        (triangular_distribution, (10, 15, 20, 1000)),
        (beta_distribution, (2, 5, 0, 100, 1000)),
        (pert_distribution, (10, 15, 20, 1000)),
        (gamma_distribution, (2, 2, 1000)),
    ])
    def test_no_nan_values(self, dist_func, args):
        """Test that distributions don't produce NaN values"""
        if dist_func == normal_distribution:
            samples = dist_func(args[0], args[1], size=args[2], random_seed=42)
        elif dist_func == lognormal_distribution:
            samples = dist_func(args[0], args[1], size=args[2], random_seed=42)
        elif dist_func == uniform_distribution:
            samples = dist_func(args[0], args[1], size=args[2], random_seed=42)
        elif dist_func == triangular_distribution:
            samples = dist_func(args[0], args[1], args[2], size=args[3], random_seed=42)
        elif dist_func == beta_distribution:
            samples = dist_func(args[0], args[1], args[2], args[3], size=args[4], random_seed=42)
        elif dist_func == pert_distribution:
            samples = dist_func(args[0], args[1], args[2], size=args[3], random_seed=42)
        elif dist_func == gamma_distribution:
            samples = dist_func(args[0], args[1], size=args[2], random_seed=42)

        assert not np.any(np.isnan(samples))

    @pytest.mark.parametrize("dist_func,args", [
        (normal_distribution, (100, 10, 1000)),
        (lognormal_distribution, (4, 0.5, 1000)),
        (uniform_distribution, (10, 20, 1000)),
        (triangular_distribution, (10, 15, 20, 1000)),
        (beta_distribution, (2, 5, 0, 100, 1000)),
        (pert_distribution, (10, 15, 20, 1000)),
        (gamma_distribution, (2, 2, 1000)),
    ])
    def test_no_inf_values(self, dist_func, args):
        """Test that distributions don't produce infinite values"""
        if dist_func == normal_distribution:
            samples = dist_func(args[0], args[1], size=args[2], random_seed=42)
        elif dist_func == lognormal_distribution:
            samples = dist_func(args[0], args[1], size=args[2], random_seed=42)
        elif dist_func == uniform_distribution:
            samples = dist_func(args[0], args[1], size=args[2], random_seed=42)
        elif dist_func == triangular_distribution:
            samples = dist_func(args[0], args[1], args[2], size=args[3], random_seed=42)
        elif dist_func == beta_distribution:
            samples = dist_func(args[0], args[1], args[2], args[3], size=args[4], random_seed=42)
        elif dist_func == pert_distribution:
            samples = dist_func(args[0], args[1], args[2], size=args[3], random_seed=42)
        elif dist_func == gamma_distribution:
            samples = dist_func(args[0], args[1], size=args[2], random_seed=42)

        assert not np.any(np.isinf(samples))

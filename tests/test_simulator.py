"""
Unit tests for MonteCarloSimulator class
"""

import pytest
import numpy as np
import warnings
from monte_carlo import MonteCarloSimulator


class TestMonteCarloSimulatorInit:
    """Tests for MonteCarloSimulator initialization"""

    def test_default_initialization(self):
        """Test initialization with default parameters"""
        sim = MonteCarloSimulator()
        assert sim.n_simulations == 10000
        assert sim.random_seed is None

    def test_custom_n_simulations(self):
        """Test initialization with custom number of simulations"""
        sim = MonteCarloSimulator(n_simulations=5000)
        assert sim.n_simulations == 5000

    def test_with_random_seed(self):
        """Test initialization with random seed"""
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)
        assert sim.random_seed == 42


class TestSimulate:
    """Tests for simulate method"""

    def test_basic_simulation(self, standard_sim, simple_model, sample_inputs):
        """Test basic simulation with simple model"""
        results = standard_sim.simulate(simple_model, sample_inputs)
        assert len(results) == 1000
        assert isinstance(results, np.ndarray)

    def test_simulation_correctness(self, standard_sim):
        """Test that simulation produces correct results"""
        # Simple deterministic test
        np.random.seed(42)
        inputs = {
            'a': np.array([1, 2, 3, 4, 5]),
            'b': np.array([10, 20, 30, 40, 50])
        }
        model = lambda a, b: a + b

        sim = MonteCarloSimulator(n_simulations=5, random_seed=42)
        results = sim.simulate(model, inputs)

        expected = np.array([11, 22, 33, 44, 55])
        np.testing.assert_array_equal(results, expected)

    def test_complex_model(self, standard_sim):
        """Test with more complex model"""
        np.random.seed(42)
        inputs = {
            'revenue': np.random.normal(1000, 100, 1000),
            'costs': np.random.normal(600, 50, 1000),
            'tax_rate': np.random.uniform(0.2, 0.3, 1000)
        }

        def complex_model(revenue, costs, tax_rate):
            profit_before_tax = revenue - costs
            return profit_before_tax * (1 - tax_rate)

        results = standard_sim.simulate(complex_model, inputs)
        assert len(results) == 1000
        assert np.all(results > 0)  # Should all be positive for this scenario

    def test_mismatched_input_lengths(self, standard_sim):
        """Test that error is raised for mismatched input array lengths"""
        inputs = {
            'a': np.array([1, 2, 3]),
            'b': np.array([10, 20])  # Different length
        }
        model = lambda a, b: a + b

        with pytest.raises(ValueError, match="All input arrays must have the same length"):
            standard_sim.simulate(model, inputs)

    def test_warning_for_length_mismatch(self, standard_sim):
        """Test that warning is issued when input length differs from n_simulations"""
        inputs = {
            'a': np.array([1, 2, 3, 4, 5]),  # Only 5, but sim expects 1000
            'b': np.array([10, 20, 30, 40, 50])
        }
        model = lambda a, b: a + b

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = standard_sim.simulate(model, inputs)
            assert len(w) == 1
            assert "differs from n_simulations" in str(w[0].message)

    def test_empty_inputs(self):
        """Test simulation with empty inputs"""
        sim = MonteCarloSimulator(n_simulations=0, random_seed=42)
        inputs = {'a': np.array([]), 'b': np.array([])}
        model = lambda a, b: a + b

        results = sim.simulate(model, inputs)
        assert len(results) == 0

    def test_single_input(self):
        """Test simulation with single input variable"""
        sim = MonteCarloSimulator(n_simulations=100, random_seed=42)
        inputs = {'x': np.random.normal(50, 10, 100)}
        model = lambda x: x * 2

        results = sim.simulate(model, inputs)
        assert len(results) == 100
        np.testing.assert_array_almost_equal(results, inputs['x'] * 2)


class TestAnalyzeResults:
    """Tests for analyze_results method"""

    def test_basic_analysis(self, standard_sim, sample_results):
        """Test basic statistical analysis"""
        analysis = standard_sim.analyze_results(sample_results)

        assert 'mean' in analysis
        assert 'median' in analysis
        assert 'std' in analysis
        assert 'min' in analysis
        assert 'max' in analysis
        assert 'percentiles' in analysis
        assert 'confidence_intervals' in analysis
        assert 'probability_positive' in analysis
        assert 'probability_negative' in analysis

    def test_mean_calculation(self, standard_sim):
        """Test mean calculation"""
        results = np.array([1, 2, 3, 4, 5])
        analysis = standard_sim.analyze_results(results)
        assert analysis['mean'] == 3.0

    def test_median_calculation(self, standard_sim):
        """Test median calculation"""
        results = np.array([1, 2, 3, 4, 5])
        analysis = standard_sim.analyze_results(results)
        assert analysis['median'] == 3.0

    def test_percentiles(self, standard_sim, sample_results):
        """Test percentile calculations"""
        analysis = standard_sim.analyze_results(sample_results)
        percentiles = analysis['percentiles']

        assert 'p5' in percentiles
        assert 'p10' in percentiles
        assert 'p25' in percentiles
        assert 'p75' in percentiles
        assert 'p90' in percentiles
        assert 'p95' in percentiles

        # Check ordering
        assert percentiles['p5'] < percentiles['p10']
        assert percentiles['p10'] < percentiles['p25']
        assert percentiles['p25'] < percentiles['p75']
        assert percentiles['p75'] < percentiles['p90']
        assert percentiles['p90'] < percentiles['p95']

    def test_confidence_intervals(self, standard_sim, sample_results):
        """Test confidence interval calculations"""
        analysis = standard_sim.analyze_results(sample_results)
        ci = analysis['confidence_intervals']

        assert '90%' in ci
        assert '95%' in ci
        assert '99%' in ci

        # Each CI should be a tuple with (lower, upper)
        for level in ['90%', '95%', '99%']:
            assert isinstance(ci[level], tuple)
            assert len(ci[level]) == 2
            assert ci[level][0] < ci[level][1]  # Lower < Upper

        # Wider confidence levels should have wider intervals
        width_90 = ci['90%'][1] - ci['90%'][0]
        width_95 = ci['95%'][1] - ci['95%'][0]
        width_99 = ci['99%'][1] - ci['99%'][0]
        assert width_90 < width_95 < width_99

    def test_custom_confidence_levels(self, standard_sim, sample_results):
        """Test custom confidence levels"""
        analysis = standard_sim.analyze_results(sample_results, confidence_levels=[0.80, 0.85])
        ci = analysis['confidence_intervals']

        assert '80%' in ci
        assert '85%' in ci
        assert '90%' not in ci  # Default levels should not be present

    def test_probability_calculations(self, standard_sim):
        """Test probability of positive/negative outcomes"""
        # All positive values
        results = np.array([1, 2, 3, 4, 5])
        analysis = standard_sim.analyze_results(results)
        assert analysis['probability_positive'] == 1.0
        assert analysis['probability_negative'] == 0.0

        # All negative values
        results = np.array([-1, -2, -3, -4, -5])
        analysis = standard_sim.analyze_results(results)
        assert analysis['probability_positive'] == 0.0
        assert analysis['probability_negative'] == 1.0

        # Mixed values (50/50)
        results = np.array([-2, -1, 0, 1, 2])
        analysis = standard_sim.analyze_results(results)
        assert analysis['probability_positive'] == 0.4  # 2 out of 5
        assert analysis['probability_negative'] == 0.4  # 2 out of 5

    def test_min_max(self, standard_sim):
        """Test min and max calculations"""
        results = np.array([10, 5, 15, 3, 20])
        analysis = standard_sim.analyze_results(results)
        assert analysis['min'] == 3
        assert analysis['max'] == 20


class TestProbabilityThreshold:
    """Tests for probability_threshold method"""

    def test_greater_than_threshold(self, standard_sim):
        """Test probability of exceeding threshold"""
        results = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        prob = standard_sim.probability_threshold(results, 5, 'greater')
        assert prob == 0.5  # 5 values > 5

    def test_less_than_threshold(self, standard_sim):
        """Test probability of falling below threshold"""
        results = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        prob = standard_sim.probability_threshold(results, 5, 'less')
        assert prob == 0.4  # 4 values < 5

    def test_threshold_at_boundary(self, standard_sim):
        """Test threshold at exact boundary values"""
        results = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        prob = standard_sim.probability_threshold(results, 10, 'greater')
        assert prob == 0.0  # No values > 10

        prob = standard_sim.probability_threshold(results, 1, 'less')
        assert prob == 0.0  # No values < 1

    def test_invalid_direction(self, standard_sim):
        """Test that error is raised for invalid direction"""
        results = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="direction must be 'greater' or 'less'"):
            standard_sim.probability_threshold(results, 3, 'equal')

    def test_all_values_exceed_threshold(self, standard_sim):
        """Test when all values exceed threshold"""
        results = np.array([10, 20, 30, 40, 50])
        prob = standard_sim.probability_threshold(results, 5, 'greater')
        assert prob == 1.0


class TestValueAtRisk:
    """Tests for value_at_risk method"""

    def test_var_95(self, standard_sim):
        """Test Value at Risk at 95% confidence level"""
        results = np.arange(1, 101)  # 1 to 100
        var = standard_sim.value_at_risk(results, 0.95)
        # At 95% confidence, we expect 5th percentile
        assert var == np.percentile(results, 5)

    def test_var_99(self, standard_sim):
        """Test Value at Risk at 99% confidence level"""
        results = np.arange(1, 101)
        var = standard_sim.value_at_risk(results, 0.99)
        assert var == np.percentile(results, 1)

    def test_var_with_losses(self, standard_sim):
        """Test VaR with loss scenarios"""
        # Simulate profit/loss scenarios
        np.random.seed(42)
        results = np.random.normal(0, 100, 1000)  # Mean 0, some losses
        var = standard_sim.value_at_risk(results, 0.95)
        # VaR should be negative (representing potential loss)
        assert var < 0

    def test_var_ordering(self, standard_sim, sample_results):
        """Test that higher confidence levels give more conservative VaR"""
        var_90 = standard_sim.value_at_risk(sample_results, 0.90)
        var_95 = standard_sim.value_at_risk(sample_results, 0.95)
        var_99 = standard_sim.value_at_risk(sample_results, 0.99)
        # Higher confidence = more conservative (lower VaR for losses)
        assert var_99 <= var_95 <= var_90


class TestExpectedShortfall:
    """Tests for expected_shortfall method"""

    def test_expected_shortfall_95(self, standard_sim):
        """Test Expected Shortfall at 95% confidence"""
        results = np.arange(-100, 1)  # -100 to 0
        es = standard_sim.expected_shortfall(results, 0.95)
        # ES should be the mean of worst 5% (roughly -100 to -95)
        assert es < -90

    def test_es_worse_than_var(self, standard_sim, sample_results):
        """Test that Expected Shortfall is worse than VaR"""
        var = standard_sim.value_at_risk(sample_results, 0.95)
        es = standard_sim.expected_shortfall(sample_results, 0.95)
        # ES should be <= VaR (worse outcome)
        assert es <= var

    def test_es_with_uniform_distribution(self, standard_sim):
        """Test ES with uniformly distributed results"""
        results = np.arange(1, 101)  # Uniform: 1 to 100
        es = standard_sim.expected_shortfall(results, 0.95)
        var = standard_sim.value_at_risk(results, 0.95)

        # For uniform distribution, ES should be mean of values <= VaR
        expected_es = np.mean(results[results <= var])
        assert np.isclose(es, expected_es)


class TestSensitivityAnalysis:
    """Tests for sensitivity_analysis method"""

    def test_basic_sensitivity(self, standard_sim):
        """Test basic sensitivity analysis"""
        np.random.seed(42)
        base_inputs = {
            'revenue': np.random.normal(1000, 100, 1000),
            'costs': np.random.normal(600, 50, 1000)
        }
        model = lambda revenue, costs: revenue - costs

        sensitivity = standard_sim.sensitivity_analysis(
            model, base_inputs, 'revenue',
            variation_range=(0.8, 1.2),
            n_points=5
        )

        assert 'multipliers' in sensitivity
        assert 'mean_results' in sensitivity
        assert 'variable' in sensitivity
        assert sensitivity['variable'] == 'revenue'
        assert len(sensitivity['multipliers']) == 5
        assert len(sensitivity['mean_results']) == 5

    def test_sensitivity_monotonic_increase(self, standard_sim):
        """Test that increasing revenue increases profit"""
        np.random.seed(42)
        base_inputs = {
            'revenue': np.random.normal(1000, 100, 1000),
            'costs': np.random.normal(600, 50, 1000)
        }
        model = lambda revenue, costs: revenue - costs

        sensitivity = standard_sim.sensitivity_analysis(
            model, base_inputs, 'revenue',
            variation_range=(0.5, 1.5),
            n_points=10
        )

        # Mean results should increase as revenue multiplier increases
        results = sensitivity['mean_results']
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_sensitivity_monotonic_decrease(self, standard_sim):
        """Test that increasing costs decreases profit"""
        np.random.seed(42)
        base_inputs = {
            'revenue': np.random.normal(1000, 100, 1000),
            'costs': np.random.normal(600, 50, 1000)
        }
        model = lambda revenue, costs: revenue - costs

        sensitivity = standard_sim.sensitivity_analysis(
            model, base_inputs, 'costs',
            variation_range=(0.5, 1.5),
            n_points=10
        )

        # Mean results should decrease as costs multiplier increases
        results = sensitivity['mean_results']
        for i in range(len(results) - 1):
            assert results[i] > results[i + 1]

    def test_custom_variation_range(self, standard_sim):
        """Test custom variation range"""
        np.random.seed(42)
        base_inputs = {
            'x': np.random.normal(100, 10, 1000)
        }
        model = lambda x: x * 2

        sensitivity = standard_sim.sensitivity_analysis(
            model, base_inputs, 'x',
            variation_range=(0.5, 2.0),
            n_points=6
        )

        multipliers = sensitivity['multipliers']
        assert multipliers[0] == 0.5
        assert multipliers[-1] == 2.0
        assert len(multipliers) == 6

    def test_sensitivity_different_n_points(self, standard_sim):
        """Test sensitivity with different number of points"""
        np.random.seed(42)
        base_inputs = {'x': np.random.normal(100, 10, 1000)}
        model = lambda x: x

        for n_points in [3, 5, 10, 20]:
            sensitivity = standard_sim.sensitivity_analysis(
                model, base_inputs, 'x',
                n_points=n_points
            )
            assert len(sensitivity['multipliers']) == n_points
            assert len(sensitivity['mean_results']) == n_points


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_complete_simulation_workflow(self):
        """Test complete simulation from start to finish"""
        # Setup
        sim = MonteCarloSimulator(n_simulations=1000, random_seed=42)

        # Create inputs
        np.random.seed(42)
        inputs = {
            'revenue': np.random.normal(100000, 10000, 1000),
            'costs': np.random.normal(60000, 5000, 1000)
        }

        # Define model
        model = lambda revenue, costs: revenue - costs

        # Run simulation
        results = sim.simulate(model, inputs)

        # Analyze
        analysis = sim.analyze_results(results)

        # Assertions
        assert analysis['mean'] > 0  # Expected profit
        assert analysis['probability_positive'] > 0.9  # High chance of profit
        assert 'confidence_intervals' in analysis

        # Risk metrics
        var = sim.value_at_risk(results, 0.95)
        es = sim.expected_shortfall(results, 0.95)
        assert var < analysis['mean']
        assert es < var

    @pytest.mark.integration
    def test_business_decision_scenario(self):
        """Test realistic business decision scenario"""
        sim = MonteCarloSimulator(n_simulations=5000, random_seed=42)

        # Investment decision
        np.random.seed(42)
        investment = 100000

        # Uncertain variables
        new_customers = np.random.triangular(500, 1000, 2000, 5000)
        revenue_per_customer = np.random.normal(300, 50, 5000)
        retention_rate = np.clip(np.random.normal(0.6, 0.1, 5000), 0, 1)

        inputs = {
            'customers': new_customers,
            'revenue': revenue_per_customer,
            'retention': retention_rate,
            'invest': np.full(5000, investment)
        }

        # ROI model
        def roi_model(customers, revenue, retention, invest):
            year1 = customers * revenue
            year2 = customers * retention * revenue * 0.8
            total = year1 + year2 - invest
            return total

        # Run simulation
        results = sim.simulate(roi_model, inputs)
        analysis = sim.analyze_results(results)

        # Decision metrics
        prob_profit = analysis['probability_positive']
        expected_return = analysis['mean']
        var_95 = sim.value_at_risk(results, 0.95)

        # Assertions - basic sanity checks
        assert 0 <= prob_profit <= 1
        assert expected_return != 0  # Should have some expected return
        assert var_95 < expected_return  # VaR should be worse than mean

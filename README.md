# Monte Carlo Simulator for Business Decision Making

A Python-based Monte Carlo simulation tool designed for business decision making and risk analysis. Built with SciPy and NumPy, this tool helps you make data-driven decisions under uncertainty.

## Features

- **Easy-to-use Monte Carlo simulation framework**
- **Multiple probability distributions** for modeling business uncertainty:
  - Normal, Lognormal, Uniform
  - Triangular, PERT (ideal for expert estimates)
  - Beta, Gamma, Discrete
  - Correlated variables support
- **Comprehensive risk analysis**:
  - Value at Risk (VaR)
  - Expected Shortfall
  - Confidence intervals
  - Probability thresholds
- **Sensitivity analysis** to understand key drivers
- **Real-world business examples**:
  - Project ROI analysis
  - Product launch decisions
  - Pricing optimization
  - Capacity planning

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the simple demo:
```bash
python demo.py
```

Run comprehensive business examples:
```bash
python examples/business_examples.py
```

## Usage Example

```python
from monte_carlo import MonteCarloSimulator
from monte_carlo.distributions import pert_distribution, normal_distribution
import numpy as np

# Create simulator with 10,000 iterations
sim = MonteCarloSimulator(n_simulations=10000, random_seed=42)

# Define uncertain inputs
revenue = pert_distribution(
    low=80000,      # pessimistic
    mode=100000,    # most likely
    high=150000,    # optimistic
    size=10000
)

costs = normal_distribution(
    mean=60000,
    std=5000,
    size=10000
)

# Define your business model
def profit_model(revenue, costs):
    return revenue - costs

# Run simulation
inputs = {'revenue': revenue, 'costs': costs}
results = sim.simulate(profit_model, inputs)

# Analyze results
analysis = sim.analyze_results(results)

print(f"Expected Profit: ${analysis['mean']:,.0f}")
print(f"Probability of profit: {analysis['probability_positive']*100:.1f}%")
print(f"95% Confidence Interval: ${analysis['confidence_intervals']['95%'][0]:,.0f} to ${analysis['confidence_intervals']['95%'][1]:,.0f}")
```

## Business Examples

### 1. Project ROI Analysis
Evaluate investment decisions considering uncertainty in costs, revenues, and timelines.

```python
from examples.business_examples import project_roi_analysis
project_roi_analysis()
```

### 2. Product Launch Decision
Assess whether to launch a new product considering market scenarios and demand uncertainty.

```python
from examples.business_examples import product_launch_decision
product_launch_decision()
```

### 3. Pricing Optimization
Find the optimal price point considering demand elasticity and cost structures.

```python
from examples.business_examples import pricing_optimization
pricing_optimization()
```

### 4. Capacity Planning
Determine optimal production capacity balancing utilization and stockout risk.

```python
from examples.business_examples import capacity_planning
capacity_planning()
```

## Available Distributions

### PERT Distribution (Recommended for Expert Estimates)
```python
from monte_carlo.distributions import pert_distribution

samples = pert_distribution(
    low=50000,      # minimum estimate
    mode=100000,    # most likely estimate
    high=200000,    # maximum estimate
    size=10000
)
```

### Normal Distribution
```python
from monte_carlo.distributions import normal_distribution

samples = normal_distribution(mean=100, std=15, size=10000)
```

### Triangular Distribution
```python
from monte_carlo.distributions import triangular_distribution

samples = triangular_distribution(low=10, mode=20, high=30, size=10000)
```

### Discrete Scenarios
```python
from monte_carlo.distributions import discrete_distribution

# Market scenarios: recession, stable, growth
market_returns = discrete_distribution(
    values=[-0.10, 0.05, 0.20],
    probabilities=[0.30, 0.50, 0.20],
    size=10000
)
```

### Correlated Variables
```python
from monte_carlo.distributions import correlated_normal

# Revenue and costs often move together
revenue, costs = correlated_normal(
    mean1=100000, std1=10000,
    mean2=60000, std2=5000,
    correlation=0.7,
    size=10000
)
```

## Core Methods

### MonteCarloSimulator

#### `simulate(model, inputs)`
Run Monte Carlo simulation with your business model.

#### `analyze_results(results)`
Get comprehensive statistical analysis including:
- Mean, median, standard deviation
- Percentiles (5th, 10th, 25th, 75th, 90th, 95th)
- Confidence intervals
- Probability of positive/negative outcomes

#### `probability_threshold(results, threshold, direction)`
Calculate probability of exceeding or falling below a threshold.

#### `value_at_risk(results, confidence_level)`
Calculate Value at Risk (VaR) for risk assessment.

#### `expected_shortfall(results, confidence_level)`
Calculate Expected Shortfall (Conditional VaR).

#### `sensitivity_analysis(model, base_inputs, variable, variation_range)`
Analyze how sensitive results are to changes in specific variables.

## When to Use Monte Carlo Simulation

Monte Carlo simulation is ideal for business decisions involving:

- **Multiple sources of uncertainty** (market conditions, costs, demand, etc.)
- **Complex interactions** between variables
- **Risk assessment** - understanding downside and upside scenarios
- **Comparing alternatives** with different risk profiles
- **Communicating uncertainty** to stakeholders

## Best Practices

1. **Choose appropriate distributions**:
   - Use PERT for expert estimates (min, most likely, max)
   - Use Normal for variables with symmetric uncertainty
   - Use Lognormal for prices or values that can't be negative
   - Use Discrete for scenario analysis

2. **Run enough simulations**: 10,000 is typically sufficient for stable results

3. **Validate your model**:
   - Check if results make sense
   - Compare with historical data if available
   - Test extreme scenarios

4. **Consider correlations**: Many business variables are correlated (e.g., revenue and costs)

5. **Focus on decisions**: Use simulation to compare alternatives, not just predict the future

## Project Structure

```
monte_carlo/
├── README.md
├── requirements.txt
├── demo.py                      # Quick demo script
├── monte_carlo/                 # Main package
│   ├── __init__.py
│   ├── simulator.py            # Core Monte Carlo simulator
│   └── distributions.py        # Probability distributions
└── examples/
    └── business_examples.py    # Comprehensive business examples
```

## Requirements

- Python 3.8+
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0 (for visualization)
- Pandas >= 2.0.0 (for data handling)

## License

MIT License - feel free to use for any business or educational purpose.

## Contributing

Contributions welcome! Feel free to submit issues or pull requests.

## Further Reading

- [Monte Carlo Simulation in Risk Analysis](https://en.wikipedia.org/wiki/Monte_Carlo_method_in_statistical_physics)
- [PERT Distribution for Project Management](https://en.wikipedia.org/wiki/PERT_distribution)
- [Value at Risk (VaR)](https://en.wikipedia.org/wiki/Value_at_risk)

---

Built with Python, SciPy, and NumPy for robust statistical analysis.

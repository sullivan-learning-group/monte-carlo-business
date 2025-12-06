"""
Business Decision Making Examples using Monte Carlo Simulation
"""

import sys
sys.path.insert(0, '../')

import numpy as np
from monte_carlo import MonteCarloSimulator
from monte_carlo.distributions import (
    normal_distribution,
    triangular_distribution,
    pert_distribution,
    discrete_distribution,
    correlated_normal
)


def project_roi_analysis():
    """
    Example: Project ROI Analysis

    Evaluate whether to invest in a new project considering uncertainty
    in costs, revenues, and timeline.
    """
    print("=" * 70)
    print("PROJECT ROI ANALYSIS")
    print("=" * 70)

    n_sims = 10000
    sim = MonteCarloSimulator(n_simulations=n_sims, random_seed=42)

    # Define uncertain inputs using PERT distribution (common in project management)
    # Initial investment
    initial_investment = 500000  # Known cost

    # Annual revenue (optimistic, most likely, pessimistic)
    annual_revenue = pert_distribution(
        low=150000,
        mode=200000,
        high=300000,
        size=n_sims
    )

    # Annual operating costs
    annual_costs = pert_distribution(
        low=80000,
        mode=100000,
        high=140000,
        size=n_sims
    )

    # Project lifetime in years
    project_lifetime = triangular_distribution(
        low=3,
        mode=5,
        high=7,
        size=n_sims
    )

    # Define ROI model
    def roi_model(revenue, costs, lifetime, investment):
        total_profit = (revenue - costs) * lifetime - investment
        roi = (total_profit / investment) * 100
        return roi

    # Run simulation
    inputs = {
        'revenue': annual_revenue,
        'costs': annual_costs,
        'lifetime': project_lifetime,
        'investment': np.full(n_sims, initial_investment)
    }

    results = sim.simulate(roi_model, inputs)

    # Analyze results
    analysis = sim.analyze_results(results)

    print(f"\nInvestment: ${initial_investment:,.0f}")
    print(f"\nROI Statistics:")
    print(f"  Mean ROI: {analysis['mean']:.1f}%")
    print(f"  Median ROI: {analysis['median']:.1f}%")
    print(f"  Std Dev: {analysis['std']:.1f}%")
    print(f"  Min ROI: {analysis['min']:.1f}%")
    print(f"  Max ROI: {analysis['max']:.1f}%")

    print(f"\nPercentiles:")
    for pct, value in analysis['percentiles'].items():
        print(f"  {pct}: {value:.1f}%")

    print(f"\nConfidence Intervals:")
    for level, (lower, upper) in analysis['confidence_intervals'].items():
        print(f"  {level}: [{lower:.1f}%, {upper:.1f}%]")

    print(f"\nRisk Metrics:")
    print(f"  Probability of positive ROI: {analysis['probability_positive']*100:.1f}%")
    print(f"  Probability of negative ROI: {analysis['probability_negative']*100:.1f}%")
    print(f"  Probability of ROI > 50%: {sim.probability_threshold(results, 50)*100:.1f}%")

    var_95 = sim.value_at_risk(results, 0.95)
    es_95 = sim.expected_shortfall(results, 0.95)
    print(f"  Value at Risk (95%): {var_95:.1f}%")
    print(f"  Expected Shortfall (95%): {es_95:.1f}%")

    return results


def product_launch_decision():
    """
    Example: New Product Launch Decision

    Decide whether to launch a new product considering market uncertainty.
    """
    print("\n\n" + "=" * 70)
    print("PRODUCT LAUNCH DECISION ANALYSIS")
    print("=" * 70)

    n_sims = 10000
    sim = MonteCarloSimulator(n_simulations=n_sims, random_seed=42)

    # Market scenarios: recession (20%), stable (60%), growth (20%)
    market_multiplier = discrete_distribution(
        values=[0.6, 1.0, 1.4],
        probabilities=[0.20, 0.60, 0.20],
        size=n_sims
    )

    # Expected market size
    base_market_size = 1000000  # units

    # Market penetration rate (%)
    penetration_rate = triangular_distribution(
        low=0.02,
        mode=0.05,
        high=0.12,
        size=n_sims
    )

    # Price per unit
    price = normal_distribution(
        mean=50,
        std=5,
        size=n_sims
    )

    # Variable cost per unit
    variable_cost = normal_distribution(
        mean=30,
        std=3,
        size=n_sims
    )

    # Fixed costs (marketing, development, etc.)
    fixed_costs = 1500000

    # Define profit model
    def profit_model(market_mult, penetration, price, var_cost, fixed):
        units_sold = base_market_size * market_mult * penetration
        revenue = units_sold * price
        total_var_costs = units_sold * var_cost
        profit = revenue - total_var_costs - fixed
        return profit

    # Run simulation
    inputs = {
        'market_mult': market_multiplier,
        'penetration': penetration_rate,
        'price': price,
        'var_cost': variable_cost,
        'fixed': np.full(n_sims, fixed_costs)
    }

    results = sim.simulate(profit_model, inputs)

    # Analyze results
    analysis = sim.analyze_results(results)

    print(f"\nFixed Costs: ${fixed_costs:,.0f}")
    print(f"Base Market Size: {base_market_size:,} units")

    print(f"\nProfit Forecast:")
    print(f"  Expected Profit: ${analysis['mean']:,.0f}")
    print(f"  Median Profit: ${analysis['median']:,.0f}")
    print(f"  Std Dev: ${analysis['std']:,.0f}")

    print(f"\nPercentiles:")
    for pct, value in analysis['percentiles'].items():
        print(f"  {pct}: ${value:,.0f}")

    print(f"\nRisk Assessment:")
    print(f"  Probability of profit: {analysis['probability_positive']*100:.1f}%")
    print(f"  Probability of loss: {analysis['probability_negative']*100:.1f}%")
    print(f"  Probability of profit > $1M: {sim.probability_threshold(results, 1000000)*100:.1f}%")

    breakeven_prob = sim.probability_threshold(results, 0, 'greater')
    print(f"  Breakeven probability: {breakeven_prob*100:.1f}%")

    # Decision recommendation
    print(f"\nDecision Recommendation:")
    if breakeven_prob > 0.70 and analysis['mean'] > 500000:
        print("  RECOMMEND LAUNCH - Strong probability of profitability")
    elif breakeven_prob > 0.50:
        print("  CONDITIONAL LAUNCH - Moderate risk, consider risk tolerance")
    else:
        print("  DO NOT LAUNCH - High risk of loss")

    return results


def pricing_optimization():
    """
    Example: Pricing Strategy Optimization

    Find optimal price point by considering demand elasticity and costs.
    """
    print("\n\n" + "=" * 70)
    print("PRICING STRATEGY OPTIMIZATION")
    print("=" * 70)

    n_sims = 10000
    sim = MonteCarloSimulator(n_simulations=n_sims, random_seed=42)

    # Test different price points
    price_points = [40, 45, 50, 55, 60, 65, 70]
    results_by_price = {}

    for price in price_points:
        # Demand decreases with higher prices (price elasticity)
        # Base demand at $50 is 10,000 units
        base_demand = 10000
        price_elasticity = -1.5  # typical for consumer products

        # Calculate demand with price elasticity and uncertainty
        price_ratio = price / 50
        demand_multiplier = price_ratio ** price_elasticity

        # Add demand uncertainty
        demand = normal_distribution(
            mean=base_demand * demand_multiplier,
            std=base_demand * demand_multiplier * 0.15,  # 15% uncertainty
            size=n_sims
        )
        demand = np.maximum(demand, 0)  # Demand can't be negative

        # Variable costs with economies of scale
        def variable_cost_per_unit(quantity):
            # Cost decreases with volume
            if quantity > 8000:
                return 25
            elif quantity > 5000:
                return 27
            else:
                return 30

        # Calculate profit for each simulation
        profits = []
        for d in demand:
            var_cost = variable_cost_per_unit(d)
            profit = d * (price - var_cost) - 50000  # Fixed costs
            profits.append(profit)

        results_by_price[price] = np.array(profits)

    # Compare results
    print(f"\nPrice Point Comparison:")
    print(f"{'Price':<10} {'Mean Profit':<15} {'P(Profit>0)':<15} {'P95':<15}")
    print("-" * 60)

    best_price = None
    best_expected_profit = -float('inf')

    for price in price_points:
        results = results_by_price[price]
        mean_profit = np.mean(results)
        prob_positive = np.mean(results > 0)
        p95 = np.percentile(results, 95)

        print(f"${price:<9} ${mean_profit:<14,.0f} {prob_positive*100:<14.1f}% ${p95:<14,.0f}")

        if mean_profit > best_expected_profit:
            best_expected_profit = mean_profit
            best_price = price

    print(f"\nRecommended Price: ${best_price}")
    print(f"Expected Profit: ${best_expected_profit:,.0f}")

    return results_by_price


def capacity_planning():
    """
    Example: Production Capacity Planning

    Determine optimal production capacity considering demand uncertainty and costs.
    """
    print("\n\n" + "=" * 70)
    print("PRODUCTION CAPACITY PLANNING")
    print("=" * 70)

    n_sims = 10000
    sim = MonteCarloSimulator(n_simulations=n_sims, random_seed=42)

    # Uncertain demand forecast
    demand = lognormal_distribution(
        mean=np.log(50000),  # median demand
        std=0.3,
        size=n_sims
    )

    # Test different capacity levels
    capacity_levels = [40000, 45000, 50000, 55000, 60000]

    print(f"\nCapacity Level Analysis:")
    print(f"{'Capacity':<12} {'Avg Profit':<15} {'Utilization':<15} {'Stockout Risk':<15}")
    print("-" * 60)

    for capacity in capacity_levels:
        # Production costs
        fixed_cost_per_unit = 5  # depreciation, facility costs
        variable_cost_per_unit = 20
        price_per_unit = 40

        # Calculate metrics for each simulation
        profits = []
        utilization_rates = []
        stockouts = 0

        for d in demand:
            actual_sales = min(d, capacity)
            revenue = actual_sales * price_per_unit
            costs = capacity * fixed_cost_per_unit + actual_sales * variable_cost_per_unit
            profit = revenue - costs

            profits.append(profit)
            utilization_rates.append(actual_sales / capacity)

            if d > capacity:
                stockouts += 1

        avg_profit = np.mean(profits)
        avg_utilization = np.mean(utilization_rates)
        stockout_risk = stockouts / n_sims

        print(f"{capacity:<12,} ${avg_profit:<14,.0f} {avg_utilization*100:<14.1f}% {stockout_risk*100:<14.1f}%")

    return demand


def lognormal_distribution(mean, std, size):
    """Helper function for lognormal distribution"""
    return np.random.lognormal(mean, std, size)


if __name__ == "__main__":
    print("\nMonte Carlo Simulation for Business Decision Making")
    print("=" * 70)

    # Run all examples
    project_roi_analysis()
    product_launch_decision()
    pricing_optimization()
    capacity_planning()

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

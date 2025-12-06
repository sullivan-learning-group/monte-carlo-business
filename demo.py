"""
Quick Demo of Monte Carlo Simulator for Business Decisions
"""

import numpy as np
from monte_carlo import MonteCarloSimulator
from monte_carlo.distributions import pert_distribution, normal_distribution


def simple_investment_analysis():
    """
    Simple example: Should we invest $100K in a marketing campaign?
    """
    print("Monte Carlo Simulation: Marketing Investment Decision")
    print("=" * 70)

    # Create simulator
    sim = MonteCarloSimulator(n_simulations=10000, random_seed=42)

    # Investment amount
    investment = 100000

    # Expected outcomes with uncertainty
    # Using PERT distribution: (min, most_likely, max)

    # New customers from campaign
    new_customers = pert_distribution(
        low=500,        # pessimistic
        mode=1000,      # most likely
        high=2000,      # optimistic
        size=10000
    )

    # Revenue per customer
    revenue_per_customer = normal_distribution(
        mean=300,
        std=50,
        size=10000
    )

    # Customer retention rate (% that stay for year 2)
    retention_rate = normal_distribution(
        mean=0.60,
        std=0.10,
        size=10000
    )
    retention_rate = np.clip(retention_rate, 0, 1)  # Keep between 0 and 1

    # Define the model
    def marketing_roi(customers, revenue, retention, invest):
        # Year 1 revenue
        year1_revenue = customers * revenue

        # Year 2 revenue (retained customers)
        year2_revenue = customers * retention * revenue * 0.8  # Assume 20% decline

        total_revenue = year1_revenue + year2_revenue
        profit = total_revenue - invest
        roi = (profit / invest) * 100

        return profit

    # Run simulation
    inputs = {
        'customers': new_customers,
        'revenue': revenue_per_customer,
        'retention': retention_rate,
        'invest': np.full(10000, investment)
    }

    results = sim.simulate(marketing_roi, inputs)

    # Analyze results
    analysis = sim.analyze_results(results)

    print(f"\nInvestment: ${investment:,}")
    print(f"\nExpected Outcomes:")
    print(f"  Mean Profit: ${analysis['mean']:,.0f}")
    print(f"  Median Profit: ${analysis['median']:,.0f}")
    print(f"  Standard Deviation: ${analysis['std']:,.0f}")

    print(f"\nRange of Outcomes:")
    print(f"  Best Case (95th percentile): ${analysis['percentiles']['p95']:,.0f}")
    print(f"  Worst Case (5th percentile): ${analysis['percentiles']['p5']:,.0f}")

    print(f"\n95% Confidence Interval:")
    ci_95 = analysis['confidence_intervals']['95%']
    print(f"  We are 95% confident profit will be between ${ci_95[0]:,.0f} and ${ci_95[1]:,.0f}")

    print(f"\nRisk Assessment:")
    prob_positive = analysis['probability_positive']
    print(f"  Probability of making a profit: {prob_positive*100:.1f}%")
    print(f"  Probability of losing money: {analysis['probability_negative']*100:.1f}%")

    prob_double = sim.probability_threshold(results, investment, 'greater')
    print(f"  Probability of doubling investment: {prob_double*100:.1f}%")

    # Value at Risk
    var_95 = sim.value_at_risk(results, 0.95)
    print(f"\n  Value at Risk (95%): ${var_95:,.0f}")
    print(f"    (There's only a 5% chance of losing more than this)")

    # Decision
    print(f"\nDecision Recommendation:")
    if prob_positive > 0.75 and analysis['mean'] > investment * 0.5:
        print("  STRONG GO - High probability of good returns")
    elif prob_positive > 0.60:
        print("  CAUTIOUS GO - Moderate risk, reasonable upside")
    elif prob_positive > 0.50:
        print("  RISKY - Consider alternatives or reduce investment")
    else:
        print("  DO NOT INVEST - High risk of loss")

    return results


def sensitivity_example():
    """
    Demonstrate sensitivity analysis
    """
    print("\n\n" + "=" * 70)
    print("Sensitivity Analysis: How sensitive is profit to customer acquisition?")
    print("=" * 70)

    sim = MonteCarloSimulator(n_simulations=5000, random_seed=42)

    # Base inputs
    base_inputs = {
        'customers': pert_distribution(500, 1000, 2000, size=5000),
        'revenue': normal_distribution(300, 50, size=5000),
        'invest': np.full(5000, 100000)
    }

    def simple_model(customers, revenue, invest):
        return customers * revenue - invest

    # Sensitivity analysis on customer acquisition
    sensitivity = sim.sensitivity_analysis(
        model=simple_model,
        base_inputs=base_inputs,
        variable='customers',
        variation_range=(0.5, 1.5),
        n_points=10
    )

    print(f"\nCustomer Acquisition Sensitivity:")
    print(f"{'Multiplier':<15} {'Expected Profit':<20}")
    print("-" * 40)

    for mult, profit in zip(sensitivity['multipliers'], sensitivity['mean_results']):
        print(f"{mult:<15.2f} ${profit:<19,.0f}")

    print(f"\nInsight: Profit is highly sensitive to customer acquisition.")
    print(f"A 20% decrease in customers significantly impacts profitability.")


if __name__ == "__main__":
    # Run the demo
    results = simple_investment_analysis()
    sensitivity_example()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("\nTry running 'python examples/business_examples.py' for more examples")
    print("=" * 70)

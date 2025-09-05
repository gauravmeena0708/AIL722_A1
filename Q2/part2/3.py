import numpy as np
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv
from rl_functions_portfolio import (
    value_iteration,
    policy_iteration,
    simulate_episode,
    plot_episode_evolution,
    plot_convergence_analysis
)

PRICE_SEQUENCES = {
    'Seq 1': [1, 3, 5, 5, 4, 3, 2, 3, 5, 8],
    'Seq 2': [2, 2, 2, 4, 2, 2, 4, 2, 2, 2],
    'Seq 3': [4, 1, 4, 1, 4, 4, 4, 1, 1, 4]
}
GAMMA_VALUES = [0.999, 1.0]

def run_value_iteration_analysis():
    """Run Value Iteration analysis for the Portfolio Optimization problem."""
    print("\n" + "="*80)
    print("VALUE ITERATION ANALYSIS")
    print("="*80)
    
    for name, prices in PRICE_SEQUENCES.items():
        for gamma in GAMMA_VALUES:
            title = f"VI_{name}_gamma{gamma}"
            print(f"\n--- Running {title} ---")
            env = DiscretePortfolioOptEnv(prices=prices)
            
            policy, _, exec_time = value_iteration(env, prices, gamma=gamma)
            
            print(f"Execution time: {exec_time:.2f}s")
            results = simulate_episode(env, policy, prices)
            print(f"Final wealth: {results['final_wealth']:.2f}")
            plot_episode_evolution(results, prices, title)
            print(f"Plot saved for {title}")

def run_policy_iteration_analysis():
    """Run Policy Iteration analysis for the Portfolio Optimization problem."""
    print("\n" + "="*80)
    print("POLICY ITERATION ANALYSIS")
    print("="*80)

    for name, prices in PRICE_SEQUENCES.items():
        for gamma in GAMMA_VALUES:
            title = f"PI_{name}_gamma{gamma}"
            print(f"\n--- Running {title} ---")
            env = DiscretePortfolioOptEnv(prices=prices)

            policy, _, exec_time, _ = policy_iteration(env, prices, gamma=gamma)

            print(f"Execution time: {exec_time:.2f}s")
            results = simulate_episode(env, policy, prices)
            print(f"Final wealth: {results['final_wealth']:.2f}")
            plot_episode_evolution(results, prices, title)
            print(f"Plot saved for {title}")

def run_variance_analysis():
    """Run variance analysis with stochastic prices using Policy Iteration."""
    print("\n" + "="*80)
    print("VARIANCE ANALYSIS (POLICY ITERATION)")
    print("="*80)
    
    env = DiscretePortfolioOptEnv(variance=1.0)
    # For the DP solver, we must use a deterministic price sequence.
    # We use the mean price sequence from the stochastic environment.
    mean_prices = env.asset_price_means.flatten()
    print(f"Using mean price sequence for training: {mean_prices}")

    _, _, _, conv_hist = policy_iteration(env, mean_prices, gamma=1.0, theta=1e-2, max_iterations=1000)
    
    plot_convergence_analysis(conv_hist, "Variance=1.0")
    print("Convergence plot saved.")

def main():
    """Main function to run all portfolio optimization analyses."""
    np.random.seed(42)
    
    # Problem 1
    run_value_iteration_analysis()
    
    # Problem 2
    run_policy_iteration_analysis()
    
    # Problem 3
    run_variance_analysis()
    
    print("\n" + "="*80)
    print("All portfolio analyses complete.")
    print("="*80)

if __name__ == "__main__":
    main()

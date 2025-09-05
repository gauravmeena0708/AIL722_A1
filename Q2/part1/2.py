import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
from rl_functions import (
    policy_iteration,
    value_iteration,
    evaluate_policy_multiple_seeds, 
    plot_value_progression,
    create_value_function_heatmap
)

def run_policy_iteration_analysis(seeds):
    """Run Policy Iteration analysis for the Online Knapsack problem."""
    print("\n" + "="*80)
    print("POLICY ITERATION ANALYSIS")
    print("="*80)
    
    env = OnlineKnapsackEnv()
    
    print("Training Policy Iteration model...")
    # A low number of iterations is used as PI is computationally expensive.
    policy, V = policy_iteration(env, gamma=0.95, theta=1e-4, max_iterations=5)
    
    print("\nEvaluating Policy Iteration model...")
    results = evaluate_policy_multiple_seeds(env, policy, seeds, step_limit=env.step_limit)
    
    print(f"\nFinal Knapsack Values (Policy Iteration): {results['final_values']}")
    mean_pi = np.mean(results['final_values'])
    std_pi = np.std(results['final_values'])
    print(f"Mean: {mean_pi:.2f} +/- {std_pi:.2f}")
    
    plot_value_progression(results, title="PI Knapsack Value Progression")
    create_value_function_heatmap(V, env, title="PI Value Function Heatmap")
    print("Policy Iteration analysis complete. Plots saved.")

def run_value_iteration_analysis(seeds):
    """Run Value Iteration analysis for the Online Knapsack problem."""
    print("\n" + "="*80)
    print("VALUE ITERATION ANALYSIS")
    print("="*80)
    
    env = OnlineKnapsackEnv()
    
    print("Training Value Iteration model...")
    policy, V = value_iteration(env, gamma=0.95, theta=1e-4, max_iterations=100)
    
    print("\nEvaluating Value Iteration model...")
    results = evaluate_policy_multiple_seeds(env, policy, seeds, step_limit=env.step_limit)
    
    print(f"\nFinal Knapsack Values (Value Iteration): {results['final_values']}")
    mean_vi = np.mean(results['final_values'])
    std_vi = np.std(results['final_values'])
    print(f"Mean: {mean_vi:.2f} +/- {std_vi:.2f}")
    
    plot_value_progression(results, title="VI Knapsack Value Progression")
    create_value_function_heatmap(V, env, title="VI Value Function Heatmap")
    print("Value Iteration analysis complete. Plots saved.")

def run_episode_steps_analysis(seed):
    """Run Value Iteration with varying episode lengths and evaluate the policies."""
    print("\n" + "-"*80)
    print("VALUE ITERATION WITH VARYING EPISODE STEPS ANALYSIS")
    print("-"*80)
    
    step_limits = [10, 50, 500]
    eval_seeds = [0, 42, 123] # Use a few seeds for a stable average
    
    for steps in step_limits:
        print(f"\n--- Running for {steps} steps ---")
        env = OnlineKnapsackEnv(step_limit=steps)
        
        print(f"Training Value Iteration model for {steps} steps...")
        policy, V = value_iteration(env, gamma=0.95, theta=1e-4, max_iterations=100)
        
        print(f"\nEvaluating policy for {steps} steps...")
        results = evaluate_policy_multiple_seeds(env, policy, eval_seeds, step_limit=steps)
        mean_reward = np.mean(results['final_values'])
        std_reward = np.std(results['final_values'])
        print(f"-> Average final reward for {steps} steps: {mean_reward:.2f} +/- {std_reward:.2f}")

        create_value_function_heatmap(V, env, title=f"VI_Heatmap_{steps}_steps")
        print(f"Heatmap for {steps} steps saved.")

def main():
    """Main function to run all analyses."""
    np.random.seed(42)
    seeds = [0, 42, 123, 555, 101]
    
    print("Starting Online Knapsack DP Analysis...")
    
    # Task 1: Policy Iteration
    run_policy_iteration_analysis(seeds)
    
    # Task 2: Value Iteration
    run_value_iteration_analysis(seeds)
    
    # Task 3: Value Iteration with varying episode steps
    run_episode_steps_analysis(seed=seeds[0])
    
    print("\n" + "="*80)
    print("All analyses complete.")
    print("="*80)

if __name__ == "__main__":
    main()
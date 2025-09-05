import numpy as np
import time
from env import FootballSkillsEnv
from rl_functions import policy_iteration, value_iteration, prioritized_value_iteration

def main():
    """
    Runs all experiments for the Prioritized Value Iteration Analysis (Part 1.4).
    """
    gamma = 0.95
    theta = 1e-6

    print("\n" + "="*50)
    print("RUNNING EXPERIMENTS FOR PRIORITIZED VI (GAMMA = 0.95)")
    print("="*50)

    env = FootballSkillsEnv()

    # --- Prioritized Value Iteration ---
    print("\n--- Running Prioritized Value Iteration ---")
    start_time = time.time()
    pvi_policy, pvi_v, pvi_updates, pvi_calls = prioritized_value_iteration(env, gamma, theta)
    pvi_time = time.time() - start_time
    print(f"Execution Time: {pvi_time:.4f} seconds")
    print(f"Converged in {pvi_updates} updates.")
    print(f"Total Transition Calls: {pvi_calls}")

    # --- Standard Value Iteration (for comparison) ---
    print("\n--- Running Standard Value Iteration ---")
    start_time = time.time()
    vi_policy, vi_v, vi_iters, vi_calls = value_iteration(env, gamma, theta)
    vi_time = time.time() - start_time
    print(f"Execution Time: {vi_time:.4f} seconds")
    print(f"Converged in {vi_iters} iterations.")
    print(f"Total Transition Calls: {vi_calls}")

    # --- Standard Policy Iteration (for comparison) ---
    print("\n--- Running Policy Iteration ---")
    start_time = time.time()
    pi_policy, pi_v, pi_iters, pi_calls = policy_iteration(env, gamma, theta)
    pi_time = time.time() - start_time
    print(f"Execution Time: {pi_time:.4f} seconds")
    print(f"Converged in {pi_iters} iterations.")
    print(f"Total Transition Calls: {pi_calls}")

    # --- Performance Comparison ---
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    print(f"{'Algorithm':<25} {'Time (s)':<12} {'Transitions':<15} {'Updates/Iters':<12}")
    print("-"*65)
    print(f"{'Prioritized VI':<25} {pvi_time:<12.4f} {pvi_calls:<15,} {pvi_updates:<12}")
    print(f"{'Standard VI':<25} {vi_time:<12.4f} {vi_calls:<15,} {vi_iters:<12}")
    print(f"{'Policy Iteration':<25} {pi_time:<12.4f} {pi_calls:<15,} {pi_iters:<12}")
    print("-"*65)

    transition_reduction = ((vi_calls - pvi_calls) / vi_calls) * 100 if vi_calls > 0 else 0
    print(f"\nPrioritized VI reduced transition calls by: {transition_reduction:.2f}%")

if __name__ == "__main__":
    main()

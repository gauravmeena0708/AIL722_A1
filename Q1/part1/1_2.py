import numpy as np
import time
from env import FootballSkillsEnv
from rl_functions import policy_iteration, value_iteration, evaluate_policy


def main():
    """
    Runs all experiments for the Stationary Environment Analysis (Part 1.2).
    """
    theta = 1e-6
    gammas_to_test = [0.95, 0.3, 0.5]

    for gamma in gammas_to_test:
        print("\n" + "="*50)
        print(f"RUNNING EXPERIMENTS FOR GAMMA = {gamma}")
        print("="*50)

        # Use the standard stationary environment
        env = FootballSkillsEnv(render_mode='gif')
        
        # --- Policy Iteration ---
        print("\n--- Running Policy Iteration ---")
        start_time = time.time()
        pi_policy, pi_v, pi_iters, pi_calls = policy_iteration(env, gamma, theta)
        end_time = time.time()
        print(f"Execution Time: {end_time - start_time:.4f} seconds")
        print(f"Converged in {pi_iters} iterations.")
        print(f"Total Transition Calls: {pi_calls}")

        # --- Value Iteration ---
        print("\n--- Running Value Iteration ---")
        start_time = time.time()
        vi_policy, vi_v, vi_iters, vi_calls = value_iteration(env, gamma, theta)
        end_time = time.time()
        print(f"Execution Time: {end_time - start_time:.4f} seconds")
        print(f"Converged in {vi_iters} iterations.")
        print(f"Total Transition Calls: {vi_calls}")

        # --- Policy Comparison ---
        policies_are_identical = (pi_policy == vi_policy)
        print(f"\nPolicies are identical: {policies_are_identical}")
        if not policies_are_identical:
            print("Observation: The policies may differ in states where multiple actions yield the same maximum value. This is due to np.argmax's tie-breaking behavior.")
        else:
            print("Observation: The policies are identical, as expected since both algorithms converge to an optimal policy.")

        # --- Performance Evaluation and GIF Generation (Only for gamma=0.95) ---
        if gamma == 0.95:
            print("\n--- Policy Performance Evaluation (20 Episodes) ---")
            pi_mean, pi_std = evaluate_policy(pi_policy, FootballSkillsEnv)
            vi_mean, vi_std = evaluate_policy(vi_policy, FootballSkillsEnv)
            print(f"Policy Iteration Mean Reward: {pi_mean:.2f} +/- {pi_std:.2f}")
            print(f"Value Iteration Mean Reward:  {vi_mean:.2f} +/- {vi_std:.2f}")

            print("\n--- Generating GIFs ---")
            gif_env = FootballSkillsEnv(render_mode='gif')
            gif_env.get_gif(pi_policy, seed=20, filename=f"policy_iteration_gamma{gamma}.gif")
            
            gif_env = FootballSkillsEnv(render_mode='gif')
            gif_env.get_gif(vi_policy, seed=20, filename=f"value_iteration_gamma{gamma}.gif")
            print("GIFs generated successfully.")
            
if __name__ == "__main__":
    main()

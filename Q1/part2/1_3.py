import time
from env import FootballSkillsEnv
from rl_functions import value_iteration, value_iteration_non_stationary, evaluate_policy

def main():
    """
    Runs all experiments for the Non-Stationary Environment Analysis (Part 1.3).
    """
    gamma = 0.95
    max_horizon = 40

    print("\n" + "="*50)
    print("RUNNING EXPERIMENTS FOR NON-STATIONARY ENVIRONMENT")
    print("="*50)

    # Instantiate the non-stationary environment for the experiments
    env_non_stationary = FootballSkillsEnv(render_mode='gif', degrade_pitch=True)

    # --- Time-Dependent Value Iteration ---
    print("\n--- Running Time-Dependent Value Iteration ---")
    start_time = time.time()
    ns_policy, ns_v, ns_iters, ns_calls = value_iteration_non_stationary(env_non_stationary, gamma, max_horizon)
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print(f"Total Transition Calls: {ns_calls}")

    # --- Standard Value Iteration on Degraded Pitch (for comparison) ---
    print("\n--- Running Standard Value Iteration (for comparison) ---")
    start_time = time.time()
    s_policy, s_v, s_iters, s_calls = value_iteration(env_non_stationary, gamma)
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print(f"Total Transition Calls: {s_calls}")

    # --- Performance Evaluation ---
    print("\n--- Policy Performance Evaluation (20 Episodes) ---")
    ns_mean, ns_std = evaluate_policy(ns_policy, FootballSkillsEnv, non_stationary=True)
    print(f"Time-Dependent VI Mean Reward: {ns_mean:.2f} +/- {ns_std:.2f}")

    s_mean, s_std = evaluate_policy(s_policy, lambda: FootballSkillsEnv(degrade_pitch=True), non_stationary=False)
    print(f"Standard VI Mean Reward:         {s_mean:.2f} +/- {s_std:.2f}")

    # --- GIF Generation ---
    print("\n--- Generating GIFs ---")
    
    # Use a new env instance for each GIF to ensure a clean state
    gif_env = FootballSkillsEnv(render_mode='gif', degrade_pitch=True)
    gif_env.get_gif(ns_policy, seed=20, filename="non_stationary_vi.gif")
    
    gif_env = FootballSkillsEnv(render_mode='gif', degrade_pitch=True)
    # Workaround: Wrap the stationary policy in a time-dependent structure for GIF generation.
    s_policy_for_gif = {t: s_policy for t in range(100)} # max_steps in get_gif is 100
    gif_env.get_gif(s_policy_for_gif, seed=20, filename="standard_vi_on_degraded.gif")
    
    print("GIFs saved as 'non_stationary_vi.gif' and 'standard_vi_on_degraded.gif'.")

if __name__ == "__main__":
    main()


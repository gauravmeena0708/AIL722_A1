import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

TRANSITION_CACHE = {}

# --- Helper Functions ---
def _get_env_params(env):
    max_weight = env.max_weight
    max_timestep = env.step_limit
    num_items = env.N
    num_states = (max_weight + 1) * max_timestep * num_items
    num_actions = 2  # 0: reject, 1: accept
    return max_weight, max_timestep, num_items, num_states, num_actions

def encode_state(weight, timestep, item_idx, env):
    """Encode (weight, timestep, item_idx) into a single state index."""
    _, max_timestep, num_items, _, _ = _get_env_params(env)
    return int(weight * (max_timestep * num_items) + timestep * num_items + item_idx)

def decode_state(state_idx, env):
    """Decode a state index back to (weight, timestep, item_idx)."""
    _, max_timestep, num_items, _, _ = _get_env_params(env)
    item_idx = state_idx % num_items
    timestep = ((state_idx - item_idx) // num_items) % max_timestep
    weight = (state_idx - item_idx - timestep * num_items) // (max_timestep * num_items)
    return int(weight), int(timestep), int(item_idx)

def calculate_q_values_with_cache(s, V, env, gamma):
    """Calculates the Q-value for each action in a given state using a cache."""
    global TRANSITION_CACHE
    max_weight, max_timestep, num_items, _, num_actions = _get_env_params(env)
    
    weight, timestep, item_idx = decode_state(s, env)
    action_values = np.zeros(num_actions)

    for a in range(num_actions):
        item_w = env.item_weights[item_idx]
        item_v = env.item_values[item_idx]

        if a == 1:  # Accept
            if weight + item_w <= max_weight:
                reward = item_v
                next_w = weight + item_w
                done = (timestep + 1 >= max_timestep) or (next_w == max_weight)
            else:
                reward = -10  # Penalty
                next_w = weight
                done = True
        else:  # Reject
            reward = 0
            next_w = weight
            done = (timestep + 1 >= max_timestep)

        if done:
            action_values[a] = reward
        else:
            next_t = timestep + 1
            cache_key = (next_w, next_t)
            
            expected_future_value = TRANSITION_CACHE.get(cache_key)
            if expected_future_value is None:
                future_val = 0
                for next_item_idx in range(num_items):
                    next_s_idx = encode_state(next_w, next_t, next_item_idx, env)
                    future_val += V[next_s_idx]
                expected_future_value = future_val / num_items
                TRANSITION_CACHE[cache_key] = expected_future_value
            
            action_values[a] = reward + gamma * expected_future_value
            
    return action_values

# --- Core Algorithms ---
def value_iteration(env, gamma=0.95, theta=1e-4, max_iterations=100):
    """Implements the Value Iteration algorithm."""
    global TRANSITION_CACHE
    TRANSITION_CACHE.clear()
    print("Running Value Iteration...")
    start_time = time.time()

    _, _, _, num_states, _ = _get_env_params(env)
    V = np.zeros(num_states)

    for i in range(max_iterations):
        delta = 0
        # Clear cache at the start of each full-sweep iteration
        TRANSITION_CACHE.clear()
        for s in range(num_states):
            v_old = V[s]
            q_values = calculate_q_values_with_cache(s, V, env, gamma)
            V[s] = np.max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        
        if delta < theta:
            print(f"Converged after {i + 1} iterations.")
            break

    # Extract policy
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        policy[s] = np.argmax(calculate_q_values_with_cache(s, V, env, gamma))

    print(f"Value Iteration training time: {time.time() - start_time:.2f} seconds")
    return policy, V

def policy_evaluation(policy, V, env, gamma, theta, max_eval_iters=100):
    """Evaluates a policy by calculating its value function."""
    global TRANSITION_CACHE
    _, _, _, num_states, _ = _get_env_params(env)
    V_eval = np.copy(V)

    for _ in range(max_eval_iters):
        delta = 0
        TRANSITION_CACHE.clear()
        for s in range(num_states):
            v_old = V_eval[s]
            a = policy[s]
            q_values = calculate_q_values_with_cache(s, V_eval, env, gamma)
            V_eval[s] = q_values[a]
            delta = max(delta, abs(v_old - V_eval[s]))
        if delta < theta:
            break
    return V_eval

def policy_iteration(env, gamma=0.95, theta=1e-4, max_iterations=5):
    """Implements the Policy Iteration algorithm."""
    print("Running Policy Iteration...")
    start_time = time.time()

    _, _, _, num_states, _ = _get_env_params(env)
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    for i in range(max_iterations):
        print(f"Iteration {i+1}...")
        V = policy_evaluation(policy, V, env, gamma, theta)
        
        policy_stable = True
        new_policy = np.copy(policy)
        for s in range(num_states):
            old_action = policy[s]
            q_values = calculate_q_values_with_cache(s, V, env, gamma)
            new_policy[s] = np.argmax(q_values)
            if old_action != new_policy[s]:
                policy_stable = False
        policy = new_policy

        if policy_stable:
            print(f"Converged after {i + 1} iterations.")
            break

    print(f"Policy Iteration training time: {time.time() - start_time:.2f} seconds")
    return policy, V

# --- Evaluation & Plotting ---
def get_action_from_policy(state, timestep, policy, env):
    """Get the action for a given state from a policy array."""
    if isinstance(state, dict):
        current_weight = int(state['state'][0])
        item_idx = int(state['state'][1])
    else:
        current_weight = int(state[0])
        item_idx = int(state[1])

    max_weight, max_timestep, num_items, _, _ = _get_env_params(env)
    current_weight = min(current_weight, max_weight)
    timestep = min(timestep, max_timestep - 1)
    item_idx = item_idx % num_items

    state_idx = encode_state(current_weight, timestep, item_idx, env)
    return policy[state_idx]

def evaluate_policy_multiple_seeds(env, policy, seeds, step_limit):
    """Evaluate a policy across multiple seeds."""
    results = {}
    episode_values = []
    final_values = []
    
    for seed in seeds:
        print(f"\nEvaluating seed {seed}...")
        env.set_seed(seed)
        state = env.reset()
        
        total_value = 0
        step_values = [0]
        
        for step in range(step_limit):
            action = get_action_from_policy(state, env.step_counter, policy, env)
            state, reward, done, _ = env.step(action)
            total_value += reward
            step_values.append(total_value)

            if done:
                while len(step_values) <= step_limit:
                    step_values.append(total_value)
                break
        
        episode_values.append(step_values[:step_limit+1])
        final_values.append(total_value)
        print(f"  Final value for seed {seed}: {total_value:.2f}")
    
    results['episode_values'] = episode_values
    results['final_values'] = final_values
    results['seeds'] = seeds
    
    return results

def plot_value_progression(results, title="Knapsack Value Over Time"):
    """Plot the accumulated value over time for multiple seeds."""
    plt.figure(figsize=(12, 8))
    
    for i, (seed, values) in enumerate(zip(results['seeds'], results['episode_values'])):
        steps = range(len(values))
        plt.plot(steps, values, label=f'Seed {seed}', linewidth=2, alpha=0.8)
    
    plt.xlabel('Time Step')
    plt.ylabel('Accumulated Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()

def create_value_function_heatmap(V, env, title="Value Function Heatmap"):
    """Create heatmaps of the value function, averaged over timesteps."""
    max_weight, max_timestep, num_items, _, _ = _get_env_params(env)
    avg_V = np.zeros((max_weight + 1, num_items))
    for w in range(max_weight + 1):
        for i in range(num_items):
            vals = []
            for t in range(max_timestep):
                state_idx = encode_state(w, t, i, env)
                vals.append(V[state_idx])
            avg_V[w, i] = np.mean(vals)

    item_weights = env.item_weights
    item_values = env.item_values
    ratios = item_values / (item_weights + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sort_indices = [np.argsort(item_weights), np.argsort(item_values), np.argsort(ratios)]
    x_labels = ["Items sorted by Weight", "Items sorted by Value", "Items sorted by Value/Weight Ratio"]
    
    for i, ax in enumerate(axes):
        sorted_V = avg_V[:, sort_indices[i]]
        cbar = i == 2
        cbar_ax = fig.add_axes([.91, .3, .02, .4]) if cbar else None
        sns.heatmap(sorted_V, ax=ax, cmap="viridis", cbar=cbar, cbar_ax=cbar_ax)
        ax.set_title(x_labels[i])
        ax.set_xlabel("Items")
        if i == 0:
            ax.set_ylabel("Knapsack Weight")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, .9, 1])
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()
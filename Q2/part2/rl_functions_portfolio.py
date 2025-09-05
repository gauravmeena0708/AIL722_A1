import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Helper Functions ---
MAX_CASH = 50
MAX_HOLDINGS = 15
ACTIONS = [-2, -1, 0, 1, 2]

def _get_env_params(env):
    """Get key parameters from the environment."""
    step_limit = env.step_limit
    num_states = step_limit * (MAX_CASH + 1) * (MAX_HOLDINGS + 1)
    return step_limit, num_states

def encode_state(t, cash, holdings):
    """Encode (timestep, cash, holdings) into a single state index."""
    return int(t * (MAX_CASH + 1) * (MAX_HOLDINGS + 1) + cash * (MAX_HOLDINGS + 1) + holdings)

def decode_state(s):
    """Decode a state index back to (timestep, cash, holdings)."""
    holdings = s % (MAX_HOLDINGS + 1)
    cash = ((s - holdings) // (MAX_HOLDINGS + 1)) % (MAX_CASH + 1)
    t = (s - holdings - cash * (MAX_HOLDINGS + 1)) // ((MAX_CASH + 1) * (MAX_HOLDINGS + 1))
    return int(t), int(cash), int(holdings)

def _get_transition_details(cash, holdings, action, price):
    """Calculate next state and reward for a given state-action-price combination."""
    buy_cost, sell_cost = 1, 1
    new_holdings = holdings + action
    if not (0 <= new_holdings <= MAX_HOLDINGS):
        return None  # Invalid transition

    if action > 0:  # Buy
        cost = action * (price + buy_cost)
    elif action < 0:  # Sell
        cost = action * (price - sell_cost)  # Negative cost is cash inflow
    else:  # Hold
        cost = 0
    
    new_cash = cash - cost
    if not (0 <= new_cash <= MAX_CASH):
        return None  # Invalid transition

    return int(new_cash), int(new_holdings)

# --- Core Algorithms ---
def value_iteration(env, prices, gamma=1.0, theta=1e-4):
    """Implements finite-horizon Value Iteration for the portfolio problem."""
    print("Running Value Iteration...")
    start_time = time.time()
    step_limit, num_states = _get_env_params(env)
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)

    # Backward induction from T-1 to 0
    for t in range(step_limit - 1, -1, -1):
        for cash in range(MAX_CASH + 1):
            for holdings in range(MAX_HOLDINGS + 1):
                s = encode_state(t, cash, holdings)
                action_values = -np.inf * np.ones(len(ACTIONS))

                for i, action in enumerate(ACTIONS):
                    transition = _get_transition_details(cash, holdings, action, prices[t])
                    if transition is None:
                        continue
                    
                    next_cash, next_holdings = transition
                    reward = 0
                    future_val = 0

                    if t == step_limit - 1:
                        reward = next_cash + next_holdings * prices[t]
                    else:
                        next_s = encode_state(t + 1, next_cash, next_holdings)
                        future_val = V[next_s]

                    action_values[i] = reward + gamma * future_val
                
                V[s] = np.max(action_values)
                policy[s] = ACTIONS[np.argmax(action_values)]

    exec_time = time.time() - start_time
    print(f"Value Iteration training time: {exec_time:.2f} seconds")
    return policy, V, exec_time

def policy_evaluation(policy, V, env, prices, gamma, theta, max_eval_iters=10):
    """Evaluates a policy for the finite-horizon problem."""
    step_limit, num_states = _get_env_params(env)
    V_eval = np.copy(V)
    for _ in range(max_eval_iters):
        delta = 0
        for t in range(step_limit - 1, -1, -1):
            for cash in range(MAX_CASH + 1):
                for holdings in range(MAX_HOLDINGS + 1):
                    s = encode_state(t, cash, holdings)
                    v_old = V_eval[s]
                    action = policy[s]
                    
                    transition = _get_transition_details(cash, holdings, action, prices[t])
                    if transition is None:
                        continue

                    next_cash, next_holdings = transition
                    reward = 0
                    future_val = 0
                    if t == step_limit - 1:
                        reward = next_cash + next_holdings * prices[t]
                    else:
                        next_s = encode_state(t + 1, next_cash, next_holdings)
                        future_val = V_eval[next_s]

                    V_eval[s] = reward + gamma * future_val
                    delta = max(delta, abs(v_old - V_eval[s]))
        if delta < theta:
            break
    return V_eval

def policy_iteration(env, prices, gamma=1.0, theta=1e-2, max_iterations=100):
    """Implements finite-horizon Policy Iteration."""
    print("Running Policy Iteration...")
    start_time = time.time()
    step_limit, num_states = _get_env_params(env)
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int) # Default action is 0 (hold)
    convergence_history = []

    for i in range(max_iterations):
        V = policy_evaluation(policy, V, env, prices, gamma, theta)
        
        policy_stable = True
        max_v_diff = 0
        new_policy = np.copy(policy)
        for s in range(num_states):
            t, cash, holdings = decode_state(s)
            old_action = policy[s]
            
            action_values = -np.inf * np.ones(len(ACTIONS))
            for j, action in enumerate(ACTIONS):
                transition = _get_transition_details(cash, holdings, action, prices[t])
                if transition is None:
                    continue
                next_cash, next_holdings = transition
                reward = 0
                future_val = 0
                if t == step_limit - 1:
                    reward = next_cash + next_holdings * prices[t]
                else:
                    next_s = encode_state(t + 1, next_cash, next_holdings)
                    future_val = V[next_s]
                action_values[j] = reward + gamma * future_val

            best_action_idx = np.argmax(action_values)
            new_policy[s] = ACTIONS[best_action_idx]
            max_v_diff = max(max_v_diff, abs(V[s] - np.max(action_values)))

            if old_action != new_policy[s]:
                policy_stable = False
        
        policy = new_policy
        convergence_history.append(max_v_diff)
        print(f"Iteration {i+1}: Max value difference = {max_v_diff:.6f}")
        if policy_stable or max_v_diff < theta:
            print(f"Converged after {i + 1} iterations.")
            break

    exec_time = time.time() - start_time
    print(f"Policy Iteration training time: {exec_time:.2f} seconds")
    return policy, V, exec_time, convergence_history

# --- Evaluation & Plotting ---
def simulate_episode(env, policy, prices):
    """Simulate one episode with the learned policy."""
    env.reset()
    env.asset_prices = np.array(prices).reshape(1, -1)
    cash = env.initial_cash
    holdings = 0

    cash_hist, holdings_hist, wealth_hist = [cash], [holdings], [cash]

    for t in range(env.step_limit):
        s = encode_state(t, min(cash, MAX_CASH), min(holdings, MAX_HOLDINGS))
        action = policy[s]
        
        # Manually step through logic to avoid env state issues
        price = prices[t]
        transition = _get_transition_details(cash, holdings, action, price)
        if transition:
            cash, holdings = transition
        
        wealth = cash + holdings * price
        cash_hist.append(cash)
        holdings_hist.append(holdings)
        wealth_hist.append(wealth)

    return {
        'cash': cash_hist, 'holdings': holdings_hist, 'wealth': wealth_hist,
        'final_wealth': wealth_hist[-1]
    }

def plot_episode_evolution(results, prices, title):
    """Plot cash, holdings, and wealth over an episode."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    time_steps = range(len(results['wealth']))

    axes[0, 0].plot(time_steps, results['cash'], 'b-', marker='o')
    axes[0, 0].set_title('Cash Over Time')
    axes[0, 0].set_ylabel('Cash')
    axes[0, 0].grid(True, alpha=0.5)

    axes[0, 1].plot(time_steps, results['holdings'], 'g-', marker='s')
    axes[0, 1].set_title('Asset Holdings Over Time')
    axes[0, 1].set_ylabel('Holdings')
    axes[0, 1].grid(True, alpha=0.5)

    axes[1, 0].plot(time_steps, results['wealth'], 'r-', marker='^')
    axes[1, 0].set_title('Total Wealth Over Time')
    axes[1, 0].set_ylabel('Wealth')
    axes[1, 0].grid(True, alpha=0.5)

    axes[1, 1].plot(range(len(prices)), prices, 'k-', marker='d')
    axes[1, 1].set_title('Asset Prices')
    axes[1, 1].set_ylabel('Price')
    axes[1, 1].grid(True, alpha=0.5)

    for ax in fig.get_axes():
        ax.set_xlabel('Time Step')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(title.replace(" ", "_") + ".png")
    plt.close()

def plot_convergence_analysis(history, title):
    """Plot maximum value difference vs iteration for convergence analysis."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(history) + 1), history, 'b-', marker='o')
    plt.axhline(y=0.01, color='r', linestyle='--', label='Convergence Threshold (0.01)')
    plt.xlabel('Policy Iteration Number')
    plt.ylabel('Maximum Value Difference')
    plt.title(f'Policy Iteration Convergence: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + "_convergence.png")
    plt.close()

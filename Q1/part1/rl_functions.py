import numpy as np
import heapq

# This global variable will be used by the main script to track transition calls.
TRANSITION_CALL_COUNT = 0
TRANSITION_CACHE = {}


def calculate_q_values(s, V, env, gamma=0.95, time_step=None):
    """
    Calculates the Q-value for each action in a given state.
    Includes optional time_step for non-stationary environments.
    """
    global TRANSITION_CALL_COUNT
    num_actions = env.action_space.n
    action_values = np.zeros(num_actions)
    state_tuple = env.index_to_state(s)

    for a in range(num_actions):
        TRANSITION_CALL_COUNT += 1
        transitions = env.get_transitions_at_time(state_tuple, a, time_step=time_step)
        
        current_action_value = 0
        for prob, next_state_tuple in transitions:
            next_s_idx = env.state_to_index(next_state_tuple)
            player_pos = (state_tuple[0], state_tuple[1])
            ball_pos = (next_state_tuple[0], next_state_tuple[1])
            reward = env._get_reward(ball_pos, a, player_pos)
            is_terminal = next_state_tuple[2]
            
            current_action_value += prob * (reward + gamma * V[next_s_idx] * (1 - is_terminal))
        
        action_values[a] = current_action_value
        
    return action_values

def calculate_q_values_with_cache(s, V, env, gamma=0.95, time_step=None):
    """
    Calculates the Q-value for each action in a given state using a cache.
    Includes optional time_step for non-stationary environments.
    """
    global TRANSITION_CALL_COUNT
    global TRANSITION_CACHE
    num_actions = env.action_space.n
    action_values = np.zeros(num_actions)
    state_tuple = env.index_to_state(s)

    for a in range(num_actions):
        cache_key = (id(env), time_step, state_tuple, a)
        transitions = TRANSITION_CACHE.get(cache_key)
        
        if transitions is None:
            TRANSITION_CALL_COUNT += 1
            transitions = env.get_transitions_at_time(state_tuple, a, time_step=time_step)
            TRANSITION_CACHE[cache_key] = transitions

        current_action_value = 0.0
        for prob, next_state_tuple in transitions:
            next_s_idx = env.state_to_index(next_state_tuple)
            player_pos = (state_tuple[0], state_tuple[1])
            ball_pos = (next_state_tuple[0], next_state_tuple[1])
            reward = env._get_reward(ball_pos, a, player_pos)
            is_terminal = next_state_tuple[2]
            
            current_action_value += prob * (reward + gamma * V[next_s_idx] * (1 - is_terminal))
        
        action_values[a] = current_action_value
        
    return action_values

def policy_evaluation(policy, V_initial, env, gamma=0.95, theta=1e-6):
    """
    Evaluates a policy by calculating its value function until convergence.
    """
    num_states = env.grid_size * env.grid_size * 2
    V = np.copy(V_initial)
    
    while True:
        delta = 0
        for s in range(num_states):
            v_old = V[s]
            state_tuple = env.index_to_state(s)

            if state_tuple[2] == 1:
                V[s] = 0
                continue

            a = policy[s]
            q_value = calculate_q_values(s, V, env, gamma, time_step=None)[a]
            
            V[s] = q_value
            delta = max(delta, abs(v_old - V[s]))
        
        if delta < theta:
            break
            
    return V

def policy_improvement(policy, V, env, gamma=0.95):
    """
    Improves a policy greedily based on a given value function.
    """
    num_states = env.grid_size * env.grid_size * 2
    new_policy = np.copy(policy)
    policy_stable = True
    
    for s in range(num_states):
        old_action = policy[s]
        state_tuple = env.index_to_state(s)

        if state_tuple[2] == 1:
            continue
        
        q_values = calculate_q_values(s, V, env, gamma)
        best_action = np.argmax(q_values)
        new_policy[s] = best_action
        
        if old_action != best_action:
            policy_stable = False
            
    return new_policy, policy_stable

def policy_iteration(env, gamma=0.95, theta=1e-6):
    """
    Implements the Policy Iteration algorithm for stationary environments.
    """
    global TRANSITION_CALL_COUNT
    TRANSITION_CALL_COUNT = 0
    TRANSITION_CACHE.clear()
    
    num_states = env.grid_size * env.grid_size * 2
    policy = np.zeros(num_states, dtype=int)
    V = np.zeros(num_states)
    num_iterations = 0

    while True:
        num_iterations += 1
        V = policy_evaluation(policy, V, env, gamma, theta)
        new_policy, policy_stable = policy_improvement(policy, V, env, gamma)
        policy = new_policy
        
        if policy_stable:
            break

    optimal_policy = {i: policy[i] for i in range(num_states)}
    return optimal_policy, V, num_iterations, TRANSITION_CALL_COUNT

def value_iteration(env, gamma=0.95, theta=1e-6):
    """
    Implements Value Iteration for stationary environments.
    """
    global TRANSITION_CALL_COUNT
    TRANSITION_CALL_COUNT = 0
    TRANSITION_CACHE.clear()
    
    num_states = env.grid_size * env.grid_size * 2
    V = np.zeros(num_states)
    num_iterations = 0

    while True:
        num_iterations += 1
        delta = 0
        for s in range(num_states):
            v_old = V[s]
            state_tuple = env.index_to_state(s)

            if state_tuple[2] == 1:
                V[s] = 0
                continue
            
            q_values = calculate_q_values(s, V, env, gamma)
            V[s] = np.max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        
        if delta < theta:
            break
            
    dummy_policy = np.zeros(num_states, dtype=int)
    optimal_policy_arr, _ = policy_improvement(dummy_policy, V, env, gamma)
    optimal_policy = {i: optimal_policy_arr[i] for i in range(num_states)}
    
    return optimal_policy, V, num_iterations, TRANSITION_CALL_COUNT

def prioritized_value_iteration(env, gamma=0.95, theta=1e-6):
    """
    Implements Prioritized Value Iteration for stationary environments.
    """
    global TRANSITION_CALL_COUNT
    TRANSITION_CALL_COUNT = 0
    TRANSITION_CACHE.clear()

    num_states = env.grid_size * env.grid_size * 2
    V = np.zeros(num_states)
    pq = []

    # 1. Build predecessor model
    predecessors = {s: set() for s in range(num_states)}
    for s in range(num_states):
        state_tuple = env.index_to_state(s)
        if state_tuple[2] == 1: continue
        for a in range(env.action_space.n):
            # This does not use the cache, correctly counting model-building calls
            transitions = env.get_transitions_at_time(state_tuple, a, time_step=None)
            TRANSITION_CALL_COUNT += 1
            for prob, next_state_tuple in transitions:
                if prob > 0:
                    next_s_idx = env.state_to_index(next_state_tuple)
                    predecessors[next_s_idx].add(s)

    # 2. Initialize priority queue
    for s in range(num_states):
        state_tuple = env.index_to_state(s)
        if state_tuple[2] == 1: continue
        q_values = calculate_q_values(s, V, env, gamma)
        best_q_value = np.max(q_values)
        error = abs(V[s] - best_q_value)
        if error > theta:
            heapq.heappush(pq, (-error, s))

    # 3. Main loop
    num_updates = 0
    while pq:
        if num_updates > num_states * 10: # Safety break
            print("Warning: Prioritized VI exceeded max updates.")
            break
        
        priority, s = heapq.heappop(pq)


        q_values_check = calculate_q_values(s, V, env, gamma)
        if abs(V[s] - np.max(q_values_check)) < theta:
            continue

        num_updates += 1
        V[s] = np.max(q_values_check)

        for p_idx in predecessors[s]:
            p_state_tuple = env.index_to_state(p_idx)
            if p_state_tuple[2] == 1: continue

            q_values_p = calculate_q_values(p_idx, V, env, gamma)
            best_q_value_p = np.max(q_values_p)
            error_p = abs(V[p_idx] - best_q_value_p)

            if error_p > theta:
                heapq.heappush(pq, (-error_p, p_idx))

    # 4. Extract final policy
    dummy_policy = np.zeros(num_states, dtype=int)
    optimal_policy_arr, _ = policy_improvement(dummy_policy, V, env, gamma)
    optimal_policy = {i: optimal_policy_arr[i] for i in range(num_states)}
    
    return optimal_policy, V, num_updates, TRANSITION_CALL_COUNT

def value_iteration_non_stationary(env, gamma=0.95, max_time_horizon=40):
    """
    Implements Value Iteration for the non-stationary 'degraded pitch' environment.
    """
    global TRANSITION_CALL_COUNT
    TRANSITION_CALL_COUNT = 0
    TRANSITION_CACHE.clear()
    
    num_states = env.grid_size * env.grid_size * 2
    V = np.zeros((max_time_horizon + 1, num_states))
    policy_arr = np.zeros((max_time_horizon, num_states), dtype=int)
    
    for t in range(max_time_horizon - 1, -1, -1):
        for s in range(num_states):
            state_tuple = env.index_to_state(s)
            
            if state_tuple[2] == 1:
                V[t, s] = 0
                continue
                
            q_values = calculate_q_values(s, V[t+1, :], env, gamma, time_step=t)
            V[t, s] = np.max(q_values)
            policy_arr[t, s] = np.argmax(q_values)
    
    optimal_policy = {t: {s: policy_arr[t, s] for s in range(num_states)} for t in range(max_time_horizon)}
    
    return optimal_policy, V, max_time_horizon, TRANSITION_CALL_COUNT


def evaluate_policy(policy, envr, num_episodes=20, non_stationary=False):
    """
    Evaluates a policy.
    If non_stationary is False (default), evaluates a stationary policy (dict: state->action).
    If non_stationary is True, evaluates a time-dependent policy (dict: time_step->state->action).
    """
    if non_stationary:
        env = envr(degrade_pitch=True)
    else:
        env = envr()

    episode_rewards = []

    for i in range(num_episodes):
        obs, _ = env.reset(seed=i)
        done = False
        total_reward = 0
        time_step = 0

        while not done:
            state_index = env.state_to_index(obs)

            if non_stationary:
                action = policy[time_step][state_index]
            else:
                action = policy[state_index]

            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            time_step += 1

            if non_stationary and time_step >= len(policy):
                break

        episode_rewards.append(total_reward)

    return np.mean(episode_rewards), np.std(episode_rewards)

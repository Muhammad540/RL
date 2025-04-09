'''
in this script, the performance of 4 different algorithms are compared on a non-stationary setting:
1. Sample Average Method
2. Constant Step-Size Method
3. UCB Method
4. Gradient Bandit Method

the performance is evaluated over 2000 episodes, with each episode having 200000 timesteps
'''
import numpy as np
import matplotlib.pyplot as plt

K = 10 # number of arms
eps = 0.1 # probability of exploration
episodes = 2000
timesteps = 200000 

random_walks = True 
step_size = 0.1
initial_true_action_values = np.zeros(K) 

# sum the rewards across all episodes
# These will store the sum of rewards received at each timestep across all episodes
total_rewards_sample_avg = np.zeros(timesteps)
total_rewards_constant_step_size = np.zeros(timesteps)
total_rewards_ucb = np.zeros(timesteps)
total_rewards_gradient_bandit = np.zeros(timesteps)

for episode in range(episodes):
    # Reset estimates and counts for each new episode
    # Start from same initial point if desired, or comment out if walk continues across episodes
    true_action_values = np.copy(initial_true_action_values) 
    
    action_counts_sample_avg = np.zeros(K)
    action_counts_constant_step_size = np.zeros(K)
    action_counts_ucb = np.zeros(K)
    
    estimated_action_values_sample_avg = np.zeros(K)
    estimated_action_values_constant_step_size = np.zeros(K)
    estimated_action_values_ucb = np.zeros(K)
    estimated_preferences_gradient_bandit = np.zeros(K)

    # baseline for gradient bandit for this episode
    avg_reward_baseline_gradient = 0
    # Gradient Bandit parameters
    alpha_gradient = 0.1 # Learning rate for preferences

    if (episode + 1) % 100 == 0:
        print(f"Running episode {episode + 1}/{episodes}")

    for timestep in range(timesteps):
        # Perform random walk on true action values at each timestep
        if random_walks:
            true_action_values += np.random.normal(0, 0.01, K)
            
        # --- Sample average method ---
        if np.random.rand() < eps:
            action_sa = np.random.randint(K) # explore
        else:
            action_sa = np.argmax(estimated_action_values_sample_avg) # exploit

        reward_sa = np.random.normal(true_action_values[action_sa], 1)
        action_counts_sample_avg[action_sa] += 1
        # Update estimates using sample average
        estimated_action_values_sample_avg[action_sa] += (reward_sa - estimated_action_values_sample_avg[action_sa]) / action_counts_sample_avg[action_sa]
        # Accumulate reward for this timestep
        total_rewards_sample_avg[timestep] += reward_sa

        # --- Constant step-size action-selection ---
        if np.random.rand() < eps:
            action_cs = np.random.randint(K) # explore
        else:
            action_cs = np.argmax(estimated_action_values_constant_step_size) # exploit

        reward_cs = np.random.normal(true_action_values[action_cs], 1)
        # Update estimates using constant step-size
        estimated_action_values_constant_step_size[action_cs] += (reward_cs - estimated_action_values_constant_step_size[action_cs]) * step_size
        # Accumulate reward for this timestep
        total_rewards_constant_step_size[timestep] += reward_cs


        # --- UCB action-selection ---
        ucb_values = estimated_action_values_ucb + np.sqrt(2 * np.log(timestep + 1 + 1e-5) / (action_counts_ucb + 1e-5))
        action_ucb = np.argmax(ucb_values)

        reward_ucb = np.random.normal(true_action_values[action_ucb], 1)
        action_counts_ucb[action_ucb] += 1
        # Update estimates using sample average (standard for UCB)
        estimated_action_values_ucb[action_ucb] += (reward_ucb - estimated_action_values_ucb[action_ucb]) / action_counts_ucb[action_ucb]
        # Accumulate reward for this timestep
        total_rewards_ucb[timestep] += reward_ucb

        # --- Gradient bandit action-selection ---
        # Calculate softmax probabilities from preferences
        # Subtract max for numerical stability before exponentiating
        exp_prefs = np.exp(estimated_preferences_gradient_bandit - np.max(estimated_preferences_gradient_bandit))
        action_probs = exp_prefs / np.sum(exp_prefs)

        # Select action probabilistically based on preferences (softmax)
        action_gb = np.random.choice(K, p=action_probs)

        reward_gb = np.random.normal(true_action_values[action_gb], 1)
        # Accumulate reward for this timestep
        total_rewards_gradient_bandit[timestep] += reward_gb

        avg_reward_baseline_gradient += (reward_gb - avg_reward_baseline_gradient) / (timestep + 1) # Incremental average update

        # Update preferences
        for a in range(K):
            if a == action_gb:
                estimated_preferences_gradient_bandit[a] += alpha_gradient * (reward_gb - avg_reward_baseline_gradient) * (1 - action_probs[a])
            else:
                estimated_preferences_gradient_bandit[a] -= alpha_gradient * (reward_gb - avg_reward_baseline_gradient) * action_probs[a]

avg_rewards_sample_avg = total_rewards_sample_avg / episodes
avg_rewards_constant_step_size = total_rewards_constant_step_size / episodes
avg_rewards_ucb = total_rewards_ucb / episodes
avg_rewards_gradient_bandit = total_rewards_gradient_bandit / episodes

start_index = timesteps - 100000
perf_sample_avg = np.mean(avg_rewards_sample_avg[start_index:])
perf_constant_step_size = np.mean(avg_rewards_constant_step_size[start_index:])
perf_ucb = np.mean(avg_rewards_ucb[start_index:])
perf_gradient_bandit = np.mean(avg_rewards_gradient_bandit[start_index:])

print(f"Average Reward over last {timesteps-start_index} steps:")
print(f"  Sample Average: {perf_sample_avg:.4f}")
print(f"  Constant Step-Size (alpha={step_size}): {perf_constant_step_size:.4f}")
print(f"  UCB (c=sqrt(2)): {perf_ucb:.4f}")
print(f"  Gradient Bandit (alpha={alpha_gradient}): {perf_gradient_bandit:.4f}")


plt.figure(figsize=(12, 8)) 
plt.plot(avg_rewards_sample_avg, label='Sample Average Method (\u03B5=0.1)') 
plt.plot(avg_rewards_constant_step_size, label=f'Constant Step-Size Method (\u03B5=0.1, \u03B1={step_size})') 
plt.plot(avg_rewards_ucb, label='UCB Method (c=\u221A2)') 
plt.plot(avg_rewards_gradient_bandit, label=f'Gradient Bandit Method (\u03B1={alpha_gradient})') 
plt.xlabel('Time Steps')
plt.ylabel('Average Reward')
plt.title('Comparison of Action-Value Methods (Non-Stationary, Random Walk q*)') 
plt.legend()
plt.grid(True) 
plt.ylim(bottom=0) 
plt.savefig('compare_algos_for_non_stationary_case_updated.png')
plt.show() 
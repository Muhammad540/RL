'''
in this script, the performance of 2 different algorithms are compared on a non-stationary setting:
1. Sample Average Method
2. Constant Step-Size Method

the performance is evaluated over 2000 episodes, with each episode having 10000 timesteps
'''
import numpy as np 
import matplotlib.pyplot as plt

K = 10 # number of arms
eps = 0.1 # probability of exploration 
episodes = 2000 
timesteps = 10000

random_walks = True
step_size = 0.1

total_rewards_sample_avg = np.zeros(timesteps)
total_rewards_constant_step_size = np.zeros(timesteps)

for episode in range(episodes):
    true_action_values = np.zeros(K) # Start fresh for each episode
    
    action_counts_sample_avg = np.zeros(K)
    estimated_action_values_sample_avg = np.zeros(K)
    estimated_action_values_constant_step_size = np.zeros(K)

    for timestep in range(timesteps):
        
        # Non-stationary part: Add random walk *at each timestep*
        if random_walks:
            true_action_values += np.random.normal(0, 0.01, K)

        # Sample average method
        if np.random.rand() < eps:
            # explore
            action_sample_avg = np.random.randint(K)
        else:
            # exploit
            action_sample_avg = np.argmax(estimated_action_values_sample_avg)
        
        reward_sample_avg = np.random.normal(true_action_values[action_sample_avg], 1)
        total_rewards_sample_avg[timestep] += reward_sample_avg
        action_counts_sample_avg[action_sample_avg] += 1

        estimated_action_values_sample_avg[action_sample_avg] += (reward_sample_avg - estimated_action_values_sample_avg[action_sample_avg]) / action_counts_sample_avg[action_sample_avg]
    
        # Constant step-size action-selection  
        if np.random.rand() < eps:  
            action_constant_step = np.random.randint(K) 
        else:  
            action_constant_step = np.argmax(estimated_action_values_constant_step_size) 
        
        # Use the same true_action_values (updated by the random walk) for reward generation
        reward_constant_step = np.random.normal(true_action_values[action_constant_step], 1) 
        total_rewards_constant_step_size[timestep] += reward_constant_step 

        estimated_action_values_constant_step_size[action_constant_step] += (reward_constant_step - estimated_action_values_constant_step_size[action_constant_step]) * step_size

avg_rewards_sample_avg = total_rewards_sample_avg / episodes
avg_rewards_constant_step_size = total_rewards_constant_step_size / episodes

plt.figure(figsize=(12, 6))  
plt.plot(avg_rewards_sample_avg, label='Sample Average Method')  
plt.plot(avg_rewards_constant_step_size, label=f'Constant Step-Size Method (α = {step_size})') 
plt.xlabel('Time Steps')  
plt.ylabel('Average Reward')  
plt.title('Action-Value Methods Performance (Non-Stationary)') 
plt.legend()  
plt.grid()  
plt.savefig('action_value_methods_performance_nonstationary.png')
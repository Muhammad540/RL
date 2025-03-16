import numpy as np 
import matplotlib.pyplot as plt

K = 10 # number of arms
eps = 0.1 # probability of exploration 
episodes = 2000 
timesteps = 10000
# we will observe the effect of eps on the average rewards
epsilons = 0.1

average_rewards = np.zeros(timesteps)
average_action_counts = np.zeros(K)
random_walks = False
constant_step_size = False
step_size = 0.1

true_action_values = np.zeros(K)
total_rewards_sample_avg = np.zeros(timesteps)
total_rewards_constant_step_size = np.zeros(timesteps)

action_counts_sample_avg = np.zeros(K)
action_counts_constant_step_size = np.zeros(K)
estimated_action_values_sample_avg = np.zeros(K)
estimated_action_values_constant_step_size = np.zeros(K)

for episode in range(episodes):
    if random_walks:
        # random walks means that the true action values are changing over time
        true_action_values += np.random.normal(0, 0.01, K)

    total_reward_sample_avg = 0
    total_reward_constant_step_size = 0
    
    for timestep in range(timesteps):
        
        # Sample average method
        if np.random.rand() < eps:
            # explore
            action = np.random.randint(K)
        else:
            # exploit
            action = np.argmax(estimated_action_values_sample_avg)
        
        reward = np.random.normal(true_action_values[action],1)
        total_reward_sample_avg += reward
        action_counts_sample_avg[action] += 1

        estimated_action_values_sample_avg[action] += (reward - estimated_action_values_sample_avg[action])/action_counts_sample_avg[action]
        total_rewards_sample_avg[timestep] += total_reward_sample_avg/(episode+1)
    
        # Constant step-size action-selection  
        if np.random.rand() < eps:  
            action = np.random.randint(K) 
        else:  
            action = np.argmax(estimated_action_values_constant_step_size) 
        
        reward = np.random.normal(true_action_values[action], 1) 
        total_reward_constant_step_size += reward 
        action_counts_constant_step_size[action] += 1 
        
        estimated_action_values_constant_step_size[action] += (reward - estimated_action_values_constant_step_size[action])*step_size

        total_rewards_constant_step_size[timestep] += total_reward_constant_step_size / (episode + 1)  


plt.figure(figsize=(12, 6))  
plt.plot(total_rewards_sample_avg, label='Sample Average Method')  
plt.plot(total_rewards_constant_step_size, label='Constant Step-Size Method (α = 0.1)')  
plt.xlabel('Time Steps')  
plt.ylabel('Average Reward')  
plt.title('Action-Value Methods Performance')  
plt.legend()  
plt.grid()  
plt.savefig('action_value_methods_performance.png')  # Save the plot instead of showing it

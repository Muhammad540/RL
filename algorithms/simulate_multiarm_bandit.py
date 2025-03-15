# In this script, we will code up a simple multi-arm bandit problem and
# observe how the exploration-exploitation trade-off plays out.
# DONT BE GREEDY, GET A FEW REWARDS FIRST :D

import numpy as np 
import matplotlib.pyplot as plt

K = 10 # number of arms
eps = 0.1 # probability of exploration 
episodes = 2000 
timesteps = 1000 
# we will observe the effect of eps on the average rewards
epsilons = [0.1, 0.05, 0.01, 0] # larger eps means more exploration probability

average_rewards = {eps: np.zeros(timesteps) for eps in epsilons}
average_action_counts = np.zeros(K)

for eps in epsilons:
    total_reward = np.zeros(timesteps)
    
    for episode in range(episodes):
        # for each episode, we have a new set of true action values
        # think of each episode as a different game of the multi-arm bandit problem
        true_action_values = np.random.normal(0, 1, K)
        
        # we have no clue about the true action values, so we initialize the estimated action values to 0
        estimated_action_values = np.zeros(K)
        action_counts = np.zeros(K)
        
        for timestep in range(timesteps):   
            if np.random.rand() < eps:
                # explore
                action = np.random.randint(K)
            else:
                # exploit
                action = np.argmax(estimated_action_values)
            
            # based on the chosen action, we get a reward 
            # the reward is sampled from a normal distribution with the mean 
            # being the true action value and the variance being 1
            reward = np.random.normal(true_action_values[action],1)
            total_reward[timestep] += reward
            action_counts[action] += 1
            
            # so intuitively, we are closing the error gap between the estimated and true action values
            # by moving the estimated action value towards the reward
            # also we normalize the update by the number of times we've chosen the action
            # so the more we've chosen the action, the less we update the estimated action value (since we are more confident)
            estimated_action_values[action] += (reward - estimated_action_values[action])/action_counts[action] # this is the sample average action value update rule

        # average reward and action counts for each episode
        average_rewards[eps] += total_reward/timesteps
        average_action_counts += action_counts/timesteps

plt.figure(figsize=(12, 6))  
for eps, rewards in average_rewards.items():  
    plt.plot(rewards, label=f'ε={eps}', linewidth=2)  

plt.title('Average Reward Over Time Steps for Different Epsilon Values', fontsize=16)  
plt.xlabel('Timesteps', fontsize=14)  
plt.ylabel('Average Reward', fontsize=14)  
plt.legend(fontsize=12)  
plt.grid(alpha=0.3)  

plt.savefig('average_reward_timesteps.png')  
plt.show()  
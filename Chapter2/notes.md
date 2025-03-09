# Chapter 2: Multi-Armed Bandits
* **Optimal Control:**  The problem of designing a controller for a dynamical system to minimize or maximize a measure of its behavior over time.

* **Bellman Equation:** A central equation in dynamic programming and optimal control that provides a recursive relationship for calculating optimal values or policies.

* **Dynamic Programming:** A class of methods for solving optimal control problems using the Bellman equation. It breaks down complex problems into simpler overlapping subproblems.

* **Markov Decision Process (MDP):**  The discrete-time stochastic version of optimal control problems. MDPs provide a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker.

* **Stochastic Optimal Control:** Optimal control problems where the system dynamics or rewards are stochastic (probabilistic). These problems are often formulated and solved using MDPs.

* **Curse of Dimensionality:** A significant challenge in dynamic programming and related methods. Computational requirements (memory and time) grow exponentially with the number of state variables, making it difficult to apply these methods to high-dimensional problems.

* **Credit Assignment Problem:** A fundamental challenge in reinforcement learning.
    * **Definition:** How to attribute credit or blame for an outcome (e.g., success or failure) to the sequence of actions that led to it, especially when many decisions are involved over time.

* **Temporal Difference (TD) Methods:** A class of reinforcement learning methods that learn by bootstrapping from temporally successive estimates of the *same* value function or quantity. They learn from incomplete episodes and can be used online.

* **Bandit Problems:** A simplified type of reinforcement learning problem where there is only a *single state*. In essence, it's about repeatedly choosing between different actions in the same situation to maximize cumulative reward.

* **Fundamental Methods for Solving MDPs:** There are three main classes of methods, each with its own strengths and weaknesses:

    1. **Dynamic Programming (DP):**
        * **Requires:** A complete and accurate model of the environment (transition probabilities and reward functions).
        * **Pros:** Guaranteed to find the optimal policy if the model is correct and computational resources are sufficient.
        * **Cons:** Suffers from the curse of dimensionality; computationally expensive for large state spaces; requires a model, which is not always available in real-world problems.

    2. **Monte Carlo (MC) Methods:**
        * **Requires:** No model of the environment (model-free).
        * **Pros:** Simple to understand and implement; can learn directly from experience without needing a model.
        * **Cons:** Not well-suited for step-by-step incremental computation; typically requires complete episodes to learn; can have high variance.

    3. **Temporal Difference (TD) Learning:**
        * **Requires:** No model of the environment (model-free).
        * **Pros:** Fully incremental (can learn online, step-by-step); doesn't require complete episodes; typically lower variance than Monte Carlo methods.
        * **Cons:** Can be more complex to analyze theoretically compared to Monte Carlo methods; can be sensitive to step-size parameters.

    * **Summary:** These methods differ in their model requirements, computational efficiency, speed of convergence, and suitability for different types of problems.

* **Combining Strengths:**
    * **Multi-step Bootstrapping:**  The strengths of Monte Carlo methods (no model, simplicity) can be combined with Temporal Difference methods (incremental learning, lower variance) through multi-step bootstrapping techniques (like TD($\lambda$)).
    * **Model Learning and Planning:** Temporal difference learning can be integrated with model learning (estimating a model from experience) and planning methods (like dynamic programming using the learned model) to create a more complete and unified solution for reinforcement learning problems, especially in tabular settings.

<br/>

## Multi-Armed Bandits

* **Reinforcement Learning vs. Supervised Learning:**  In reinforcement learning, training information is used to *evaluate* the actions taken (how good was the action?) rather than to *instruct* by providing the correct actions (as in supervised learning).

* **Evaluative Feedback:**  RL agents receive evaluative feedback in the form of rewards, indicating the quality of their actions, but not necessarily telling them the best action to take.

* **Exploration Encouraged:**  The evaluative nature of feedback encourages RL algorithms to actively explore different actions to discover better behaviors and maximize cumulative reward.

* **Non-Associative Setting:** A simplified setting where the agent does not need to learn different actions in different situations. This implies that the state remains constant or is not relevant to the decision-making process. Bandit problems typically operate in a non-associative setting.

* **Focus on Evaluative Feedback:** In non-associative settings (like bandit problems), the agent relies purely on evaluative feedback (rewards) without considering the context or situation in which the agent finds itself.

* **Stationary Probability Distribution:**  In the context of bandit problems, it means that the probability distribution of rewards for each action remains constant over time. The underlying reward mechanism doesn't change.

### k-Armed Bandit Problem

* **Problem Description:** You are repeatedly faced with a choice among *k* different options (arms/actions). Each action, when selected, yields a reward drawn from a stationary probability distribution specific to that action.

* **Objective:** Maximize the expected *total* reward over some time period (e.g., over a sequence of action selections).

* **Exploitation vs. Exploration:** To achieve the objective, you need to balance:
    * **Exploitation:**  Concentrating action selections on the options (arms) that are currently believed to be the best (highest expected reward).
    * **Exploration:** Trying out different options, including those that are currently believed to be suboptimal, to gather more information and potentially discover even better options in the long run.

* **Action Value (True Value):**  The *true* value of an action *a*, denoted as $q_*(a)$, is the expected reward when action *a* is selected. It is defined as:

    $$
    q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]
    $$

    Where:
    * $q_*(a)$ is the true value (expected reward) of action $a$.
    * $\mathbb{E}[\cdot]$ denotes the expected value.
    * $R_t$ is the reward received at time step $t$.
    * $A_t$ is the action taken at time step $t$.
    * The notation $\doteq$ means "defined as" or "approximately equal to" in this context of definition.

* **Estimated Action Value:** The *estimated* value of action *a* at time step *t* is denoted as $Q_t(a)$.  The goal of learning is to make $Q_t(a)$ as close as possible to the true value $q_*(a)$.

* **Greedy Action Selection (Exploitation):** Choosing the action with the currently highest estimated value.  This rule is defined as:

    $$
    A_t \doteq \arg\max_a Q_t(a)
    $$

    * $\arg\max_a$ means selecting the action *a* that maximizes the expression $Q_t(a)$.

* **Exploration Benefits:** Choosing non-greedy actions (exploration) allows you to improve your estimates of the values of less-explored actions, which can lead to the discovery of better actions and greater cumulative reward in the long run.

* **Exploration vs. Exploitation Trade-off:** The optimal balance between exploration and exploitation depends on factors like:
    1. **Action value estimates:** How confident you are in your current value estimates.
    2. **Uncertainties:** How much uncertainty there is about the true values of actions.
    3. **Number of remaining steps:** How many more action selections you have left.

* **Balancing Exploration and Exploitation:**  Various strategies exist to balance these two competing objectives.

### Action-Value Methods

* **Core Idea:** Methods that estimate the values of actions and use these estimates to make action selection decisions.

* **Sample-Average Method for Estimating Action Values:** A simple method to estimate the value of an action *a*. It averages the rewards received after taking action *a* in the past:

    $$
    Q_t(a) \doteq \frac{\text{sum of rewards when } a \text{ taken prior to } t}{\text{number of times } a \text{ taken prior to } t} = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}}
    $$

    Where:
    * $\mathbb{1}_{A_i=a}$ is an indicator function that is 1 if action $A_i$ was equal to $a$, and 0 otherwise.

* **Law of Large Numbers:** As the number of times action *a* is taken approaches infinity (denominator goes to infinity), the estimated value $Q_t(a)$ converges to the true value $q_*(a)$.

### $\epsilon$-Greedy Action Selection

* **$\epsilon$-Greedy Strategy:** A simple way to balance exploration and exploitation.
    * With a small probability $\epsilon$ (epsilon), select a random action from among all possible actions with equal probability (exploration).
    * Otherwise (with probability $1 - \epsilon$), select the greedy action (exploitation) - the action with the highest estimated value.

* **Near-Greedy:** $\epsilon$-greedy is considered a "near-greedy" approach because it mostly acts greedily but occasionally explores.

* **Benefit of $\epsilon$-Greedy (Convergence in the Limit):**  In the long run (as the number of steps increases), $\epsilon$-greedy ensures that every action will be sampled infinitely many times (if $\epsilon > 0$). This guarantees that all estimated action values $Q_t(a)$ will eventually converge to their respective true values $q_*(a)$. This convergence property is important for finding optimal policies in the long run.
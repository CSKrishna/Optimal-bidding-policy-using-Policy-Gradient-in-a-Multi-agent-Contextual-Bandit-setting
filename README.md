# Optimal-bidding-policy-using-Policy-Gradient-in-a-Multi-agent-Contextual-Bandit-setting
## Problem Definition
We consider the multi-agent competitive contextual bandit setting where each agent must learn a policy that maps from the state to action space such that each agent’s expected reward is maximized given the other agents’ bidding policy. We also assume that none of the agents have any visibility to the actions of the other agents.

Concretely, we consider a 2 agent setting (this can be easily extended to a multi-agent setting) where each agent submits bids comprising price-quantity tuples for a product in the day-ahead market.  The market regulator stacks up the bids of all the agents to construct the supply curve. The demand curve is similarly constituted and the market clearing price and quantities are obtained. Agents which bid at prices less than the market clearing prices get their quantities sold in the day ahead market and earn income. The agent’s profit is computed by deducting the cost of production which is assumed to be the product of the marginal cost of production and the quantity sold.

In determining an optimal policy so as to maximize its expected profit, the agent must take into account the value of the state that influences both its cost of production and the demand curve. For instance, oil prices could determine the cost of production while weather conditions could influence the shape and position of the demand curve for the product. 

In a single agent setting, this would constitute a contextual bandit problem where the agent simply needs to learn a mapping from the state space to the action space in a stationary environment. In a multi-agent setting, apart from state, the agent’s profit will also be influenced by the bids submitted by competing agents. These bids are not visible and would be part of the agent’s environment. Since agents will continue to explore and tweak their policies, each agent’s environment will be non-stationary. In the multi-agent setting, therefore, the agents must explore policies and arrive at a Nash Equilibrium at which point none of the agent has an incentive to change its policy given the other agent’s bidding policy.

Given that the agent’s profit is a non-linear function of the state (and the actions of the other agent), we use a neural network to learn a parameterized policy for each agent. We then use a policy gradient algorithm, [REINFORCE] (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), to jointly optimize each agent’s policy network until we attain a Nash equilibrium. 

## Algorithm for multi-agent contextual bandits
The base algorithm of the basic multi-agent bidding problem is presented below:
* Randomly initialize all the agents’ policy networks
* Repeat until convergence
    * Generate a batch of states and for each state, sample each agent’s action based on the current state of its policy network
    * Submit the state-action tuples to the regulator to get the corresponding batch of rewards for each agent
    * Use REINFORCE to update each agent’s policy network parameters

As noted in [1] (https://www.ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf), we find that modelling the policy using the Beta distribution instead of the Gaussian improves training. This is especially useful in settings where the action space is constrained within a range, say the range of positive real numbers, and using the Gaussian creates training inefficiencies. Finally, the variance of the reward can be mitigated by partitioning the state space and training separate agent models for each sub-space. This idea has been explored in the full blown MDP setting as well in [2] (https://arxiv.org/abs/1711.09874).

## Technology Stack
We implement the model using Tensorflow. We have taken helper code for implementing REINFORCE with a baseline from [CS294 Assignment 2] (http://rll.berkeley.edu/deeprlcourse/f17docs/hw2_final.pdf).


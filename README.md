## Reward Shaping with Dynamic Guidance to Accelerate Learning for Multi-Agent Systems

### Overview

This repository is derived from [DennyBritz's Reinforcement Learning Library](https://github.com/dennybritz/reinforcement-learning)

multi_gridworld.py contains the code for the Dynamic Guidance with Reward Shaping using the sum of Euclidean Distances from the goals as the guidance function. A policy for it is trained using SARSA in SARSA-MultiGridWorld.ipynb and takes the linear combination of the guidance and the real reward function to train the policy. It slowly decays the weight given to the guidance shaping over the 500 episodes. 

![With Reward Shaping on modified GridWorld](https://github.com/Tanay1998/Dynamic-Reward-Shaping/plots/multi-with.png "With Reward Shaping on modified GridWorld")


curriculum_multi_gridworld.py contains the code for the Dynamic Guidance with Curriculum Training which uses three different rewards, forming the training curriculum of the agent: the sum of Euclidean Distances from the goals, the sum of the inverse Euclidean Distances from other agents, and the actual reward on getting to the goals. A policy for it is trained using SARSA in SARSA-MultiGridWorld-Curriculum.ipynb and takes the linear combination of the rewards, while slowly decaying the weights given to curriculum rewards over the 500 episodes. 

![With Curriculum Training](https://github.com/Tanay1998/Dynamic-Reward-Shaping/plots/curriculum.png "With Curriculum Training")

Contribution: 
- Tanay: Designed the environment, ideated and implemented guided reward shaping, and did data analysis. 
- Jordan: Tested various environments and algorithms and ideated and implemented curriculum based reward shaping

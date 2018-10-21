# Navigation-p1
Udacity Reinforcment learning online course project one solution

### Project description

For this project, the task is to train an agent to navigate in a large, square world, while collecting yellow bananas, and avoiding blue bananas. A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. Thus, the goal is to collect as many yellow bananas as possible while avoiding blue bananas.

- **State space** is `37` dimensional and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 

- **Action space** is `4` dimentional. Four discrete actions correspond to:
  - `0` - move forward
  - `1` - move backward
  - `2` - move left
  - `3` - move right

- **Solution criteria**: the environment is considered as solved when the agent gets an average score of **+13 over 100 consecutive episodes**.

### Environment visualisation (untrained agent)
![Navigation-p1](https://github.com/betadecay1993/Navigation-p1/blob/master/results/banana_gatherer_untrained.gif)

### Introduction
The described above task was solved using value-based reinforcement learning algorithm Deep Q-Network (DQN)
The idea behind an algorithm:
Using neural network, approximate value of making an action from some state ( q(s,a) ).



### Hyperparameters

### Performance of a trained agent
![Navigation-p1](https://github.com/betadecay1993/Navigation-p1/blob/master/results/banana_gatherer.gif)
![Navigation-p1](https://github.com/betadecay1993/Navigation-p1/blob/master/results/scores.png)
### Suggested further improvements

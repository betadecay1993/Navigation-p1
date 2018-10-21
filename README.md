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
![environment](https://github.com/betadecay1993/Navigation-p1/blob/master/results/banana_gatherer_untrained.gif)

### Environment setup
Download pre-built Unity Environment:
  - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
  - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
  - [Win x32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
  - [Win x64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

### Theoretical background
The described above task was solved using value-based reinforcement learning algorithm Deep Q-Network (**DQN**)
The idea behind an algorithm:
Using neural network, iteratively approximate value **q(s,a)** of taking an action from some state.

The update rule for weights of neural network is:

**![equation](https://latex.codecogs.com/gif.latex?\Delta&space;\omega&space;=&space;\alpha&space;(R&space;&plus;&space;\gamma&space;\max_a&space;q(S',a,\omega^-)&space;-&space;q(S,A,\omega))\nabla_w&space;q(S,A,\omega))**

where ![equation](https://latex.codecogs.com/gif.latex?\omega^-) − are the weights of a separate target network that are not changed during the learning step, and **(S, A, R, S')** is an experience tuple (State, Action, Reward, Next State).

Code implementation:
```python
...
states, actions, rewards, next_states, dones, TD_errors = experiences
Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))  # if done == True: second term is equal to 0
Q_expected = self.qnetwork_local(states).gather(1, actions)  # gets one value from each row in Q function for
TD_errors = (Q_targets-Q_expected).abs()
...
```

To make the learning more stable, the idea of experience replay was used. Instead of online learning, an agent collects the experiences into internal buffer and then from time to time learns from some radomly sampled experiences.

____________________________________________________________________________________________________________________
 - To further improve learning stability, the **Double DQN** algorithm was imploemented. Instead of originaly described update rule, one utilizes the following equation:
**![equation](https://latex.codecogs.com/gif.latex?\Delta&space;\omega&space;=&space;\alpha&space;(R&space;&plus;&space;\gamma&space;q(S',arg&space;\&space;\text{max}_a&space;q(S',A,\omega),\omega^-)&space;-&space;q(S,A,\omega))\nabla_w&space;q(S,A,\omega))**
Code implementation:
```python
...
states, actions, rewards, next_states, dones, TD_errors = experiences
next_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))  # if done == True: second term is equal to 0
Q_expected = self.qnetwork_local(states).gather(1, actions)  # gets one value from each row in Q function for
TD_errors = (Q_targets-Q_expected).abs()
...
```
[More about Double DQN](https://arxiv.org/abs/1509.06461)

____________________________________________________________________________________________________________________

- The next idea used in the implementation was **Dueling DQN**. Instead of evaluating **q(s,a)** directly, one may evaluate state value **v(s)** and then evaluate an advantage fucntion **A(s,a)**, so that **q(s,a) = v(s) + A(s,a)**.
Code implementation:
```python
def forward(self, state):  # get action values from the neural network given a state
...
return y+(x-x.mean()) # y - value function of a state, x - vector of advantage values given an action and a state
```
According this article, it increases the performance of an agent:
[More about Duelling DQN](https://arxiv.org/abs/1511.06581)
____________________________________________________________________________________________________________________

 - Theoretically, training may also be accelerated by employing prioritised experience replay strategy: instead of uniform sampling from the experience buffer, one may sample the experiences which were more "surprising" (defined by TD error) with higher probability.
In practice, this algorithm requires an efficient data structure to implement sampling, updating a priority, and adding of a new experience for log(n) operations. 
This data structure is described here: 
[SumTree](https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/)
[More on prioritised experience replay](https://arxiv.org/abs/1511.05952)

____________________________________________________________________________________________________________________
- In addition, gradual decrease of a learning rate and an exploration factor was employed.

### Hyperparameters

The following hyperparameters were used:

```python
num_episodes = 2000       # number of episodes
buffer_size = int(2**18)  # replay buffer size
batch_size = 100          # minibatch size
gamma = 0.99              # discount factor
tau = 1e-3                # for soft update of target parameters
lr = 5e-4                 # learning rate
epsilon_init = 0.05       # initial epsilon greedy exploration factor
epsilon_final = 0.0005    # epsilon expolration
epsilon_decay = np.exp((np.log(epsilon_final/epsilon_init)/(0.9*num_episodes))) #epsilon decay factor
# (computed so that epsiloon decays to 'epsilon final' during 90% of num_episodes
a = 0.0  # parameter which identifies to what extent to use prioritised replay 
# (1.0 is fully use prioritised experience replay)
b = 0.0  # parameter which controls update weights
b_step = (1.0 - b)/(0.9*num_episodes) # step of b after each episode
update_every = 3  # how often to update the network
seed = random.randint(0,100)
max_t = 1000 # maximum length of sequence of states till episode is finished
```

The network architecture:

| Layer   | (in, out)          | Activation|
|---------|--------------------|-----------|
| Layer 1 | (`state_size`, 64) | `relu`    |
| Layer 2 | (64, 128)          | `relu`    |
| Layer 3 | (128, 64)          | `relu`    |
| Layer 4.1 | (64, `action_size`)| -         |
| Layer 4.2 | (64, 1)            | -         |

Output: Layer 5 + Layer 4


### Code organisation
The implementation is stored in the folder 'code', which includes:
- `navigation.py`- the main file, to run the training of reinforcment learning agent. It includes hyperparameters and fucntion 'interact_and_train' which makes created Agent and Environment to interact.
- `Agent.py` - contains the implementation of an agent. It also includes parameter for the neural network to estimate q-function.
- `ReplayBuffer.py` - implementation of internal buffer to sample the experiences from it.
- `QNetwork.py` - an ANN to evaluate q-function.
- `SumTree.py` - a data structure for storing, updating and sampling experiences, which is utilised in ReplayBuffer.py.
- `plotter.py` - generates plot of acquired scores during the training.
- `run_one_time.py` - Initialises an agent with specified state dictionary and architecture and run visualisation of the agent's performance.


### Performance of a trained agent
![performance](https://github.com/betadecay1993/Navigation-p1/blob/master/results/banana_gatherer.gif)

![scores](https://github.com/betadecay1993/Navigation-p1/blob/master/results/scores.png)

Weights and parameters of used trained network can be found here:
[QNetwork-weights](https://github.com/betadecay1993/Navigation-p1/blob/master/results/Banana_17.1.pth)

### Additional comments
While implemeting this algorithm I've noticed a couple of interesting phenomena:
- Initial epsilon may be successfully set almsot to 0 and agent will still lear to gather above 13 bananas. This may be the result of aproximation of q values using neural network, and it's update already may introduce sufficient amount of noice needed for maintaining exploration.
- Prioritised experience replay doesn't improve the performance of an algorithm. In fact, the agent does even worse with it (Maybe there is a bug in the code? You may help to isolate it!)

### Suggested further improvements
There a many possible venues of boosting the algorithm's performance:
- Reward Shaping (give more reward for collecting 20th banana than 1s)
- Using LSTM (or other ways of including information about previous states). In current implementation an agent don't remember locations of bananas if he turns from them. This results in learning some kind-of greedy policy.
- Using noisy nets to replaceclumsy epsilon-greedy exploration, eliminating a couple of hyperparameters.
- Make agent learn the model of an environment for the agent to create an extra volume of fictive experiences.
- Combine policy-basedt methods with value based methods (Actor-critic algorithm) 

from unityagents import UnityEnvironment
import numpy as np
import random

# load necessary files

from Agent import Agent
env = UnityEnvironment(file_name="/home/pavel/Documents/0Study/Reinforcement_Learning/"
                                 "deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
for i in range(np.random.randint(0, 100)):
    env_info = env.reset(train_mode=True)[brain_name]
# number of actions
action_size = brain.vector_action_space_size
# examine the state space
state = env_info.vector_observations[0]
state_size = len(state)
buffer_size = int(2 ** 20)  # replay buffer size
batch_size = 100  # minibatch size
gamma = 0.99  # discount factor
tau = 1e-3  # for soft update of target parameters
lr = 5e-4  # learning rate
epsilon_init = 0.0001
epsilon_final = 0.0001
epsilon_decay = 0
a = 0.0
b = 0.0
b_step = 0
update_every = 0  # how often to update the network
seed = random.randint(0, 100)
max_t = 1000
print_every = 1

RL_Agent = Agent(state_size, action_size, buffer_size, batch_size, gamma,
                 tau, lr, epsilon_init, epsilon_final, epsilon_decay,
                 a, b, b_step, update_every, seed)

RL_Agent.qnetwork_local.load('Banana_16.0.pth')
# # reset the environment
# env_info = env.reset(train_mode=True)[brain_name]
RL_Agent.epsilon = 0.0


def run_one_episode_in_test_mode(Agent, Env, brain_name):
    env_info = Env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    action = Agent.choose_action(state, 'train')
    score = 0
    for t in range(max_t):
        env_info = Env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        state = next_state
        if done:  # exit loop if episode finished
            return score
        action = Agent.choose_action(state, 'train')
    return score

score = run_one_episode_in_test_mode(RL_Agent, env, brain_name)
print("Score achieved : {}".format(score))
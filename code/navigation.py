from unityagents import UnityEnvironment
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from Agent import Agent
from scipy.signal import savgol_filter
import pickle


def interact_and_train(Agent, Env, brain_name, num_episodes, max_t, save_to):
    '''
    :param Agent: Agant to train
    :param Env: Environment to train an Agent on
    :param brain_name: the parameter needed to address the environment Env to get rewards and next states
    :param num_episodes: Number of episodes to train an Agent
    :param max_t: maximum length of sequency of states till done, in one episode
    :param save_to: save the state dict and architecture parameters if the the Agent reached some specified score
    :return: scores for each episode
    '''
    env_info = Env.reset(train_mode=True)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]
    action = Agent.choose_action(state, mode='train')
    scores = []
    scores_window = deque(maxlen=100)
    best_score = 0
    for e in range(num_episodes):
        env_info = Env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        score = 0
        for t in range(max_t):
            env_info = Env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            # get the next state
            score += reward  # get the reward
            # see if episode has finished
            Agent.memorize_experience(state, action, reward, next_state, done)
            Agent.learn_from_past_experiences()
            state = next_state  # roll over the state to next time step
            if done:  # exit loop if episode finished
                break
            action = Agent.choose_action(state, mode='train')  # get new action form the next state

        Agent.update_epsilon()
        Agent.update_b()
        if e % 200 == 0:
            Agent.update_lr()

        scores.append(score)
        scores_window.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}\t Current Score: {}'.format(e + 1, np.mean(scores_window), score), end="")
        if (e + 1) % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e + 1, np.mean(scores_window)))
        if np.mean(scores_window) >= 15.5 and (np.mean(scores_window) > best_score):
            best_score = np.mean(scores_window)
            print('\nEnvironment achieved average score {:.2f} in {:d} episodes!\t '.format(np.mean(scores_window),(e + 1)))
            file_name = str(save_to) + '_' + str(np.round(np.mean(scores_window), 1)) + str('.pth')
            Agent.qnetwork_local.save(file_name)
    return scores

#HYPER PARAMS
num_episodes = 2000
buffer_size = int(2**18)  # replay buffer size
batch_size = 100          # minibatch size
gamma = 0.99              # discount factor
tau = 1e-3                # for soft update of target parameters
lr = 5e-4                 # learning rate
epsilon_init = 0.05        # initial epsilon greedy exploration factor
epsilon_final = 0.0005    # epsilon expolration
epsilon_decay = np.exp((np.log(epsilon_final/epsilon_init)/(0.9*num_episodes))) #epsilon decay factor
# (computed so that epsiloon decays to 'epsilon final' during 90% of num_episodes
a = 0.0  # parameter which identifies to what extent to use prioritised replay (1.0 is fully use prioritised experience replay)
b = 0.0  # parameter which controls update weights
b_step = (1.0 - b)/(0.9*num_episodes) # step of b after each episode
update_every = 3  # how often to update the network
seed = random.randint(0,100)
max_t = 1000 # maximum length of sequency of states till done, in one episode

#ENVIRONMENT
env = UnityEnvironment(file_name="/home/pavel/Documents/0Study/Reinforcement_Learning/"
                                 "deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

#AGENT
RL_Agent = Agent(state_size, action_size, buffer_size, batch_size, gamma,
                 tau, lr, epsilon_init, epsilon_final, epsilon_decay,
                 a, b, b_step, update_every, seed)

#TRAIN AGENT AND GET SCORES
scores = interact_and_train(RL_Agent, env, brain_name, num_episodes, max_t, save_to = './Banana')


#SAVE SOCRES IN PICKLE
pickle.dump(scores, open('banana_scores','wb+'))
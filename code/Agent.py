import numpy as np
import random
from collections import OrderedDict

from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
import torch
import torch.optim as optim


def weighted_MSE(inp, target, weights, beta):
    '''Custom made weighted Mean Squared Error loss'''
    weights = weights.pow(-beta)
    weights = weights*(weights.shape[0]/weights.sum())
    return torch.mean(weights.detach().double() * (inp.double() - target.double()) ** 2)


class Agent():
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, tau, lr, epsilon_init, epsilon_final,
                 epsilon_decay, a, b, b_step, update_every, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.epsilon = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.a = a
        self.b = b
        self.b_step = b_step
        random.seed(seed)
        self.update_every = update_every
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Q-Network
        arch_params = OrderedDict(
            {'state_and_action_sizes': (state_size, action_size), 'Linear_2': 64, 'ReLU_2': None, 'Linear_3': 128,
             'ReLU_3': None, 'Linear_4': 64, 'ReLU_4': None, 'Linear_5': action_size})
        self.qnetwork_local = QNetwork(seed, arch_params).to(device)  # decision_maker
        self.qnetwork_target = QNetwork(seed, arch_params).to(device)  # fixed
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, seed)
        self.t_step = 0
        self.average_TD_error = 1.0

    def memorize_experience(self, state, action, reward, next_state, done):
        if self.a != 0:
            TD_error = 10 * (self.average_TD_error ** self.a)
        else:
            TD_error = 1.0

        self.memory.add(state, action, reward, next_state, done, TD_error)
        self.t_step = (self.t_step + 1)

    def learn_from_past_experiences(self):
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                idxs, experiences = self.memory.sample()  # self.update_every
                TD_errors_new = self.update_QNetwork(experiences)
                if self.a != 0:
                    self.memory.update_multiple(idxs, TD_errors_new)  #

    def choose_action(self, state, mode='test'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if mode == 'test':
            eps = 0
        elif mode == 'train':
            eps = self.epsilon
        else:
            raise KeywordError()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update_QNetwork(self, experiences):
        states, actions, rewards, next_states, dones, TD_errors = experiences
        next_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))  # if done == True: second term is equal to 0
        Q_expected = self.qnetwork_local(states).gather(1, actions)  # gets one value from each row in Q function for
        self.optimizer.zero_grad()
        loss = weighted_MSE(Q_expected, Q_targets, TD_errors, self.b)
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        TD_errors_new = ((Q_expected - Q_targets).detach().abs()).pow(self.a) + 0.01
        self.average_TD_error += 0.1 * (TD_errors_new.mean().numpy() - self.average_TD_error)
        return TD_errors_new

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_final)

    def update_b(self):
        self.b = min(self.b + self.b_step, 1.0)

    def update_lr(self):
        self.lr /= 1.5

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

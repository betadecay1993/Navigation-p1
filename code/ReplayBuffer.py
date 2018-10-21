from collections import namedtuple
import numpy as np
from SumTree import SumTree
import random
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        '''

        :param buffer_size: size of the replay buffer
        :param batch_size: size of the batch to do the update of the neural network
        :param seed: seed
        '''
        self.memory = SumTree(capacity=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", 'TD_error'])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, TD_error):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, TD_error)
        # self.memory.append(e)
        self.memory.add(p=e.TD_error, data=e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        idxs, experiences = self.memory.sample(self.batch_size)  # self.memory.sample(self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        TD_errors = torch.from_numpy(np.vstack([e.TD_error for e in experiences if e is not None])).float().to(device)

        if len(idxs) != TD_errors.size()[0]:
            raise IndexError(len(idxs), TD_errors.size()[0])

        return idxs, (states, actions, rewards, next_states, dones, TD_errors)

    def update_multiple(self, idxs, ps):
        self.memory.update_multiple(idxs, ps)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


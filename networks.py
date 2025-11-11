import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    data_names = ('s', 'a', 'r', 's_', 'done')
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, data):
        # data is a tuple (s, a, r, s_, done)
        self.buffer.append(data)
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_, done = zip(*batch)
        return np.array(s), np.array(a), np.array(r), np.array(s_), np.array(done)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, X)
        self.fc2 = nn.Linear(X, Y)
        self.fc3 = nn.Linear(Y, action_dim)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, X)
        self.fc2 = nn.Linear(X, Y)
        self.fc3 = nn.Linear(Y, 1)

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from config import ACTOR_LR, CRITIC_LR, TAU, BUFFER_SIZE, BATCH_SIZE, OPTIMIZER, HIDDEN_LAYERS


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
        self.fc1 = nn.Linear(state_dim, HIDDEN_LAYERS[0])
        self.fc2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
        self.fc3 = nn.Linear(HIDDEN_LAYERS[1], action_dim)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, HIDDEN_LAYERS[0])
        self.fc2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
        self.fc3 = nn.Linear(HIDDEN_LAYERS[1], 1)

class DDPG_Agent:
    def __init__(self, state_dim, action_dim, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR):
        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        if OPTIMIZER == 'Adam':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        self.update_targets(tau=1.0)

        def update_targets(self):
            #soft update function
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

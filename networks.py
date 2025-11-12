import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from config import *


class ReplayBuffer:
    #done and r same as DQN
    #changes for s, a, s'
    data_names = ('s', 'a', 'r', 's_', 'done')
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def push(self, data):
        # data is a tuple (s, a, r, s_, done)
        self.buffer.append(data)
    
    def sample(self):
        #adjust sampling for DDPG
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s_, done = zip(*batch)
        # return np.array(s), np.array(a), np.array(r), np.array(s_), np.array(done)
        return (np.array(s, dtype=np.float32), #4d-state ([position, velocity, angle, angular velocity])
                np.array(a, dtype=np.float32), #1d-action (force applied to cart)
                np.array(r, dtype=np.float32).reshape(-1, 1), #reward (scalar)
                np.array(s_, dtype=np.float32), #next state (4d)
                np.array(done, dtype=np.float32).reshape(-1, 1)) #true or false, epsiode done (scalar)

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        #input state, output action
        self.fc1 = nn.Linear(state_dim, HIDDEN_LAYERS[0])
        # self.fc2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
        self.fc2 = nn.Linear(HIDDEN_LAYERS[0], action_dim)

    def forward(self, state):
        #relu activation for hidden layers (faster for non-saturated gradients, cheaper computationally)
        x = nn.ReLU()(self.fc1(state))
        # x = nn.ReLU()(self.fc2(x))
        return nn.Tanh()(self.fc2(x)) #action range of cartpole is [-1, 1], so tanh activation


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        #input state-action pair, output Q-value
        self.fc1 = nn.Linear(state_dim + action_dim, HIDDEN_LAYERS[0])
        # self.fc2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
        self.fc2 = nn.Linear(HIDDEN_LAYERS[0], 1)

    def forward(self, state, action):
        #relu adds non-linearity, no tanh on output bc Q-value has no range restriction
        x = torch.cat([state, action], dim=-1)
        x = nn.ReLU()(self.fc1(x))
        # x = nn.ReLU()(self.fc2(x))
        return self.fc2(x) #no tanh bc its for Q-value

class DDPG_Agent:
    def __init__(self, state_dim, action_dim, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, tau=TAU):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
    
        #for stats
        self.final_actor_loss = 0
        self.final_critic_loss = 0
        self.final_q_value = 0

        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        if OPTIMIZER == 'Adam':
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        self.replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)

        if NOISE_TYPE == 'Gaussian':
            self.noise_std = EXPL_NOISE #standard deviation of Gaussian noise for exploration

    def update_noise_std(self, episode):
        noise_start = 0.2
        noise_end = 0.05
        decay_episodes = 300
        self.noise_std = max(noise_end, noise_start - (noise_start - noise_end) * (episode / decay_episodes))


    #given env state, return action 
    #noise=True adds exploration noise, not for evaluation
    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0) #add batch dimension
        action = self.actor(state).detach().cpu().numpy()[0]
        if noise:
            action += np.random.normal(0, self.noise_std, size=self.action_dim)
        return np.clip(action, -1, 1) #return valid action within range of DM_control cartpole env
    
    def train(self, step):
        #1 - check if enough samples in replay buffer
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        #2 - sample buffer
        s, a, r, s_, done = self.replay_buffer.sample()

        #3 - convert to tensors for nn processign
        state = torch.FloatTensor(s)
        action = torch.FloatTensor(a)
        reward = torch.FloatTensor(r)#.unsqueeze(1)
        next_state = torch.FloatTensor(s_)
        done = torch.FloatTensor(done)#.unsqueeze(1)

        #4 - compute target Q-values for critic
        with torch.no_grad(): #save computation for unnecessary gradient calcs
            next_action = self.target_actor(next_state)
            target_Q = self.target_critic(next_state, next_action)
            # target_Q = torch.clamp(target_Q, min=-50, max=50) #avoid extreme Q-values
            target_Q = reward + (1 - done) * self.gamma * target_Q #r_t + gamma * Q(s', a')
            

        #5 - compute current Q-values from critic
        current_Q = self.critic(state, action)

        #6 - compute critic loss (MSE)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.final_critic_loss = critic_loss.item()
        # print("Critic loss:", critic_loss.item())

        #7 - update critic network - standard backprop
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        if (step % A_UPDATE) == 0:
            #8 - compute actor loss (policy gradient)
            self.actor_optimizer.zero_grad()
            actor_chosen_action = self.actor(state)
            actor_loss = -self.critic(state, actor_chosen_action).mean() #maximize Q
            self.final_actor_loss = actor_loss.item()
            # print("Actor loss:", actor_loss.item())

            #9 - update actor network
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            #10 - soft update target networks
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        
        

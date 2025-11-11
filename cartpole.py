import os, sys

from dm_control import suite
import numpy as np 
from dm_control import viewer
import matplotlib.pyplot as plt
import numpy as np
from networks import Actor, Critic, ReplayBuffer, DDPG_Agent
from config import *

seed = sys.argv[1] if len(sys.argv > 0) else 0
print(f"Using Seed: {seed}")

#set seed for reproducibility
np.random.seed(seed)

#Start environment/episode
env = suite.load(domain_name="cartpole", task_name="balance")
agent = DDPG_Agent(state_dim=4, action_dim=1)

time_step = env.reset()
done = False
steps = 0
while not done:
    action = np.random.uniform(-1, 1, size=env.action_spec().shape)
    time_step = env.step(action)
    
    obs = time_step.observation
    cart_position = obs['position'][0]
    pole_angle = obs['position'][1]
    # obs['position'][2]
    cart_vel = obs['velocity'][0]
    pole_ang_vel = obs['velocity'][1]
    
    # Consider it "balanced" if angle is small
    if abs(pole_angle) < 0.05:
        balance = True
    else:
        balance = False
    
    print(f"Step {steps}: angle={pole_angle:.3f} rad, balanced={balance}, done={time_step.last()}")
    
    done = time_step.last()
    steps += 1





# action_spec = env.action_spec()
# print("Action spec:", action_spec) 
# time_step = env.reset()
# print("Initial time step:", time_step)
# while not time_step.last():
#     action = np.random.uniform(action_spec.minimum,
#                                action_spec.maximum,
#                                size=action_spec.shape)
#     # print(action.task_name)
#     time_step = env.step(action)
#     print(time_step.reward, time_step.discount, time_step.observation)


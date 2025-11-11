import os, sys

from dm_control import suite
import numpy as np 
from dm_control import viewer
import matplotlib.pyplot as plt
import math
import random
import torch
from time import sleep
from networks import Actor, Critic, ReplayBuffer, DDPG_Agent
from config import *

seed = int(sys.argv[1]) if len(sys.argv) > 0 else 0
print(f"Using Seed: {seed}")

#set seed for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

#Start environment/episode
env = suite.load(domain_name="cartpole", task_name="balance")
agent = DDPG_Agent(state_dim=4, action_dim=1)



#stats for logging and plotting
reward_history = []
episode_lengths = []
loss_history = []
reward_history_eval = []

def get_state(time_step):
    obs = time_step.observation
    cart_position = obs['position'][0]
    pole_angle = obs['position'][1]
    # obs['position'][2]
    cart_vel = obs['velocity'][0]
    pole_ang_vel = obs['velocity'][1]
    return np.array([cart_position, pole_angle, cart_vel, pole_ang_vel])


for episode in range(MAX_EPISODES):
    time_step = env.reset()
    observation = time_step.observation
    state = get_state(time_step) #0 - cart pos, 1 - pole angle, 2 - cart vel, 3 - pole ang vel
    done = False
    steps = 0
    episode_reward = 0
    while not done and steps < MAX_STEPS:
        #1 - select action
        action = agent.select_action(state) # actor selects action based on curr state

        #2 - step environment
        time_step = env.step(action)
        next_state = get_state(time_step) 
        reward = time_step.reward #env defined reward 
        episode_reward += reward
        done = time_step.last() #env defined done signal
        pole_angle_rad = next_state[1]
        pole_angle_deg = pole_angle_rad * 180 / math.pi


        #3 - store transition in replay buffer
        agent.replay_buffer.push((state, action, reward, next_state, done))

        #4 - train agent 
        #if not enough samples for trainning, train randomly until enough samples
        if len(agent.replay_buffer) >= BATCH_SIZE:
            agent.train()

        #5 - updates
        state = next_state
        steps += 1
        # print(f"Episode {episode+1}: step_reward={reward:.2f}, Steps={steps}, Pole Angle = {pole_angle_rad:.2f}, Pole Angle={pole_angle_deg:.2f} deg")

    print(f"Episode {episode+1} finished with reward {episode_reward} in {steps} steps.")
    print(f"actor loss {agent.final_actor_loss}, critic loss {agent.final_critic_loss}")
    sleep(2)
    
    # Log episode reward
    reward_history.append(episode_reward)
    # Print running average every 10 episodes
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(reward_history[-10:])
        print(f"Average reward (last 10 episodes): {avg_reward:.2f}")

    





# action_spec = env.action_spec()
# print("Action spec:", action_spec) 
# time_step = env.reset()
# print("Initial time step:", time_step)
# Plot episode rewards after training
if len(reward_history) > 0:
    plt.figure(figsize=(10,5))
    plt.plot(reward_history, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Reward Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('episode_rewards.png')
# while not time_step.last():
#     action = np.random.uniform(action_spec.minimum,
#                                action_spec.maximum,
#                                size=action_spec.shape)
#     # print(action.task_name)
#     time_step = env.step(action)
#     print(time_step.reward, time_step.discount, time_step.observation)


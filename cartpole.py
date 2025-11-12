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
            agent.train(steps)
        else: 
            print("Filling Replay Buffer...")
        #5 - updates
        state = next_state
        steps += 1
        agent.update_noise_std(episode)

        # print(f"Episode {episode+1}: step_reward={reward:.2f}, Steps={steps}, Pole Angle = {pole_angle_rad:.2f}, Pole Angle={pole_angle_deg:.2f} deg")

    with torch.no_grad():
        # sample a batch of transitions to compute Q-value stats
        if len(agent.replay_buffer) >= BATCH_SIZE:
            s, a, r, s_, done = agent.replay_buffer.sample()

            #3 - convert to tensors for nn processign
            state = torch.FloatTensor(s)
            action = torch.FloatTensor(a)
            reward = torch.FloatTensor(r)#.unsqueeze(1)
            next_state = torch.FloatTensor(s_)
            done = torch.FloatTensor(done)#.unsqueeze(1)
            q_values = agent.critic(state, action)
            q_min = q_values.min().item()
            q_max = q_values.max().item()
            q_mean = q_values.mean().item()
        else:
            q_min = q_max = q_mean = float('nan')

    print(f"Episode {episode+1} finished: Reward={episode_reward:.2f}, "
          f"Actor loss={agent.final_actor_loss:.3f}, Critic loss={agent.final_critic_loss:.3f}, "
          f"Q_min={q_min:.2f}, Q_max={q_max:.2f}, Q_mean={q_mean:.2f}, "
          f"Noise_std={agent.noise_std:.3f}")
    
    reward_history.append(episode_reward)
    episode_lengths.append(steps)
    loss_history.append((agent.final_actor_loss, agent.final_critic_loss))
    
    sleep(0.5)
    





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
    plot_name = f"by2softupdate_{MAX_EPISODES}_episode_rewards_seed_{seed}_actorlr_{ACTOR_LR}_criticlr_{CRITIC_LR}_tau_{TAU}_batchsize_{BATCH_SIZE}_buffersize_{BUFFER_SIZE}.png"
    plt.savefig(plot_name)
# while not time_step.last():
#     action = np.random.uniform(action_spec.minimum,
#                                action_spec.maximum,
#                                size=action_spec.shape)
#     # print(action.task_name)
#     time_step = env.step(action)
#     print(time_step.reward, time_step.discount, time_step.observation)


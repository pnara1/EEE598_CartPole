import os
import sys
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from dm_control import suite
from networks import TD3_Agent      # same agent class you used for training
from config import *                # REWARD_SCALE, MAX_STEPS, etc.


# import numpy as np
# print(np.load("logs/seed0/train_rewards.npy"))
# exit()

TRAINING_SEEDS = [0, 1, 2]   # seeds used during training
EVAL_SEED = 10               # fixed evaluation seed
EVAL_EPISODES = 10           # how many episodes per policy to eval
LOG_ROOT = "logs"            # where seed folders live (logs/seed0, logs/seed1, ...)


def get_state(time_step):
    obs = time_step.observation
    cart_position = obs['position'][0]
    pole_angle = obs['position'][1]
    # obs['position'][2]
    cart_vel = obs['velocity'][0]
    pole_ang_vel = obs['velocity'][1]
    return np.array([cart_position, pole_angle, cart_vel, pole_ang_vel])


def evaluate_policy(env, agent, episodes=EVAL_EPISODES):
    """Run deterministic evaluation (no exploration noise)."""
    returns = []

    for ep in range(episodes):
        ts = env.reset()
        s = get_state(ts)
        done = False
        steps = 0
        ep_ret = 0.0

        while not done and steps < MAX_STEPS:
            # IMPORTANT: noise=False → no exploration noise
            a = agent.select_action(s, noise=False)
            ts = env.step(a)
            s = get_state(ts)
            ep_ret += ts.reward * REWARD_SCALE
            done = ts.last()
            steps += 1

        returns.append(ep_ret)
        print(f"[Eval] Episode {ep+1}: return = {ep_ret:.2f}")

    returns = np.array(returns, dtype=np.float32)
    print(f"[Eval] Mean return = {returns.mean():.2f} ± {returns.std():.2f}")
    return returns


def main():
    # Set eval RNGs (for env randomness, etc.)
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    random.seed(EVAL_SEED)

    all_eval_returns = []
    all_train_curves = []

    for train_seed in TRAINING_SEEDS:
        print(f"\n=== Evaluating policy trained with seed {train_seed} ===")

        seed_dir = os.path.join(LOG_ROOT, f"seed{train_seed}")
        actor_path = os.path.join(seed_dir, "actor.pth")
        train_rewards_path = os.path.join(seed_dir, "train_rewards.npy")
        eval_rewards_path = os.path.join(seed_dir, "eval_rewards.npy")

        if not os.path.exists(actor_path):
            print(f"  [WARN] No actor.pth found for seed {train_seed} at {actor_path}, skipping.")
            continue

        # --- Create eval env with fixed seed ---
        env = suite.load(
            domain_name="cartpole",
            task_name="balance",
            task_kwargs={'random': EVAL_SEED}
        )

        # --- Recreate agent and load weights ---
        agent = TD3_Agent(state_dim=4, action_dim=1)
        state_dict = torch.load(actor_path, map_location=torch.device("cpu"))
        agent.actor.load_state_dict(state_dict)
        agent.actor.eval()

        # --- Run evaluation ---
        eval_returns = evaluate_policy(env, agent, episodes=EVAL_EPISODES)
        all_eval_returns.append(eval_returns)

        # Save eval returns for later reuse
        os.makedirs(seed_dir, exist_ok=True)
        np.save(eval_rewards_path, eval_returns)
        print(f"  Saved eval rewards to {eval_rewards_path}")

        # --- Try to load training curve (if available) ---
        if os.path.exists(train_rewards_path):
            train_rewards = np.load(train_rewards_path)
            all_train_curves.append(train_rewards)
        else:
            print(f"  [WARN] No train_rewards.npy found for seed {train_seed}, "
                  f"training curve won't be included in mean plot.")

    # If no eval data collected, just exit
    if len(all_eval_returns) == 0:
        print("\nNo evaluation data collected. Check your log folders and seeds.")
        return

    # ------------------ PLOT EVAL MEAN ± STD ------------------
    all_eval_returns = np.stack(all_eval_returns, axis=0)  # (num_seeds, EVAL_EPISODES)
    eval_mean = all_eval_returns.mean(axis=0)
    eval_std = all_eval_returns.std(axis=0)
    eval_episodes = np.arange(1, EVAL_EPISODES + 1)

    plt.figure(figsize=(10, 5))
    plt.title("TD3 Cartpole – Evaluation Rewards (mean ± std over seeds)")

    plt.plot(eval_episodes, eval_mean, label="Eval mean return")
    plt.fill_between(
        eval_episodes,
        eval_mean - eval_std,
        eval_mean + eval_std,
        alpha=0.2,
        label="Eval ± std"
    )
    plt.xlabel("Evaluation episode")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("td3_cartpole_eval_mean_std.png", bbox_inches="tight")
    # plt.show()
    print("\nSaved evaluation plot to td3_cartpole_eval_mean_std.png")

    # ------------------ OPTIONAL: PLOT TRAINING MEAN ± STD ------------------
    if len(all_train_curves) > 0:
        # Make all training curves same length (truncate to shortest if needed)
        for tr in all_train_curves:
            print(tr)
            print(f"  Training curve length: {len(tr)} episodes")
        min_len = min(len(tr) for tr in all_train_curves)
        train_mat = np.stack([tr[:min_len] for tr in all_train_curves], axis=0)

        train_mean = train_mat.mean(axis=0)
        train_std = train_mat.std(axis=0)
        episodes = np.arange(1, min_len + 1)

        plt.figure(figsize=(10, 5))
        plt.title("TD3 Cartpole – Training Rewards (mean ± std over seeds)")

        plt.plot(episodes, train_mean, label="Train mean return")
        plt.fill_between(
            episodes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            label="Train ± std"
        )
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("td3_cartpole_train_mean_std.png", bbox_inches="tight")
        # plt.show()
        print("Saved training plot to td3_cartpole_train_mean_std.png")


if __name__ == "__main__":
    main()
"""Train DQN agent (with fixed exploration) and log everything for the report."""
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

from mountain_car.agents.dqn import DQNAgent

SEED = 0
EPISODES = 2_500

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

t0 = time.time()
agent = DQNAgent("MountainCar-v0")
history = agent.train(total_episodes=EPISODES, log_interval=25)
train_time = time.time() - t0
print(f"\n=== DQN training finished in {train_time:.1f}s ===")

# Evaluation
env = gym.make("MountainCar-v0")
eval_rewards = []
eval_solved = 0
for i in range(100):
    obs, _ = env.reset(seed=SEED + 1000 + i)
    R, done = 0.0, False
    while not done:
        a, _ = agent.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        R += r
        done = term or trunc
    eval_rewards.append(R)
    if R > -200:
        eval_solved += 1
env.close()

mean_eval = float(np.mean(eval_rewards))
print(f"Eval: mean {mean_eval:.2f}, reached flag {eval_solved}/100")

Path("saves").mkdir(exist_ok=True)
agent.save(Path("saves/dqn.pt"))

with open("saves/dqn_metrics.pkl", "wb") as f:
    pickle.dump({
        "history": history,
        "eval_rewards": eval_rewards,
        "eval_mean": mean_eval,
        "eval_solved": eval_solved,
        "train_time_s": train_time,
        "seed": SEED,
        "episodes": EPISODES,
        "hyperparams": {
            "lr": agent.lr, "gamma": agent.gamma,
            "epsilon_end": agent.epsilon_end, "epsilon_decay": agent.epsilon_decay,
            "batch_size": agent.batch_size, "buffer_capacity": agent.buffer_capacity,
            "target_update_freq": agent.target_update_freq, "hidden": agent.hidden,
            "explore_hold_min": agent.explore_hold_min, "explore_hold_max": agent.explore_hold_max,
        },
    }, f)
print("Saved saves/dqn_metrics.pkl")

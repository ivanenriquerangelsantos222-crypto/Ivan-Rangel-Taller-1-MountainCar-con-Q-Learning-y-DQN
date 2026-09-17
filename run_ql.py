"""Train Q-Learning tabular agent and log everything for the report."""
import pickle
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

from mountain_car.agents.qlearning import QLearningAgent

SEED = 0
EPISODES = 20_000

np.random.seed(SEED)
t0 = time.time()
agent = QLearningAgent("MountainCar-v0")
history = agent.train(total_episodes=EPISODES, log_interval=500)
train_time = time.time() - t0
print(f"\n=== Q-Learning training finished in {train_time:.1f}s ===")

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
    if R > -200:  # Reached flag before truncation
        eval_solved += 1
env.close()

mean_eval = float(np.mean(eval_rewards))
print(f"Eval: mean {mean_eval:.2f}, reached flag {eval_solved}/100")

# Save agent + full metrics
Path("saves").mkdir(exist_ok=True)
agent.save(Path("saves/qlearning.pkl"))

with open("saves/qlearning_metrics.pkl", "wb") as f:
    pickle.dump({
        "history": history,
        "eval_rewards": eval_rewards,
        "eval_mean": mean_eval,
        "eval_solved": eval_solved,
        "train_time_s": train_time,
        "seed": SEED,
        "episodes": EPISODES,
        "states_visited": len(agent.q_table),
        "hyperparams": {
            "n_bins": agent.n_bins, "lr": agent.lr, "gamma": agent.gamma,
            "epsilon_end": agent.epsilon_end, "epsilon_decay": agent.epsilon_decay,
        },
    }, f)
print("Saved saves/qlearning_metrics.pkl")

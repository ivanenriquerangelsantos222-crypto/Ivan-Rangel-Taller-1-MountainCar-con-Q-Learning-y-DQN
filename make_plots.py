"""Generate learning curves for both agents."""
import pickle
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rolling(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def plot_agent(pkl_path, out_path, title, window):
    with open(pkl_path, "rb") as f:
        m = pickle.load(f)
    hist = m["history"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist, color="steelblue", alpha=0.25, linewidth=0.6, label="Recompensa por episodio")
    smooth = rolling(hist, window)
    ax.plot(np.arange(window - 1, len(hist)), smooth, color="firebrick", linewidth=2.0,
            label=f"Media móvil ({window} episodios)")
    ax.axhline(-110, color="green", linestyle="--", alpha=0.7, label="Umbral 'solved' (-110)")
    ax.axhline(-200, color="grey", linestyle=":", alpha=0.6, label="Piso (-200)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa acumulada")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Saved {out_path}")

    # Print summary
    print(f"  Episodios: {m['episodes']}")
    print(f"  Tiempo:    {m['train_time_s']:.1f} s")
    print(f"  Eval:      media {m['eval_mean']:.2f}, bandera {m['eval_solved']}/100")


if len(sys.argv) > 1 and sys.argv[1] == "ql":
    plot_agent("saves/qlearning_metrics.pkl", "qlearning_curve.png",
               "Q-Learning tabular — Curva de aprendizaje (MountainCar-v0)", window=500)
elif len(sys.argv) > 1 and sys.argv[1] == "dqn":
    plot_agent("saves/dqn_metrics.pkl", "dqn_curve.png",
               "DQN con exploración por ráfagas — Curva de aprendizaje (MountainCar-v0)", window=50)
elif len(sys.argv) > 1 and sys.argv[1] == "cmp":
    with open("saves/qlearning_metrics.pkl", "rb") as f:
        ql = pickle.load(f)
    with open("saves/dqn_metrics.pkl", "rb") as f:
        dqn = pickle.load(f)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ql_smooth = rolling(ql["history"], 500)
    dqn_smooth = rolling(dqn["history"], 50)
    # Plot both on the same axis (episode count)
    ax.plot(np.arange(499, len(ql["history"])), ql_smooth,
            color="steelblue", linewidth=2.0, label="Q-Learning (media móvil 500 ep)")
    ax.plot(np.arange(49, len(dqn["history"])), dqn_smooth,
            color="firebrick", linewidth=2.0, label="DQN (media móvil 50 ep)")
    ax.axhline(-110, color="green", linestyle="--", alpha=0.7, label="Umbral 'solved' (-110)")
    ax.axhline(-200, color="grey", linestyle=":", alpha=0.6)
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa acumulada (media móvil)")
    ax.set_title("Comparación Q-Learning vs DQN — MountainCar-v0")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("comparison.png", dpi=140)
    print("Saved comparison.png")

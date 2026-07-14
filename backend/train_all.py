"""
Trains all 6 runs -- DQN/REINFORCE/Actor-Critic, each baseline AND improved --
across multiple seeds, then exports one combined JSON consumed by the
frontend (3 views: baseline comparison, improved comparison, per-algorithm
before/after) and by docs/baseline/ + docs/improved/.

Run:  py -3.11 train_all.py
"""
import json
import os
import time

import numpy as np

from algorithms.dqn import train_dqn
from algorithms.dqn_double import train_double_dqn
from algorithms.reinforce import train_reinforce
from algorithms.reinforce_improved import train_reinforce_improved
from algorithms.actor_critic import train_actor_critic
from algorithms.actor_critic_gae import train_actor_critic_gae

N_EPISODES = 400
SEEDS = [42, 43, 44, 45, 46]
SOLVED_THRESHOLD = 195.0
SOLVED_WINDOW = 20
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "data", "results.json")

# (key, family, variant, label, train_fn)
RUN_SPECS = [
    ("DQN", "value-based", "baseline", "DQN (Baseline)", train_dqn),
    ("DQN", "value-based", "improved", "Double DQN", train_double_dqn),
    ("REINFORCE", "policy-based", "baseline", "REINFORCE (Baseline)", train_reinforce),
    ("REINFORCE", "policy-based", "improved", "REINFORCE + Baseline + Entropy", train_reinforce_improved),
    ("Actor-Critic", "actor-critic", "baseline", "Actor-Critic (Baseline)", train_actor_critic),
    ("Actor-Critic", "actor-critic", "improved", "Actor-Critic + GAE", train_actor_critic_gae),
]


def per_seed_metrics(result: dict, elapsed_s: float, seed: int) -> dict:
    rewards = result["rewards"]
    solved_at = result["solved_at"]
    last_window = rewards[-SOLVED_WINDOW:]
    steps_to_solve = sum(rewards[: solved_at + 1]) if solved_at is not None else None
    mean_last = float(np.mean(last_window))
    std_last = float(np.std(last_window))
    return {
        "seed": seed,
        "solved_at_episode": solved_at,
        "env_steps_to_solve": steps_to_solve,
        "final_avg_reward_last20": mean_last,
        "final_std_reward_last20": std_last,
        "best_episode_reward": float(max(rewards)),
        "training_time_sec": round(elapsed_s, 2),
    }


def aggregate_metric(values, ignore_none=False):
    clean = [v for v in values if v is not None] if ignore_none else values
    if not clean:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(clean)), "std": float(np.std(clean)), "n": len(clean)}


def run_one(key, family, variant, label, fn):
    print(f"Training {label} ({len(SEEDS)} seeds x {N_EPISODES} episodes)...")
    seed_results = []
    seed_metrics = []
    for seed in SEEDS:
        start = time.perf_counter()
        result = fn(n_episodes=N_EPISODES, seed=seed)
        elapsed = time.perf_counter() - start
        seed_results.append(result)
        seed_metrics.append(per_seed_metrics(result, elapsed, seed))
        print(
            f"    seed {seed}: final avg(last20)={seed_metrics[-1]['final_avg_reward_last20']:.1f}"
            f" solved_at={seed_metrics[-1]['solved_at_episode']} time={elapsed:.1f}s"
        )

    rewards_matrix = np.array([r["rewards"] for r in seed_results])  # (n_seeds, n_episodes)
    moving_avg_matrix = np.array([r["moving_avg"] for r in seed_results])

    solved_ats = [m["solved_at_episode"] for m in seed_metrics]
    solved_count = sum(1 for s in solved_ats if s is not None)

    metrics = {
        "solved_rate": solved_count / len(SEEDS),
        "solved_at_episode": aggregate_metric(solved_ats, ignore_none=True),
        "env_steps_to_solve": aggregate_metric([m["env_steps_to_solve"] for m in seed_metrics], ignore_none=True),
        "final_avg_reward_last20": aggregate_metric([m["final_avg_reward_last20"] for m in seed_metrics]),
        "final_std_reward_last20": aggregate_metric([m["final_std_reward_last20"] for m in seed_metrics]),
        "best_episode_reward": aggregate_metric([m["best_episode_reward"] for m in seed_metrics]),
        "training_time_sec": aggregate_metric([m["training_time_sec"] for m in seed_metrics]),
    }

    return {
        "key": key,
        "family": family,
        "variant": variant,
        "label": label,
        "seeds": SEEDS,
        "rewards_mean": rewards_matrix.mean(axis=0).round(2).tolist(),
        "moving_avg_mean": moving_avg_matrix.mean(axis=0).round(2).tolist(),
        "moving_avg_std": moving_avg_matrix.std(axis=0).round(2).tolist(),
        "per_seed": seed_metrics,
        "metrics": metrics,
    }


def run_all():
    runs = [run_one(*spec) for spec in RUN_SPECS]

    payload = {
        "environment": "CartPole-v1",
        "n_episodes": N_EPISODES,
        "seeds": SEEDS,
        "solved_threshold": SOLVED_THRESHOLD,
        "solved_window": SOLVED_WINDOW,
        "runs": runs,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    run_all()

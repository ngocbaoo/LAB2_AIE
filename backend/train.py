"""
Trains DQN, REINFORCE (Policy Gradient) and Actor-Critic on CartPole-v1 with the
same episode budget and seed, then exports a single JSON consumed by the
frontend dashboard and referenced by docs/03_results.md.

Run:  py -3.11 train.py
"""
import json
import os
import time

from algorithms.dqn import train_dqn
from algorithms.reinforce import train_reinforce
from algorithms.actor_critic import train_actor_critic

N_EPISODES = 400
SEED = 42
SOLVED_THRESHOLD = 195.0  # moving-average (window 20) episode reward
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "data", "results.json")


def summarize(result: dict, elapsed_s: float) -> dict:
    rewards = result["rewards"]
    solved_at = result["solved_at"]
    last_window = rewards[-20:]
    steps_to_solve = sum(rewards[: solved_at + 1]) if solved_at is not None else None

    return {
        "name": result["name"],
        "family": result["family"],
        "rewards": rewards,
        "moving_avg": result["moving_avg"],
        "metrics": {
            "solved_at_episode": solved_at,
            "env_steps_to_solve": steps_to_solve,
            "final_avg_reward_last20": sum(last_window) / len(last_window),
            "final_std_reward_last20": (
                sum((r - sum(last_window) / len(last_window)) ** 2 for r in last_window)
                / len(last_window)
            )
            ** 0.5,
            "best_episode_reward": max(rewards),
            "training_time_sec": round(elapsed_s, 2),
        },
    }


def run_all():
    runs = []
    for label, fn in [
        ("DQN", train_dqn),
        ("REINFORCE", train_reinforce),
        ("Actor-Critic", train_actor_critic),
    ]:
        print(f"Training {label} for {N_EPISODES} episodes...")
        start = time.perf_counter()
        result = fn(n_episodes=N_EPISODES, seed=SEED)
        elapsed = time.perf_counter() - start
        summary = summarize(result, elapsed)
        print(f"  done in {elapsed:.1f}s | final avg (last 20) = {summary['metrics']['final_avg_reward_last20']:.1f}"
              f" | solved at episode {summary['metrics']['solved_at_episode']}")
        runs.append(summary)

    payload = {
        "environment": "CartPole-v1",
        "n_episodes": N_EPISODES,
        "seed": SEED,
        "solved_threshold": SOLVED_THRESHOLD,
        "solved_window": 20,
        "runs": runs,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved results to {OUT_PATH}")


if __name__ == "__main__":
    run_all()

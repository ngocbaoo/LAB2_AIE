"""
Trains Actor-Critic (baseline) on TradingEnv (real SPY daily-close
trading-strategy problem, see backend/trading_env.py) -- a quick 1-seed
sanity check. The authoritative 5-seed data source is train_all.py.

Run:  py -3.11 train.py
"""
import json
import os
import time

from algorithms.actor_critic import train_actor_critic

N_EPISODES = 400
SEED = 42
SOLVED_THRESHOLD = 2.0  # moving-average (window 20) % return per episode
ENV_EPISODE_LEN = 200  # TradingEnv.episode_len -- every episode has this fixed length
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "data", "results.json")


def summarize(result: dict, elapsed_s: float) -> dict:
    rewards = result["rewards"]
    solved_at = result["solved_at"]
    last_window = rewards[-20:]
    steps_to_solve = (solved_at + 1) * ENV_EPISODE_LEN if solved_at is not None else None

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
        ("Actor-Critic (Baseline)", train_actor_critic),
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
        "environment": "TradingEnv (real SPY daily close, 2015-2024, window=10, episode_len=200)",
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

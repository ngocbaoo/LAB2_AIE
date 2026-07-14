"""Shared helpers: seeding and a rolling-average tracker used by all three algorithms."""
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RewardTracker:
    """Keeps per-episode returns and a moving average for convergence comparisons."""

    def __init__(self, window: int = 20):
        self.window = window
        self.rewards = []

    def add(self, reward: float):
        self.rewards.append(reward)

    @property
    def moving_avg(self):
        out = []
        for i in range(len(self.rewards)):
            lo = max(0, i - self.window + 1)
            out.append(float(np.mean(self.rewards[lo:i + 1])))
        return out

    def solved_at(self, threshold: float = 195.0):
        """First episode index where the moving average crosses `threshold`, else None."""
        ma = self.moving_avg
        for i, v in enumerate(ma):
            if v >= threshold:
                return i
        return None

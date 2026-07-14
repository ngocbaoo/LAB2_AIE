"""
REINFORCE -- vanilla Policy Gradient, the pure counterpart to DQN.

Core idea: learn pi(a|s) directly as a probability distribution over actions,
and push up the log-probability of actions that led to high return. There is
no Q(s, a) here at all -- only a policy network and the episode return used
as a (high-variance) learning signal.
"""
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import set_seed, RewardTracker


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


def discount_returns(rewards, gamma: float):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def train_reinforce(
    n_episodes: int = 400,
    gamma: float = 0.99,
    lr: float = 1e-3,
    seed: int = 42,
):
    set_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    tracker = RewardTracker()

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        log_probs = []
        rewards = []

        while not done:
            probs = policy(torch.tensor(state, dtype=torch.float32, device=device))
            dist = Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            rewards.append(reward)

        returns = discount_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        loss = -torch.stack([lp * g for lp, g in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.add(sum(rewards))

    env.close()
    return {
        "name": "Policy Gradient (REINFORCE)",
        "family": "policy-based",
        "rewards": tracker.rewards,
        "moving_avg": tracker.moving_avg,
        "solved_at": tracker.solved_at(),
    }


if __name__ == "__main__":
    result = train_reinforce(n_episodes=50)
    print(result["moving_avg"][-1])

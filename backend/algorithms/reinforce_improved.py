"""
REINFORCE + learned baseline + entropy bonus -- improved Policy Gradient.

The baseline results (docs/baseline/03_results.md) showed REINFORCE's core
weakness clearly: it reached the solved threshold *earliest* of all three
algorithms, then its policy collapsed again before training ended. Two
independent fixes target that instability, without turning this into a
full Actor-Critic (there is still one Monte-Carlo update per whole episode,
not a bootstrap after every step):

1. **Learned baseline V(s)** -- vanilla REINFORCE (algorithms/reinforce.py)
   only standardizes the return within an episode. Here a small value
   network is trained (via MSE against the same Monte-Carlo return) and its
   prediction is subtracted from the return before it multiplies the
   log-probability: `advantage = G_t - V(s_t)`. This is the textbook
   variance-reduction trick -- it does not bias the gradient (E[V(s)] term
   cancels out) but shrinks its variance a lot, so a handful of unusually
   good/bad episodes can no longer swing the policy as far.

2. **Entropy bonus** -- an extra `- entropy_coef * H(pi(.|s))` term in the
   loss rewards the policy for staying stochastic a little longer. This
   directly targets the "collapsed too early" failure mode: without it,
   nothing stops the policy from driving action probabilities to ~0/~1 the
   moment it finds a decent trajectory, which is exactly what produced the
   instability in the baseline run.
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
        return torch.softmax(self.net(x), dim=-1)


class ValueBaseline(nn.Module):
    """Learned V(s) used only to reduce variance -- not a bootstrapped critic."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def discount_returns(rewards, gamma: float):
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def train_reinforce_improved(
    n_episodes: int = 400,
    gamma: float = 0.99,
    lr: float = 1e-3,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    seed: int = 42,
):
    set_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    baseline = ValueBaseline(obs_dim).to(device)
    optimizer = optim.Adam(list(policy.parameters()) + list(baseline.parameters()), lr=lr)
    tracker = RewardTracker()

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        states, log_probs, entropies, rewards = [], [], [], []

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            probs = policy(state_t)
            dist = Categorical(probs)
            action = dist.sample()

            states.append(state_t)
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            rewards.append(reward)

        returns = discount_returns(rewards, gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        values = baseline(torch.stack(states))
        advantage = returns_t - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        policy_loss = -torch.stack(
            [lp * a for lp, a in zip(log_probs, advantage)]
        ).sum()
        entropy_bonus = torch.stack(entropies).sum()
        value_loss = nn.functional.mse_loss(values, returns_t)

        loss = policy_loss - entropy_coef * entropy_bonus + value_coef * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tracker.add(sum(rewards))

    env.close()
    return {
        "name": "REINFORCE",
        "family": "policy-based",
        "rewards": tracker.rewards,
        "moving_avg": tracker.moving_avg,
        "solved_at": tracker.solved_at(),
    }


if __name__ == "__main__":
    result = train_reinforce_improved(n_episodes=50)
    print(result["moving_avg"][-1])

"""
Actor-Critic (one-step A2C) -- combines both worlds.

- Actor:  pi(a|s), a policy network that picks actions (same role as in REINFORCE).
- Critic: V(s), a value network that scores how good a state is (a scaled-down
  cousin of the Q-network in DQN, but state-value only, not per-action).

Instead of waiting for the full episode return (REINFORCE's high-variance
signal), the Critic gives an immediate estimate of the advantage after every
single step:

    advantage = r + gamma * V(s') * (1 - done) - V(s)

The Actor is updated with that advantage in place of the raw return, which is
what makes Actor-Critic learn in fewer episodes with lower variance than pure
policy gradient, while still being a genuine policy-gradient method (unlike DQN).
"""
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import set_seed, RewardTracker
from trading_env import TradingEnv


class Actor(nn.Module):
    """pi(a|s): outputs a categorical action distribution over Sell/Hold/Buy."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)


class Critic(nn.Module):
    """V(s): scores how good the current state (recent returns + position) is."""

    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_networks(obs_dim: int, n_actions: int, hidden: int = 128):
    """Builds the Actor (policy) and Critic (state-value) networks."""
    actor = Actor(obs_dim, n_actions, hidden)
    critic = Critic(obs_dim, hidden)
    return actor, critic


def custom_loss(log_prob: torch.Tensor, advantage: torch.Tensor, value: torch.Tensor,
                 target: torch.Tensor, value_coef: float = 0.5):
    """Actor-Critic loss = policy log-likelihood weighted by Advantage + Critic's value error.

        actor_loss  = -log(pi(a|s)) * A          (A = target - V(s), detached: Actor doesn't
                                                    backprop through the Critic's own error)
        critic_loss = (target - V(s))^2           (Critic learns to predict the TD target)
        loss        = actor_loss + value_coef * critic_loss
    """
    actor_loss = -log_prob * advantage.detach()
    critic_loss = (target - value).pow(2)
    return actor_loss + value_coef * critic_loss


def train_actor_critic(
    n_episodes: int = 400,
    gamma: float = 0.99,
    lr: float = 1e-3,
    value_coef: float = 0.5,
    seed: int = 42,
):
    set_seed(seed)
    env = TradingEnv()
    obs_dim = env.obs_dim
    n_actions = env.n_actions

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor, critic = build_networks(obs_dim, n_actions)
    actor, critic = actor.to(device), critic.to(device)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
    tracker = RewardTracker()

    for episode in range(n_episodes):
        state = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            probs = actor(state_t)
            value = critic(state_t)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, done = env.step(int(action.item()))
            ep_reward += reward

            with torch.no_grad():
                next_value = critic(torch.tensor(next_state, dtype=torch.float32, device=device))
                target = reward + gamma * next_value * (1 - float(done))

            advantage = target - value
            loss = custom_loss(dist.log_prob(action), advantage, value, target, value_coef)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        tracker.add(ep_reward)

    return {
        "name": "Actor-Critic",
        "family": "actor-critic",
        "rewards": tracker.rewards,
        "moving_avg": tracker.moving_avg,
        "solved_at": tracker.solved_at(),
    }


if __name__ == "__main__":
    result = train_actor_critic(n_episodes=50)
    print(result["moving_avg"][-1])

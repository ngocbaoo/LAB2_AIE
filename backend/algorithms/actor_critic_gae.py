"""
Actor-Critic + GAE (Generalized Advantage Estimation) -- improved Actor-Critic.

The baseline Actor-Critic (algorithms/actor_critic.py) uses the crudest
possible advantage estimate -- one step of bootstrapping:

    advantage = r_t + gamma * V(s_{t+1}) - V(s_t)

This has low variance but high bias: it trusts the Critic's (still-learning)
value estimate for everything beyond one step ahead. REINFORCE-with-baseline
sits at the other extreme -- the full Monte-Carlo return has high variance
but no bootstrapping bias. GAE(lambda) interpolates between the two with a
single knob:

    delta_t   = r_t + gamma * V(s_{t+1}) - V(s_t)                  (1-step TD error)
    A_t^GAE   = delta_t + (gamma*lambda) * delta_{t+1}
                        + (gamma*lambda)^2 * delta_{t+2} + ...

lambda=0 reduces exactly to the baseline's 1-step advantage; lambda=1
reduces to the full Monte-Carlo advantage (REINFORCE-with-baseline). This
file uses lambda=0.95: mostly Monte-Carlo, with just enough bootstrapping to
keep variance in check.

Unlike the baseline (which updates online after every single step), this
version collects a full episode, computes GAE backward through it, then does
one update -- necessary because GAE needs the entire reward sequence to
compute each advantage. Entropy bonus and gradient clipping are added on top
for the same reasons as in reinforce_improved.py: keep exploration alive and
keep updates from overshooting.
"""
import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import set_seed, RewardTracker


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, n_actions)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.trunk(x)
        action_probs = torch.softmax(self.actor_head(features), dim=-1)
        state_value = self.critic_head(features).squeeze(-1)
        return action_probs, state_value


def compute_gae(rewards, values, gamma: float, lam: float):
    """values has length len(rewards)+1 (bootstrap value appended, 0 for terminal)."""
    advantages = torch.zeros(len(rewards), device=values.device)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns


def train_actor_critic_gae(
    n_episodes: int = 400,
    gamma: float = 0.99,
    lam: float = 0.95,
    lr: float = 1e-3,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 42,
):
    set_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ActorCriticNetwork(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    tracker = RewardTracker()

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        log_probs, entropies, rewards, values = [], [], [], []

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            probs, value = net(state_t)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value)

            state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            rewards.append(reward)

        with torch.no_grad():
            bootstrap = torch.tensor(0.0, device=device)  # episode always ends (fall or truncation) -> no continuation value
        values_t = torch.cat([torch.stack(values), bootstrap.unsqueeze(0)])
        advantages, returns = compute_gae(rewards, values_t, gamma, lam)
        advantages = advantages.to(device)
        returns = returns.to(device)
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -torch.stack(
            [lp * a for lp, a in zip(log_probs, advantages_norm)]
        ).sum()
        entropy_bonus = torch.stack(entropies).sum()
        critic_loss = nn.functional.mse_loss(torch.stack(values), returns)

        loss = actor_loss - entropy_coef * entropy_bonus + value_coef * critic_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
        optimizer.step()

        tracker.add(sum(rewards))

    env.close()
    return {
        "name": "Actor-Critic",
        "family": "actor-critic",
        "rewards": tracker.rewards,
        "moving_avg": tracker.moving_avg,
        "solved_at": tracker.solved_at(),
    }


if __name__ == "__main__":
    result = train_actor_critic_gae(n_episodes=50)
    print(result["moving_avg"][-1])

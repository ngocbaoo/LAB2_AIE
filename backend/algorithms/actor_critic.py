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
import gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import set_seed, RewardTracker


class ActorCriticNetwork(nn.Module):
    """Shared trunk with two heads: the Actor (policy) and the Critic (value)."""

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
        state_value = self.critic_head(features)
        return action_probs, state_value


def train_actor_critic(
    n_episodes: int = 400,
    gamma: float = 0.99,
    lr: float = 1e-3,
    value_coef: float = 0.5,
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
        ep_reward = 0.0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device)
            probs, value = net(state_t)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated
            ep_reward += reward

            with torch.no_grad():
                _, next_value = net(torch.tensor(next_state, dtype=torch.float32, device=device))
                target = reward + gamma * next_value.squeeze() * (1 - float(done))

            advantage = target - value.squeeze()

            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = advantage.pow(2)
            loss = actor_loss + value_coef * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        tracker.add(ep_reward)

    env.close()
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

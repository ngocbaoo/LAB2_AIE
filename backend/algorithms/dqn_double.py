"""
Double DQN -- improved DQN that fixes Q-value overestimation bias.

Vanilla DQN (algorithms/dqn.py) both *selects* and *evaluates* the next
action with the same target network:

    target = r + gamma * max_a' Q_target(s', a')

Because max() is applied to noisy estimates, this systematically
overestimates Q-values -- the network keeps picking whichever action
happens to have an inflated value that day, and bootstraps on it.

Double DQN decouples the two roles: the *online* (policy) network picks
which action looks best, the *target* network only scores that specific
action:

    a* = argmax_a' Q_policy(s', a')
    target = r + gamma * Q_target(s', a*)

This one-line change to the Bellman target is the entire difference from
dqn.py -- same network architecture, same replay buffer, same epsilon
schedule, so any change in results is attributable to this fix alone.
"""
import sys
import os
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import set_seed, RewardTracker

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class QNetwork(nn.Module):
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
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


def train_double_dqn(
    n_episodes: int = 400,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 20000,
    min_buffer: int = 1000,
    eps_start: float = 1.0,
    eps_end: float = 0.02,
    eps_decay_episodes: int = 250,
    target_update_every: int = 10,
    seed: int = 42,
):
    set_seed(seed)
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_net = QNetwork(obs_dim, n_actions).to(device)
    target_net = QNetwork(obs_dim, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_capacity)
    tracker = RewardTracker()

    eps = eps_start
    eps_step = (eps_start - eps_end) / eps_decay_episodes

    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        ep_reward = 0.0

        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy_net(torch.tensor(state, dtype=torch.float32, device=device))
                    action = int(torch.argmax(q).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            if len(buffer) >= min_buffer:
                batch = buffer.sample(batch_size)
                states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
                actions = torch.tensor(batch.action, dtype=torch.int64, device=device).unsqueeze(1)
                rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
                next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
                dones = torch.tensor(batch.done, dtype=torch.float32, device=device)

                q_values = policy_net(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1, keepdim=True)
                    next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
                    target = rewards + gamma * next_q * (1 - dones)

                loss = nn.functional.smooth_l1_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        eps = max(eps_end, eps - eps_step)
        if episode % target_update_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        tracker.add(ep_reward)

    env.close()
    return {
        "name": "DQN",
        "family": "value-based",
        "rewards": tracker.rewards,
        "moving_avg": tracker.moving_avg,
        "solved_at": tracker.solved_at(),
    }


if __name__ == "__main__":
    result = train_double_dqn(n_episodes=50)
    print(result["moving_avg"][-1])

"""
TradingEnv -- moi truong giao dich dung du lieu gia thuc te (SPY, 2015-2024,
gia dong cua hang ngay, lay tu Yahoo Finance qua yfinance, cache san trong
backend/data/spy.csv).

Moi episode lay ngau nhien 1 doan lien tuc trong lich su gia that (do dai
window_size + episode_len + 1 ngay), vi tri bat dau duoc chon boi seed +
episode -> nhieu episode/seed khac nhau se roi vao cac giai doan thi truong
khac nhau (tang/giam/di ngang, bien dong cao/thap that), nhung van la du lieu
gia that 100%, khong sinh tong hop.

State:  cua so `window_size` log-return gan nhat (thuc te) + vi the hien tai
        (scalar) -> obs_dim = window_size + 1
Action: 0 = Sell (vi the muc tieu -1), 1 = Hold (0), 2 = Buy (+1)
Reward: (vi the truoc do) * (log-return buoc nay that) - phi giao dich khi
        doi vi the, nhan 100 de bieu dien theo % loi nhuan.
"""
import csv
import os

import numpy as np

ACTIONS = (-1.0, 0.0, 1.0)  # Sell, Hold, Buy -> vi the muc tieu
REWARD_SCALE = 100.0  # bieu dien reward theo % thay vi so thap phan nho

_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "spy.csv")


def _load_log_returns(path: str = _DATA_PATH) -> np.ndarray:
    closes = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            closes.append(float(row["Close"]))
    closes = np.asarray(closes, dtype=np.float64)
    return np.diff(np.log(closes))


class TradingEnv:
    def __init__(self, window_size: int = 10, episode_len: int = 200,
                 transaction_cost: float = 0.0002, data_path: str = _DATA_PATH):
        self.window_size = window_size
        self.episode_len = episode_len
        self.transaction_cost = transaction_cost
        self.obs_dim = window_size + 1
        self.n_actions = len(ACTIONS)
        self.all_log_returns = _load_log_returns(data_path)

        n_steps = episode_len + window_size + 1
        if n_steps > len(self.all_log_returns):
            raise ValueError(
                f"episode_len+window_size+1={n_steps} exceeds available history "
                f"({len(self.all_log_returns)} log-returns)"
            )
        self._max_start = len(self.all_log_returns) - n_steps

    def reset(self, seed: int):
        rng = np.random.default_rng(seed)
        n_steps = self.episode_len + self.window_size + 1
        start = int(rng.integers(0, self._max_start + 1))
        self.log_returns = self.all_log_returns[start:start + n_steps]
        self.t = self.window_size
        self.position = 0.0
        self.steps_done = 0
        return self._obs()

    def _obs(self):
        window = self.log_returns[self.t - self.window_size:self.t]
        return np.concatenate([window, [self.position]]).astype(np.float32)

    def step(self, action_idx: int):
        target_position = ACTIONS[action_idx]
        cost = self.transaction_cost * abs(target_position - self.position)

        realized_return = self.log_returns[self.t]
        reward = (self.position * realized_return - cost) * REWARD_SCALE

        self.position = target_position
        self.t += 1
        self.steps_done += 1
        done = self.steps_done >= self.episode_len

        return self._obs(), float(reward), done

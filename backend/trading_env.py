"""
TradingEnv -- moi truong giao dich tong hop dung chung cho ca 6 cau hinh
(DQN/REINFORCE/Actor-Critic x baseline/improved).

Khong phu thuoc gym. Gia duoc sinh tu mot qua trinh log-return tu-tuong-quan
bac 1 (AR(1), rieng cho moi episode, seed hoa boi seed + episode):

    r_t = phi * r_{t-1} + eps_t,   eps_t ~ N(mu, sigma)

Voi phi > 0, r_t co xu huong tiep dien dau cua r_{t-1} (momentum) -- day la
"alpha" ma agent co the hoc duoc tu state (cua so log-return gan nhat da bao
gom r_{t-1}): chinh sach kieu "theo da" (di theo dau cua return gan nhat) co
loi nhuan ky vong duong. Neu chi dung nhieu Gauss doc lap (khong tu tuong
quan), khong co tin hieu nao trong state du bao duoc return tiep theo, va
chinh sach toi uu se suy bien ve "Hold" -- day la ly do file nay dung AR(1)
thay vi random walk thuan.

State:  cua so `window_size` log-return gan nhat + vi the hien tai (scalar)
        -> obs_dim = window_size + 1
Action: 0 = Sell (vi the muc tieu -1), 1 = Hold (0), 2 = Buy (+1)
Reward: (vi the truoc do) * (log-return buoc nay) - phi giao dich khi doi vi
        the, nhan 100 de bieu dien theo % loi nhuan (de doc hon so voi so
        thap phan nho).
"""
import numpy as np

ACTIONS = (-1.0, 0.0, 1.0)  # Sell, Hold, Buy -> vi the muc tieu
REWARD_SCALE = 100.0  # bieu dien reward theo % thay vi so thap phan nho


class TradingEnv:
    def __init__(self, window_size: int = 10, episode_len: int = 200,
                 transaction_cost: float = 0.0002, mu: float = 0.0, sigma: float = 0.01,
                 phi: float = 0.4):
        self.window_size = window_size
        self.episode_len = episode_len
        self.transaction_cost = transaction_cost
        self.mu = mu
        self.sigma = sigma
        self.phi = phi  # he so tu tuong quan (momentum) cua log-return
        self.obs_dim = window_size + 1
        self.n_actions = len(ACTIONS)

    def reset(self, seed: int):
        rng = np.random.default_rng(seed)
        n_steps = self.episode_len + self.window_size + 1
        eps = rng.normal(self.mu, self.sigma, size=n_steps)
        log_returns = np.empty(n_steps, dtype=np.float64)
        log_returns[0] = eps[0]
        for i in range(1, n_steps):
            log_returns[i] = self.phi * log_returns[i - 1] + eps[i]
        self.log_returns = log_returns
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

"""Tabular Q-learning agent that learns a CORRECTION to the forecaster's order-up-to target.

State  : how full the inventory position is relative to the target (8 buckets) x recent forecast bias (3 buckets)
Action : multiply the target by one of ACTIONS, then order up to that level
Reward : minus the day's holding + stockout cost
Trained in the simulator on bootstrapped windows of TRAINING data only, then run greedily on the test period.
"""
import numpy as np

ACTIONS = np.array([0.85, 1.0, 1.15])           # small corrections: the forecaster stays in charge
IDENTITY = 1
RATIO_EDGES = np.array([0.25, 0.5, 0.75, 0.9, 1.0, 1.15, 1.4])    # -> 8 buckets
BIAS_EDGES = np.array([-0.15, 0.15])                               # -> 3 buckets


def _state(pos, target, recent_bias):
    r = np.digitize(pos / np.maximum(target, 1e-9), RATIO_EDGES)
    b = np.digitize(recent_bias, BIAS_EDGES)
    return r * 3 + b


class RLPolicy:
    def __init__(self, base_target, demand, H, holding, stockout, lead, seed=0):
        self.target, self.demand, self.H = base_target, demand, H
        self.h, self.p, self.lead = holding, stockout, lead
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((8 * 3, len(ACTIONS)))
        self.Q[:, IDENTITY] = 0.5            # start by trusting the forecaster; deviate only when experience says so
        self.trained_steps = 0
        self.curve = []                       # mean episode return (negative cost) per block of episodes
        self.action_counts = np.zeros(len(ACTIONS))

    # ---- recent bias: mean of (actual - planned daily demand) / planned, over the last 7 days
    def _bias(self, t):
        lo = max(0, t - 7)
        planned = self.target[lo:t] / self.H
        return (self.demand[lo:t] - planned).mean(axis=0) / np.maximum(planned.mean(axis=0), 1e-9)

    def train(self, t_lo, t_hi, episodes=6000, steps=60, alpha=0.1, gamma=0.95):
        T, N = self.demand.shape
        block = max(1, episodes // 60)
        returns = []
        for ep in range(episodes):
            eps = max(0.05, 1.0 - ep / (0.7 * episodes))
            i = self.rng.integers(N)
            t = self.rng.integers(t_lo + 8, t_hi - steps - self.lead - 1)
            on_hand = float(self.target[t, i])
            pipe = np.zeros(self.lead + 1)                      # pipe[k] arrives in k days
            ep_return = 0.0
            for _ in range(steps):
                on_hand += pipe[0]
                pipe = np.append(pipe[1:], 0.0)
                pos = on_hand + pipe.sum()
                bias = (self.demand[t - 7:t, i] - self.target[t - 7:t, i] / self.H).mean() / max(self.target[t - 7:t, i].mean() / self.H, 1e-9)
                s = int(_state(np.array([pos]), np.array([self.target[t, i]]), np.array([bias]))[0])
                a = self.rng.integers(len(ACTIONS)) if self.rng.random() < eps else int(np.argmax(self.Q[s]))
                q = max(ACTIONS[a] * self.target[t, i] - pos, 0.0)
                pipe[self.lead - 1 if self.lead >= 1 else 0] += q
                sales = min(on_hand, self.demand[t, i])
                cost = self.h * (on_hand - sales) + self.p * (self.demand[t, i] - sales)
                on_hand -= sales
                t += 1
                pos2 = on_hand + pipe.sum()
                bias2 = (self.demand[t - 7:t, i] - self.target[t - 7:t, i] / self.H).mean() / max(self.target[t - 7:t, i].mean() / self.H, 1e-9)
                s2 = int(_state(np.array([pos2]), np.array([self.target[t, i]]), np.array([bias2]))[0])
                scale = max(self.target[t, i] / self.H, 1.0)    # normalize reward so one table serves all SKUs
                self.Q[s, a] += alpha * (-cost / scale + gamma * self.Q[s2].max() - self.Q[s, a])
                ep_return += -cost / scale
                self.trained_steps += 1
            returns.append(ep_return / steps)
            if len(returns) == block:
                self.curve.append(round(float(np.mean(returns)), 4))
                returns = []
        return self

    def order(self, t, on_hand, pos):
        s = _state(pos, self.target[t], self._bias(t))
        idx = np.argmax(self.Q[s], axis=1)
        self.action_counts += np.bincount(idx, minlength=len(ACTIONS))
        return np.maximum(ACTIONS[idx] * self.target[t] - pos, 0.0)

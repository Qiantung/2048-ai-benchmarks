"""DQN agent for 2048 reinforcement learning."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np

from cnn_model import TORCH_AVAILABLE


if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


if TORCH_AVAILABLE:
    # 【修改1：统一网络结构】替换为与cnn_model.py中CNNPolicyNetwork一致的CNN结构
    class QNetwork(nn.Module):
        def __init__(self, dropout: float = 0.2):  # 对齐CNNPolicyNetwork参数
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 4),
            )

        def forward(self, x):
            # 对齐CNNPolicyNetwork的输入维度处理（适配4x4棋盘）
            if x.dim() == 2:  # 输入为展平的16维时，reshape为4x4
                x = x.view(-1, 1, 4, 4)
            elif x.dim() == 3:
                x = x.unsqueeze(1)
            return self.net(x.float())


@dataclass
class DQNConfig:
    state_dim: int = 16
    action_dim: int = 4
    gamma: float = 0.99
    lr: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 50000
    batch_size: int = 64
    target_update_freq: int = 200
    seed: int = 42


class DQNAgent:
    def __init__(self, config: DQNConfig | None = None):
        self.config = config or DQNConfig()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.epsilon = self.config.epsilon_start
        self.memory: Deque[Transition] = deque(maxlen=self.config.buffer_size)
        self.learn_steps = 0
        self.use_torch = TORCH_AVAILABLE
        if self.use_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 【适配修改1】初始化CNN结构的QNetwork
            self.policy_net = QNetwork().to(self.device)
            self.target_net = QNetwork().to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        else:
            rng = np.random.default_rng(self.config.seed)
            self.weights = rng.normal(0, 0.01, size=(self.config.state_dim, self.config.action_dim)).astype(np.float32)
            self.bias = np.zeros(self.config.action_dim, dtype=np.float32)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        if self.use_torch:
            with torch.no_grad():
                # 【适配修改1】展平的16维state reshape为4x4，适配CNN输入
                tensor = torch.tensor(state.reshape(1, -1), device=self.device, dtype=torch.float32)
                return self.policy_net(tensor).cpu().numpy()[0]
        return state @ self.weights + self.bias

    def select_action(self, state: np.ndarray, legal_actions: List[int] | None = None, training: bool = True) -> int:
        legal = legal_actions or list(range(self.config.action_dim))
        if training and random.random() < self.epsilon:
            return random.choice(legal)
        q = self.q_values(state)
        masked = np.full(self.config.action_dim, -1e9, dtype=np.float32)
        masked[legal] = q[legal]
        return int(masked.argmax())

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state.astype(np.float32), int(action), float(reward), next_state.astype(np.float32), bool(done)))

    def learn(self) -> float | None:
        if len(self.memory) < self.config.batch_size:
            return None
        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
        if self.use_torch:
            return self._learn_torch(states, actions, rewards, next_states, dones)
        return self._learn_numpy(states, actions, rewards, next_states, dones)

    def _learn_torch(self, states, actions, rewards, next_states, dones) -> float:
        states_t = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions_t = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_t = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones_t = torch.tensor(dones, device=self.device, dtype=torch.float32)
        q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            target = rewards_t + self.config.gamma * self.target_net(next_t).max(dim=1).values * (1 - dones_t)
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._post_learn()
        return float(loss.item())

    def _learn_numpy(self, states, actions, rewards, next_states, dones) -> float:
        q = states @ self.weights + self.bias
        next_q = next_states @ self.weights + self.bias
        target = q.copy()
        td = rewards + self.config.gamma * next_q.max(axis=1) * (1 - dones.astype(np.float32))
        target[np.arange(len(actions)), actions] = td
        error = q - target
        grad_w = states.T @ error / len(states)
        grad_b = error.mean(axis=0)
        self.weights -= self.config.lr * grad_w
        self.bias -= self.config.lr * grad_b
        self._post_learn()
        return float((error**2).mean())

    def _post_learn(self) -> None:
        self.learn_steps += 1
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
        if self.use_torch and self.learn_steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if self.use_torch:
            torch.save(self.policy_net.state_dict(), path)
        else:
            np.savez(path, weights=self.weights, bias=self.bias, epsilon=self.epsilon)

    # 【修改2：新增预训练加载功能】
    def load(self, path: str | Path) -> None:
        """加载预训练权重，适配PyTorch/NumPy两种模式"""
        path = Path(path)
        if self.use_torch:
            # 加载CNN结构的PyTorch权重，自动适配设备
            state_dict = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)  # 同步target_net权重
        else:
            # 加载NumPy全连接权重（兼容原有fallback逻辑）
            data = np.load(path)
            self.weights = data["weights"]
            self.bias = data["bias"]
            if "epsilon" in data:  # 兼容save的epsilon参数
                self.epsilon = data["epsilon"]


if __name__ == "__main__":
    agent = DQNAgent(DQNConfig(batch_size=4))
    s = np.zeros(16, dtype=np.float32)
    for _ in range(5):
        agent.store(s, 0, 1.0, s, False)
    print("action:", agent.select_action(s))
    print("loss:", agent.learn())
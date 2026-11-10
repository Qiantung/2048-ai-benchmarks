"""Training loop for DQN on Env2048."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from dqn_agent import DQNAgent, DQNConfig
from env2048 import Env2048


def train_dqn(episodes: int, max_steps: int, out_dir: Path, seed: int = 42, pretrained_path: str | None = None) -> List[Dict[str, float]]:
    env = Env2048(max_steps=max_steps)
    agent = DQNAgent(DQNConfig(seed=seed))
    # 加载预训练模型（监督学习训练好的CNN权重）
    if pretrained_path is not None:
        agent.load(pretrained_path)
    history: List[Dict[str, float]] = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        losses = []
        done = False
        info = {"score": 0, "max_tile": 0}
        while not done:
            legal = env.game.get_legal_actions() if hasattr(env.game, "get_legal_actions") else None
            action = agent.select_action(state, legal_actions=legal, training=True)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            state = next_state
        row = {
            "episode": episode,
            "reward": total_reward,
            "score": float(info["score"]),
            "max_tile": float(info["max_tile"]),
            "steps": float(info["steps"]),
            "epsilon": float(agent.epsilon),
            "loss": float(sum(losses) / len(losses)) if losses else 0.0,
        }
        history.append(row)
        print(row)
        if episode % 50 == 0:
            agent.save(out_dir / ("dqn_policy.pt" if agent.use_torch else "dqn_policy.npz"))
    agent.save(out_dir / ("dqn_policy.pt" if agent.use_torch else "dqn_policy.npz"))
    return history


def save_history(path: Path, history: List[Dict[str, float]]) -> None:
    fields = ["episode", "reward", "score", "max_tile", "steps", "epsilon", "loss"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN for 2048.")
    parser.add_argument("--episodes", type=int, default=50, help="Use 3000 for the formal experiment.")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--pretrained", default=None, help="预训练模型文件路径")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = train_dqn(args.episodes, args.max_steps, out_dir, seed=args.seed, pretrained_path=args.pretrained)
    save_history(out_dir / "dqn_history.csv", history)
    print("saved DQN history and model in", out_dir)


if __name__ == "__main__":
    main()
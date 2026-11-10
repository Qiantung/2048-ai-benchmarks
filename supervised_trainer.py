"""Supervised-learning data collection and policy training for 2048."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from cnn_model import CNNPolicyNetwork, NumpySoftmaxPolicy, TORCH_AVAILABLE
# 1. 接口适配：移除SimpleHeuristicAI，导入heuristic_ai中的HeuristicAI
from utils import collect_dataset, evaluate_ai, save_dataset_csv, set_seed
from heuristic_ai import HeuristicAI  # 替换原SimpleHeuristicAI，适配接口


if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset


# ========== 新增：接口适配类，仅补全方法名和格式转换，不修改原有逻辑 ==========
class HeuristicAIAdapter:
    """适配HeuristicAI与collect_dataset的接口规范，补齐select_action方法"""
    def __init__(self, weights=None):
        self.ai = HeuristicAI(weights)
        # 方向字符串转动作ID，与game.py定义严格对应：0=上 1=下 2=左 3=右
        self._dir2id = {"up": 0, "down": 1, "left": 2, "right": 3}

    def select_action(self, game):
        """符合collect_dataset调用规范的动作选择方法，返回整数动作ID"""
        action_str = self.ai.choose_action(game)
        if action_str is None:
            return 0
        return self._dir2id[action_str]
# ==============================================================================


def split_dataset(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    test_size = max(1, int(len(x) * test_ratio))
    test_idx, train_idx = idx[:test_size], idx[test_size:]
    # 仅修复此处：将未定义的 y_train 改为正确的 y[train_idx]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def train_torch_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    model_path: Path,
) -> Tuple[object, list[Dict[str, float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNPolicyNetwork(dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    history: list[Dict[str, float]] = []
    best_acc = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, device=device))
            test_loss = float(criterion(logits, torch.tensor(y_test, device=device)).item())
            test_acc = float((logits.argmax(dim=1).cpu().numpy() == y_test).mean())
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "test_loss": test_loss, "test_acc": test_acc}
        history.append(row)
        print(row)
        if test_acc > best_acc:
            best_acc = test_acc
            stale = 0
            torch.save(model.state_dict(), model_path)
        else:
            stale += 1
            if stale >= patience:
                break
    return model, history


def train_numpy_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    model_path: Path,
) -> Tuple[NumpySoftmaxPolicy, list[Dict[str, float]]]:
    model = NumpySoftmaxPolicy(lr=lr)
    history: list[Dict[str, float]] = []
    best_acc = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
        train_loss, _ = model.train_epoch(x_train, y_train, batch_size=batch_size)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        row = {"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc}
        history.append(row)
        print(row)
        if test_acc > best_acc:
            best_acc = test_acc
            stale = 0
            model.save(model_path)
        else:
            stale += 1
            if stale >= patience:
                break
    return model, history


def save_history(path: Path, history: list[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_loss", "test_acc"])
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised 2048 policy.")
    parser.add_argument("--games", type=int, default=20, help="Use 150 for the formal experiment.")
    parser.add_argument("--epochs", type=int, default=20, help="Use 80 for the formal experiment.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001 if TORCH_AVAILABLE else 0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. 替换数据采集的老师AI：使用适配后的启发式AI，加载遗传算法最优权重
    import json
    with open("best_weights.json", "r", encoding="utf-8") as f:
        best_weights = json.load(f)
    teacher = HeuristicAIAdapter(best_weights)  # 用适配器包装，解决接口不兼容问题

    x, y = collect_dataset(teacher, games=args.games, seed=args.seed)
    save_dataset_csv(out_dir / "supervised_dataset.csv", x, y)
    x_train, y_train, x_test, y_test = split_dataset(x, y, seed=args.seed)

    model_path = out_dir / ("supervised_policy.pt" if TORCH_AVAILABLE else "supervised_policy.npz")
    if TORCH_AVAILABLE:
        _, history = train_torch_model(x_train, y_train, x_test, y_test, args.epochs, args.batch_size, args.lr, args.patience, model_path)
    else:
        _, history = train_numpy_model(x_train, y_train, x_test, y_test, args.epochs, args.batch_size, args.lr, args.patience, model_path)
    save_history(out_dir / "supervised_history.csv", history)
    print("teacher baseline:", evaluate_ai(teacher, games=5))
    print("saved:", model_path)


if __name__ == "__main__":
    main()
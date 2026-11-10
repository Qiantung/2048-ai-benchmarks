"""Common utilities for member B's 2048 machine-learning modules."""

from __future__ import annotations

import csv
import importlib
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ACTIONS = ["up", "down", "left", "right"]
ACTION_TO_ID = {name: i for i, name in enumerate(ACTIONS)}
ID_TO_ACTION = {i: name for i, name in enumerate(ACTIONS)}


def get_game_class():
    """Prefer member A's real Game2048, otherwise use member B's placeholder."""
    try:
        return importlib.import_module("game").Game2048
    except Exception:
        from game_dummy import Game2048

        return Game2048


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def board_to_log2(board: Sequence[Sequence[int]]) -> np.ndarray:
    """Convert a 4x4 tile board to log2 features, keeping empty cells at 0."""
    arr = np.asarray(board, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    mask = arr > 0
    out[mask] = np.log2(arr[mask])
    return out


def flatten_board(board: Sequence[Sequence[int]]) -> np.ndarray:
    return board_to_log2(board).reshape(-1).astype(np.float32)


def one_hot_action(action: int, num_actions: int = 4) -> np.ndarray:
    vec = np.zeros(num_actions, dtype=np.float32)
    vec[int(action)] = 1.0
    return vec


def get_legal_actions(game: Any) -> List[int]:
    if hasattr(game, "get_legal_actions"):
        return list(game.get_legal_actions())
    legal: List[int] = []
    for action in range(4):
        sim = clone_game(game)
        if sim.move(action):
            legal.append(action)
    return legal


def clone_game(game: Any) -> Any:
    if hasattr(game, "clone"):
        return game.clone()
    GameClass = game.__class__
    new_game = GameClass()
    if hasattr(new_game, "set_board"):
        new_game.set_board(game.get_board(), getattr(game, "score", None))
    else:
        new_game.board = [row[:] for row in game.get_board()]
        if hasattr(game, "get_score"):
            new_game.score = game.get_score()
    return new_game


def max_tile(board: Sequence[Sequence[int]]) -> int:
    return int(np.asarray(board).max())


def empty_count(board: Sequence[Sequence[int]]) -> int:
    return int((np.asarray(board) == 0).sum())


def simple_heuristic_action(game: Any) -> int:
    """A lightweight teacher policy used before member A's heuristic AI is available."""
    legal = get_legal_actions(game)
    if not legal:
        return 0
    best_action = legal[0]
    best_score = -float("inf")
    for action in legal:
        sim = clone_game(game)
        before = sim.get_score()
        sim.move(action)
        board = sim.get_board()
        score = (
            (sim.get_score() - before) * 3.0
            + empty_count(board) * 12.0
            + math.log2(max_tile(board) or 2) * 8.0
            + corner_bonus(board) * 20.0
        )
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def corner_bonus(board: Sequence[Sequence[int]]) -> float:
    arr = np.asarray(board)
    largest = arr.max()
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    return 1.0 if largest in corners else 0.0


class RandomAI:
    def select_action(self, game: Any) -> int:
        legal = get_legal_actions(game)
        return random.choice(legal) if legal else 0


class SimpleHeuristicAI:
    def select_action(self, game: Any) -> int:
        return simple_heuristic_action(game)


def play_one_game(ai: Any, game_cls: Callable[[], Any] | None = None, max_steps: int = 1000) -> Dict[str, Any]:
    GameClass = game_cls or get_game_class()
    game = GameClass()
    steps = 0
    while not game.is_game_over() and steps < max_steps:
        action = ai.select_action(game) if hasattr(ai, "select_action") else ai(game)
        changed = game.move(action)
        if not changed:
            legal = get_legal_actions(game)
            if not legal:
                break
            game.move(random.choice(legal))
        steps += 1
    board = game.get_board()
    return {"score": game.get_score(), "max_tile": max_tile(board), "steps": steps, "board": board}


def evaluate_ai(ai: Any, games: int = 10, seed: int = 42, max_steps: int = 1000) -> Dict[str, float]:
    set_seed(seed)
    results = [play_one_game(ai, max_steps=max_steps) for _ in range(games)]
    return {
        "games": games,
        "avg_score": float(np.mean([r["score"] for r in results])),
        "avg_max_tile": float(np.mean([r["max_tile"] for r in results])),
        "best_max_tile": float(np.max([r["max_tile"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
    }


def collect_dataset(ai: Any, games: int = 20, max_steps: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    GameClass = get_game_class()
    states: List[np.ndarray] = []
    actions: List[int] = []
    for _ in range(games):
        game = GameClass()
        steps = 0
        while not game.is_game_over() and steps < max_steps:
            action = ai.select_action(game) if hasattr(ai, "select_action") else ai(game)
            legal = get_legal_actions(game)
            if action not in legal and legal:
                action = random.choice(legal)
            states.append(flatten_board(game.get_board()))
            actions.append(int(action))
            if not game.move(action):
                break
            steps += 1
    return np.vstack(states).astype(np.float32), np.asarray(actions, dtype=np.int64)


def save_dataset_csv(path: str | Path, states: np.ndarray, actions: np.ndarray) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"cell_{i}" for i in range(16)] + ["action"])
        for x, y in zip(states, actions):
            writer.writerow(list(map(float, x)) + [int(y)])


def load_dataset_csv(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, :16].astype(np.float32), data[:, 16].astype(np.int64)


if __name__ == "__main__":
    set_seed(1)
    ai = SimpleHeuristicAI()
    print("Simple heuristic evaluation:", evaluate_ai(ai, games=3, max_steps=300))
    states, actions = collect_dataset(ai, games=2, max_steps=50)
    print("dataset", states.shape, actions.shape, "labels", np.bincount(actions, minlength=4))

"""Reinforcement-learning environment wrapper for 2048."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from utils import flatten_board, get_game_class, max_tile


class Env2048:
    """A lightweight Gym-like interface without requiring gym installation."""

    def __init__(self, max_steps: int = 1000):
        self.GameClass = get_game_class()
        self.max_steps = max_steps
        self.game = self.GameClass()
        self.steps = 0
        self.reached_2048 = False

    def reset(self) -> np.ndarray:
        self.game = self.GameClass()
        self.steps = 0
        self.reached_2048 = False
        return self._state()

    def _state(self) -> np.ndarray:
        return flatten_board(self.game.get_board())

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        before_score = self.game.get_score()
        changed = self.game.move(int(action))
        self.steps += 1
        score_gain = self.game.get_score() - before_score
        board = self.game.get_board()

        reward = float(score_gain)
        if not changed:
            reward -= 1.0
        current_max = max_tile(board)
        if current_max >= 2048 and not self.reached_2048:
            reward += 1000.0
            self.reached_2048 = True

        done = self.game.is_game_over() or self.steps >= self.max_steps
        if self.game.is_game_over():
            reward -= 100.0
        info = {
            "score": self.game.get_score(),
            "score_gain": score_gain,
            "changed": changed,
            "max_tile": current_max,
            "steps": self.steps,
        }
        return self._state(), reward, done, info


if __name__ == "__main__":
    env = Env2048(max_steps=20)
    state = env.reset()
    print("state shape:", state.shape)
    for action in [0, 1, 2, 3]:
        state, reward, done, info = env.step(action)
        print(action, reward, done, info)

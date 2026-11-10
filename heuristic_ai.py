"""Heuristic search player for 2048.

The AI evaluates each legal one-step move and chooses the move with the best
weighted board score. The weights are deliberately exposed so the genetic
algorithm can optimize them directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

from game import Board, Game2048


DIRECTIONS = ("up", "down", "left", "right")


@dataclass(frozen=True)
class HeuristicWeights:
    empty_cells: float = 2.7
    smoothness: float = 0.10
    monotonicity: float = 1.0
    max_in_corner: float = 2.5
    merge_potential: float = 1.0
    score_gain: float = 1.0

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "HeuristicWeights":
        fields = cls.__dataclass_fields__
        defaults = cls()
        padded = list(values) + [getattr(defaults, name) for name in fields]
        return cls(*padded[: len(fields)])

    def as_list(self) -> List[float]:
        return [
            self.empty_cells,
            self.smoothness,
            self.monotonicity,
            self.max_in_corner,
            self.merge_potential,
            self.score_gain,
        ]


class HeuristicAI:
    def __init__(self, weights: Optional[Sequence[float] | HeuristicWeights] = None) -> None:
        if weights is None:
            self.weights = HeuristicWeights()
        elif isinstance(weights, HeuristicWeights):
            self.weights = weights
        else:
            self.weights = HeuristicWeights.from_sequence(weights)

    def choose_action(self, game: Game2048) -> Optional[str]:
        scored_moves = self.score_moves(game)
        if not scored_moves:
            return None
        return max(scored_moves, key=scored_moves.get)

    def score_moves(self, game: Game2048) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for direction in DIRECTIONS:
            simulated = game.simulate_move(direction)
            if simulated is None:
                continue
            score_gain = simulated.get_score() - game.get_score()
            scores[direction] = self.evaluate_board(simulated.get_board(), score_gain)
        return scores

    def evaluate_board(self, board: Board, score_gain: int = 0) -> float:
        features = extract_features(board)
        weights = self.weights
        return (
            weights.empty_cells * features["empty_cells"]
            + weights.smoothness * features["smoothness"]
            + weights.monotonicity * features["monotonicity"]
            + weights.max_in_corner * features["max_in_corner"]
            + weights.merge_potential * features["merge_potential"]
            + weights.score_gain * math.log2(score_gain + 1)
        )

    def play_game(
        self,
        seed: Optional[int] = None,
        max_steps: int = 5000,
        verbose: bool = False,
    ) -> Dict[str, int]:
        game = Game2048(seed=seed)
        while not game.is_game_over() and game.steps < max_steps:
            action = self.choose_action(game)
            if action is None:
                break
            game.move(action)
            if verbose:
                print(game.render())
        return {
            "score": game.get_score(),
            "max_tile": game.get_max_tile(),
            "steps": game.steps,
            "won": int(game.won),
        }


def extract_features(board: Board) -> Dict[str, float]:
    log_board = [[log_tile(value) for value in row] for row in board]
    empty_cells = sum(value == 0 for row in board for value in row)
    return {
        "empty_cells": float(empty_cells),
        "smoothness": smoothness(log_board),
        "monotonicity": monotonicity(log_board),
        "max_in_corner": max_in_corner(board),
        "merge_potential": merge_potential(board),
    }


def log_tile(value: int) -> float:
    return 0.0 if value == 0 else math.log2(value)


def smoothness(log_board: List[List[float]]) -> float:
    """Higher is better: neighboring non-empty tiles should be close in value."""

    penalty = 0.0
    for row in range(4):
        for col in range(4):
            current = log_board[row][col]
            if current == 0:
                continue
            if row + 1 < 4 and log_board[row + 1][col] != 0:
                penalty += abs(current - log_board[row + 1][col])
            if col + 1 < 4 and log_board[row][col + 1] != 0:
                penalty += abs(current - log_board[row][col + 1])
    return -penalty


def monotonicity(log_board: List[List[float]]) -> float:
    """Reward rows/columns that consistently increase or decrease."""

    totals = [0.0, 0.0, 0.0, 0.0]
    for row in range(4):
        for col in range(3):
            diff = log_board[row][col + 1] - log_board[row][col]
            if diff > 0:
                totals[0] += diff
            else:
                totals[1] -= diff

    for col in range(4):
        for row in range(3):
            diff = log_board[row + 1][col] - log_board[row][col]
            if diff > 0:
                totals[2] += diff
            else:
                totals[3] -= diff

    return -min(totals[0], totals[1]) - min(totals[2], totals[3])


def max_in_corner(board: Board) -> float:
    max_tile = max(max(row) for row in board)
    if max_tile == 0:
        return 0.0

    corners = (board[0][0], board[0][3], board[3][0], board[3][3])
    if max_tile in corners:
        return math.log2(max_tile)

    edge_values = board[0] + board[3] + [board[row][0] for row in range(1, 3)] + [
        board[row][3] for row in range(1, 3)
    ]
    return math.log2(max_tile) * 0.45 if max_tile in edge_values else 0.0


def merge_potential(board: Board) -> float:
    potential = 0.0
    for row in range(4):
        for col in range(4):
            current = board[row][col]
            if current == 0:
                continue
            if row + 1 < 4 and board[row + 1][col] == current:
                potential += log_tile(current) + 1
            if col + 1 < 4 and board[row][col + 1] == current:
                potential += log_tile(current) + 1
    return potential


def evaluate_ai(
    ai: HeuristicAI,
    games: int = 10,
    seed: Optional[int] = 2026,
    max_steps: int = 5000,
) -> Dict[str, float]:
    rng = random.Random(seed)
    results = [ai.play_game(seed=rng.randrange(1_000_000_000), max_steps=max_steps) for _ in range(games)]
    return {
        "average_score": mean(result["score"] for result in results),
        "best_score": max(result["score"] for result in results),
        "average_max_tile": mean(result["max_tile"] for result in results),
        "best_max_tile": max(result["max_tile"] for result in results),
        "average_steps": mean(result["steps"] for result in results),
        "win_rate": mean(result["won"] for result in results),
    }


def main() -> None:
    ai = HeuristicAI()
    metrics = evaluate_ai(ai, games=10)
    print("Heuristic AI metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main()

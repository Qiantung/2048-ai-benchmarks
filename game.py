"""Core 2048 game logic.

This module is intentionally dependency-free so every AI method can share the
same environment. Directions accept either strings ("up", "down", "left",
"right"), WASD keys, Chinese direction names, or integer action ids 0..3.
"""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Iterable, List, Optional, Sequence, Tuple, Union


Board = List[List[int]]
Direction = Union[str, int]


ACTION_TO_DIRECTION = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

DIRECTION_ALIASES = {
    "w": "up",
    "s": "down",
    "a": "left",
    "d": "right",
    "u": "up",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "上": "up",
    "下": "down",
    "左": "left",
    "右": "right",
}


class Game2048:
    """A complete 4x4 2048 game implementation."""

    size: int = 4

    def __init__(
        self,
        board: Optional[Sequence[Sequence[int]]] = None,
        score: int = 0,
        seed: Optional[int] = None,
        add_initial_tiles: bool = True,
    ) -> None:
        self.rng = random.Random(seed)
        self.score = score
        self.steps = 0
        self.won = False

        if board is None:
            self.board: Board = [[0 for _ in range(self.size)] for _ in range(self.size)]
            if add_initial_tiles:
                self.generate_new_tile()
                self.generate_new_tile()
        else:
            self.board = [list(row) for row in board]
            self._validate_board()
            self.won = any(value >= 2048 for row in self.board for value in row)

    def _validate_board(self) -> None:
        if len(self.board) != self.size or any(len(row) != self.size for row in self.board):
            raise ValueError("board must be a 4x4 matrix")
        for row in self.board:
            for value in row:
                if not isinstance(value, int) or value < 0:
                    raise ValueError("board values must be non-negative integers")

    def clone(self) -> "Game2048":
        clone = Game2048(self.board, self.score, add_initial_tiles=False)
        clone.steps = self.steps
        clone.won = self.won
        clone.rng.setstate(self.rng.getstate())
        return clone

    def reset(self) -> Board:
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.score = 0
        self.steps = 0
        self.won = False
        self.generate_new_tile()
        self.generate_new_tile()
        return self.get_board()

    def get_board(self) -> Board:
        return deepcopy(self.board)

    def set_board(self, board: Sequence[Sequence[int]], score: Optional[int] = None) -> None:
        self.board = [list(row) for row in board]
        self._validate_board()
        if score is not None:
            self.score = score
        self.won = any(value >= 2048 for row in self.board for value in row)

    def get_score(self) -> int:
        return self.score

    def get_max_tile(self) -> int:
        return max(max(row) for row in self.board)

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        return [
            (row_index, col_index)
            for row_index, row in enumerate(self.board)
            for col_index, value in enumerate(row)
            if value == 0
        ]

    def generate_new_tile(self) -> bool:
        empty_cells = self.get_empty_cells()
        if not empty_cells:
            return False

        row, col = self.rng.choice(empty_cells)
        self.board[row][col] = 4 if self.rng.random() < 0.10 else 2
        return True

    def move(self, direction: Direction, add_random_tile: bool = True) -> bool:
        """Move tiles in the given direction.

        Returns True only when the board changes. For AI simulation, pass
        add_random_tile=False to evaluate the deterministic post-move board.
        """

        normalized = normalize_direction(direction)
        before = self.get_board()
        new_board, gained_score = self._move_board(before, normalized)

        if new_board == before:
            return False

        self.board = new_board
        self.score += gained_score
        self.steps += 1
        self.won = self.won or any(value >= 2048 for row in self.board for value in row)
        if add_random_tile:
            self.generate_new_tile()
        return True

    def simulate_move(self, direction: Direction) -> Optional["Game2048"]:
        game = self.clone()
        if not game.move(direction, add_random_tile=False):
            return None
        return game

    def get_valid_moves(self) -> List[str]:
        return [
            direction
            for direction in ("up", "down", "left", "right")
            if self.simulate_move(direction) is not None
        ]

    def is_game_over(self) -> bool:
        return len(self.get_valid_moves()) == 0

    def render(self) -> str:
        line = "+------+------+------+------+"
        rows = [line]
        for row in self.board:
            rows.append("|" + "|".join(f"{value or '.':^6}" for value in row) + "|")
            rows.append(line)
        rows.append(f"score={self.score} steps={self.steps} max_tile={self.get_max_tile()}")
        return "\n".join(rows)

    @classmethod
    def _move_board(cls, board: Board, direction: str) -> Tuple[Board, int]:
        if direction == "left":
            return cls._move_left(board)
        if direction == "right":
            reversed_board = [list(reversed(row)) for row in board]
            moved, score = cls._move_left(reversed_board)
            return [list(reversed(row)) for row in moved], score
        if direction == "up":
            transposed = transpose(board)
            moved, score = cls._move_left(transposed)
            return transpose(moved), score
        if direction == "down":
            transposed = transpose(board)
            reversed_board = [list(reversed(row)) for row in transposed]
            moved, score = cls._move_left(reversed_board)
            restored = [list(reversed(row)) for row in moved]
            return transpose(restored), score
        raise ValueError(f"unsupported direction: {direction}")

    @classmethod
    def _move_left(cls, board: Board) -> Tuple[Board, int]:
        moved_rows = []
        gained_score = 0
        for row in board:
            moved_row, row_score = merge_line_left(row)
            moved_rows.append(moved_row)
            gained_score += row_score
        return moved_rows, gained_score


def normalize_direction(direction: Direction) -> str:
    if isinstance(direction, int):
        if direction not in ACTION_TO_DIRECTION:
            raise ValueError("integer direction must be in 0..3")
        return ACTION_TO_DIRECTION[direction]

    key = str(direction).strip().lower()
    if key not in DIRECTION_ALIASES:
        raise ValueError(f"unknown direction: {direction}")
    return DIRECTION_ALIASES[key]


def transpose(board: Board) -> Board:
    return [list(row) for row in zip(*board)]


def merge_line_left(line: Iterable[int], size: int = 4) -> Tuple[List[int], int]:
    """Compress and merge one row according to official 2048 rules."""

    values = [value for value in line if value != 0]
    merged: List[int] = []
    gained_score = 0
    index = 0

    while index < len(values):
        if index + 1 < len(values) and values[index] == values[index + 1]:
            merged_value = values[index] * 2
            merged.append(merged_value)
            gained_score += merged_value
            index += 2
        else:
            merged.append(values[index])
            index += 1

    merged.extend([0] * (size - len(merged)))
    return merged, gained_score


def play_console(seed: Optional[int] = None) -> None:
    game = Game2048(seed=seed)
    print("2048 console. Use W/A/S/D, Q to quit.")
    while True:
        print(game.render())
        if game.is_game_over():
            print("Game over.")
            return
        command = input("move> ").strip().lower()
        if command == "q":
            return
        try:
            moved = game.move(command)
        except ValueError as exc:
            print(exc)
            continue
        if not moved:
            print("Invalid move.")


if __name__ == "__main__":
    play_console()

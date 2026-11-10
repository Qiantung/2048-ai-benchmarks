"""Inference wrapper for the supervised 2048 policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cnn_model import CNNPolicyNetwork, NumpySoftmaxPolicy, TORCH_AVAILABLE
from utils import flatten_board, get_legal_actions, play_one_game


if TORCH_AVAILABLE:
    import torch


class SupervisedAI:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.use_torch = TORCH_AVAILABLE and self.model_path is not None and self.model_path.suffix == ".pt"
        if self.use_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = CNNPolicyNetwork().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
        elif self.model_path and self.model_path.exists():
            self.model = NumpySoftmaxPolicy.load(self.model_path)
        else:
            self.model = NumpySoftmaxPolicy()

    def action_scores(self, board) -> np.ndarray:
        x = flatten_board(board).reshape(1, -1)
        if self.use_torch:
            with torch.no_grad():
                logits = self.model(torch.tensor(x, device=self.device)).cpu().numpy()[0]
            return logits
        return self.model.predict_proba(x)[0]

    def select_action(self, game: Any) -> int:
        legal = get_legal_actions(game)
        if not legal:
            return 0
        scores = self.action_scores(game.get_board())
        masked = np.full(4, -1e9, dtype=np.float32)
        masked[legal] = scores[legal]
        return int(masked.argmax())


if __name__ == "__main__":
    ai = SupervisedAI()
    print(play_one_game(ai, max_steps=100))

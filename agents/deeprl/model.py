"""Small CNN and the board/action transformations used by DQN."""
import numpy as np
import torch
from torch import nn

# Empty + exponents 1..16 + overflow (131072 and above).
CHANNELS = 18


def encode(board: list[list[int]]) -> np.ndarray:
    """One-hot (channels, rows, columns); 65536 has its own channel."""
    values = np.asarray(board)
    if values.shape != (4, 4):
        raise ValueError("Expected a 4x4 board")
    exponents = np.zeros((4, 4), dtype=np.int64)
    for r in range(4):
        for c in range(4):
            value = int(values[r, c])
            if value != values[r, c] or value < 0 or (value and (value < 2 or value & (value - 1))):
                raise ValueError("Tiles must be zero or powers of two >= 2")
            exponents[r, c] = min(value.bit_length() - 1, 17) if value else 0
    return np.eye(CHANNELS, dtype=np.float32)[exponents].transpose(2, 0, 1).copy()


def action_map(symmetry: int) -> np.ndarray:
    """Old action -> transformed action: optional left/right flip, then CCW turns."""
    actions = np.arange(4)
    if symmetry >= 4:
        actions = (-actions) % 4
    return (actions - symmetry % 4) % 4


def transform(array: np.ndarray, symmetry: int) -> np.ndarray:
    """Apply one of eight square symmetries to the final two dimensions."""
    if symmetry >= 4:
        array = np.flip(array, axis=-1)
    return np.rot90(array, symmetry % 4, axes=(-2, -1)).copy()


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 2), nn.ReLU(),
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4),
        )

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        return self.layers(board)

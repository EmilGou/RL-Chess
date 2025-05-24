    
import torch
import numpy as np
import random


def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_fen(fen_str: str, default_side: str = "w") -> str:
    """
    Return a six-field FEN.  If the side-to-move field is missing or blank,
    insert `default_side` ("w" by default, set to "b" if you prefer Black).
    """
    # strip outer whitespace and collapse any double spaces
    parts = [p for p in fen_str.strip().split(' ') if p != '']

    # A legal FEN must have six tokens; five means the side-to-move is missing
    if len(parts) == 5:
        parts.insert(1, default_side)

    return " ".join(parts)


def parse_fen(fen_str: str):
  parts = fen_str.split(' ')
  position = parts[0]
  turn = parts[1]
  half_move_clock = parts[4]
  full_moves = parts[5]

  return {
      "position": position,
      "turn": turn,
      "half_move_clock": int(half_move_clock),
      "full_moves": int(full_moves)
  }




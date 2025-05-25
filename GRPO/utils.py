    
import torch
import numpy as np
import random
import chess
from .tokenize import untokenize
from .vocab import SPECIAL_TOKENS, UCI_IDS


def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def play_moves_get_final_board(input_ids: list):

    board = chess.board(extract_fen_from_game(input_ids))
    
    moves_idx = SPECIAL_TOKENS["<moves>"]
    end_moves_idx = SPECIAL_TOKENS["</moves>"]

    if end_moves_idx in input_ids:
        end = input_ids.index(end_moves_idx)
    else:
        end = len(input_ids)
    
    moves_ids = input_ids[input_ids.index(moves_idx) + 1:end]
    for move_id in moves_ids:
        move = UCI_IDS[move_id]
        board.push_uci(move)
    
    return board.fen()



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


def extract_fen_from_game(game_ids):
    '''
    Takes example of token ids in the format of 
    idx(<board>)idx(fen)idx(</board>)idx(<moves>) idx(e2e4) - - idx(</moves>) 
    and returns the fen string.

    Args:
        game_ids (list): List of token ids.

    Returns:
        str: The FEN string.

    '''
    game = untokenize(game_ids)

    board_idx = 0
    for i, token in enumerate(game):
        if token == '<board>':
            moves_idx = i
            break
    end_board_idx = 0
    for i, token in enumerate(game):
        if token == '</board>':
            end_moves_idx = i
            break
    fen_str = ''.join(game[moves_idx+1:end_moves_idx])

    fen_str = fix_fen(fen_str)

    return fen_str


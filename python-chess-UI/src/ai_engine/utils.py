import chess
import torch
from genericpath import exists

from stockfish import Stockfish
from contextlib import nullcontext
from .uci_moves import  UCI_MOVES
from .model_specific_utils import (
    SPECIAL_TOKENS, UCI_IDS, FEN_CHAR_TO_ID,
    ID_TO_FEN_CHAR, ID_TO_SPECIAL, untokenize # <<< ADD untokenize HERE
)
import torch.nn.functional as F

def tokenize_fen(fen: str) -> list[int]:
    """
    Takes the full FEN, e.g.:
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    and returns a list of character-IDs for every character that
    appears in our FEN_CHAR_TO_ID vocabulary.
    """
    return [FEN_CHAR_TO_ID[c] for c in fen if c in FEN_CHAR_TO_ID]
#Evaluations
from tqdm import tqdm
import chess
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


# Inside python-chess-UI/src/ai_engine/utils.py

def extract_fen_from_game(game_ids: list[int]) -> str: # Added type hint
    '''
    Takes example of token ids in the format of <board>fen</board><moves> e2e4 - -  </moves> and returns the fen string.

    Args:
        game_ids (list): List of token ids.

    Returns:
        str: The FEN string.
    '''
    game_str_list = untokenize(game_ids) # Now untokenize is correctly imported and used

    board_start_idx = -1
    board_end_idx = -1

    try:
        board_start_idx = game_str_list.index("<board>") + 1
        board_end_idx = game_str_list.index("</board>")
    except ValueError:
        print("Error in extract_fen_from_game: <board> or </board> tags not found.")
        return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # Return default FEN on error

    fen_str = "".join(game_str_list[board_start_idx:board_end_idx])
    fen_str = fix_fen(fen_str) # Ensure fix_fen is also available in this file
    return fen_str


def evaluate_legal_move_accuracy(model, val_loader, n_games, seed=None, top_k=None, temperature=1.0):
    """
    Returns
    -------
    acc : float
        Fraction of sampled moves that are legal.
    mistakes : dict[str, str]
        {FEN : illegal_predicted_move}  (only entries that had mistakes)
    """
    if seed is not None:
        set_seeds(seed)
    model.eval()
    total, legal = 0, 0
    mistakes = {}

    moves_tok     = SPECIAL_TOKENS['<moves>']
    end_moves_tok = SPECIAL_TOKENS['</moves>']

    with torch.no_grad():
        games_seen = 0

        for idx, (x, y) in enumerate(val_loader):          # x: (B, L)
            if games_seen == n_games:
                break

            # ---- choose one random game in this batch ------------------------
            random_idx  = random.randint(0, x.size(0) - 1)

            # keep original names ── just slice to batch-size 1 before GPU
            input_seq   = x[random_idx:random_idx + 1, :-1].cuda(non_blocking=True)
            target_seq  = y[random_idx:random_idx + 1, 1:]        # still CPU
            logits      = model(input_seq)                        # (1, L-1, |V|)
            logits      = logits.squeeze(0)                       # (L-1, |V|)
            seq         = input_seq.squeeze(0).cpu()              # (L-1,)

            # ---- locate sentinel tokens --------------------------------------
            moves_idx_tens      = (seq == moves_tok).nonzero(as_tuple=True)[0]
            end_moves_idx_tens  = (seq == end_moves_tok).nonzero(as_tuple=True)[0]
            if moves_idx_tens.nelement() == 0 or end_moves_idx_tens.nelement() == 0:
                continue

            moves_idx      = moves_idx_tens.item()
            end_moves_idx  = end_moves_idx_tens.item()

            # ---- rebuild board up to a random position -----------------------
            current_fen = extract_fen_from_game(x[random_idx].tolist())
            board       = chess.Board(current_fen)

            pos              = random.randint(moves_idx, end_moves_idx - 2)
            num_moves_ahead  = pos - moves_idx
            for step in range(num_moves_ahead):
                tok_id = target_seq[0, moves_idx + step].item()
                board.push_uci(UCI_IDS[tok_id])            # faster than push(Move)

            current_fen = board.fen()

            if board.is_game_over():
                continue

            # ---- sample the model’s move at `pos` ----------------------------
            logits_vec = logits[pos] / temperature                      # shape: (|V|,)

            if top_k is not None and top_k > 0 and top_k < logits_vec.size(0):
                kth_val = torch.topk(logits_vec, top_k).values[-1]      # value of k-th largest logit
                logits_vec = torch.where(
                    logits_vec < kth_val,
                    logits_vec.new_full((), -float('inf')),              # mask everything outside top-k
                    logits_vec
                )

            probs   = torch.softmax(logits_vec, dim=-1)
            pred_id = torch.multinomial(probs, 1).item()

            pred_uci  = UCI_IDS.get(pred_id, None)


            total += 1
            try:
                move = chess.Move.from_uci(pred_uci)
            except Exception:                              # malformed UCI
                mistakes[current_fen] = pred_uci
                games_seen += 1
                continue

            if move in board.legal_moves:
                legal += 1
            else:
                mistakes[current_fen] = pred_uci

            games_seen += 1

    acc = legal / total if total else 0.0
    return acc, mistakes

def get_piece_moved(fen, uci_move):
  '''
  Returns the piece moved from the FEN and UCI move.

  Args:
    fen (str): The FEN string.
    uci_move (str): The UCI move string.

  Returns:
    str: The piece moved.
  '''
  board = chess.Board(fen)
  move = chess.Move.from_uci(uci_move)
  piece_moved = board.piece_at(move.from_square)
  if piece_moved is None:
    return 'emp'
  return piece_moved.symbol()

def evaluate_fn_across_epochs(model, val_loader, epochs, checkpoints, eval_fn, *eval_args, **eval_kwargs):
  pass #too complicated for now



def sample_move_from_model(model, seq, seq_tensor,
                           top_k=10, temperature=1.0, alpha=30,
                           mask_illegal=True, last_fen=None):
    """
    (doc‑string unchanged)
    """
    device = seq_tensor.device
    if last_fen is None and mask_illegal:
        raise ValueError("FEN must be provided if mask_illegal is True")

    board = chess.Board(last_fen) if last_fen else None


    if alpha is not None:
        moves_tok = SPECIAL_TOKENS['<moves>']
        moves_idx = (seq_tensor == moves_tok).nonzero(as_tuple=False)[0, 1]
        num_moves = seq_tensor.size(1) - moves_idx - 1
        if num_moves > alpha:
            cutoff = seq_tensor.size(1) - 1 - alpha
            to_replay_ids = seq_tensor[0, moves_idx+1:cutoff+1].cpu().tolist()
            temp_board = chess.Board(extract_fen_from_game(seq))
            for tok in to_replay_ids:
                temp_board.push_uci(UCI_IDS[tok])
            new_fen = temp_board.fen()
            prefix = ([SPECIAL_TOKENS["<board>"]] +
                      tokenize_fen(new_fen) +
                      [SPECIAL_TOKENS["</board>"], SPECIAL_TOKENS["<moves>"]])
            prefix_tensor = torch.tensor(prefix, device=device).unsqueeze(0)
            seq_tensor = torch.cat([prefix_tensor,
                                    seq_tensor[:, cutoff+1:]], dim=1)
            seq[:] = prefix + seq[cutoff+1:]

    x = seq_tensor
    logits = model(x)[0, -1, :] / temperature

    legal_ids = [UCI_MOVES[mv.uci()] for mv in board.legal_moves
                 if mv.uci() in UCI_MOVES]
    if not legal_ids:
        return "<end>", seq_tensor
    if mask_illegal:
        mask = torch.full_like(logits, float('-inf'))
        mask[legal_ids] = 0.0
        logits += mask

    if top_k is not None and 0 < top_k < logits.size(0):
        kth_val = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth_val,
                             logits.new_full((), -float('inf')), logits)
    probs = F.softmax(logits, dim=-1)
    token = torch.multinomial(probs, num_samples=1).item()

    return token, seq_tensor

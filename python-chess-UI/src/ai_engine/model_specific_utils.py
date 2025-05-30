# python-chess-UI/src/ai_engine/model_specific_utils.py
import chess
from .uci_moves import UCI_MOVES
UCI_IDS = {v: k for k, v in UCI_MOVES.items()}

# 2) BUILD FEN VOCABULARY (covers full FEN)
FEN_CHARS = [
    '/', ' ', '-',                             # separators & dash
    'P','N','B','R','Q','K',
    'p','n','b','r','q','k',                   # pieces
    '0','1','2','3','4','5','6','7','8','9',    # digits for counters
    'a','b','c','d','e','f','g','h'            # files (en passant targets)
]
FEN_CHAR_TO_ID = {c: i + len(UCI_MOVES) for i, c in enumerate(FEN_CHARS)}
ID_TO_FEN_CHAR = {v: k for k, v in FEN_CHAR_TO_ID.items()}

# Compute next available index
max_idx = max(FEN_CHAR_TO_ID.values())

# ——————————————————————————————————————————————————————————————————————
# 3) SPECIAL TOKENS
SPECIAL_TOKENS = {
    "<board>":   max_idx + 1,
    "</board>":  max_idx + 2,
    "<moves>":   max_idx + 3,
    "</moves>":  max_idx + 4,
    "<pad>":     max_idx + 5,
}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

# ——————————————————————————————————————————————————————————————————————
# 4) TOKENIZERS / UNTOKENIZER
def tokenize_fen(fen: str) -> list[int]:
    """
    Turn the full FEN string (all 6 fields) into token IDs,
    one per character, dropping only chars not in our vocab.
    """
    return [FEN_CHAR_TO_ID[c] for c in fen if c in FEN_CHAR_TO_ID]

def tokenize_uci(moves: list[str]) -> list[int]:
    return [UCI_MOVES[m] for m in moves if m in UCI_MOVES]

def untokenize(tokens: list[int]) -> list[str]:
    out = []
    for t in tokens:
        if t in ID_TO_SPECIAL:
            out.append(ID_TO_SPECIAL[t])
        elif t in ID_TO_FEN_CHAR:
            out.append(ID_TO_FEN_CHAR[t])
        elif t in UCI_IDS:
            out.append(UCI_IDS[t])
        else:
            out.append(f"<unk:{t}>")
    return out

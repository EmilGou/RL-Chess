from .vocab import *
def tokenize_fen(fen: str) -> list[int]:
 
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

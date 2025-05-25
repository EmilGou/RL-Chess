import torch
import random
import chess
from .vocab import *
from .tokenize import tokenize_fen, tokenize_uci

class ChessGameDataset(torch.utils.data.Dataset):
    def __init__(self, games: list[list[str]], max_seq_len: int):
        self.games = games
        self.max_seq_len = max_seq_len
        self.pad_token = SPECIAL_TOKENS["<pad>"]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        moves = [m.lower() for m in self.games[idx]]
        cutoff = random.randint(1, len(moves) - 1)
        past, future = moves[:cutoff], moves[cutoff:]

        board = chess.Board()
        for m in past:
            board.push_uci(m)
        fen = board.fen() 


        fen_tokens    = tokenize_fen(fen)
        future_tokens = tokenize_uci(future)

        input_seq = (
            [SPECIAL_TOKENS["<board>"]] +
            fen_tokens +
            [SPECIAL_TOKENS["</board>"],
             SPECIAL_TOKENS["<moves>"]] +
            future_tokens +
            [SPECIAL_TOKENS["</moves>"]]
        )
        moves_start = len(fen_tokens) + 2  
        labels = (
            [self.pad_token] * (moves_start + 1) +
            future_tokens +
            [self.pad_token]                    
        )
        input_seq = (input_seq + [self.pad_token] * self.max_seq_len)[:self.max_seq_len]
        labels    = (labels    + [self.pad_token] * self.max_seq_len)[:self.max_seq_len]

        return torch.tensor(input_seq, dtype=torch.long), \
               torch.tensor(labels,    dtype=torch.long)



class RLThinkDataset(torch.utils.data.Dataset):
    def __init__(self, games, max_len=128):
        self.games    = games
        self.max_len  = max_len
        self.pad_token = SPECIAL_TOKENS["<pad>"]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):

        moves = [m.lower() for m in self.games[idx]]
        cutoff = random.randint(1, len(moves) - 1)
        past = moves[:cutoff]

        board = chess.Board()
        for m in past:
            board.push_uci(m)
        fen = board.fen()  

        fen_tokens = tokenize_fen(fen)


        input_seq = (
            [SPECIAL_TOKENS["<board>"]] +
            fen_tokens +
            [SPECIAL_TOKENS["</board>"],
             SPECIAL_TOKENS["<moves>"]] 
        )
        
        # TODO: swap right padding with left padding 
        input_seq = (input_seq + [self.pad_token] * self.max_seq_len)[:self.max_seq_len]
    
        return torch.tensor(input_seq, dtype=torch.long)
           

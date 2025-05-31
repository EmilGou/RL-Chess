from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.nn import functional as F
import chess
import math
from .vocab import UCI_MOVES, UCI_IDS, SPECIAL_TOKENS, ID_TO_SPECIAL
from .tokenize import tokenize_fen
from .utils import extract_fen_from_game


@dataclass
class ChessConfig:
    vocab_size: int = 2008
    d_model: int = 1024
    n_heads: int = 8
    num_layers: int = 8
    d_ff: int = 4096
    max_len: int = 257
    dropout: float = 0.1
    pad_id: int = 2006



class AutoregressiveTransformer(nn.Module):
    def __init__(self, config: ChessConfig) -> None:
        super().__init__()
        self.pad_id = config.pad_id
        self.max_len = config.max_len

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx = self.pad_id)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        self.n_heads = config.n_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)



    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
      bsz, seq_len = tokens.shape
      if seq_len > self.max_len:
          raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}.")

      pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(bsz, seq_len)
      x = self.token_emb(tokens) + self.pos_emb(pos)

      attn_mask = self._causal_mask(size=seq_len, device=tokens.device)

      pad_mask = tokens.eq(self.pad_id)         

      x = self.transformer(x, mask=attn_mask, src_key_padding_mask=pad_mask)
      return self.lm_head(x)
   

    def predict_move(self, 
                     seq: list, 
                     seq_tensor: torch.tensor,
                     boards: list, 
                     top_k: int = 10, 
                     temperature: int = 1.0, 
                     alpha : int = None,
                     mask_illegal: bool = True,
                     ):
        """
        Predict the next move based on the sequence tensor and the current board state.

        Args:
            seq (list): List of token ids representing the sequence.
            seq_tensor (torch.Tensor): Tensor representation of the sequence. (B, L)
            boards (list): List of chess.Board objects corresponding to the sequences.
            top_k (int, optional): Number of top tokens to consider for sampling. Defaults to 10.
            temperature (float, optional): Temperature for softmax sampling. Defaults to 1.0.
            alpha (int, optional): If provided, will truncate the sequence to the last alpha moves.
            mask_illegal (bool, optional): If True, masks illegal moves in the logits.
        Returns:
            tokens (list): List of predicted token ids.
            new_seq_tensor (torch.Tensor): Updated sequence tensor with the predicted token appended. (B, L+1)
        """
        device = seq_tensor.device

        if alpha is not None:
            moves_tok = SPECIAL_TOKENS['<moves>']
            for i in range(seq_tensor.size(0)):
                row = seq_tensor[i]
                moves_idx = (row == moves_tok).nonzero(as_tuple=False)[0].item()
                num_moves = row.size(0) - moves_idx - 1
                if num_moves > alpha:
                    cutoff = row.size(0) - 1 - alpha
                    to_replay_ids = row[moves_idx + 1:cutoff + 1].cpu().tolist()
                    temp_board = chess.Board(extract_fen_from_game(seq[i]))
                    for tok in to_replay_ids:
                        temp_board.push_uci(UCI_IDS[tok])
                    new_fen = temp_board.fen()
                    prefix = ([SPECIAL_TOKENS["<board>"]] +
                              tokenize_fen(new_fen) +
                              [SPECIAL_TOKENS["</board>"], SPECIAL_TOKENS["<moves>"]])
                    prefix_tensor = torch.tensor(prefix, device=device)
                    new_row = torch.cat([prefix_tensor, row[cutoff + 1:]])
                    seq[i] = prefix + seq[i][cutoff + 1:]
                    if new_row.size(0) < row.size(0):
                        left_pad = torch.full((row.size(0) - new_row.size(0),),
                                            self.pad_id,
                                            device=device,
                                            dtype=row.dtype)
                        new_row = torch.cat([left_pad, new_row])    # ← pad on the left
                    else:
                        new_row = new_row[-row.size(0):]            # truncate on the left if too long
                    seq_tensor[i] = new_row
               

        logits = self(seq_tensor)[:, -1, :] / temperature
        
        new_seq_tensor = seq_tensor.clone()
        tokens = []
        for i in range(logits.size(0)):
            logit_i = logits[i]
            board = boards[i]
            legal_ids = [UCI_MOVES[mv.uci()] for mv in board.legal_moves
                         if mv.uci() in UCI_MOVES] if board else []
            if not legal_ids:
                tokens.append(self.pad_id)  # If no legal moves, return pad token
                continue
            if mask_illegal:
                mask = torch.full_like(logit_i, float("-inf"))
                mask[legal_ids] = 0.0
                logit_i += mask
            if top_k is not None and 0 < top_k < logit_i.size(0):
                kth_val = torch.topk(logit_i, top_k).values[-1]
                logit_i = torch.where(logit_i < kth_val,
                                      logit_i.new_full((), -float("inf")),
                                      logit_i)
            probs = F.softmax(logit_i, dim=-1)
            token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(token)
        
        new_seq_tensor = torch.cat([new_seq_tensor, torch.tensor(tokens, device = device).unsqueeze(1)], dim = 1)

        return tokens, new_seq_tensor
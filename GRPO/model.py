from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.nn import functional as F
import math

class AutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.n_heads = n_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)


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

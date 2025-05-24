#Adapted from https://github.com/sgrvinod/chess-transformers/blob/main/chess_transformers/data/prep.py

import tables as tb
from tqdm import tqdm
from .uci_moves import UCI_MOVES
from .utils import fen_transform, get_vocab
from .tokenizer import Tokenizer
import os


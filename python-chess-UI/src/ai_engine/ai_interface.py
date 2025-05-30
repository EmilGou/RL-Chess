# python-chess-UI/src/ai_engine/ai_interface.py
import os
import torch
import torch.nn.functional as F # Often needed by sample_move_from_model or related utils
from .model_definition import AutoregressiveTransformer
from .model_specific_utils import SPECIAL_TOKENS, UCI_IDS, ID_TO_SPECIAL, FEN_CHAR_TO_ID, ID_TO_FEN_CHAR, tokenize_fen as tokenize_fen_for_model_init # Renamed to avoid conflict
from .uci_moves import UCI_MOVES

try:
    from .utils import sample_move_from_model, extract_fen_from_game, tokenize_fen as tokenize_fen_from_utils
except ImportError as e:
    print(f"ERROR: Could not import from .utils in ai_interface.py: {e}. Ensure utils.py is correctly placed.")
    def sample_move_from_model(*_args, **_kwargs):
        print("Critical Error: sample_move_from_model (from utils) not available!")
        return None, None
    def tokenize_fen_from_utils(*_args, **_kwargs): # Fallback for the one utils.py provides
        print("Critical Error: tokenize_fen (from utils) not available!")
        return []
    def extract_fen_from_game(*_args, **_kwargs):
        print("Critical Error: extract_fen_from_game (from utils) not available!")
        return ""


MODEL_INSTANCE = None
INVERSE_UCI_MOVES = None # Will be UCI_IDS
MODEL_TRAINING_CONFIG = {} # Stores block_size, pad_id etc.
MODEL_VOCAB_INV = None # For decoding any token ID for debugging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def load_model(model_filename="model_checkpoint.pt"):
    global MODEL_INSTANCE, INVERSE_UCI_MOVES, MODEL_TRAINING_CONFIG, DEVICE, MODEL_VOCAB_INV

    if MODEL_INSTANCE is not None:
        print(f"AI Model ('{model_filename}') already loaded.")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_full_path = os.path.join(base_dir, "trained_models", model_filename)

    print(f"Attempting to load AI model from: {model_full_path} onto device: {DEVICE}")

    try:
        # Build the inverse vocabulary for debugging any token ID
        MODEL_VOCAB_INV = {**UCI_IDS, **ID_TO_FEN_CHAR, **ID_TO_SPECIAL}

        cfg_vocab_size_trained = 2008 # From notebook
        cfg_pad_id_trained = SPECIAL_TOKENS.get("<pad>", 2006) # from model_specific_utils via SPECIAL_TOKENS
        cfg_d_model = 1024
        cfg_d_ff = 4096
        cfg_num_layers = 8
        cfg_block_size_trained = 257 # max_len from sampling notebook (256+1)

        MODEL_TRAINING_CONFIG['pad_id'] = cfg_pad_id_trained
        MODEL_TRAINING_CONFIG['block_size'] = cfg_block_size_trained
        # vocab_size_trained not directly stored in MODEL_TRAINING_CONFIG, but used for model init

        MODEL_INSTANCE = AutoregressiveTransformer(
            vocab_size=cfg_vocab_size_trained,
            pad_id=cfg_pad_id_trained,
            d_model=cfg_d_model,
            d_ff=cfg_d_ff,
            num_layers=cfg_num_layers,
            max_len=cfg_block_size_trained,
        ).to(DEVICE)

        state_dict_checkpoint = torch.load(model_full_path, map_location=DEVICE)
        actual_model_state = state_dict_checkpoint['model_state']
        MODEL_INSTANCE.load_state_dict(actual_model_state)
        MODEL_INSTANCE.eval()

        INVERSE_UCI_MOVES = UCI_IDS # UCI_IDS is already the inverse: {token_id: uci_string}
        print(f"AI Model ('{model_filename}') loaded successfully to {DEVICE}.")

    except FileNotFoundError:
        print(f"ERROR: Trained model file not found at '{model_full_path}'.")
        MODEL_INSTANCE = None
    except RuntimeError as e:
        print(f"ERROR: Runtime error loading model state_dict from '{model_full_path}'.")
        print(f"Details: {e}")
        MODEL_INSTANCE = None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading the AI model: {e}")
        import traceback
        traceback.print_exc()
        MODEL_INSTANCE = None

def get_ai_prediction(current_fen: str, current_seq_ids: list[int], current_seq_tensor: torch.Tensor) -> tuple[str | None, torch.Tensor | None]:
    global MODEL_INSTANCE, INVERSE_UCI_MOVES, MODEL_TRAINING_CONFIG, DEVICE, SPECIAL_TOKENS, MODEL_VOCAB_INV

    if MODEL_INSTANCE is None or not MODEL_TRAINING_CONFIG or SPECIAL_TOKENS is None:
        print("AI Model, Model Config, or Special Tokens not loaded. Cannot get AI move.")
        return None, current_seq_tensor

    try:
        top_k_sampling = 10
        temperature_sampling = 1.0
        alpha_sampling = 30
        mask_illegal_moves = True

        MODEL_INSTANCE.eval()
        
        # sample_move_from_model is imported from .utils
        # It uses tokenize_fen from utils, which uses FEN_CHAR_TO_ID from model_specific_utils
        predicted_token_id, updated_seq_tensor = sample_move_from_model(
            model=MODEL_INSTANCE,
            seq=current_seq_ids, # list of token IDs
            seq_tensor=current_seq_tensor,
            top_k=top_k_sampling,
            temperature=temperature_sampling,
            alpha=alpha_sampling,
            mask_illegal=mask_illegal_moves,
            last_fen=current_fen
        )

        if predicted_token_id is None:
            print("Warning: sample_move_from_model returned None for token_id.")
            return None, updated_seq_tensor

        if isinstance(predicted_token_id, str) and predicted_token_id == "<end>":
            print("DEBUG: Model predicted <end>.")
            return "<end>", updated_seq_tensor

        predicted_uci_move = INVERSE_UCI_MOVES.get(predicted_token_id) # INVERSE_UCI_MOVES is UCI_IDS

        if predicted_uci_move:
            # print(f"DEBUG: Model sampled token ID {predicted_token_id}, maps to UCI: {predicted_uci_move}")
            return predicted_uci_move, updated_seq_tensor
        else:
            predicted_token_str = MODEL_VOCAB_INV.get(predicted_token_id, f"UnknownTokenID({predicted_token_id})")
            print(f"Warning: Model sampled token ID {predicted_token_id} ('{predicted_token_str}'), not a valid UCI move. Fallback needed.")
            return None, updated_seq_tensor

    except Exception as e:
        print(f"Error during AI move generation (get_ai_prediction): {e}")
        import traceback
        traceback.print_exc()
        return None, current_seq_tensor
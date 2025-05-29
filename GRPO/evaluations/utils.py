
import torch
import chess
import chess.engine
import cairosvg
import json
import imageio.v2 as imageio
import os
from IPython.display import Video, display
from tempfile import TemporaryDirectory
import torch.nn.functional as F
import imageio.v2 as imageio
from contextlib import nullcontext
import random
from ..vocab import SPECIAL_TOKENS, UCI_IDS, UCI_MOVES
from ..utils import set_seeds, extract_fen_from_game
from ..tokenize import tokenize_fen

# TODO: Separate eval functions to eval.py instead of having them in utils.py
LICHESS_LEVELS = {
    1: {"skill_level": -9, "depth": 5, "time_constraint": .050},
    2: {"skill_level": -5, "depth": 5, "time_constraint": .100},
    3: {"skill_level": -1, "depth": 5, "time_constraint": .150},
    4: {"skill_level": 3, "depth": 5, "time_constraint": .200}
}

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

# TODO: Refactor to just use the model method instead of having this function
def sample_move_from_model(model, seq, seq_tensor,
                           top_k=10, temperature=1.0, alpha=None,
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


def model_vs_engine(model,
                    engine_type="fairy",
                    model_color="w",
                    max_moves=50,
                    skill_level=1,
                    depth = 15,
                    time_constraint = 100,
                    record=False,
                    show=False,
                    video_path="game.mp4",
                    *sample_args,
                    **sample_kwargs):
    device = next(model.parameters()).device

    board = chess.Board()
    fen = board.fen()
    seq = [SPECIAL_TOKENS["<board>"]] + tokenize_fen(fen) + [
        SPECIAL_TOKENS["</board>"], SPECIAL_TOKENS["<moves>"]]
    seq_tensor = torch.tensor(seq, device=device).unsqueeze(0)

    board = chess.Board(fen)

    if engine_type == "stockfihs":
        engine = chess.engine.SimpleEngine.popen_uci("/content/stockfish-ubuntu-x86-64-sse41-popcnt")
    elif engine_type == "fairy":
        engine = chess.engine.SimpleEngine.popen_uci("/content/fairy-stockfish-largeboard_x86-64")
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")

    engine.configure({
                    "Skill Level": skill_level,
                    "Threads": 8,
                    "Hash": 8000,})





    ctx = TemporaryDirectory() if record else nullcontext()
    with ctx as tmpdir:
        if record:
            game = chess.pgn.Game()
            last_move = None
            frame_duration = 1.2
            fps = 1.0 / frame_duration
            frames = []
            svg0 = chess.svg.board(board=board, size=350, lastmove=last_move)
            p0, png0 = os.path.join(tmpdir, "frame_000.svg"), os.path.join(tmpdir, "frame_000.png")
            open(p0, "w").write(svg0); cairosvg.svg2png(url=p0, write_to=png0)
            frames.append(png0)

        for _ in range(2 * max_moves):
            if board.is_game_over():
                break
            if board.turn == chess.WHITE:
                if model_color == "w":
                    token, seq_tensor = sample_move_from_model(
                        model, seq, seq_tensor, last_fen=board.fen(),
                        *sample_args, **sample_kwargs)
                    move = UCI_IDS.get(token, None)
                    if move in ("<end>", None):
                        break
                    board.push_uci(move)
                    seq.append(token)
                    seq_tensor = torch.cat([seq_tensor,
                                            torch.tensor([[token]], device=device)], dim=1)
                    if record:
                        game.add_variation(move)
                        svg_i = chess.svg.board(board=board, size=350, lastmove=last_move)
                        pi, pngi = os.path.join(tmpdir, f"frame_{_:03}.svg"), os.path.join(tmpdir, f"frame_{_:03}.png")
                        open(pi, "w").write(svg_i); cairosvg.svg2png(url=pi, write_to=pngi)
                        frames.append(pngi)
                else:  # model_color == "b"
                    move = engine.play(board, chess.engine.Limit(depth=depth, time=time_constraint)).move.uci()
                    board.push_uci(move)
                    seq.append(UCI_MOVES[move])
                    seq_tensor = torch.cat([seq_tensor,
                                            torch.tensor([[UCI_MOVES[move]]], device=device)], dim=1)
                    if record:
                        game.add_variation(move)
                        svg_i = chess.svg.board(board=board, size=350, lastmove=last_move)
                        pi, pngi = os.path.join(tmpdir, f"frame_{_:03}.svg"), os.path.join(tmpdir, f"frame_{_:03}.png")
                        open(pi, "w").write(svg_i); cairosvg.svg2png(url=pi, write_to=pngi)
                        frames.append(pngi)
            else:  # chess.BLACK
                if model_color == "b":
                    token, seq_tensor = sample_move_from_model(
                        model, seq, seq_tensor, last_fen=board.fen(),
                        *sample_args, **sample_kwargs)
                    move = UCI_IDS.get(token, None)
                    if move == "<end>":
                        break
                    board.push_uci(move)
                    seq.append(token)
                    seq_tensor = torch.cat([seq_tensor,
                                            torch.tensor([[token]], device=device)], dim=1)
                    if record:
                        game.add_variation(move)
                        svg_i = chess.svg.board(board=board, size=350, lastmove=last_move)
                        pi, pngi = os.path.join(tmpdir, f"frame_{_:03}.svg"), os.path.join(tmpdir, f"frame_{_:03}.png")
                        open(pi, "w").write(svg_i); cairosvg.svg2png(url=pi, write_to=pngi)
                        frames.append(pngi)
                else:  # model_color == "w"
                    move = engine.play(board, chess.engine.Limit(depth=depth, time=time_constraint)).move.uci()
                    board.push_uci(move)
                    seq.append(UCI_MOVES[move])
                    seq_tensor = torch.cat([seq_tensor,
                                            torch.tensor([[UCI_MOVES[move]]], device=device)], dim=1)
                    if record:
                        game.add_variation(move)
                        svg_i = chess.svg.board(board=board, size=350, lastmove=last_move)
                        pi, pngi = os.path.join(tmpdir, f"frame_{_:03}.svg"), os.path.join(tmpdir, f"frame_{_:03}.png")
                        open(pi, "w").write(svg_i); cairosvg.svg2png(url=pi, write_to=pngi)
                        frames.append(pngi)

        if record:
            with imageio.get_writer(video_path, format='ffmpeg', fps=fps) as writer:
                for p in frames:
                    writer.append_data(imageio.imread(p))
    if show:
        display(Video(video_path, embed=True, html_attributes="controls autoplay loop"))
        print(f"✅ video saved to {video_path} @ {fps:.2f} fps")
    engine.quit()

    return board.result()


def evaluate_model_vs_engine(model,
                        n_games = 100,
                        evals_path = "/content/drive/MyDrive/A_Chess_Transformer/Evals/",
                        save_every = 25,
                        engine_config_dict = LICHESS_LEVELS,
                        engine_type="fairy",
                        *sample_args,
                        **sample_kwargs):

    os.makedirs(evals_path + f'model_eval_vids', exist_ok=True)
    results_across_elo = {tier: None for tier in engine_config_dict}
    for tier, config in engine_config_dict.items():
      skill_level, depth, time_constraint = config.values()
      results = {'win': 0, 'lose': 0, 'draw': 0}
      for i in range(n_games // 2 + 1): # white games
          record = False
          if i % save_every == 0:
            video_path = evals_path + f'model_eval_vids/game_{i}_col=w_tier={tier}.mp4'
            record = True
          result = model_vs_engine(model,
                                           engine_type = engine_type,
                                           model_color='w',
                                           skill_level=skill_level,
                                           depth=depth,
                                           time_constraint=time_constraint,
                                           record = record,
                                           video_path=video_path,
                                           *sample_args,
                                           **sample_kwargs)
          if result == '1-0':
            results['win'] += 1
          elif result == '0-1':
            results['lose'] += 1
          else:
            results['draw'] += 1

      for i in range(n_games // 2 + 1): # black games
          record = False
          if i % save_every == 0:
              video_path = evals_path + f'model_eval_vids/game_{i}_col=b_tier={tier}.mp4'
              record = True

          result = model_vs_engine(model,
                                            engine_type = engine_type,
                                            model_color='b',
                                            skill_level=skill_level,
                                            depth=depth,
                                            time_constraint=time_constraint,
                                            record = record,
                                            video_path=video_path,
                                            *sample_args,
                                            **sample_kwargs)

          if result == '0-1':
            results['win'] += 1
          elif result == '1-0':
            results['lose'] += 1
          else:
            results['draw'] += 1

      results_across_elo[tier] = results

    with open(evals_path + 'results_across_elo.json', 'w') as f:
      json.dump(results_across_elo,f)


    return results_across_elo
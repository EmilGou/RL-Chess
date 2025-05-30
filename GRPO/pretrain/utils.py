from uci_moves import UCI_MOVES
import math
import torch
import chess
import chess.svg
import chess.pgn
import torch.nn.functional as F
import cairosvg
import imageio.v2 as imageio
import os
import random
import base64
from typing import Optional
from tempfile import TemporaryDirectory
from IPython.display import SVG, display, Video, HTML

from GRPO.tokenize import tokenize_fen, tokenize_uci, SPECIAL_TOKENS, UCI_IDS, UCI_MOVES


def uci_moves_to_fen(moves: list[str], show_board: bool = False) -> str:
    """
    Given a list of UCI moves, returns the resulting FEN and optionally shows the board.
    Args:
        moves: List of moves in UCI format, e.g., ['e2e4', 'e7e5']
        show_board: If True, display the board with IPython SVG (for notebooks)
    Returns:
        FEN string of the final board position
    """
    board = chess.Board()
    try:
        for move in moves:
            board.push_uci(move)
    except ValueError as e:
        raise ValueError(f"Invalid move '{move}': {e}")

    if show_board:
        display(SVG(chess.svg.board(board=board)))

    return board.fen()

@torch.no_grad()
def sample_game_to_video(
    model,
    max_moves: int = 50,
    temperature: float = 1.0,
    top_k: int = 10,
    video_path: str = "sample_game.mp4",
    frame_duration: float = 1.2
) -> Optional[chess.pgn.Game]:
    model.eval()
    device = next(model.parameters()).device

    board = chess.Board()

    fen_ids = tokenize_fen(board.fen())
    seq = ([SPECIAL_TOKENS["<board>"]] +
           fen_ids +
           [SPECIAL_TOKENS["</board>"], SPECIAL_TOKENS["<moves>"]])
    gen = torch.tensor(seq, device=device).unsqueeze(0)[0].tolist()

    game = chess.pgn.Game()
    last = None

    fps = 1.0/frame_duration
    with TemporaryDirectory() as tmp:
        frames = []
        svg0 = chess.svg.board(board=board, size=350, lastmove=last)
        p0   = os.path.join(tmp, "f000.svg")
        png0 = os.path.join(tmp, "f000.png")
        open(p0,"w").write(svg0)
        cairosvg.svg2png(url=p0, write_to=png0)
        frames.append(png0)

        for i in range(1, max_moves+1):
            x = torch.tensor(gen, device=device).unsqueeze(0)
            lg = model(x)[0,-1,:]/temperature
            if top_k:
                v,iid = torch.topk(lg, top_k)
                pr = F.softmax(v, dim=0)
                tok = iid[torch.multinomial(pr,1)].item()
            else:
                pr = F.softmax(lg,dim=-1)
                tok = torch.multinomial(pr,1).item()
            if tok in (SPECIAL_TOKENS["</moves>"], SPECIAL_TOKENS["<pad>"]):
                break
            gen.append(tok)
            if tok not in UCI_IDS:
                print("unk", tok); break
            m = chess.Move.from_uci(UCI_IDS[tok])
            if not board.is_legal(m):
                print("illegal", UCI_IDS[tok]); break
            board.push(m); last=m; game.add_variation(m)

            svgi = chess.svg.board(board=board, size=350, lastmove=last)
            pi = os.path.join(tmp, f"f{i:03}.svg")
            pngi = os.path.join(tmp, f"f{i:03}.png")
            open(pi,"w").write(svgi)
            cairosvg.svg2png(url=pi, write_to=pngi)
            frames.append(pngi)

        writer = imageio.get_writer(video_path, format="ffmpeg", fps=fps)
        for p in frames:
            writer.append_data(imageio.imread(p))
        writer.close()

    display(Video(video_path, embed=True, html_attributes="controls autoplay loop"))
    print(f"Video saved @ {fps:.2f} fps → {video_path}")
    return game

@torch.no_grad()
def sample_game_masked(
    model,
    max_moves: int = 50,
    temperature: float = 1.0,
    video_path: str = "sample_game_masked.mp4",
    frame_duration: float = 1.2  # seconds per frame
) -> Optional[chess.pgn.Game]:
    """
    Samples a full game, but at each step:
      • Enumerates board.legal_moves
      • Converts them to your UCI token IDs
      • Masks out all other logits
      • Samples from the remaining legal‐move distribution
    Saves as MP4 at fps = 1/frame_duration.
    """
    model.eval()
    device = next(model.parameters()).device

    board = chess.Board()

    fen_ids = tokenize_fen(board.fen())
    seq = [SPECIAL_TOKENS["<board>"]] + fen_ids + [
        SPECIAL_TOKENS["</board>"], SPECIAL_TOKENS["<moves>"]
    ]
    generated = seq.copy()
    game = chess.pgn.Game()
    last_move = None

    fps = 1.0 / frame_duration

    with TemporaryDirectory() as tmpdir:
        frames = []
  
        svg0 = chess.svg.board(board=board, size=350, lastmove=last_move)
        p0   = os.path.join(tmpdir, "frame_000.svg")
        png0 = os.path.join(tmpdir, "frame_000.png")
        open(p0, "w").write(svg0)
        cairosvg.svg2png(url=p0, write_to=png0)
        frames.append(png0)

        for i in range(1, max_moves+1):
            x = torch.tensor(generated, device=device).unsqueeze(0)
            logits = model(x)[0, -1, :] / temperature  # (V,)

            legal_ids = []
            for mv in board.legal_moves:
                uci = mv.uci()
                if uci in UCI_MOVES:
                    legal_ids.append(UCI_MOVES[uci])
            if not legal_ids:
                print("No legal moves tokenized – stopping.")
                break

            mask = torch.full_like(logits, float('-inf'))
            mask[legal_ids] = 0.0
            filtered_logits = logits + mask

            probs = F.softmax(filtered_logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1).item()

            if token in (SPECIAL_TOKENS["</moves>"], SPECIAL_TOKENS["<pad>"]):
                break
            generated.append(token)

            uci = UCI_IDS.get(token, None)
            if uci is None:
                print(f"Generated unknown token {token}.")
                break
            move = chess.Move.from_uci(uci)
            if not board.is_legal(move):
                print(f"Illegal generated move: {uci}.")
                break

            board.push(move)
            last_move = move
            game.add_variation(move)

            svg_i = chess.svg.board(board=board, size=350, lastmove=last_move)
            pi   = os.path.join(tmpdir, f"frame_{i:03}.svg")
            pngi = os.path.join(tmpdir, f"frame_{i:03}.png")
            open(pi, "w").write(svg_i)
            cairosvg.svg2png(url=pi, write_to=pngi)
            frames.append(pngi)

        with imageio.get_writer(video_path, format='ffmpeg', fps=fps) as writer:
            for p in frames:
                writer.append_data(imageio.imread(p))

    display(Video(video_path, embed=True, html_attributes="controls autoplay loop"))
    print(f"Masked‐sampling video saved to {video_path} @ {fps:.2f} fps")
    return game



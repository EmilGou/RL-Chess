
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
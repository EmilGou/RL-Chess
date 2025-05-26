import chess.engine
from .utils import extract_fen_from_game

# TODO: Formulate this reward function to work properly
def reward_fn(batch_prompts: list[str],
              batch_completions: list[str],
              engine_path: str,
              limit: int = 3,
              depth: int = 12,
              ) -> list[float]:



    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    rewards = []

    for prompt, completion in zip(batch_prompts, batch_completions):
        fen_txt = extract_fen_from_game(prompt)
        board = chess.Board(fen_txt)
        

        agent = board.turn 

        base  = engine.analyse(board, chess.engine.Limit(depth=depth))
        base_eval  = base["score"].pov(agent).score(mate_score=10000)

        board.push(move)
        after = engine.analyse(board, chess.engine.Limit(depth=depth))
        after_eval = after["score"].pov(agent).score(mate_score=10000)
        board.pop()

        delta = max(min((after_eval - base_eval) / 100.0,  limit), -limit)
        rewards.append(float(delta))

    return rewards
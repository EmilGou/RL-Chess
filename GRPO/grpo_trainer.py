import torch
import chess
import chess.engine
from dataclasses import dataclass
from .utils import selective_log_softmax, nanmax, nanmin, extract_fen_from_game
from .vocab import UCI_IDS

@dataclass
class GRPOArgs:
    epsilon_low: float = 0.2
    epsilon_high: float = 0.2
    temperature: float = 1.0
    beta: float = 0.03
    loss_type: str = "grpo"
    num_generations: int = 4
    num_moves: int = 10
    total_steps: int = 10000
    log_every: int = 5
    save_every: int = 500
    device: str   = "cuda"
    engine_path: str = "stockfish/stockfish-ubuntu-x86-64-sse41-popcnt"

# TODO: If time, clean up the init a little bit
class GRPOTrainer:
    def __init__(self, model, ref_model, args):
        self.pad_id = model.pad_id
        self.model = model
        self.ref_model = ref_model
        self.epsilon_low = args.epsilon_low
        self.epsilon_high = args.epsilon_high
        self.temperature = args.temperature
        self.beta = args.beta
        self.loss_type = args.loss_type
        self.num_generations = args.num_generations
        self.num_moves = args.num_moves
        self.total_steps = args.total_steps
        self.log_every = args.log_every
        self.save_every = args.save_every
        self.device = args.device
        self._metrics = {"train": {}, "eval": {}}
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        self.global_step = 0
        self.engine = chess.engine.SimpleEngine.popen_uci(args.engine_path)

    def step(self, batch):

        loss = self._compute_loss(self.model, batch)   # (scalar tensor)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log
        loss_val = loss.detach().cpu().item()
        self._metrics["train"].setdefault("loss", []).append(loss_val)

        self.global_step += 1
        if self.global_step % self.log_every == 0:
            print(f"step {self.global_step:>6} | loss {loss_val:8.4f}")

        return loss_val
    
    def train(self, dataloader, engine_path):
        """
        dataloader yields prompt tensors of shape (B,T) on CPU
        engine_path path to Stockfish binary for rollouts
        """
        while self.global_step < self.total_steps:
            for prompts in dataloader:
                prompts = prompts.to(self.device)
                # rollout → batch dict
                
                batch = self._generate_completions_and_score(
                    prompts,
                    engine_path   = engine_path,
                    depth         = 12,
                    num_generations = self.num_generations,
                    num_moves       = self.num_moves,
                    limit           = 3,
                )
                self.step(batch)

                if self.global_step >= self.total_steps:
                    break


        
    def _get_per_token_logps(self, model, input_ids, logits_to_keep, batch_size=None) -> torch.Tensor:
            was_training = model.training
            model.eval()  # set model to eval mode
            batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
            all_logps = []
            for i in range(0, input_ids.size(0), batch_size):
                input_ids_batch = input_ids[i : i + batch_size]

                logits = model(input_ids_batch)
                logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
                input_ids_batch = input_ids_batch[:, -logits_to_keep:]
                # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
                # See https://github.com/huggingface/trl/issues/2770
                logits = logits[:, -logits_to_keep:]
                # Divide logits by sampling temperature.
                # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
                logits = logits / self.temperature
                logps = selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
                all_logps.append(logps)
            model.train(was_training)
            return torch.cat(all_logps, dim=0)


    def _compute_loss(self, inputs):

            mode = "train" if self.model.training else "eval"

            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            input_ids = inputs["input_ids"]
        
            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

            per_token_logps = self._get_per_token_logps(self.model, input_ids, logits_to_keep)

            # Compute the KL divergence between the model and the reference model
            if self.beta != 0.0:
                with torch.no_grad():
                
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, logits_to_keep
                    )
                    
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                )

            # Compute the loss
            advantages = inputs["advantages"]

            old_per_token_logps = per_token_logps.detach() 
        
            coef_1 = torch.exp(per_token_logps - old_per_token_logps)
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

          
            # Original GRPO clipping (only lower bound implicitly applied by the final min)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            reward = -((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            # log reward
            self._metrics[mode].setdefault("reward", []).append(reward.item())

            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl

            if self.loss_type == "grpo":
                loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            elif self.loss_type == "bnpo":
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            elif self.loss_type == "dr_grpo":
                loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.num_moves)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            

            if self.beta != 0.0:
                mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
                self._metrics[mode].setdefault("kl", []).append(mean_kl.nanmean().item())


            is_low_clipped    = (coef_1 < 1 - self.epsilon_low)  & (advantages < 0)
            is_high_clipped   = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip   = (is_low_clipped    * completion_mask).sum() / completion_mask.sum()
            high_clip  = (is_high_clipped   * completion_mask).sum() / completion_mask.sum()
            clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

            # 3. record metrics ----------------------------------------------------------
            metrics = self._metrics[mode]  # shorthand

            metrics.setdefault("clip_ratio/low_mean",   []).append(low_clip.nanmean().item())
            metrics.setdefault("clip_ratio/low_min",    []).append(nanmin(low_clip).item())

            metrics.setdefault("clip_ratio/high_mean",  []).append(high_clip.nanmean().item())
            metrics.setdefault("clip_ratio/high_max",   []).append(nanmax(high_clip).item())

            metrics.setdefault("clip_ratio/region_mean", []).append(clip_ratio.nanmean().item())
            
            return loss

    def _pad(self, seqs):
        max_len = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [self.pad_id] * (max_len - len(s)) for s in seqs],
            dtype=torch.long,
        )

    @torch.no_grad()
    def _generate_completions_and_score(
        self,
        prompt_ids,                  # (B, T)
        engine_path,
        depth           = 10,        # for Stockfish analyse
        num_generations = 4,
        num_moves       = 10,
        limit = 2
    ):
        was_training = self.model.training
        self.model.eval()
        self.ref_model.eval()
        device, pad_id = prompt_ids.device, self.model.pad_id

        # 1) repeat prompts: (B, T) → (B·G, T)
        B, T       = prompt_ids.shape
        input_ids  = prompt_ids.repeat_interleave(num_generations, dim=0)
        batch_size = input_ids.size(0)                 # B·G
        seqs       = input_ids.tolist()

        # 2) boards
        boards = [chess.Board(extract_fen_from_game(s)) for s in seqs]

        # 3) Stockfish eval before roll‑outs

        base_eval = torch.tensor(
            [
                self.engine.analyse(b, chess.engine.Limit(depth=depth))["score"]
                    .pov(b.turn).score(mate_score=10000)
                for b in boards
            ],
            dtype=torch.float32, device=device
        )

        # establish model color
        turns = [b.turn for b in boards]

        # 4) roll‑out:  num_moves × 2 half‑moves
        completions, completion_masks = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]
        pair_rewards = []
        for _ in range(num_moves):
            # 4‑a) *policy* move  (mask = 1)
            move_ids, input_ids = self.model.predict_move(
                seqs,
                self._pad(seqs).to(device),
                boards,
            )
            for i, tok in enumerate(move_ids):
                completions[i].append(tok)
                completion_masks[i].append(1 if tok != pad_id else 0)
                if tok != pad_id and not boards[i].is_game_over():
                    boards[i].push_uci(UCI_IDS[tok])
                seqs[i].append(tok)

            # 4‑b) *model* response move  (mask = 0)
            resp_ids, input_ids = self.ref_model.predict_move(
                seqs,
                self._pad(seqs).to(device),
                boards,
            )
            for i, tok in enumerate(resp_ids):
                completions[i].append(tok)
                completion_masks[i].append(0)           # never optimised
                if tok != pad_id and not boards[i].is_game_over():
                    boards[i].push_uci(UCI_IDS[tok])
                seqs[i].append(tok)

            # 5) Stockfish eval after roll‑outs
            after_eval = torch.tensor(
                [
                    self.engine.analyse(b, chess.engine.Limit(depth=depth))["score"]
                        .pov(turns[i]).score(mate_score=10000)
                    for i, b in enumerate(boards)
                ],
                dtype=torch.float32, device=device
            )
            pair_rewards.append((after_eval - base_eval) / 100.0)
            base_eval = after_eval  # update for the next round


        # 6) centred rewards → advantages

        delta   = torch.stack(pair_rewards, dim=1)                 # (B·G, M)
        delta   = torch.clamp(delta, -limit, limit)

        rewards   = delta.view(B, num_generations, num_moves)      # (B, G, M)
        centered  = rewards - rewards.mean(dim=1, keepdim=True)    # baseline

        expanded  = centered.reshape(B*num_generations, num_moves) # (B·G, M)
        expanded  = expanded.repeat_interleave(2, dim=1)           # (B·G, M*2)
        expanded[:, 1::2] = 0                                       # mask replies

        
        # 7) pad everything for the backward pass
        input_ids  = self._pad(seqs).to(device)                        
        max_L      = max(len(c) for c in completions)
        completion_ids  = torch.full((batch_size, max_L), pad_id,
                                    dtype=torch.long, device=device)
        completion_mask = torch.zeros_like(completion_ids, dtype=torch.float32)
            # Zero‑out tokens that are masked out (responses & padding)

        for i, (c, m) in enumerate(zip(completions, completion_masks)):
            L = len(c)
            completion_ids[i, :L]  = torch.tensor(c, dtype=torch.long, device=device)
            completion_mask[i, :L] = torch.tensor(m, dtype=torch.float32, device=device)

        advantages = torch.flip(torch.cumsum(torch.flip(expanded, [1]), dim=1), [1])

        # … build completion_ids / completion_mask as before …
        advantages *= completion_mask

        self.model.train(was_training)  # set model back to train mode

        return {
            "input_ids"       : input_ids,       
            "completion_ids"  : completion_ids,   
            "completion_mask" : completion_mask,  
            "advantages"      : advantages,       
        }
            
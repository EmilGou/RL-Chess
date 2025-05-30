# python-chess-UI/src/game.py
import pygame
import chess
import random
import os
import torch

from const import *
from board import Board
from piece import Piece
from dragger import Dragger
from config import Config

from ai_engine.ai_interface import (
    get_ai_prediction, DEVICE, SPECIAL_TOKENS, UCI_MOVES, MODEL_TRAINING_CONFIG,
    tokenize_fen_from_utils as tokenize_fen
)

AI_MOVE_EVENT = pygame.USEREVENT + 1

class Game:
    def __init__(self, config: Config):
        self.config = config
        self.chess_board = chess.Board()
        self.board = Board()
        self.dragger = Dragger()

        self.move_sound_obj = None
        self.capture_sound_obj = None
        try:
            if not pygame.mixer.get_init(): pygame.mixer.init()
            if self.config.move_sound_path and os.path.exists(self.config.move_sound_path):
                self.move_sound_obj = pygame.mixer.Sound(self.config.move_sound_path)
            if self.config.capture_sound_path and os.path.exists(self.config.capture_sound_path):
                self.capture_sound_obj = pygame.mixer.Sound(self.config.capture_sound_path)
        except Exception as e: print(f"Error initializing sounds in Game: {e}")

        self.hovered_sqr_ui_pos = None
        self.playing = True
        self.ai_is_black = self.config.ai_player_is_black

        if self.ai_is_black:
            print("Game Setup: Human plays White, AI (Model) plays Black.")
        else:
            print("Game Setup: Human plays Black, AI (Model) plays White.")
        
        self._initialize_model_sequence_if_needed()
        # print(f"Initial FEN: {self.chess_board.fen()}") # Already printed by console

    def _initialize_model_sequence_if_needed(self):
        if hasattr(self.config, 'ai_player_is_black'):
            starting_fen = self.chess_board.fen()
            self.current_seq_ids = [SPECIAL_TOKENS["<board>"]] + tokenize_fen(starting_fen) + \
                                   [SPECIAL_TOKENS["</board>"], SPECIAL_TOKENS["<moves>"]]
            self.current_seq_tensor = torch.tensor([self.current_seq_ids], dtype=torch.long).to(DEVICE)

    def _update_model_sequence_after_move(self, chess_move_obj: chess.Move, updated_tensor_from_ai: torch.Tensor | None = None):
        if not hasattr(self, 'current_seq_ids'): return
        move_uci = chess_move_obj.uci()
        if move_uci not in UCI_MOVES: return
        move_token_id = UCI_MOVES[move_uci]

        self.current_seq_ids.append(move_token_id)
        base_tensor_for_concat = updated_tensor_from_ai if updated_tensor_from_ai is not None else self.current_seq_tensor
        new_token_tensor = torch.tensor([[move_token_id]], dtype=torch.long).to(DEVICE)
        self.current_seq_tensor = torch.cat((base_tensor_for_concat, new_token_tensor), dim=1)
        
        # block_size = MODEL_TRAINING_CONFIG.get('block_size', 257)
        # if self.current_seq_tensor.shape[1] > block_size:
            # print(f"Warning: Sequence tensor length {self.current_seq_tensor.shape[1]} exceeds block size {block_size}.")

    def _ui_sq_to_chess_sq(self, ui_row: int, ui_col: int) -> chess.Square | None:
        if not (0 <= ui_row < ROWS and 0 <= ui_col < COLS): return None
        return chess.square(ui_col, (ROWS - 1) - ui_row)

    def _chess_sq_to_ui_sq(self, chess_sq: chess.Square) -> tuple[int, int] | None:
        if not (0 <= chess_sq < 64): return None
        return (ROWS - 1) - chess.square_rank(chess_sq), chess.square_file(chess_sq)

    def _uci_to_chess_move(self, uci_move_str: str) -> chess.Move | None:
        try: return chess.Move.from_uci(uci_move_str)
        except: return None

    
    def handle_player_drag_release(self, initial_ui_row_col: tuple[int, int], final_ui_row_col: tuple[int, int]):
        # print(f"DEBUG Game: handle_player_drag_release called with initial=({initial_ui_row_col}), final=({final_ui_row_col})")
        initial_ui_row, initial_ui_col = initial_ui_row_col
        final_ui_row, final_ui_col = final_ui_row_col

        # Correctly determine human player's color for turn checking
        if self.ai_is_black: # AI is Black, so Human is White
            human_color_chess = chess.WHITE
        else: # AI is White, so Human is Black
            human_color_chess = chess.BLACK
        
        # Debug print to verify the derived human color
        # print(f"DEBUG Game: handle_player_drag_release - ai_is_black: {self.ai_is_black}, Derived human_color_chess: {'WHITE' if human_color_chess == chess.WHITE else 'BLACK'}")

        if not self.playing or not (self.chess_board.turn == human_color_chess):
            # This print was in your debug output, let's keep it for now
            print(f"DEBUG Game: handle_player_drag_release - Not playing or not human's turn. Playing: {self.playing}, Board Turn: {'WHITE' if self.chess_board.turn == chess.WHITE else 'BLACK'}, Expected Human Color: {'WHITE' if human_color_chess == chess.WHITE else 'BLACK'}")
            self.dragger.undrag_piece()
            return

        from_chess_sq = self._ui_sq_to_chess_sq(initial_ui_row, initial_ui_col)
        to_chess_sq = self._ui_sq_to_chess_sq(final_ui_row, final_ui_col)
        # print(f"DEBUG Game: Converted UI to chess squares: from_sq={from_chess_sq}, to_sq={to_chess_sq}")

        if from_chess_sq is None or to_chess_sq is None or from_chess_sq == to_chess_sq:
            # print(f"DEBUG Game: Invalid from/to square or same square. from_sq={from_chess_sq}, to_sq={to_chess_sq}")
            self.dragger.undrag_piece()
            return

        promotion_piece_type = None
        piece_on_from_sq = self.chess_board.piece_at(from_chess_sq)
        if piece_on_from_sq and piece_on_from_sq.piece_type == chess.PAWN:
            target_rank = chess.square_rank(to_chess_sq)
            if (piece_on_from_sq.color == chess.WHITE and target_rank == 7) or \
               (piece_on_from_sq.color == chess.BLACK and target_rank == 0):
                promotion_piece_type = chess.QUEEN
        
        chess_move_attempt = chess.Move(from_chess_sq, to_chess_sq, promotion=promotion_piece_type)
        # print(f"DEBUG Game: Constructed chess.Move: {chess_move_attempt.uci()}")

        is_legal_on_board = self.chess_board.is_legal(chess_move_attempt)
        # print(f"DEBUG Game: Move {chess_move_attempt.uci()} is_legal_on_board? {is_legal_on_board}")

        if is_legal_on_board:
            # print(f"DEBUG Game: Move is legal. Pushing to board: {chess_move_attempt.uci()}")
            is_capture = self.chess_board.is_capture(chess_move_attempt)
            self.chess_board.push(chess_move_attempt)
            self.board.last_move = chess_move_attempt 
            self.play_sound(is_capture)
            self._update_model_sequence_after_move(chess_move_attempt, updated_tensor_from_ai=None)
            self.check_game_status() 
            if self.playing:
                # print("DEBUG Game: Move legal and game still playing, calling next_turn()")
                self.next_turn()
            # else:
                # print("DEBUG Game: Move legal BUT game is no longer playing after check_game_status.")
        # else:
            # print(f"DEBUG Game: Move {chess_move_attempt.uci()} is ILLEGAL.")
        
        # print("DEBUG Game: Calling dragger.undrag_piece() at end of handle_player_drag_release")
        self.dragger.undrag_piece()
    

    def trigger_ai_move(self):
        if not self.playing or self.chess_board.is_game_over(): return

        ai_color_chess = chess.BLACK if self.ai_is_black else chess.WHITE
        if self.chess_board.turn != ai_color_chess: return # Not AI's turn
            
        print(f"AI ({self.config.ai_color}) is thinking...")
        current_fen = self.chess_board.fen()
        
        seq_ids_copy = list(self.current_seq_ids)
        seq_tensor_copy = self.current_seq_tensor.clone()

        ai_uci_move_str, returned_seq_tensor = get_ai_prediction(
            current_fen, seq_ids_copy, seq_tensor_copy
        )
        
        if returned_seq_tensor is not None:
            self.current_seq_tensor = returned_seq_tensor
            self.current_seq_ids = seq_ids_copy # Reflect changes made by sample_move_from_model
        else: print("CRITICAL: AI prediction did not return an updated sequence tensor.")


        if ai_uci_move_str and ai_uci_move_str != "<end>":
            ai_chess_move = self._uci_to_chess_move(ai_uci_move_str)
            if ai_chess_move and self.chess_board.is_legal(ai_chess_move):
                
                print(f"SUCCESS: AI (Model) predicts and plays: {ai_uci_move_str}")
                
                is_capture = self.chess_board.is_capture(ai_chess_move)
                self.chess_board.push(ai_chess_move)
                self.board.last_move = ai_chess_move
                self.play_sound(is_capture)
                self._update_model_sequence_after_move(ai_chess_move, updated_tensor_from_ai=self.current_seq_tensor)
                self.check_game_status()
                if self.playing: self.next_turn()
            else:
                reason = "malformed/None" if not ai_chess_move else f"illegal ({ai_uci_move_str})"
                print(f"AI predicted an {reason} move. Fallback.")
                self._handle_ai_fallback(self.current_seq_tensor) 
        elif ai_uci_move_str == "<end>":
            print("AI signals end of game."); self.playing = False; self.check_game_status()
        else:
            print("AI did not provide a valid move. Fallback."); self._handle_ai_fallback(self.current_seq_tensor)

        self.dragger.undrag_piece()

    
    def show_pieces(self, surface): # CRITICAL: Ensures Piece object identity
        for r_ui in range(ROWS):
            for c_ui in range(COLS):
                ui_square = self.board.squares[r_ui][c_ui]
                chess_sq_idx = self._ui_sq_to_chess_sq(r_ui, c_ui)

                logical_chess_piece = self.chess_board.piece_at(chess_sq_idx)
                existing_ui_piece_on_square = ui_square.piece
                ui_piece_to_render_this_frame = None

                if logical_chess_piece:
                    color_str = 'white' if logical_chess_piece.color == chess.WHITE else 'black'
                    type_map = {
                        chess.PAWN: 'pawn', chess.KNIGHT: 'knight', chess.BISHOP: 'bishop',
                        chess.ROOK: 'rook', chess.QUEEN: 'queen', chess.KING: 'king'}
                    name_str = type_map.get(logical_chess_piece.piece_type, 'unknown')
                    texture_path = self.config.get_texture_for_piece(name_str, color_str)

                    if existing_ui_piece_on_square and \
                       existing_ui_piece_on_square.name == name_str and \
                       existing_ui_piece_on_square.color == color_str:
                        if existing_ui_piece_on_square.texture != texture_path:
                             existing_ui_piece_on_square.texture = texture_path
                        ui_piece_to_render_this_frame = existing_ui_piece_on_square
                    else: 
                        new_ui_piece = Piece(name_str, color_str, 0, texture_path)
                        ui_square.piece = new_ui_piece
                        ui_piece_to_render_this_frame = new_ui_piece
                else:
                    ui_square.piece = None
                    ui_piece_to_render_this_frame = None

                if ui_piece_to_render_this_frame:
                    is_this_exact_piece_being_dragged = (
                        self.dragger.dragging and
                        self.dragger.piece == ui_piece_to_render_this_frame and 
                        self.dragger.initial_row == r_ui and # Ensure drag started from this square
                        self.dragger.initial_col == c_ui)

                    if not is_this_exact_piece_being_dragged:
                        try:
                            if ui_piece_to_render_this_frame.texture and os.path.exists(ui_piece_to_render_this_frame.texture):
                                img = pygame.image.load(ui_piece_to_render_this_frame.texture)
                                img_center = c_ui * SQSIZE + SQSIZE // 2, r_ui * SQSIZE + SQSIZE // 2
                                ui_piece_to_render_this_frame.texture_rect = img.get_rect(center=img_center)
                                surface.blit(img, ui_piece_to_render_this_frame.texture_rect)
                        except Exception as e:
                            print(f"Error drawing piece {ui_piece_to_render_this_frame.name}: {e}")
    
    def next_turn(self):
        self.next_player_ui = 'white' if self.chess_board.turn == chess.WHITE else 'black'
        print(f"DEBUG Game: next_turn called. Current board turn is now: {self.next_player_ui.upper()}") # Add
        
        ai_color_chess = chess.BLACK if self.ai_is_black else chess.WHITE
        if self.playing and self.chess_board.turn == ai_color_chess:
            print(f"DEBUG Game: It's AI's turn. Setting AI_MOVE_EVENT timer.") # Add
            pygame.time.set_timer(AI_MOVE_EVENT, 100, loops=1) # AI moves quickly
        else:
            print(f"DEBUG Game: Not AI's turn OR game not playing. Playing: {self.playing}, Board Turn: {self.chess_board.turn}, AI Color: {ai_color_chess}") # Add


    def trigger_ai_move(self):
        if not self.playing or self.chess_board.is_game_over(): return

        ai_color_chess = chess.BLACK if self.ai_is_black else chess.WHITE
        if self.chess_board.turn != ai_color_chess: return
            
        print(f"AI ({self.config.ai_color}) is thinking...")
        current_fen = self.chess_board.fen()
        
        # Pass copies to prevent unintended modification if sample_move_from_model is destructive beyond its design
        seq_ids_copy = list(self.current_seq_ids)
        seq_tensor_copy = self.current_seq_tensor.clone()

        ai_uci_move_str, returned_seq_tensor = get_ai_prediction(
            current_fen, seq_ids_copy, seq_tensor_copy
        )
        
        # After get_ai_prediction, seq_ids_copy might have been modified by sample_move_from_model (due to alpha logic).
        # We should use this modified list as the basis for our new self.current_seq_ids if a move is made.
        if returned_seq_tensor is not None:
            self.current_seq_tensor = returned_seq_tensor # This is the base tensor for the *next* move token
            self.current_seq_ids = seq_ids_copy # This list was modified in place by sample_move_from_model
        else:
            print("CRITICAL: AI prediction did not return an updated sequence tensor.")
            # Potentially revert or handle error, for now, we might be in a bad state for next AI move

        if ai_uci_move_str and ai_uci_move_str != "<end>":
            ai_chess_move = self._uci_to_chess_move(ai_uci_move_str)
            if ai_chess_move and self.chess_board.is_legal(ai_chess_move):
                is_capture = self.chess_board.is_capture(ai_chess_move)
                self.chess_board.push(ai_chess_move)
                self.board.last_move = ai_chess_move
                self.play_sound(is_capture)
                
                # self.current_seq_tensor IS the tensor returned by get_ai_prediction,
                # which is the state *before* this ai_chess_move token.
                # self.current_seq_ids was also updated by sample_move_from_model.
                self._update_model_sequence_after_move(ai_chess_move, updated_tensor_from_ai=self.current_seq_tensor)
                
                self.check_game_status()
                if self.playing: self.next_turn()
            else:
                self._handle_ai_fallback(self.current_seq_tensor) # Pass the latest tensor
        elif ai_uci_move_str == "<end>":
            self.playing = False; self.check_game_status()
        else:
            self._handle_ai_fallback(self.current_seq_tensor)

        self.dragger.undrag_piece()

    def _handle_ai_fallback(self, base_tensor_for_update: torch.Tensor):
        if not self.playing or self.chess_board.is_game_over(): return
        legal_moves = list(self.chess_board.legal_moves)
        if legal_moves:
            fallback_move = random.choice(legal_moves)
            is_capture = self.chess_board.is_capture(fallback_move)
            print(f"AI Fallback plays: {fallback_move.uci()}")
            self.chess_board.push(fallback_move)
            self.board.last_move = fallback_move
            self.play_sound(is_capture)
            # Update sequence with the fallback move
            self._update_model_sequence_after_move(fallback_move, updated_tensor_from_ai=base_tensor_for_update)
            self.check_game_status()
            if self.playing: self.next_turn()
        else:
            self.check_game_status()

    def check_game_status(self):
        game_over = False; message = ""
        if self.chess_board.is_checkmate():
            winner = "White" if self.chess_board.turn == chess.BLACK else "Black"
            message = f"Checkmate! {winner} wins."; game_over = True
        elif self.chess_board.is_stalemate(): message = "Stalemate! Draw."; game_over = True
        elif self.chess_board.is_insufficient_material(): message = "Draw: Insufficient material."; game_over = True
        elif self.chess_board.is_seventyfive_moves(): message = "Draw: 75-move rule."; game_over = True
        elif self.chess_board.is_fivefold_repetition(): message = "Draw: Fivefold repetition."; game_over = True
        
        if game_over: print(message); self.playing = False
        elif self.chess_board.is_check(): pass # print("Check!")

    def next_turn(self):
        self.next_player_ui = 'white' if self.chess_board.turn == chess.WHITE else 'black'
        ai_color_chess = chess.BLACK if self.ai_is_black else chess.WHITE
        if self.playing and self.chess_board.turn == ai_color_chess:
            pygame.time.set_timer(AI_MOVE_EVENT, 100, loops=1)

    def handle_ai_timer_event(self):
        if self.playing: self.trigger_ai_move()

    def show_bg(self, surface):
        theme = self.config.theme
        for r in range(ROWS):
            for c in range(COLS):
                color = theme.bg.light if (r + c) % 2 == 0 else theme.bg.dark
                pygame.draw.rect(surface, color, (c * SQSIZE, r * SQSIZE, SQSIZE, SQSIZE))
                if self.config.font:
                     alphacols = "abcdefgh"
                     if c == 0:
                         lbl = self.config.font.render(str(ROWS-r), 1, theme.bg.dark if r % 2 == 0 else theme.bg.light)
                         surface.blit(lbl, (5, 5 + r * SQSIZE))
                     if r == 7:
                         lbl = self.config.font.render(alphacols[c], 1, theme.bg.dark if (r + c) % 2 == 0 else theme.bg.light)
                         surface.blit(lbl, (c * SQSIZE + SQSIZE - self.config.font.size(alphacols[c])[0] - 5, HEIGHT - self.config.font.get_height() - 5))

    
    def show_moves(self, surface):
        if not (self.dragger.dragging and self.dragger.piece and self.dragger.initial_row is not None):
            return

        theme = self.config.theme
        light_move_color_rgb = getattr(theme.moves, 'light', (186, 202, 68))[:3]
        dark_move_color_rgb = getattr(theme.moves, 'dark', (100, 110, 64))[:3]
        move_alpha = 120 # Semi-transparent

        from_sq_idx_chess = self._ui_sq_to_chess_sq(self.dragger.initial_row, self.dragger.initial_col)
        if from_sq_idx_chess is None: return

        for move in self.chess_board.legal_moves:
            if move.from_square == from_sq_idx_chess:
                to_ui_pos = self._chess_sq_to_ui_sq(move.to_square)
                if to_ui_pos:
                    to_ui_row, to_ui_col = to_ui_pos
                    center_x = to_ui_col * SQSIZE + SQSIZE // 2
                    center_y = to_ui_row * SQSIZE + SQSIZE // 2
                    radius = SQSIZE // 7 # Smaller circles

                    is_capture = self.chess_board.is_capture(move)
                    base_color = dark_move_color_rgb if is_capture else light_move_color_rgb
                    
                    # Draw semi-transparent circle
                    temp_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surface, base_color + (move_alpha,), (radius, radius), radius)
                    surface.blit(temp_surface, (center_x - radius, center_y - radius))

    def show_last_move(self, surface):
        if not (self.board.last_move and isinstance(self.board.last_move, chess.Move)):
            return
        theme = self.config.theme
        from_ui = self._chess_sq_to_ui_sq(self.board.last_move.from_square)
        to_ui = self._chess_sq_to_ui_sq(self.board.last_move.to_square)
        
        highlight_surface = pygame.Surface((SQSIZE, SQSIZE), pygame.SRCALPHA)
        alpha = 100

        if from_ui:
            r, c = from_ui
            color = theme.trace.light if (r + c) % 2 == 0 else theme.trace.dark
            highlight_surface.fill(color + (alpha,))
            surface.blit(highlight_surface, (c * SQSIZE, r * SQSIZE))
        if to_ui:
            r, c = to_ui
            color = theme.trace.light if (r + c) % 2 == 0 else theme.trace.dark
            highlight_surface.fill(color + (alpha,))
            surface.blit(highlight_surface, (c * SQSIZE, r * SQSIZE))

    def show_hover(self, surface):
        if self.hovered_sqr_ui_pos:
            r, c = self.hovered_sqr_ui_pos
            hover_surface = pygame.Surface((SQSIZE, SQSIZE), pygame.SRCALPHA)
            hover_surface.fill((180, 180, 180, 70)) # Light gray, semi-transparent
            surface.blit(hover_surface, (c * SQSIZE, r * SQSIZE))

    def set_hover(self, motion_row_ui, motion_col_ui):
        if 0 <= motion_row_ui < ROWS and 0 <= motion_col_ui < COLS:
            self.hovered_sqr_ui_pos = (motion_row_ui, motion_col_ui)
        else: self.hovered_sqr_ui_pos = None

    def play_sound(self, captured=False):
        sound = self.capture_sound_obj if captured else self.move_sound_obj
        if sound:
            try: sound.play()
            except Exception as e: print(f"Error playing sound: {e}")
            
    def reset(self):
        self.__init__(self.config) # Re-initializes the game object
        print("--- Game Reset ---")

    def change_theme(self):
        self.config.change_theme()
# python-chess-UI/src/main.py
import pygame
import sys
import os
import chess # For chess.WHITE, chess.BLACK constants

from const import *
from game import Game
from config import Config
# When running as `python -m src.main` from `python-chess-UI` directory:
from ai_engine.ai_interface import load_model

AI_MOVE_EVENT = pygame.USEREVENT + 1

class Main:
    def __init__(self):
        pygame.init()
        self.config = Config()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Chess Game')
        
        print("--- Attempting to load AI model ---")
        load_model("model_checkpoint.pt") # Ensure model is in src/ai_engine/trained_models/
        print("--- Finished attempt to load AI model ---")
        self.game = Game(self.config)

    def mainloop(self):
        screen = self.screen
        game = self.game
        dragger = self.game.dragger # Get dragger from game instance
        
        clock = pygame.time.Clock()

        while True:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == AI_MOVE_EVENT:
                    game.handle_ai_timer_event()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left mouse button
                        # If a piece is already being dragged, a new click cancels it and starts fresh.
                        if dragger.dragging:
                            dragger.undrag_piece() # Cancel previous drag

                        dragger.update_mouse(event.pos) # Update mouse position for dragger
                        clicked_row_ui = dragger.mouseY // SQSIZE
                        clicked_col_ui = dragger.mouseX // SQSIZE

                        if not (0 <= clicked_row_ui < ROWS and 0 <= clicked_col_ui < COLS):
                            continue # Click outside board

                        # Determine human player's color based on AI config
                        human_color_chess = chess.WHITE if game.ai_is_black else chess.BLACK
                        
                        is_players_turn_on_board = (game.chess_board.turn == human_color_chess)

                        if game.playing and is_players_turn_on_board:
                            clicked_chess_sq_idx = game._ui_sq_to_chess_sq(clicked_row_ui, clicked_col_ui)
                            # clicked_chess_sq_idx check isn't strictly needed again due to bounds check

                            logical_piece_on_click = game.chess_board.piece_at(clicked_chess_sq_idx)

                            if logical_piece_on_click and logical_piece_on_click.color == human_color_chess:
                                # Get the UI Piece instance from the game's UI board
                                # This relies on game.show_pieces() correctly populating game.board.squares
                                ui_square_clicked = game.board.squares[clicked_row_ui][clicked_col_ui]
                                ui_piece_to_drag = ui_square_clicked.piece
                                
                                if ui_piece_to_drag: # This should be the correct Piece instance
                                    dragger.save_initial(event.pos, (clicked_row_ui, clicked_col_ui))
                                    dragger.drag_piece(ui_piece_to_drag)
                                # else:
                                    # This can happen if show_pieces hasn't run yet or there's a sync issue.
                                    # print(f"DEBUG Main: No UI piece to drag at ({clicked_row_ui},{clicked_col_ui}) even if logical piece exists.")


                elif event.type == pygame.MOUSEMOTION:
                    game.set_hover(event.pos[1] // SQSIZE, event.pos[0] // SQSIZE)
                    if dragger.dragging and dragger.piece:
                        dragger.update_mouse(event.pos)

                # python-chess-UI/src/main.py
# ... (other code is the same as your uploaded version) ...

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1: # Left mouse button release
                        print(f"DEBUG Main: MOUSEBUTTONUP event detected. Dragger.dragging={dragger.dragging}, Dragger.piece={dragger.piece}") # Add this
                        if dragger.dragging and dragger.piece: # Ensure a piece was being dragged
                            dragger.update_mouse(event.pos) # Final mouse position for drop
                            released_row_ui = dragger.mouseY // SQSIZE
                            released_col_ui = dragger.mouseX // SQSIZE
                            print(f"DEBUG Main: Attempting drop at UI ({released_row_ui},{released_col_ui}). Initial was ({dragger.initial_row},{dragger.initial_col})") # Add this

                            if dragger.initial_row is not None and dragger.initial_col is not None:
                                initial_pos_tuple = (dragger.initial_row, dragger.initial_col)
                                final_pos_tuple = (released_row_ui, released_col_ui)
                                print("DEBUG Main: Calling game.handle_player_drag_release()") # Add this
                                game.handle_player_drag_release(initial_pos_tuple, final_pos_tuple)
                                # game.handle_player_drag_release calls dragger.undrag_piece() and game.next_turn()
                            else:
                                print("DEBUG Main ERROR: MOUSEBUTTONUP while dragging, but dragger's initial_row/col was None. Undragging.") # Add this
                                dragger.undrag_piece() # Reset dragger state
                        # else: # Add this for clarity
                            # print(f"DEBUG Main: MOUSEBUTTONUP but not considered dragging a piece. dragger.dragging={dragger.dragging}, dragger.piece={dragger.piece}")

# ... (rest of main.py)
                          
                            # Mouse up but not dragging, or no piece was being dragged
                            # print("DEBUG Main: MOUSEBUTTONUP but not dragging a piece.")


                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t: game.change_theme()
                    if event.key == pygame.K_r: game.reset()
            
            # --- Drawing Order ---
            game.show_bg(screen)
            game.show_last_move(screen)
            game.show_hover(screen)
            if dragger.dragging and dragger.piece: # Show valid moves if dragging
                game.show_moves(screen)
            game.show_pieces(screen) # Draw all pieces on board (not the one being dragged)
            if dragger.dragging and dragger.piece: # Draw dragged piece on top
                dragger.update_blit(screen)
            
            pygame.display.flip()
            clock.tick(60) # Target 60 FPS

if __name__ == '__main__':
    # Ensure current working directory is the project root (python-chess-UI)
    # This helps with relative asset paths. Best practice: run with `python -m src.main`
    expected_cwd_end = "python-chess-UI" 
    if not os.getcwd().endswith(expected_cwd_end):
        # A simple check; might need adjustment based on your exact execution context
        if "src" in os.listdir("..") and os.path.basename(os.getcwd()) == "src": # If in src, go up
            os.chdir("..")
            print(f"Adjusted CWD to: {os.getcwd()} (for asset paths)")
        # else:
            # print(f"Warning: CWD is {os.getcwd()}. Asset paths might be incorrect if not running from '{expected_cwd_end}' or using `python -m src.main`.")

    main = Main()
    main.mainloop()
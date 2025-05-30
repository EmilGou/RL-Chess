# python-chess-UI/src/board.py

import pygame
import chess # For type hinting chess.Move


from const import * 
from square import Square 


class Board:
    """
    Manages the visual representation of the board grid (UI Squares) 
    and stores the last move made for highlighting purposes.
    Does NOT handle game logic, piece placement logic, or move validation.
    """

    def __init__(self):
        """Initializes the grid of UI Square objects."""
        # Create a 2D list of Square objects (from square.py)
        self.squares = [[Square(row, col) for col in range(COLS)] for row in range(ROWS)]
        
        # Stores the last move made (as a python-chess Move object) for highlighting
        self.last_move: chess.Move | None = None 
        
        # No longer adds pieces here; Game.show_pieces handles that
        # self._add_pieces('white') 
        # self._add_pieces('black')

    # The _create method might be redundant if __init__ creates squares directly
    # def _create(self):
    #     for row in range(ROWS):
    #         for col in range(COLS):
    #             self.squares[row][col] = Square(row, col)

    # --- REMOVED METHODS ---
    # _add_pieces: Piece placement is now driven by Game.show_pieces reading game.chess_board
    # move: The old move logic is removed. Game class handles pushing moves to game.chess_board.
    #       We only store last_move here now.
    # valid_move: Validation is done by game.chess_board.is_legal() in Game class.
    # check_promotion: Promotion logic is handled in Game class.
    # castling: Handled by python-chess.
    # set_true_en_passant: Handled by python-chess.
    # in_check: Handled by game.chess_board.is_check() etc. in Game class.
    # calc_moves: Move generation is done by game.chess_board.legal_moves in Game class.

    # --- UI Helper Methods (Can remain if used by Game or Main) ---

    def set_last_move(self, move: chess.Move | None):
        """Stores the last move made (a python-chess Move object)."""
        self.last_move = move

    # clear_moves and set_hover were likely related to the old move calculation display.
    # The new Game.show_moves handles highlighting dynamically.
    # These might be removable unless used for other UI purposes.
    
    # def clear_moves(self): 
    #     """Clears the possible_move flag on all squares (if used)."""
    #     for row in range(ROWS):
    #         for col in range(COLS):
    #             # Assuming Square object has this attribute from old system
    #             if hasattr(self.squares[row][col], 'possible_move'):
    #                  self.squares[row][col].possible_move = False

    # set_hover is now handled directly within the Game class (game.set_hover)
    # def set_hover(self, row, col): ... REMOVE ...


# python-chess-UI/src/dragger.py
import pygame
import os
from const import SQSIZE
from piece import Piece 

class Dragger:
    def __init__(self):
        self.piece: Piece | None = None    # The UI Piece instance being dragged
        self.dragging = False
        self.mouseX = 0
        self.mouseY = 0
        self.initial_pos_pixel = (0, 0)    # Mouse click position in pixels
        self.initial_row: int | None = None # UI row of the piece when drag started
        self.initial_col: int | None = None # UI col of the piece when drag started

    def update_mouse(self, pos: tuple[int, int]):
        self.mouseX, self.mouseY = pos

    def save_initial(self, pixel_pos: tuple[int, int], ui_row_col: tuple[int, int]):
        self.initial_pos_pixel = pixel_pos
        self.mouseX, self.mouseY = pixel_pos # Mouse is initially at the click position
        self.initial_row, self.initial_col = ui_row_col
        # print(f"Dragger: Saved initial - UI Square ({self.initial_row},{self.initial_col}), Mouse ({self.mouseX},{self.mouseY})")

    def drag_piece(self, piece_instance: Piece):
        if isinstance(piece_instance, Piece):
            self.piece = piece_instance
            self.dragging = True
            # print(f"Dragger: Started drag with piece: {self.piece}")
        else:
            print(f"Error Dragger: Attempted to drag invalid object: {piece_instance}")
            self.undrag_piece()


    def undrag_piece(self):
        # if self.dragging:
            # print(f"Dragger: Undragging piece: {self.piece}")
        self.piece = None
        self.dragging = False
        self.initial_row = None
        self.initial_col = None
        self.initial_pos_pixel = (0, 0)

    def update_blit(self, surface): # Draws the piece being dragged
        if not self.piece or not self.dragging:
            return

        try:
            texture_path = self.piece.texture
            if texture_path and os.path.exists(texture_path):
                img = pygame.image.load(texture_path)
                # Center the image on the current mouse position
                img_rect = img.get_rect(center=(self.mouseX, self.mouseY))
                surface.blit(img, img_rect)
            else:
                # Fallback if texture fails to load or path is bad
                # print(f"Warning Dragger: Texture not found or invalid for {self.piece}: {texture_path}")
                pygame.draw.circle(surface, (255, 0, 0), (self.mouseX, self.mouseY), SQSIZE // 3) # Red circle
        except Exception as e:
            print(f"Error in Dragger.update_blit for {self.piece}: {e}")
            # Fallback drawing on error
            pygame.draw.circle(surface, (255, 0, 0), (self.mouseX, self.mouseY), SQSIZE // 3)
            # self.undrag_piece() # Optionally reset drag on blit error, but might be too aggressive
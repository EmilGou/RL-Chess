
import os
import os
import pygame
from const import SQSIZE # Make sure SQSIZE is accessible

class Piece:
    def __init__(self, name: str, color: str, value: float = 0.0, texture_path: str | None = None):
        self.name = name  # e.g., 'pawn', 'rook'
        self.color = color  # 'white' or 'black'
        self.value = value # Optional
        self.texture = texture_path
        # Initialize texture_rect. Game.show_pieces will update its position.
        self.texture_rect = pygame.Rect(0, 0, SQSIZE, SQSIZE)

        if self.texture is None and self.name and self.color:
            self.set_default_texture() # Tries to set a default texture

    def set_default_texture(self, size=80):
        """
        Sets a default texture path.
        Assumes 'assets' is at the project root (python-chess-UI/assets).
        """
        # This path construction assumes you run `python -m src.main` from `python-chess-UI`
        base_asset_path = "assets" 
        if self.name and self.color:
            try:
                self.texture = os.path.join(
                    base_asset_path, 'images', f'imgs-{size}px', f'{self.color}_{self.name}.png'
                )
                # print(f"DEBUG Piece: Set texture to {self.texture}")
            except Exception as e:
                print(f"Error constructing texture path for {self.color} {self.name}: {e}")
                self.texture = None
        # else:
            # print(f"DEBUG Piece: Cannot set texture, name or color missing for {self.name} {self.color}")


    def __repr__(self):
        return f"<Piece {self.color} {self.name} at {self.texture_rect.topleft if self.texture_rect else 'N/A'}>"
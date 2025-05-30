# python-chess-UI/src/config.py
import pygame
import os
from color import Color
from theme import Theme

class Config:
    def __init__(self):
        self.assets_base_path = "assets" # Relative to python-chess-UI directory
        self.themes = []
        self._add_themes()
        self.idx = 0
        self.theme: Theme = self.themes[self.idx] if self.themes else None
        
        font_filename = "DejaVuSansMono.ttf"
        font_path = os.path.join(self.assets_base_path, "fonts", font_filename)
        try:
            if not pygame.font.get_init(): pygame.font.init()
            self.font = pygame.font.Font(font_path, 18)
            # print(f"Successfully loaded font: {font_path}")
        except Exception as e:
            print(f"Warning: Font '{font_path}' failed to load: {e}. Using SysFont.")
            self.font = pygame.font.SysFont('monospace', 18, bold=True)

        # --- AI Configuration --- 
        self.ai_player_is_black = True  # AI plays Black, Human plays White
        self.ai_color = 'black' if self.ai_player_is_black else 'white'
        
        # Sounds are better handled by the Game class using pygame.mixer.Sound directly
        # For simplicity, we'll let Game class handle it.
        # These paths will be used by Game class.
        self.move_sound_path = os.path.join(self.assets_base_path, 'sounds', 'move.wav')
        self.capture_sound_path = os.path.join(self.assets_base_path, 'sounds', 'capture.wav')

    def _add_themes(self):
        try:
            # Standard move highlight colors (adjust if you have specific theme values)
            default_light_move_color = (186, 202, 68, 150) # Added alpha for transparency
            default_dark_move_color = (100, 110, 64, 150)   # Added alpha

            green = Theme(
                light_bg_value=(234, 235, 200), dark_bg_value=(119, 154, 88),
                light_trace_value=(244, 247, 116), dark_trace_value=(172, 195, 51),
                light_moves_value=default_light_move_color,
                dark_moves_value=default_dark_move_color,
                piece_style_folder='imgs-80px', name='green'
            )
            brown = Theme(
                light_bg_value=(235, 209, 166), dark_bg_value=(165, 117, 80),
                light_trace_value=(245, 234, 100), dark_trace_value=(209, 185, 59),
                light_moves_value=default_light_move_color,
                dark_moves_value=default_dark_move_color,
                piece_style_folder='imgs-80px', name='brown'
            )
            blue = Theme(
                light_bg_value=(229, 228, 200), dark_bg_value=(60, 95, 135),
                light_trace_value=(123, 187, 227), dark_trace_value=(43, 119, 191),
                light_moves_value=default_light_move_color,
                dark_moves_value=default_dark_move_color,
                piece_style_folder='imgs-80px', name='blue'
            )
            self.themes = [green, brown, blue]
        except Exception as e:
             print(f"Error initializing themes: {e}")
             self.themes = []

    def change_theme(self):
        if not self.themes: return
        self.idx = (self.idx + 1) % len(self.themes)
        self.theme = self.themes[self.idx]
        # print(f"Theme changed to: {self.theme.name}")

    def get_texture_for_piece(self, piece_name: str, piece_color: str) -> str | None:
        if not self.theme or not hasattr(self.theme, 'piece_style_folder'):
             return None
        filename = f'{piece_color}_{piece_name}.png'
        full_path = os.path.join(self.assets_base_path, "images", self.theme.piece_style_folder, filename)
        return full_path